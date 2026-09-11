import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import scipy.stats as stats

BASE = Path("/srv/ids_research")
PREP = BASE / "data/preprocessed"
MDIR = BASE / "checkpoints/models"
RES_DAT = BASE / "results/domain_adaptation"

FLOW_DATASETS = [
    "CIC-IDS-2017", "UNSW-NB15",
    "TON-IoT-2021", "CICIoT2023",
]
SHORT = {
    "CIC-IDS-2017":"CIC-2017", "UNSW-NB15":"UNSW",
    "TON-IoT-2021":"TON-IoT",  "CICIoT2023":"CICIoT23",
}
FINAL_FEATURES = [
    "duration_ms","total_packets","total_bytes",
    "fwd_packets","fwd_bytes","bwd_packets","bwd_bytes",
    "bytes_per_sec","packets_per_sec","protocol","dst_port",
]
INPUT_DIM = len(FINAL_FEATURES)
MAX_TEST = 5000
RIDGE_LAMBDA = 0.1
COND_NUMBER_THRESHOLD = 50

class MLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=[256,128,64]):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(0.3)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def load_model(ds_name, device="cpu"):
    path = MDIR / f"{ds_name}_MLP.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    n_cls = ckpt["n_classes"]
    model = MLP(INPUT_DIM, n_cls)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt["classes"], n_cls

def load_test_data(ds_name, n_samples):
    df = pd.read_parquet(PREP/"test"/f"{ds_name}.parquet")
    if len(df) > n_samples:
        df, _ = train_test_split(
            df, train_size=n_samples,
            stratify=df["label_unified"], random_state=42)
    X = df[FINAL_FEATURES].values.astype(np.float32)
    y = df["label_unified"].values
    return X, y

def predict_proba(model, X, device="cpu", chunk=8192):
    model.eval().to(device)
    probs = []
    for i in range(0, len(X), chunk):
        Xb = torch.FloatTensor(X[i:i+chunk]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(Xb), dim=1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)

def estimate_confusion_matrix(model, ds_own, tgt_classes, device="cpu"):
    X, y_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder(); le.fit(tgt_classes)
    y = le.transform(y_raw)
    n_cls = len(tgt_classes)

    probs = predict_proba(model, X, device)
    preds = probs.argmax(1)

    C = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        mask = (y == i)
        if mask.sum() == 0:
            C[i, i] = 1.0
            continue
        for j in range(n_cls):
            C[j, i] = (preds[mask] == j).mean()
    return C

def bbse_correction(model, ds_own, X_foreign, tgt_classes,
                     device="cpu", ridge=RIDGE_LAMBDA):
    n_cls = len(tgt_classes)
    C = estimate_confusion_matrix(model, ds_own, tgt_classes, device)
    cond_num = np.linalg.cond(C)

    probs_foreign = predict_proba(model, X_foreign, device)
    preds_foreign = probs_foreign.argmax(1)
    q_tilde = np.array([(preds_foreign == j).mean()
                        for j in range(n_cls)])

    C_reg = C + ridge * np.eye(n_cls)
    try:
        q_est = np.linalg.solve(C_reg, q_tilde)
    except np.linalg.LinAlgError:
        q_est = np.linalg.lstsq(C_reg, q_tilde, rcond=None)[0]
    q_est = np.clip(q_est, 1e-6, None)
    q_est = q_est / q_est.sum()

    _, y_own_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder(); le.fit(tgt_classes)
    y_own = le.transform(y_own_raw)
    q_own = np.array([(y_own == c).mean() for c in range(n_cls)])
    q_own = np.clip(q_own, 1e-6, None)

    weights = q_est / q_own
    weights = np.clip(weights, 0.2, 5.0)

    probs_corrected = probs_foreign * weights[np.newaxis, :]
    preds_corrected = probs_corrected.argmax(1)

    return preds_corrected, weights, cond_num

def compute_error(preds, y_mapped):
    return float((preds != y_mapped).mean())

data_cache = {ds: load_test_data(ds, MAX_TEST)
              for ds in FLOW_DATASETS}

bbse_results = {}
for tgt_ds in FLOW_DATASETS:
    model, tgt_classes, n_cls = load_model(tgt_ds)
    le = LabelEncoder(); le.fit(tgt_classes)
    tgt_set = set(tgt_classes)

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds:
            continue

        X_src, y_src_raw = data_cache[src_ds]
        mask = np.array([l in tgt_set for l in y_src_raw])
        if mask.sum() == 0:
            continue
        y_mapped = le.transform(y_src_raw[mask])
        X_valid = X_src[mask]

        probs_before = predict_proba(model, X_valid)
        preds_before = probs_before.argmax(1)
        error_before = compute_error(preds_before, y_mapped)

        preds_after, weights, cond_num = bbse_correction(
            model, tgt_ds, X_valid, tgt_classes)
        error_after = compute_error(preds_after, y_mapped)

        improvement = error_before - error_after

        bbse_results[f"{src_ds}__{tgt_ds}"] = {
            "src": src_ds, "tgt": tgt_ds,
            "error_before": round(error_before, 4),
            "error_after_bbse": round(error_after, 4),
            "improvement": round(improvement, 4),
            "condition_number": round(float(cond_num), 2),
        }

    del model

with open(RES_DAT/"bbse_results.json", "w") as f:
    json.dump(bbse_results, f, indent=2)

improvements = [v["improvement"] for v in bbse_results.values()]
cond_numbers = [v["condition_number"] for v in bbse_results.values()]

print(f"Mean improvement: {np.mean(improvements):+.4f}")
print(f"Pct improved: {100*np.mean([i>0 for i in improvements]):.1f}%")

t_s, t_p = stats.ttest_rel(
    [v["error_before"] for v in bbse_results.values()],
    [v["error_after_bbse"] for v in bbse_results.values()])
print(f"Paired t-test: t={t_s:.4f} p={t_p:.4f}")

r_cond, p_cond = stats.pearsonr(cond_numbers, improvements)
print(f"Correlation condition_number vs improvement: "
      f"r={r_cond:+.3f} p={p_cond:.4f}")

print(f"Saved: {RES_DAT}/bbse_results.json")

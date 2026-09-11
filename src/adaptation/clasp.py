import gc
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm

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
ALL_MODELS = ["RF","XGB","MLP","CNN"]
MAX_TEST = 5000
MIN_CLASS_SAMPLES = 15
RIDGE_LAMBDA = 0.3
COND_NUMBER_THRESHOLD = 50
GATING_THRESHOLD = 0.65

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

class CNN1D(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,3,padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*input_dim,256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.conv(x).view(x.size(0),-1))

def load_model(ds_name, mn, device="cpu"):
    if mn in ["MLP","CNN"]:
        path = MDIR / f"{ds_name}_{mn}.pt"
        if not path.exists(): return None, None
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = (MLP(INPUT_DIM, ckpt["n_classes"]) if mn=="MLP"
                 else CNN1D(INPUT_DIM, ckpt["n_classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        return model, ckpt["classes"]
    else:
        path = MDIR / f"{ds_name}_{mn}.pkl"
        if not path.exists(): return None, None
        with open(path,"rb") as f:
            model = pickle.load(f)
        res_path = BASE/"results/baseline"/f"{ds_name}_{mn}.json"
        with open(res_path) as f:
            res = json.load(f)
        return model, res["classes"]

def load_test_data(ds_name, n_samples):
    df = pd.read_parquet(PREP/"test"/f"{ds_name}.parquet")
    if len(df) > n_samples:
        df, _ = train_test_split(
            df, train_size=n_samples,
            stratify=df["label_unified"], random_state=42)
    X = df[FINAL_FEATURES].values.astype(np.float32)
    y = df["label_unified"].values
    return X, y

def predict_proba_generic(model, mn, X, device="cpu", chunk=8192):
    if mn in ["MLP","CNN"]:
        model.eval().to(device)
        probs = []
        for i in range(0, len(X), chunk):
            Xb = torch.FloatTensor(X[i:i+chunk]).to(device)
            with torch.no_grad():
                p = torch.softmax(model(Xb), dim=1).cpu().numpy()
            probs.append(p)
        return np.vstack(probs)
    else:
        return model.predict_proba(X)

def matrix_sqrt(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T

def matrix_sqrt_inv(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(1.0/np.sqrt(eigval)) @ eigvec.T

def coral_1class(X_s, X_t, eps=1e-4):
    d = X_s.shape[1]
    cov_s = np.cov(X_s, rowvar=False) + eps*np.eye(d)
    cov_t = np.cov(X_t, rowvar=False) + eps*np.eye(d)
    mean_s, mean_t = X_s.mean(0), X_t.mean(0)
    X_white = (X_s - mean_s) @ matrix_sqrt_inv(cov_s)
    return (X_white @ matrix_sqrt(cov_t) + mean_t).astype(np.float32)

def class_conditional_coral(X_src, y_src, X_tgt, y_tgt):
    X_out = X_src.copy()
    shared = set(y_src) & set(y_tgt)
    global_t = coral_1class(X_src, X_tgt)
    for cls in shared:
        m_s, m_t = y_src==cls, y_tgt==cls
        if m_s.sum() < MIN_CLASS_SAMPLES or m_t.sum() < MIN_CLASS_SAMPLES:
            X_out[m_s] = global_t[m_s]
            continue
        X_out[m_s] = coral_1class(X_src[m_s], X_tgt[m_t])
    return X_out, len(shared)

def estimate_confusion_matrix(model, mn, ds_own, tgt_classes, device="cpu"):
    X, y_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder(); le.fit(tgt_classes)
    y = le.transform(y_raw)
    n_cls = len(tgt_classes)
    probs = predict_proba_generic(model, mn, X, device)
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

def bbse_reliable_correction(model, mn, ds_own, X_foreign, tgt_classes,
                              device="cpu", ridge=RIDGE_LAMBDA,
                              cond_threshold=COND_NUMBER_THRESHOLD):
    n_cls = len(tgt_classes)
    C = estimate_confusion_matrix(model, mn, ds_own, tgt_classes, device)
    cond_num = np.linalg.cond(C)
    if cond_num >= cond_threshold:
        return np.ones(n_cls), False, cond_num

    probs_foreign = predict_proba_generic(model, mn, X_foreign, device)
    preds_foreign = probs_foreign.argmax(1)
    q_tilde = np.array([(preds_foreign == j).mean() for j in range(n_cls)])

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
    return weights, True, cond_num

def compute_mce_from_preds(preds, y_mapped, n_cls):
    errs = [float((preds[y_mapped==c] != c).mean())
            for c in np.unique(y_mapped)]
    return round(np.mean(errs), 4)

def clasp_correct(model, mn, ds_own, X_foreign, y_foreign_raw,
                  X_tgt_own, y_tgt_own_raw, tgt_classes,
                  device="cpu", gating_threshold=GATING_THRESHOLD):
    n_cls = len(tgt_classes)
    le = LabelEncoder(); le.fit(tgt_classes)
    tgt_set = set(tgt_classes)
    mask = np.array([l in tgt_set for l in y_foreign_raw])
    if mask.sum() == 0:
        return None, None, None, {"gated_off": True, "reason": "no shared classes"}
    y_mapped = le.transform(y_foreign_raw[mask])
    X_valid = X_foreign[mask]

    probs_base = predict_proba_generic(model, mn, X_valid, device)
    preds_base = probs_base.argmax(1)
    mce_base = compute_mce_from_preds(preds_base, y_mapped, n_cls)

    if mce_base <= gating_threshold:
        return preds_base, mce_base, mce_base, {
            "gated_off": True,
            "reason": f"baseline MCE={mce_base:.3f} <= threshold"
        }

    X_coral, n_shared = class_conditional_coral(
        X_valid, y_foreign_raw[mask], X_tgt_own, y_tgt_own_raw)

    probs_coral = predict_proba_generic(model, mn, X_coral, device)

    weights, bbse_reliable, cond_num = bbse_reliable_correction(
        model, mn, ds_own, X_coral, tgt_classes, device)

    probs_final = probs_coral * weights[np.newaxis, :]
    preds_final = probs_final.argmax(1)
    mce_final = compute_mce_from_preds(preds_final, y_mapped, n_cls)

    info = {
        "gated_off": False,
        "n_shared_classes": n_shared,
        "bbse_applied": bbse_reliable,
        "condition_number": round(float(cond_num), 2),
    }
    return preds_final, mce_base, mce_final, info

data_cache = {ds: load_test_data(ds, MAX_TEST) for ds in FLOW_DATASETS}

clasp_all = {}
for mn in ALL_MODELS:
    for tgt_ds in FLOW_DATASETS:
        model, tgt_classes = load_model(tgt_ds, mn)
        if model is None:
            continue
        X_tgt_own, y_tgt_own_raw = data_cache[tgt_ds]

        for src_ds in FLOW_DATASETS:
            if src_ds == tgt_ds:
                continue
            X_src, y_src_raw = data_cache[src_ds]

            preds, mce_b, mce_a, info = clasp_correct(
                model, mn, tgt_ds, X_src, y_src_raw,
                X_tgt_own, y_tgt_own_raw, tgt_classes)

            if mce_b is None:
                continue

            improvement = mce_b - mce_a

            clasp_all[f"{mn}__{src_ds}__{tgt_ds}"] = {
                "model": mn, "src": src_ds, "tgt": tgt_ds,
                "mce_before": mce_b, "mce_after": mce_a,
                "improvement": round(improvement, 4),
                **info,
            }

        if mn in ["MLP","CNN"]:
            del model
    vals = [v["improvement"] for v in clasp_all.values() if v["model"]==mn]
    if vals:
        print(f"{mn}: mean delta={np.mean(vals):+.4f} "
              f"pct_improved={100*np.mean([i>0 for i in vals]):.1f}% "
              f"n={len(vals)}")

with open(RES_DAT/"clasp_all_models.json", "w") as f:
    json.dump(clasp_all, f, indent=2)

df = pd.DataFrame(list(clasp_all.values()))
df["pair_id"] = df["src"] + "__" + df["tgt"]

t_naive, p_naive = stats.ttest_rel(df["mce_before"], df["mce_after"])
print(f"Naive paired t-test: t={t_naive:.4f} p={p_naive:.6f}")

X1 = sm.add_constant(np.ones(len(df)))
m1 = sm.OLS(df["improvement"].values, X1).fit(
    cov_type="cluster", cov_kwds={"groups": df["pair_id"].values})
print(f"Cluster-robust mean improvement: "
      f"{m1.params[0]:+.4f}  p={m1.pvalues[0]:.6f}")

np.random.seed(42)
unique_pairs = df["pair_id"].unique()
boot_means = []
for _ in range(2000):
    sampled = np.random.choice(unique_pairs, len(unique_pairs), replace=True)
    bdf = pd.concat([df[df["pair_id"]==p] for p in sampled])
    boot_means.append(bdf["improvement"].mean())
ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
print(f"Cluster bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}]")
robust = ci_l > 0 or ci_h < 0
print(f"Robust: {robust}")

mce_c = df["mce_before"].values - df["mce_before"].mean()
X4 = sm.add_constant(mce_c)
m4 = sm.OLS(df["improvement"].values, X4).fit(
    cov_type="cluster", cov_kwds={"groups": df["pair_id"].values})
print(f"Regression-to-mean control: const={m4.params[0]:+.4f} "
      f"p={m4.pvalues[0]:.4f}, coef={m4.params[1]:+.4f} "
      f"p={m4.pvalues[1]:.4f}")

boot_const = []
for _ in range(2000):
    sampled = np.random.choice(unique_pairs, len(unique_pairs), replace=True)
    bdf = pd.concat([df[df["pair_id"]==p] for p in sampled])
    mc = bdf["mce_before"].values - bdf["mce_before"].mean()
    Xb = sm.add_constant(mc)
    try:
        mb = sm.OLS(bdf["improvement"].values, Xb).fit()
        boot_const.append(mb.params[0])
    except Exception:
        continue
ci_l2, ci_h2 = np.percentile(boot_const, [2.5, 97.5])
print(f"Bootstrap CI for const: [{ci_l2:+.4f}, {ci_h2:+.4f}]")
survives = ci_l2 > 0 or ci_h2 < 0
print(f"Survives regression-to-mean control: {survives}")

print("Per-model breakdown (Bonferroni alpha=0.0125):")
for mn in ALL_MODELS:
    sub = df[df["model"]==mn]
    if len(sub) < 3: continue
    t_m, p_m = stats.ttest_rel(sub["mce_before"], sub["mce_after"])
    sig = "passes" if p_m < 0.0125 else "fails"
    print(f"  {mn}: delta={sub['improvement'].mean():+.4f} "
          f"p={p_m:.4f}  {sig}")

n_gated_total = df["gated_off"].sum()
print(f"Gated off: {n_gated_total}/{len(df)}")

df.to_csv(RES_DAT/"clasp_full_diagnostic.csv", index=False)
print(f"Saved: {RES_DAT}/clasp_full_diagnostic.csv")

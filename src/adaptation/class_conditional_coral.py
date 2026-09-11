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
ALL_MODELS = ["RF","XGB","MLP","CNN"]
MAX_TEST = 5000
MIN_CLASS_SAMPLES = 15

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

def load_model(ds_name, mn):
    if mn in ["MLP","CNN"]:
        path = MDIR / f"{ds_name}_{mn}.pt"
        if not path.exists(): return None, None
        ckpt = torch.load(str(path), map_location="cpu",
                          weights_only=False)
        model = (MLP(INPUT_DIM, ckpt["n_classes"]) if mn=="MLP"
                 else CNN1D(INPUT_DIM, ckpt["n_classes"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval().cpu()
        return model, ckpt["classes"]
    else:
        path = MDIR / f"{ds_name}_{mn}.pkl"
        if not path.exists(): return None, None
        import pickle
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
    shared_classes = set(y_src) & set(y_tgt)
    global_transformed = coral_1class(X_src, X_tgt)
    for cls in shared_classes:
        mask_s = y_src == cls
        mask_t = y_tgt == cls
        n_s, n_t = mask_s.sum(), mask_t.sum()
        if n_s < MIN_CLASS_SAMPLES or n_t < MIN_CLASS_SAMPLES:
            X_out[mask_s] = global_transformed[mask_s]
            continue
        X_out[mask_s] = coral_1class(
            X_src[mask_s], X_tgt[mask_t])
    return X_out, len(shared_classes)

def compute_mce_from_preds(preds, y_mapped, n_cls):
    errs = [float((preds[y_mapped==c] != c).mean())
            for c in np.unique(y_mapped)]
    return round(np.mean(errs), 4)

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

data_cache = {ds: load_test_data(ds, MAX_TEST)
              for ds in FLOW_DATASETS}

results = {}
for mn in ALL_MODELS:
    for tgt_ds in FLOW_DATASETS:
        model, tgt_classes = load_model(tgt_ds, mn)
        if model is None:
            continue
        X_tgt_own, y_tgt_own = data_cache[tgt_ds]

        for src_ds in FLOW_DATASETS:
            if src_ds == tgt_ds:
                continue

            X_src, y_src_raw = data_cache[src_ds]
            X_src_cc, n_shared = class_conditional_coral(
                X_src, y_src_raw, X_tgt_own, y_tgt_own)

            le = LabelEncoder(); le.fit(tgt_classes)
            tgt_set = set(tgt_classes)
            mask = np.array([l in tgt_set for l in y_src_raw])
            if mask.sum() == 0:
                continue
            y_mapped = le.transform(y_src_raw[mask])

            probs_before = predict_proba_generic(
                model, mn, X_src[mask])
            probs_after = predict_proba_generic(
                model, mn, X_src_cc[mask])

            mce_before = compute_mce_from_preds(
                probs_before.argmax(1), y_mapped, len(tgt_classes))
            mce_after = compute_mce_from_preds(
                probs_after.argmax(1), y_mapped, len(tgt_classes))

            improvement = round(mce_before - mce_after, 4)

            results[f"{mn}__{src_ds}__{tgt_ds}"] = {
                "model": mn, "src": src_ds, "tgt": tgt_ds,
                "mce_before": mce_before, "mce_after": mce_after,
                "improvement": improvement,
                "n_shared_classes": n_shared,
            }

    vals = [v["improvement"] for v in results.values()
            if v["model"]==mn]
    if vals:
        print(f"{mn}: mean delta={np.mean(vals):+.4f} "
              f"pct_improved={100*np.mean([i>0 for i in vals]):.1f}%")

with open(RES_DAT/"cc_coral_mce_all_models.json","w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(list(results.values()))
df["pair_id"] = df["src"] + "__" + df["tgt"]

t_naive, p_naive = stats.ttest_rel(df["mce_before"], df["mce_after"])
print(f"Naive paired t-test: t={t_naive:.4f} p={p_naive:.6f}")

import statsmodels.api as sm
X = sm.add_constant(np.ones(len(df)))
ols_cluster = sm.OLS(df["improvement"].values, X).fit(
    cov_type="cluster", cov_kwds={"groups": df["pair_id"].values})
print(f"Cluster-robust mean improvement: "
      f"{ols_cluster.params[0]:.4f}  p={ols_cluster.pvalues[0]:.6f}")

np.random.seed(42)
unique_pairs = df["pair_id"].unique()
boot_means = []
for _ in range(2000):
    sampled = np.random.choice(unique_pairs, len(unique_pairs), replace=True)
    boot_df = pd.concat([df[df["pair_id"]==p] for p in sampled])
    boot_means.append(boot_df["improvement"].mean())
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Cluster bootstrap 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")

print("Per-model breakdown (Bonferroni alpha=0.0125):")
for mn in ALL_MODELS:
    sub = df[df["model"]==mn]
    if len(sub) < 3: continue
    t_m, p_m = stats.ttest_rel(sub["mce_before"], sub["mce_after"])
    sig = "passes" if p_m < 0.0125 else "fails"
    print(f"  {mn}: delta={sub['improvement'].mean():+.4f} "
          f"p={p_m:.4f}  {sig}")

df.to_csv(RES_DAT/"cc_coral_mce_diagnostic.csv", index=False)
print(f"Saved: {RES_DAT}/cc_coral_mce_diagnostic.csv")

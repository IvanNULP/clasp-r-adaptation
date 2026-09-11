import gc, os, ctypes, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE     = Path("/srv/ids_research")
PREP     = BASE / "data/preprocessed"
MDIR     = BASE / "checkpoints/models"
MDIR_DAT = BASE / "checkpoints/models_coral"
RES_DAT  = BASE / "results/domain_adaptation"
MDIR_DAT.mkdir(parents=True, exist_ok=True)
RES_DAT.mkdir(parents=True, exist_ok=True)

def ram_gb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

def flush():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass

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
DEVICES = {
    "CIC-IDS-2017":"cuda:0", "UNSW-NB15":"cuda:1",
    "TON-IoT-2021":"cuda:2", "CICIoT2023":"cuda:3",
}
MAX_TEST = 5000

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

def load_model_cpu(ds_name, model_name):
    path = MDIR / f"{ds_name}_{model_name}.pt"
    ckpt = torch.load(str(path), map_location="cpu",
                      weights_only=False)
    n_cls   = ckpt["n_classes"]
    classes = ckpt["classes"]
    model   = MLP(INPUT_DIM, n_cls)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cpu()
    return model, classes, n_cls

def load_test_data(ds_name, n_samples):
    df = pd.read_parquet(PREP/"test"/f"{ds_name}.parquet")
    if len(df) > n_samples:
        df, _ = train_test_split(
            df, train_size=n_samples,
            stratify=df["label_unified"], random_state=42)
    X = df[FINAL_FEATURES].values.astype(np.float32)
    y = df["label_unified"].values
    return X, y

def coral_transform(X_src, X_tgt, eps=1e-5):
    d = X_src.shape[1]
    cov_src = np.cov(X_src, rowvar=False) + eps*np.eye(d)
    cov_tgt = np.cov(X_tgt, rowvar=False) + eps*np.eye(d)

    def matrix_sqrt_inv(cov):
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, eps, None)
        return eigvec @ np.diag(1.0/np.sqrt(eigval)) @ eigvec.T

    def matrix_sqrt(cov):
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, eps, None)
        return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T

    src_mean = X_src.mean(axis=0)
    tgt_mean = X_tgt.mean(axis=0)

    X_whitened = (X_src - src_mean) @ matrix_sqrt_inv(cov_src)
    X_coral = X_whitened @ matrix_sqrt(cov_tgt) + tgt_mean

    return X_coral.astype(np.float32)

def compute_error(model, model_name, X, y_src_raw, tgt_classes):
    tgt_set = set(tgt_classes)
    mask = np.array([l in tgt_set for l in y_src_raw])
    n_valid = mask.sum()
    if n_valid == 0:
        return None, 0
    le = LabelEncoder(); le.fit(tgt_classes)
    y_mapped = le.transform(y_src_raw[mask])
    X_valid = X[mask]
    model.eval().cpu()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_valid)).argmax(1).numpy()
    return float((preds != y_mapped).mean()), int(n_valid)

with open(BASE/"results/adversarial"/
          "clean_transfer_baseline.json") as f:
    cter_baseline = json.load(f)

data_cache = {ds: load_test_data(ds, MAX_TEST)
              for ds in FLOW_DATASETS}

coral_results = {}
for tgt_ds in FLOW_DATASETS:
    model, tgt_classes, _ = load_model_cpu(tgt_ds, "MLP")
    X_tgt_own, _ = data_cache[tgt_ds]

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds:
            continue

        X_src, y_src_raw = data_cache[src_ds]
        X_src_coral = coral_transform(X_src, X_tgt_own)

        cter_key = f"{src_ds}__{tgt_ds}"
        cter_before = cter_baseline.get(
            cter_key, {}).get("cter", None)

        cter_after, n_valid = compute_error(
            model, "MLP", X_src_coral, y_src_raw, tgt_classes)

        if cter_after is None:
            continue

        improvement = (cter_before - cter_after
                       if cter_before is not None else None)

        coral_results[cter_key] = {
            "src": src_ds, "tgt": tgt_ds,
            "cter_before": cter_before,
            "cter_after_coral": round(cter_after, 4),
            "improvement": round(improvement, 4)
                           if improvement is not None else None,
            "n_valid": n_valid,
        }

    del model; flush()

with open(RES_DAT/"coral_results.json","w") as f:
    json.dump(coral_results, f, indent=2)

improvements = [v["improvement"] for v in coral_results.values()
                if v["improvement"] is not None]
cter_after_all = [v["cter_after_coral"]
                   for v in coral_results.values()]
cter_before_all = [v["cter_before"] for v in coral_results.values()
                   if v["cter_before"] is not None]

import scipy.stats as stats
t_s, t_p = stats.ttest_rel(cter_before_all, cter_after_all)

print(f"Mean improvement: {np.mean(improvements):+.4f}")
print(f"Paired t-test: t={t_s:.4f} p={t_p:.4f}")
print(f"Saved: {RES_DAT}/coral_results.json")

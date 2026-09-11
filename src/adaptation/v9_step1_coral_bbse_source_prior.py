import gc, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

BASE = Path("/srv/ids_research")
PREP = BASE / "data/preprocessed"
MDIR = BASE / "checkpoints/models"
RES_DAT = BASE / "results/domain_adaptation"

FLOW_DATASETS = ["CIC-IDS-2017","UNSW-NB15","TON-IoT-2021","CICIoT2023"]
SHORT = {"CIC-IDS-2017":"CIC-2017","UNSW-NB15":"UNSW",
         "TON-IoT-2021":"TON-IoT","CICIoT2023":"CICIoT23"}
FINAL_FEATURES = [
    "duration_ms","total_packets","total_bytes",
    "fwd_packets","fwd_bytes","bwd_packets","bwd_bytes",
    "bytes_per_sec","packets_per_sec","protocol","dst_port",
]
INPUT_DIM = len(FINAL_FEATURES)
FROZEN_C_SEED = 7777
MAX_FROZEN_C_SAMPLE = 8000

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

def load_mlp(ds_name, device="cpu"):
    path = MDIR / f"{ds_name}_MLP.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = MLP(INPUT_DIM, ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, list(ckpt["classes"])

def load_sklearn(ds_name, mn):
    path = MDIR / f"{ds_name}_{mn}.pkl"
    with open(path,"rb") as f:
        model = pickle.load(f)
    res_path = BASE/"results/baseline"/f"{ds_name}_{mn}.json"
    with open(res_path) as f:
        res = json.load(f)
    return model, list(res["classes"])

def verify_class_alignment(model_classes, tgt_classes):
    model_classes = list(model_classes); tgt_classes = list(tgt_classes)
    if set(model_classes) != set(tgt_classes):
        raise ValueError("Class sets differ")
    le = LabelEncoder(); le.fit(tgt_classes)
    target_order = list(le.classes_)
    target_index = {c: i for i, c in enumerate(target_order)}
    remap = np.array([target_index[c] for c in model_classes], dtype=int)
    return remap, target_order

def remap_proba(probs, remap, n_classes):
    out = np.zeros((len(probs), n_classes))
    for model_col, target_col in enumerate(remap):
        out[:, target_col] = probs[:, model_col]
    return out

def predict_proba_mlp(model, X, device="cpu", chunk=8192):
    model.eval().to(device)
    probs = []
    for i in range(0, len(X), chunk):
        Xb = torch.FloatTensor(X[i:i+chunk]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(Xb), dim=1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)

def predict_proba_aligned(model, mn, X, model_classes, tgt_classes, device="cpu"):
    raw = predict_proba_mlp(model, X, device) if mn in ["MLP","CNN"] \
          else model.predict_proba(X)
    remap, target_order = verify_class_alignment(model_classes, tgt_classes)
    return remap_proba(raw, remap, len(target_order))

print("="*90)
print("CLASP-R v9 — Крок 1 (доповнено джерелом source_prior)")
print("="*90)

with open(RES_DAT/"v8_fixed_eval_and_ref_pools.pkl", "rb") as f:
    v8_artifacts = pickle.load(f)
v8_fixed_eval_sets  = v8_artifacts["fixed_eval_sets"]
v8_target_ref_pools = v8_artifacts["target_ref_pools"]

manifest_df = pd.read_csv(RES_DAT/"v8_instance_manifest.csv")
TARGET_REF_SEEDS = sorted(manifest_df["target_ref_seed"].unique().tolist())

frozen_matrices = {}

for tgt_ds in FLOW_DATASETS:
    print(f"\n{'─'*70}\nDATASET: {SHORT[tgt_ds]}")

    df_full = pd.read_parquet(PREP/"test"/f"{tgt_ds}.parquet")
    df_full = df_full.reset_index(drop=True)

    df_eval_actual = v8_fixed_eval_sets[tgt_ds]
    eval_values_set = set(map(
        tuple, df_eval_actual[FINAL_FEATURES].values.round(6).tolist()))

    v8_pool_excl_eval = v8_target_ref_pools[tgt_ds]

    all_target_ref_values = set()
    for ref_seed in TARGET_REF_SEEDS:
        n_ref = min(5000, len(v8_pool_excl_eval))
        df_ref, _ = train_test_split(
            v8_pool_excl_eval, train_size=n_ref,
            stratify=v8_pool_excl_eval["label_unified"], random_state=int(ref_seed))
        ref_values = set(map(
            tuple, df_ref[FINAL_FEATURES].values.round(6).tolist()))
        all_target_ref_values.update(ref_values)

    df_full_values = list(map(
        tuple, df_full[FINAL_FEATURES].values.round(6).tolist()))
    exclusion_set = eval_values_set | all_target_ref_values
    keep_mask = np.array([v not in exclusion_set for v in df_full_values])
    frozen_pool = df_full[keep_mask].reset_index(drop=True)

    n_frozen = min(MAX_FROZEN_C_SAMPLE, len(frozen_pool))
    df_frozen, _ = train_test_split(
        frozen_pool, train_size=n_frozen,
        stratify=frozen_pool["label_unified"], random_state=FROZEN_C_SEED)
    df_frozen = df_frozen.reset_index(drop=True)

    X_frozen = df_frozen[FINAL_FEATURES].values.astype(np.float32)
    y_frozen_raw = df_frozen["label_unified"].values

    frozen_values_set = set(map(tuple, X_frozen.round(6).tolist()))
    overlap_eval = len(frozen_values_set & eval_values_set)
    overlap_ref  = len(frozen_values_set & all_target_ref_values)
    assert overlap_eval == 0 and overlap_ref == 0, "Overlap detected!"
    print(f"  ✓ Non-overlap confirmed (eval={overlap_eval}, ref={overlap_ref})")

    for mn in ["MLP","RF","XGB"]:
        if mn == "MLP":
            model, model_classes = load_mlp(tgt_ds)
        else:
            model, model_classes = load_sklearn(tgt_ds, mn)

        _, canonical_classes = verify_class_alignment(model_classes, model_classes)
        class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
        n_cls = len(canonical_classes)

        mask = np.array([x in class_to_idx for x in y_frozen_raw], dtype=bool)
        X_valid = X_frozen[mask]
        y_valid_raw = y_frozen_raw[mask]
        y_valid = np.array([class_to_idx[x] for x in y_valid_raw], dtype=int)

        probs = predict_proba_aligned(
            model, mn, X_valid, model_classes, canonical_classes)
        preds = probs.argmax(axis=1)

        C = np.zeros((n_cls, n_cls), dtype=np.float64)
        for true_idx in range(n_cls):
            m = y_valid == true_idx
            if m.sum() == 0:
                C[true_idx, true_idx] = 1.0
            else:
                counts = np.bincount(preds[m], minlength=n_cls)
                C[:, true_idx] = counts / counts.sum()

        cond_num = np.linalg.cond(C)

        class_counts = np.bincount(y_valid, minlength=n_cls)
        source_prior = class_counts / class_counts.sum()

        key = f"{tgt_ds}__{mn}"
        frozen_matrices[key] = {
            "dataset": tgt_ds, "model": mn,
            "classes": canonical_classes,
            "class_order": "canonical_alphabetical",
            "C": C,
            "condition_number": float(cond_num),
            "n_samples": int(len(y_valid)),
            "calibration_class_counts": class_counts.tolist(),
            "source_prior": source_prior.tolist(),  # ── НОВЕ
            "frozen_c_seed": FROZEN_C_SEED,
            "calibration_subset_type": "BBSE calibration subset",
        }

        print(f"    {mn:5s}: cond(C)={cond_num:.2f}, "
              f"source_prior={np.round(source_prior,3).tolist()}")

        if mn == "MLP":
            del model

    del df_full, frozen_pool, df_frozen
    gc.collect()

with open(RES_DAT/"v9_frozen_confusion_matrices.pkl", "wb") as f:
    pickle.dump(frozen_matrices, f)

print(f"\n✓ Крок 1 (доповнено) завершено. Saved.")
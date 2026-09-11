import gc, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from scipy.spatial import cKDTree

BASE = Path("/srv/ids_research")
PREP = BASE / "data/preprocessed"
MDIR = BASE / "checkpoints/models"
RES_DAT = BASE / "results/domain_adaptation"

FLOW_DATASETS = ["CIC-IDS-2017","UNSW-NB15","TON-IoT-2021","CICIoT2023"]
SHORT = {"CIC-IDS-2017":"CIC-2017","UNSW-NB15":"UNSW",
         "TON-IoT-2021":"TON-IoT","CICIoT2023":"CICIoT23"}

FEATURES_CONTINUOUS = [
    "duration_ms","total_packets","total_bytes",
    "fwd_packets","fwd_bytes","bwd_packets","bwd_bytes",
    "bytes_per_sec","packets_per_sec",
]
FEATURES_DISCRETE = ["protocol","dst_port"]
FINAL_FEATURES = FEATURES_CONTINUOUS + FEATURES_DISCRETE
CONT_IDX = [FINAL_FEATURES.index(f) for f in FEATURES_CONTINUOUS]
DISC_IDX = [FINAL_FEATURES.index(f) for f in FEATURES_DISCRETE]

INPUT_DIM = len(FINAL_FEATURES)
MAX_FOREIGN_SAMPLE = 5000
EPS = 1e-9

POT_N_SUBSAMPLE = 500     # той самий, що й UOT
POT_MASS_FRACTION = 0.8   # m — частка маси для транспортування, фіксовано
POT_SEED = 5555           # той самий, що й UOT

BBSE_RIDGE = 1e-6
BBSE_PINV_RCOND = 1e-8
WEIGHT_CLIP_LOW, WEIGHT_CLIP_HIGH = 0.5, 2.0

with open(RES_DAT/"v9_frozen_confusion_matrices.pkl", "rb") as f:
    frozen_matrices = pickle.load(f)
audit_df_stability = pd.read_csv(RES_DAT/"v9_bbse_stability_audit.csv")
audit_lookup = {row["key"]: row["recommended_bbse_action"]
                for _, row in audit_df_stability.iterrows()}

manifest_df = pd.read_csv(RES_DAT/"v8_instance_manifest.csv")
with open(RES_DAT/"v8_fixed_eval_and_ref_pools.pkl", "rb") as f:
    v8_artifacts = pickle.load(f)
target_ref_pools = v8_artifacts["target_ref_pools"]

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
    import pickle as pkl
    path = MDIR / f"{ds_name}_{mn}.pkl"
    with open(path,"rb") as f:
        model = pkl.load(f)
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

def standardize_pair(Xs, Xt):
    mean_s, std_s = Xs.mean(0), Xs.std(0) + 1e-8
    return (Xs - mean_s) / std_s, (Xt - mean_s) / std_s, mean_s, std_s

def partial_ot_transport(X_source, X_target, n_sub=POT_N_SUBSAMPLE,
                          m=POT_MASS_FRACTION, seed=POT_SEED):
    """
    Partial OT на continuous features.
    "Partial OT was solved on a fixed 500-sample subsample with
    mass fraction m=0.8, followed by barycentric projection and
    nearest-neighbor extension to the remaining source samples."
    """
    X_adapted = X_source.copy()
    Xs_cont = X_source[:, CONT_IDX].astype(np.float64)
    Xt_cont = X_target[:, CONT_IDX].astype(np.float64)

    Xs_std, Xt_std, mean_s, std_s = standardize_pair(Xs_cont, Xt_cont)

    rng = np.random.RandomState(seed)
    idx_s = rng.choice(len(Xs_std), min(n_sub, len(Xs_std)), replace=False)
    idx_t = rng.choice(len(Xt_std), min(n_sub, len(Xt_std)), replace=False)

    Xs_sub = Xs_std[idx_s]
    Xt_sub = Xt_std[idx_t]

    a = np.ones(len(Xs_sub)) / len(Xs_sub)
    b = np.ones(len(Xt_sub)) / len(Xt_sub)
    M = ot.dist(Xs_sub, Xt_sub)
    M /= (M.max() + 1e-12)

    diag = {"total_mass": np.nan, "mean_row_mass": np.nan,
            "min_row_mass": np.nan, "max_row_mass": np.nan,
            "n_unmatched_source": np.nan}

    try:
        G = ot.partial.partial_wasserstein(a, b, M, m=m)
    except Exception:
        return X_source.astype(np.float32), False, diag

    row_mass = G.sum(axis=1)
    diag = {
        "total_mass": float(G.sum()),
        "mean_row_mass": float(row_mass.mean()),
        "min_row_mass": float(row_mass.min()),
        "max_row_mass": float(row_mass.max()),
        "n_unmatched_source": int((row_mass < 1e-9).sum()),
    }

    row_sums = row_mass.reshape(-1, 1).copy()
    unmatched_mask = (row_sums < 1e-9).flatten()
    row_sums[row_sums < 1e-9] = 1.0
    Xs_sub_mapped = (G @ Xt_sub) / row_sums
    Xs_sub_mapped[unmatched_mask] = Xs_sub[unmatched_mask]  # identity fallback

    tree = cKDTree(Xs_sub)
    _, nn_idx = tree.query(Xs_std, k=1)
    Xs_all_mapped_std = Xs_sub_mapped[nn_idx]

    Xs_all_mapped = Xs_all_mapped_std * std_s + mean_s
    X_adapted[:, CONT_IDX] = Xs_all_mapped
    return X_adapted.astype(np.float32), True, diag

def unsupervised_bbse(probs_foreign_all, frozen_entry, stability_action,
                       ridge=BBSE_RIDGE, pinv_rcond=BBSE_PINV_RCOND):
    C = frozen_entry["C"]
    n_cls = len(frozen_entry["classes"])
    p_source = np.array(frozen_entry["source_prior"])

    if stability_action == "BBSE_DISABLED":
        return np.ones(n_cls), stability_action

    preds_f = probs_foreign_all.argmax(1)
    q_tilde = np.array([(preds_f == j).mean() for j in range(n_cls)])

    if stability_action == "RIDGE_SOLVE":
        C_reg = C + ridge * np.eye(n_cls)
        q_est = np.linalg.solve(C_reg, q_tilde)
    else:
        q_est = np.linalg.pinv(C, rcond=pinv_rcond) @ q_tilde

    q_est = np.clip(q_est, 0, 1)
    q_sum = q_est.sum()
    q_est = q_est / q_sum if q_sum > 1e-9 else np.ones(n_cls) / n_cls

    weights = q_est / np.maximum(p_source, EPS)
    weights = np.clip(weights, WEIGHT_CLIP_LOW, WEIGHT_CLIP_HIGH)
    return weights, stability_action

def compute_macro_error(preds, y_mapped, n_cls):
    errs = [float((preds[y_mapped==c]!=c).mean()) for c in np.unique(y_mapped)]
    return float(np.mean(errs)) if errs else np.nan

def mean_ks_shift(X_a, X_b):
    return float(np.mean([ks_2samp(X_a[:,j], X_b[:,j])[0] for j in range(X_a.shape[1])]))

print("="*90)
print("CLASP-R — Step 4D: Partial OT Benchmark (ФІНАЛЬНИЙ transport-метод)")
print(f"POT params: n_sub={POT_N_SUBSAMPLE}, m={POT_MASS_FRACTION}, seed={POT_SEED}")
print("МЕТОД: subsampled Partial OT + barycentric projection + NN extension")
print("Unmatched source points (row_mass≈0) залишаються identity")
print(f"Continuous (POT applied): {FEATURES_CONTINUOUS}")
print(f"Discrete (identity): {FEATURES_DISCRETE}")
print("="*90)

results_path = RES_DAT / "v13_partial_ot_240.csv"
existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
done_keys = set(existing["instance_key"]) if len(existing) else set()
print(f"Checkpoint: {len(done_keys)} instances вже оброблено")

all_records = existing.to_dict("records") if len(existing) else []
t_start = time.time()
n_fallback_total = 0

for tgt_ds in FLOW_DATASETS:
    mlp_model, mlp_classes = load_mlp(tgt_ds)
    rf_model, rf_classes = load_sklearn(tgt_ds, "RF")
    xgb_model, xgb_classes = load_sklearn(tgt_ds, "XGB")
    pool_ref = target_ref_pools[tgt_ds]

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds: continue
        pair_manifest = manifest_df[
            (manifest_df["src"]==src_ds) & (manifest_df["tgt"]==tgt_ds)]

        df_src_full = pd.read_parquet(PREP/"test"/f"{src_ds}.parquet")
        df_src_full = df_src_full.reset_index(drop=True)

        for _, row in pair_manifest.iterrows():
            instance_key = f"{SHORT[src_ds]}__{SHORT[tgt_ds]}__k{row['replication_idx']}"
            if instance_key in done_keys:
                continue

            foreign_seed = int(row["foreign_seed"])
            target_ref_seed = int(row["target_ref_seed"])

            df_foreign = df_src_full
            if len(df_foreign) > MAX_FOREIGN_SAMPLE:
                df_foreign, _ = train_test_split(
                    df_foreign, train_size=MAX_FOREIGN_SAMPLE,
                    stratify=df_foreign["label_unified"], random_state=foreign_seed)
            X_foreign = df_foreign[FINAL_FEATURES].values.astype(np.float32)
            y_foreign_raw = df_foreign["label_unified"].values

            tgt_set = set(mlp_classes)
            eval_mask = np.array([x in tgt_set for x in y_foreign_raw])
            if eval_mask.sum() == 0:
                continue
            le = LabelEncoder(); le.fit(mlp_classes)
            y_mapped = le.transform(y_foreign_raw[eval_mask])

            n_ref = min(5000, len(pool_ref))
            df_ref, _ = train_test_split(
                pool_ref, train_size=n_ref,
                stratify=pool_ref["label_unified"], random_state=target_ref_seed)
            X_tgt_ref = df_ref[FINAL_FEATURES].values.astype(np.float32)

            ks_shift = mean_ks_shift(X_tgt_ref, X_foreign)

            rec = {
                "instance_key": instance_key,
                "pair_id": f"{SHORT[src_ds]}__{SHORT[tgt_ds]}",
                "src": SHORT[src_ds], "tgt": SHORT[tgt_ds],
                "replication_idx": int(row["replication_idx"]),
                "foreign_seed": foreign_seed, "target_ref_seed": target_ref_seed,
                "ks_shift": round(ks_shift, 4),
            }

            X_pot_all, pot_converged, mass_diag = partial_ot_transport(
                X_foreign, X_tgt_ref)
            rec["pot_converged"] = pot_converged
            rec["pot_total_mass"] = mass_diag["total_mass"]
            rec["pot_mean_row_mass"] = mass_diag["mean_row_mass"]
            rec["pot_n_unmatched_source"] = mass_diag["n_unmatched_source"]
            if not pot_converged:
                n_fallback_total += 1

            for mn, model, classes in [("MLP", mlp_model, mlp_classes),
                                        ("RF", rf_model, rf_classes),
                                        ("XGB", xgb_model, xgb_classes)]:
                probs_base_all = predict_proba_aligned(
                    model, mn, X_foreign, classes, mlp_classes)
                probs_pot_all = predict_proba_aligned(
                    model, mn, X_pot_all, classes, mlp_classes)

                frozen_key = f"{tgt_ds}__{mn}"
                frozen_entry = frozen_matrices[frozen_key]
                stability_action = audit_lookup[frozen_key]

                weights, applied = unsupervised_bbse(
                    probs_pot_all, frozen_entry, stability_action)
                probs_pot_bbse_all = probs_pot_all * weights[np.newaxis, :]

                preds_base    = probs_base_all[eval_mask].argmax(1)
                preds_pot     = probs_pot_all[eval_mask].argmax(1)
                preds_pot_bbse= probs_pot_bbse_all[eval_mask].argmax(1)

                rec[f"{mn}_BASE_error"] = compute_macro_error(preds_base, y_mapped, len(mlp_classes))
                rec[f"{mn}_POT_error"] = compute_macro_error(preds_pot, y_mapped, len(mlp_classes))
                rec[f"{mn}_POT_BBSE_error"] = compute_macro_error(preds_pot_bbse, y_mapped, len(mlp_classes))
                rec[f"{mn}_bbse_policy"] = applied

            all_records.append(rec)

        elapsed = time.time() - t_start
        print(f"  {SHORT[src_ds]:10s}→{SHORT[tgt_ds]:10s}: "
              f"total={len(all_records)}/240 | elapsed={elapsed/60:.1f}min | "
              f"fallbacks={n_fallback_total}")
        pd.DataFrame(all_records).to_csv(results_path, index=False)

    del mlp_model, rf_model, xgb_model; gc.collect()

print(f"\n✓ Step 4D (Partial OT) завершено: {len(all_records)} instances")
print(f"  Fallback count: {n_fallback_total}/240")

pot_df = pd.DataFrame(all_records)
print(f"\nMASS DIAGNOSTICS:")
print(f"  Mean total_mass (з m={POT_MASS_FRACTION}): {pot_df['pot_total_mass'].mean():.4f}")
print(f"  Mean n_unmatched_source (з 500): {pot_df['pot_n_unmatched_source'].mean():.1f}")
print(f"\nSaved: {results_path}")

v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
print(f"\n{'='*90}")
print("ПОПЕРЕДНЄ ПОРІВНЯННЯ: BASE vs Global CORAL+BBSE vs Partial OT")
print(f"{'='*90}")
for mn in ["MLP","RF","XGB"]:
    base = v9_df[f"{mn}_BASE_error"].mean()
    g_clasp = v9_df[f"{mn}_CLASP_error"].mean()
    pot = pot_df[f"{mn}_POT_error"].mean()
    pot_bbse = pot_df[f"{mn}_POT_BBSE_error"].mean()
    print(f"\n{mn}: BASE={base:.4f}  CORAL+BBSE={g_clasp:.4f}  "
          f"POT={pot:.4f} (Δ={pot-base:+.4f})  POT+BBSE={pot_bbse:.4f} (Δ={pot_bbse-base:+.4f})")
import gc, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, spearmanr
import statsmodels.api as sm

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
MAX_FOREIGN_SAMPLE = 5000
EPS = 1e-9
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

v9_gt = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")

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

def matrix_sqrt(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T

def matrix_sqrt_inv(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(1.0/np.sqrt(eigval)) @ eigvec.T

def unsupervised_coral(X_source, X_target, eps=1e-4):
    d = X_source.shape[1]
    cov_s = np.cov(X_source, rowvar=False) + eps*np.eye(d)
    cov_t = np.cov(X_target, rowvar=False) + eps*np.eye(d)
    mean_s, mean_t = X_source.mean(0), X_target.mean(0)
    X_white = (X_source - mean_s) @ matrix_sqrt_inv(cov_s)
    return (X_white @ matrix_sqrt(cov_t) + mean_t).astype(np.float32)

def unsupervised_bbse(probs_foreign_all, frozen_entry, stability_action,
                       ridge=BBSE_RIDGE, pinv_rcond=BBSE_PINV_RCOND):
    C = frozen_entry["C"]
    n_cls = len(frozen_entry["classes"])
    p_source = np.array(frozen_entry["source_prior"])

    if stability_action == "BBSE_DISABLED":
        return np.ones(n_cls), None, stability_action

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
    return weights, q_est, stability_action

def row_entropy(probs):
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def mean_ks_shift(X_a, X_b):
    return float(np.mean([ks_2samp(X_a[:,j], X_b[:,j])[0] for j in range(X_a.shape[1])]))

print("="*90)
print("CLASP-R v9 — Крок 3: Reliability Signal Audit")
print("Обчислення label-free features + adaptation_margin (неперервний)")
print("="*90)

reliability_path = RES_DAT / "v9_reliability_features_240.csv"
existing = pd.read_csv(reliability_path) if reliability_path.exists() else pd.DataFrame()
done_keys = set(existing["instance_key"]) if len(existing) else set()

all_records = existing.to_dict("records") if len(existing) else []
t_start = time.time()

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

            n_ref = min(5000, len(pool_ref))
            df_ref, _ = train_test_split(
                pool_ref, train_size=n_ref,
                stratify=pool_ref["label_unified"], random_state=target_ref_seed)
            X_tgt_ref = df_ref[FINAL_FEATURES].values.astype(np.float32)

            X_coral_all = unsupervised_coral(X_foreign, X_tgt_ref)
            ks_shift = mean_ks_shift(X_tgt_ref, X_foreign)

            rec = {
                "instance_key": instance_key,
                "pair_id": f"{SHORT[src_ds]}__{SHORT[tgt_ds]}",
                "src": SHORT[src_ds], "tgt": SHORT[tgt_ds],
                "replication_idx": int(row["replication_idx"]),
                "ks_shift": round(ks_shift, 4),
            }

            for mn, model, classes in [("MLP", mlp_model, mlp_classes),
                                        ("RF", rf_model, rf_classes),
                                        ("XGB", xgb_model, xgb_classes)]:
                probs_base_all = predict_proba_aligned(
                    model, mn, X_foreign, classes, mlp_classes)
                probs_coral_all = predict_proba_aligned(
                    model, mn, X_coral_all, classes, mlp_classes)

                preds_base = probs_base_all.argmax(1)
                preds_coral = probs_coral_all.argmax(1)

                prediction_flip_rate = float((preds_base != preds_coral).mean())
                probability_shift = float(
                    np.mean(np.abs(probs_base_all - probs_coral_all)))

                conf_base = probs_base_all.max(1)
                conf_coral = probs_coral_all.max(1)
                confidence_change = float(np.mean(conf_coral - conf_base))

                ent_base = row_entropy(probs_base_all)
                ent_coral = row_entropy(probs_coral_all)
                entropy_change = float(np.mean(ent_coral - ent_base))

                frozen_key = f"{tgt_ds}__{mn}"
                frozen_entry = frozen_matrices[frozen_key]
                stability_action = audit_lookup[frozen_key]

                weights, q_est, applied = unsupervised_bbse(
                    probs_coral_all, frozen_entry, stability_action)

                bbse_weight_magnitude = float(np.mean(np.abs(np.log(weights))))
                if q_est is not None:
                    n_cls_local = len(q_est)
                    uniform = np.ones(n_cls_local) / n_cls_local
                    bbse_prior_extremity = float(np.max(np.abs(q_est - uniform)))
                else:
                    bbse_prior_extremity = 0.0

                rec[f"{mn}_prediction_flip_rate"] = prediction_flip_rate
                rec[f"{mn}_probability_shift"] = probability_shift
                rec[f"{mn}_confidence_change"] = confidence_change
                rec[f"{mn}_entropy_change"] = entropy_change
                rec[f"{mn}_bbse_action"] = applied
                rec[f"{mn}_bbse_weight_magnitude"] = bbse_weight_magnitude
                rec[f"{mn}_bbse_prior_extremity"] = bbse_prior_extremity

            all_records.append(rec)

        elapsed = time.time() - t_start
        print(f"  {SHORT[src_ds]:10s}→{SHORT[tgt_ds]:10s}: total={len(all_records)}/240 "
              f"| elapsed={elapsed/60:.1f}min")
        pd.DataFrame(all_records).to_csv(reliability_path, index=False)

    del mlp_model, rf_model, xgb_model; gc.collect()

rel_df = pd.DataFrame(all_records)

merged = rel_df.merge(
    v9_gt[["instance_key"] + [f"{mn}_{s}_error" for mn in ["MLP","RF","XGB"]
                                for s in ["BASE","CORAL","CLASP"]]],
    on="instance_key")

for mn in ["MLP","RF","XGB"]:
    merged[f"{mn}_adaptation_margin"] = (
        merged[f"{mn}_BASE_error"] - merged[f"{mn}_CLASP_error"])

merged.to_csv(RES_DAT/"v9_reliability_merged_240.csv", index=False)

print(f"\n{'='*100}")
print("АНАЛІЗ A: Pair-level кореляція label-free features з adaptation_margin")
print(f"{'='*100}")

feature_names = ["ks_shift"]
model_specific_feats = ["prediction_flip_rate","probability_shift",
                         "confidence_change","entropy_change",
                         "bbse_weight_magnitude","bbse_prior_extremity"]

rng = np.random.default_rng(42)
pair_ids = merged["pair_id"].unique()

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn} — reliability signal audit\n{'='*90}")

    all_feats = feature_names + [f"{mn}_{f}" for f in model_specific_feats]
    target_col = f"{mn}_adaptation_margin"

    print(f"\n{'Feature':32s} {'Spearman r':>12s} {'p (naive)':>10s} "
          f"{'ClusterRobust p':>16s}")
    print("─"*75)

    for feat in all_feats:
        r_s, p_s = spearmanr(merged[feat], merged[target_col])

        X_reg = sm.add_constant(merged[feat].values)
        y_reg = merged[target_col].values
        model_reg = sm.OLS(y_reg, X_reg).fit(
            cov_type="cluster", cov_kwds={"groups": merged["pair_id"].values})
        p_cluster = model_reg.pvalues[1]

        flag = "→ дивитись уважніше" if abs(r_s) > 0.3 and p_cluster < 0.10 else ""
        print(f"{feat:32s} {r_s:>+12.3f} {p_s:>10.4f} {p_cluster:>16.4f} {flag}")

print(f"\nSaved: {RES_DAT}/v9_reliability_merged_240.csv")
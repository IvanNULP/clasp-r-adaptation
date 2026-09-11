import gc, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

INPUT_DIM = len(FINAL_FEATURES)
MAX_FOREIGN_SAMPLE = 5000
EPS = 1e-9

BBSE_RIDGE = 1e-6
BBSE_PINV_RCOND = 1e-8
WEIGHT_CLIP_LOW, WEIGHT_CLIP_HIGH = 0.5, 2.0
COND_MAX = 1e16

UOT_N_SUBSAMPLE, UOT_REG, UOT_REG_M, UOT_SEED = 500, 0.05, 1.0, 5555
POT_N_SUBSAMPLE, POT_MASS_FRACTION, POT_SEED = 500, 0.8, 5555

ALL_CANDIDATES = ["BASE", "CORAL", "CORAL+BBSE", "TAC", "TAC+BBSE",
                   "QT", "QT+BBSE", "UOT", "UOT+BBSE", "POT", "POT+BBSE"]

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
ks_shift_lookup = v9_gt.set_index("instance_key")["ks_shift"].to_dict()

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

def global_coral(X_source, X_target, eps=1e-4):
    d = X_source.shape[1]
    cov_s = np.cov(X_source, rowvar=False) + eps*np.eye(d)
    cov_t = np.cov(X_target, rowvar=False) + eps*np.eye(d)
    mean_s, mean_t = X_source.mean(0), X_target.mean(0)
    X_white = (X_source - mean_s) @ matrix_sqrt_inv(cov_s)
    return (X_white @ matrix_sqrt(cov_t) + mean_t).astype(np.float32)

def type_aware_coral(X_source, X_target, eps=1e-4):
    X_adapted = X_source.copy()
    Xs_cont, Xt_cont = X_source[:, CONT_IDX], X_target[:, CONT_IDX]
    d = Xs_cont.shape[1]
    cov_s = np.cov(Xs_cont, rowvar=False) + eps*np.eye(d)
    cov_t = np.cov(Xt_cont, rowvar=False) + eps*np.eye(d)
    mean_s, mean_t = Xs_cont.mean(0), Xt_cont.mean(0)
    Xs_white = (Xs_cont - mean_s) @ matrix_sqrt_inv(cov_s)
    X_adapted[:, CONT_IDX] = Xs_white @ matrix_sqrt(cov_t) + mean_t
    return X_adapted.astype(np.float32)

def quantile_transport(X_source, X_target):
    X_adapted = X_source.copy()
    for j in CONT_IDX:
        x_s_sorted = np.sort(X_source[:, j])
        x_t_sorted = np.sort(X_target[:, j])
        n = len(x_s_sorted)
        ranks = np.searchsorted(x_s_sorted, X_source[:, j], side="left")
        u = np.clip(ranks / n, 1.0/(2*n), 1 - 1.0/(2*n))
        nt = len(x_t_sorted)
        idx = np.clip((u * nt).astype(int), 0, nt - 1)
        X_adapted[:, j] = x_t_sorted[idx]
    return X_adapted.astype(np.float32)

def standardize_pair(Xs, Xt):
    mean_s, std_s = Xs.mean(0), Xs.std(0) + 1e-8
    return (Xs - mean_s) / std_s, (Xt - mean_s) / std_s, mean_s, std_s

def unbalanced_ot_transport(X_source, X_target, n_sub=UOT_N_SUBSAMPLE,
                              reg=UOT_REG, reg_m=UOT_REG_M, seed=UOT_SEED):
    X_adapted = X_source.copy()
    Xs_cont = X_source[:, CONT_IDX].astype(np.float64)
    Xt_cont = X_target[:, CONT_IDX].astype(np.float64)
    Xs_std, Xt_std, mean_s, std_s = standardize_pair(Xs_cont, Xt_cont)
    rng = np.random.RandomState(seed)
    idx_s = rng.choice(len(Xs_std), min(n_sub, len(Xs_std)), replace=False)
    idx_t = rng.choice(len(Xt_std), min(n_sub, len(Xt_std)), replace=False)
    Xs_sub, Xt_sub = Xs_std[idx_s], Xt_std[idx_t]
    a = np.ones(len(Xs_sub)) / len(Xs_sub)
    b = np.ones(len(Xt_sub)) / len(Xt_sub)
    M = ot.dist(Xs_sub, Xt_sub); M /= (M.max() + 1e-12)
    try:
        G = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m, numItermax=1000)
    except Exception:
        return X_source.astype(np.float32)
    row_sums = G.sum(axis=1, keepdims=True); row_sums[row_sums < 1e-12] = 1.0
    Xs_sub_mapped = (G @ Xt_sub) / row_sums
    tree = cKDTree(Xs_sub); _, nn_idx = tree.query(Xs_std, k=1)
    Xs_all_mapped = Xs_sub_mapped[nn_idx] * std_s + mean_s
    X_adapted[:, CONT_IDX] = Xs_all_mapped
    return X_adapted.astype(np.float32)

def partial_ot_transport(X_source, X_target, n_sub=POT_N_SUBSAMPLE,
                          m=POT_MASS_FRACTION, seed=POT_SEED):
    X_adapted = X_source.copy()
    Xs_cont = X_source[:, CONT_IDX].astype(np.float64)
    Xt_cont = X_target[:, CONT_IDX].astype(np.float64)
    Xs_std, Xt_std, mean_s, std_s = standardize_pair(Xs_cont, Xt_cont)
    rng = np.random.RandomState(seed)
    idx_s = rng.choice(len(Xs_std), min(n_sub, len(Xs_std)), replace=False)
    idx_t = rng.choice(len(Xt_std), min(n_sub, len(Xt_std)), replace=False)
    Xs_sub, Xt_sub = Xs_std[idx_s], Xt_std[idx_t]
    a = np.ones(len(Xs_sub)) / len(Xs_sub)
    b = np.ones(len(Xt_sub)) / len(Xt_sub)
    M = ot.dist(Xs_sub, Xt_sub); M /= (M.max() + 1e-12)
    try:
        G = ot.partial.partial_wasserstein(a, b, M, m=m)
    except Exception:
        return X_source.astype(np.float32)
    row_mass = G.sum(axis=1)
    unmatched = row_mass < 1e-9
    row_sums = row_mass.reshape(-1,1).copy(); row_sums[row_sums < 1e-9] = 1.0
    Xs_sub_mapped = (G @ Xt_sub) / row_sums
    Xs_sub_mapped[unmatched] = Xs_sub[unmatched]
    tree = cKDTree(Xs_sub); _, nn_idx = tree.query(Xs_std, k=1)
    Xs_all_mapped = Xs_sub_mapped[nn_idx] * std_s + mean_s
    X_adapted[:, CONT_IDX] = Xs_all_mapped
    return X_adapted.astype(np.float32)

def unsupervised_bbse(probs_foreign_all, frozen_entry, stability_action,
                       ridge=BBSE_RIDGE, pinv_rcond=BBSE_PINV_RCOND):
    C = frozen_entry["C"]; n_cls = len(frozen_entry["classes"])
    p_source = np.array(frozen_entry["source_prior"])
    cond_num = float(np.linalg.cond(C))
    if stability_action == "BBSE_DISABLED":
        return np.ones(n_cls), cond_num
    preds_f = probs_foreign_all.argmax(1)
    q_tilde = np.array([(preds_f == j).mean() for j in range(n_cls)])
    if stability_action == "RIDGE_SOLVE":
        q_est = np.linalg.solve(C + ridge*np.eye(n_cls), q_tilde)
    else:
        q_est = np.linalg.pinv(C, rcond=pinv_rcond) @ q_tilde
    q_est = np.clip(q_est, 0, 1); s = q_est.sum()
    q_est = q_est/s if s > 1e-9 else np.ones(n_cls)/n_cls
    weights = np.clip(q_est / np.maximum(p_source, EPS), WEIGHT_CLIP_LOW, WEIGHT_CLIP_HIGH)
    return weights, cond_num

def row_entropy(probs):
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def response_features(probs_base, probs_other):
    pred_base = probs_base.argmax(1)
    pred_other = probs_other.argmax(1)
    flip_rate = float((pred_base != pred_other).mean())
    prob_shift = float(np.mean(np.abs(probs_base - probs_other)))
    conf_change = float(np.mean(probs_other.max(1) - probs_base.max(1)))
    ent_change = float(np.mean(row_entropy(probs_other) - row_entropy(probs_base)))
    return flip_rate, prob_shift, conf_change, ent_change

def safe_log_condition(cond_num, cond_max=COND_MAX):
    if not np.isfinite(cond_num):
        return np.log10(cond_max)
    return np.log10(max(cond_num, 1.0))

print("="*90)
print("STEP B.1c — Candidate-wise Canonical Feature Matrix")
print("="*90)

TRANSPORT_FUNCS = {
    "CORAL": global_coral, "TAC": type_aware_coral,
    "QT": quantile_transport, "UOT": unbalanced_ot_transport,
    "POT": partial_ot_transport,
}

results_path = RES_DAT / "step_b1c_candidate_matrix.csv"

existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
if len(existing):
    existing = existing.drop_duplicates(
        subset=["instance_key","model","candidate"], keep="first")
    candidate_counts = existing.groupby(["instance_key","model"])["candidate"].nunique()
    complete_tasks = set(candidate_counts[candidate_counts == 11].index)
    existing = existing[
        existing.set_index(["instance_key","model"]).index.isin(complete_tasks)]
    print(f"Checkpoint: {len(complete_tasks)} повних (instance,model) задач "
          f"з {len(candidate_counts)} знайдених у CSV")
else:
    complete_tasks = set()

all_rows = existing.to_dict("records") if len(existing) else []
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
            pair_id = f"{SHORT[src_ds]}__{SHORT[tgt_ds]}"

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

            ks_shift = ks_shift_lookup.get(instance_key, np.nan)

            for mn, model, classes in [("MLP", mlp_model, mlp_classes),
                                        ("RF", rf_model, rf_classes),
                                        ("XGB", xgb_model, xgb_classes)]:
                if (instance_key, mn) in complete_tasks:
                    continue

                probs_base_all = predict_proba_aligned(
                    model, mn, X_foreign, classes, mlp_classes)

                frozen_key = f"{tgt_ds}__{mn}"
                frozen_entry = frozen_matrices[frozen_key]
                stability_action = audit_lookup[frozen_key]

                task_rows = []
                task_rows.append({
                    "instance_key": instance_key, "pair_id": pair_id,
                    "model": mn, "candidate": "BASE",
                    "ks_shift": ks_shift,
                    "prediction_flip_rate": 0.0, "probability_shift": 0.0,
                    "confidence_change": 0.0, "entropy_change": 0.0,
                    "bbse_weight_magnitude": 0.0, "bbse_prior_extremity": 0.0,
                    "bbse_condition_log": 0.0, "bbse_disabled": 1,
                    "candidate_is_base": 1,
                })

                for method_name, transport_fn in TRANSPORT_FUNCS.items():
                    X_adapted = transport_fn(X_foreign, X_tgt_ref)
                    probs_adapted_all = predict_proba_aligned(
                        model, mn, X_adapted, classes, mlp_classes)

                    weights, cond_num = unsupervised_bbse(
                        probs_adapted_all, frozen_entry, stability_action)

                    fr, ps, cc, ec = response_features(probs_base_all, probs_adapted_all)
                    task_rows.append({
                        "instance_key": instance_key, "pair_id": pair_id,
                        "model": mn, "candidate": method_name,
                        "ks_shift": ks_shift,
                        "prediction_flip_rate": fr, "probability_shift": ps,
                        "confidence_change": cc, "entropy_change": ec,
                        "bbse_weight_magnitude": 0.0, "bbse_prior_extremity": 0.0,
                        "bbse_condition_log": 0.0, "bbse_disabled": 1,
                        "candidate_is_base": 0,
                    })

                    probs_bbse_raw = probs_adapted_all * weights[np.newaxis, :]
                    probs_bbse = probs_bbse_raw / (
                        probs_bbse_raw.sum(axis=1, keepdims=True) + EPS)

                    fr_b, ps_b, cc_b, ec_b = response_features(probs_base_all, probs_bbse)

                    n_cls = len(weights)
                    uniform = np.ones(n_cls) / n_cls
                    weight_magnitude = float(np.mean(np.abs(np.log(weights))))
                    prior_extremity = float(np.max(np.abs(weights*uniform - uniform)))
                    cond_log = safe_log_condition(cond_num)
                    bbse_disabled_flag = int(stability_action == "BBSE_DISABLED")

                    task_rows.append({
                        "instance_key": instance_key, "pair_id": pair_id,
                        "model": mn, "candidate": f"{method_name}+BBSE",
                        "ks_shift": ks_shift,
                        "prediction_flip_rate": fr_b, "probability_shift": ps_b,
                        "confidence_change": cc_b, "entropy_change": ec_b,
                        "bbse_weight_magnitude": weight_magnitude,
                        "bbse_prior_extremity": prior_extremity,
                        "bbse_condition_log": cond_log,
                        "bbse_disabled": bbse_disabled_flag,
                        "candidate_is_base": 0,
                    })

                assert len(task_rows) == 11, \
                    f"Task {instance_key}/{mn}: очікувалось 11 rows, отримано {len(task_rows)}"
                all_rows.extend(task_rows)
                complete_tasks.add((instance_key, mn))

        elapsed = time.time() - t_start
        print(f"  {SHORT[src_ds]:10s}→{SHORT[tgt_ds]:10s}: "
              f"complete_tasks={len(complete_tasks)}/720 | rows={len(all_rows)} | "
              f"elapsed={elapsed/60:.1f}min")
        pd.DataFrame(all_rows).to_csv(results_path, index=False)

    del mlp_model, rf_model, xgb_model; gc.collect()

print(f"\n✓ STEP B.1c завершено: {len(all_rows)} рядків, "
      f"{len(complete_tasks)}/720 complete tasks")
print(f"Saved: {results_path}")
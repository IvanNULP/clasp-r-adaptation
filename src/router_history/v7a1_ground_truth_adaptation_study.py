import gc, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

BASE = Path("/srv/ids_research")
PREP = BASE / "data/preprocessed"
MDIR = BASE / "checkpoints/models"
RES_DAT = BASE / "results/domain_adaptation"
RES_DAT.mkdir(parents=True, exist_ok=True)

FLOW_DATASETS = ["CIC-IDS-2017","UNSW-NB15","TON-IoT-2021","CICIoT2023"]
SHORT = {"CIC-IDS-2017":"CIC-2017","UNSW-NB15":"UNSW",
         "TON-IoT-2021":"TON-IoT","CICIoT2023":"CICIoT23"}
FINAL_FEATURES = [
    "duration_ms","total_packets","total_bytes",
    "fwd_packets","fwd_bytes","bwd_packets","bwd_bytes",
    "bytes_per_sec","packets_per_sec","protocol","dst_port",
]
INPUT_DIM = len(FINAL_FEATURES)
MAX_TEST = 5000
MIN_CLASS_SAMPLES = 15
RIDGE_LAMBDA = 0.3
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
        raise ValueError(f"Class sets differ: {model_classes} vs {tgt_classes}")
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

def load_test_data(ds_name, n_samples):
    df = pd.read_parquet(PREP/"test"/f"{ds_name}.parquet")
    if len(df) > n_samples:
        df, _ = train_test_split(
            df, train_size=n_samples,
            stratify=df["label_unified"], random_state=42)
    X = df[FINAL_FEATURES].values.astype(np.float32)
    y = df["label_unified"].values
    return X, y

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

def coral_1class(X_s, X_t, eps=1e-4):
    d = X_s.shape[1]
    cov_s = np.cov(X_s, rowvar=False) + eps*np.eye(d)
    cov_t = np.cov(X_t, rowvar=False) + eps*np.eye(d)
    mean_s, mean_t = X_s.mean(0), X_t.mean(0)
    X_white = (X_s - mean_s) @ matrix_sqrt_inv(cov_s)
    return (X_white @ matrix_sqrt(cov_t) + mean_t).astype(np.float32)

def class_conditional_coral(X_src, y_src, X_tgt, y_tgt):
    """
    ПРИМІТКА (offline, mітки дозволені для v7A):
    Обчислюється ОДИН РАЗ на пару src->tgt, спільно для всіх
    3 моделей (MLP/RF/XGB) — це навмисний дизайн, не помилка,
    бо CORAL-трансформація ознак не залежить від того, яка
    модель потім робитиме передбачення. BBSE, натомість,
    моделе-специфічний і рахується окремо для кожної.
    """
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

def estimate_confusion_matrix(model, mn, ds_own, model_classes, tgt_classes, device="cpu"):
    X, y_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder(); le.fit(tgt_classes)
    mask = np.array([x in set(tgt_classes) for x in y_raw])
    y = le.transform(y_raw[mask]); X = X[mask]
    probs = predict_proba_aligned(model, mn, X, model_classes, tgt_classes, device)
    preds = probs.argmax(1)
    n_cls = len(tgt_classes)
    C = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        m = y == i
        if m.sum() == 0: C[i, i] = 1.0; continue
        for j in range(n_cls): C[j, i] = (preds[m] == j).mean()
    return C

def bbse_correction(model, mn, ds_own, X_foreign, model_classes, tgt_classes,
                     device="cpu", ridge=RIDGE_LAMBDA, cond_threshold=COND_NUMBER_THRESHOLD):
    n_cls = len(tgt_classes)
    C = estimate_confusion_matrix(model, mn, ds_own, model_classes, tgt_classes, device)
    cond_num = np.linalg.cond(C)
    if cond_num >= cond_threshold:
        return np.ones(n_cls), False, cond_num
    probs_f = predict_proba_aligned(model, mn, X_foreign, model_classes, tgt_classes, device)
    preds_f = probs_f.argmax(1)
    q_tilde = np.array([(preds_f == j).mean() for j in range(n_cls)])
    C_reg = C + ridge * np.eye(n_cls)
    try:
        q_est = np.linalg.solve(C_reg, q_tilde)
    except np.linalg.LinAlgError:
        q_est = np.linalg.lstsq(C_reg, q_tilde, rcond=None)[0]
    q_est = np.clip(q_est, 1e-6, None); q_est /= q_est.sum()
    _, y_own_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder(); le.fit(tgt_classes)
    mask_own = np.array([x in set(tgt_classes) for x in y_own_raw])
    y_own = le.transform(y_own_raw[mask_own])
    q_own = np.clip(np.array([(y_own==c).mean() for c in range(n_cls)]), 1e-6, None)
    weights = np.clip(q_est / q_own, 0.2, 5.0)
    return weights, True, cond_num

def compute_macro_error(preds, y_mapped, n_cls):
    """
    ПЕРЕЙМЕНОВАНО з compute_mce: це macro-averaged per-class
    error rate (рівна вага кожному класу), НЕ mean calibration
    error і не звичайна (micro) accuracy-based помилка.
    """
    errs = [float((preds[y_mapped==c]!=c).mean()) for c in np.unique(y_mapped)]
    return round(float(np.mean(errs)), 4) if errs else np.nan

def mean_ks_shift(X_a, X_b):
    return float(np.mean([ks_2samp(X_a[:,j], X_b[:,j])[0] for j in range(X_a.shape[1])]))

print("="*95)
print("CLASP-R v7A.1 — Ground-Truth Adaptation Study (з ревізією)")
print("Метрика: macro_error (macro-averaged per-class error rate)")
print("="*95)

data_cache = {ds: load_test_data(ds, MAX_TEST) for ds in FLOW_DATASETS}
records = []

for tgt_ds in FLOW_DATASETS:
    mlp_model, mlp_classes = load_mlp(tgt_ds)
    rf_model, rf_classes = load_sklearn(tgt_ds, "RF")
    xgb_model, xgb_classes = load_sklearn(tgt_ds, "XGB")
    X_tgt_own, y_tgt_own_raw = data_cache[tgt_ds]

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds: continue
        X_src, y_src_raw = data_cache[src_ds]
        tgt_set = set(mlp_classes)
        mask = np.array([x in tgt_set for x in y_src_raw])
        X_foreign = X_src[mask]; y_foreign_raw = y_src_raw[mask]
        if len(X_foreign) == 0: continue

        le = LabelEncoder(); le.fit(mlp_classes)
        y_mapped = le.transform(y_foreign_raw)
        ks_gap = mean_ks_shift(X_tgt_own, X_foreign)

        X_coral, n_shared = class_conditional_coral(
            X_foreign, y_foreign_raw, X_tgt_own, y_tgt_own_raw)

        print(f"\n{'─'*80}")
        print(f"{SHORT[src_ds]} → {SHORT[tgt_ds]}  "
              f"(KS_gap={ks_gap:.4f}, n_shared_classes={n_shared})")
        print(f"{'─'*80}")
        print(f"{'Model':6s} {'BASE':>8s} {'CORAL':>8s} {'+BBSE':>8s} "
              f"{'ΔCORAL':>9s} {'ΔBBSE':>9s} {'ΔTotal':>9s} "
              f"{'BBSE_rel':>9s} {'CondNum':>9s}")

        pair_results = {}
        for mn, model, classes in [("MLP", mlp_model, mlp_classes),
                                    ("RF", rf_model, rf_classes),
                                    ("XGB", xgb_model, xgb_classes)]:
            probs_base = predict_proba_aligned(
                model, mn, X_foreign, classes, mlp_classes)
            preds_base = probs_base.argmax(1)
            err_base = compute_macro_error(preds_base, y_mapped, len(mlp_classes))

            probs_coral = predict_proba_aligned(
                model, mn, X_coral, classes, mlp_classes)
            preds_coral = probs_coral.argmax(1)
            err_coral = compute_macro_error(preds_coral, y_mapped, len(mlp_classes))

            weights, bbse_ok, cond_num = bbse_correction(
                model, mn, tgt_ds, X_coral, classes, mlp_classes)
            probs_final = probs_coral * weights[np.newaxis, :]
            preds_final = probs_final.argmax(1)
            err_full = compute_macro_error(preds_final, y_mapped, len(mlp_classes))

            d_coral = round(err_coral - err_base, 4)
            d_bbse  = round(err_full - err_coral, 4)
            d_total = round(err_full - err_base, 4)

            print(f"{mn:6s} {err_base:>8.4f} {err_coral:>8.4f} {err_full:>8.4f} "
                  f"{d_coral:>+9.4f} {d_bbse:>+9.4f} {d_total:>+9.4f} "
                  f"{'YES' if bbse_ok else 'no':>9s} {cond_num:>9.2f}")

            pair_results[mn] = err_full
            records.append({
                "src": SHORT[src_ds], "tgt": SHORT[tgt_ds], "model": mn,
                "macro_error_base": err_base,
                "macro_error_coral": err_coral,
                "macro_error_coral_bbse": err_full,
                "delta_coral": d_coral, "delta_bbse": d_bbse, "delta_total": d_total,
                "n_shared_classes": n_shared, "ks_gap": round(ks_gap,4),
                "bbse_reliable": bbse_ok, "condition_number": round(float(cond_num),2),
            })

        oracle_model = min(pair_results, key=pair_results.get)
        print(f"  → Oracle (min macro_error CORAL+BBSE): "
              f"{oracle_model} = {pair_results[oracle_model]:.4f}")

    del mlp_model, rf_model, xgb_model; gc.collect()

df = pd.DataFrame(records)
df.to_csv(RES_DAT/"clasp_v7a1_ground_truth.csv", index=False)

print(f"\n{'='*95}\nЗВЕДЕНА ТАБЛИЦЯ: macro_error (CORAL+BBSE) по 12 парах × 3 моделі\n{'='*95}")
pivot = df.pivot_table(index=["src","tgt"], columns="model", values="macro_error_coral_bbse")
pivot["oracle_model"] = pivot[["MLP","RF","XGB"]].idxmin(axis=1)
pivot["oracle_error"] = pivot[["MLP","RF","XGB"]].min(axis=1)
print(pivot.to_string())

print(f"\n{'='*95}\nΔ АНАЛІЗ ПО МОДЕЛЯХ: чи CORAL/BBSE допомагають чи шкодять\n{'='*95}")
for mn in ["MLP","RF","XGB"]:
    sub = df[df["model"]==mn]
    print(f"\n{mn}:")
    print(f"  Mean Δ_CORAL:  {sub['delta_coral'].mean():+.4f}  "
          f"(% допомогло: {100*(sub['delta_coral']<0).mean():.1f}%)")
    print(f"  Mean Δ_BBSE:   {sub['delta_bbse'].mean():+.4f}  "
          f"(% допомогло: {100*(sub['delta_bbse']<0).mean():.1f}%)")
    print(f"  Mean Δ_Total:  {sub['delta_total'].mean():+.4f}  "
          f"(% допомогло: {100*(sub['delta_total']<0).mean():.1f}%)")

print(f"\nSaved: {RES_DAT}/clasp_v7a1_ground_truth.csv")
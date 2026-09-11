import gc
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import scipy.stats as stats

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

def load_mlp(ds_name, device="cpu"):
    path = MDIR / f"{ds_name}_MLP.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = MLP(INPUT_DIM, ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt["classes"]

def load_sklearn(ds_name, mn):
    path = MDIR / f"{ds_name}_{mn}.pkl"
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

def predict_proba_mlp(model, X, device="cpu", chunk=8192):
    model.eval().to(device)
    probs = []
    for i in range(0, len(X), chunk):
        Xb = torch.FloatTensor(X[i:i+chunk]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(Xb), dim=1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)

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
    if mn in ["MLP","CNN"]:
        probs = predict_proba_mlp(model, X, device)
    else:
        probs = model.predict_proba(X)
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

    if mn in ["MLP","CNN"]:
        probs_foreign = predict_proba_mlp(model, X_foreign, device)
    else:
        probs_foreign = model.predict_proba(X_foreign)
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

    if mn in ["MLP","CNN"]:
        probs_base = predict_proba_mlp(model, X_valid, device)
    else:
        probs_base = model.predict_proba(X_valid)
    preds_base = probs_base.argmax(1)
    mce_base = compute_mce_from_preds(preds_base, y_mapped, n_cls)

    if mce_base <= gating_threshold:
        return preds_base, mce_base, mce_base, {
            "gated_off": True,
            "reason": f"baseline MCE={mce_base:.3f} <= threshold"
        }

    X_coral, n_shared = class_conditional_coral(
        X_valid, y_foreign_raw[mask], X_tgt_own, y_tgt_own_raw)

    if mn in ["MLP","CNN"]:
        probs_coral = predict_proba_mlp(model, X_coral, device)
    else:
        probs_coral = model.predict_proba(X_coral)

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

def fit_platt_calibrator(probs_own, y_own):
    max_probs = probs_own.max(axis=1)
    preds = probs_own.argmax(axis=1)
    correct = (preds == y_own).astype(int)
    if len(np.unique(correct)) < 2:
        return lambda x: x
    lr = LogisticRegression()
    lr.fit(max_probs.reshape(-1, 1), correct)
    def calibrate(raw_confidence):
        return lr.predict_proba(raw_confidence.reshape(-1, 1))[:, 1]
    return calibrate

def get_calibrated_confidence(model, mn, ds_own, X_foreign, device="cpu"):
    X_own, y_own_raw = load_test_data(ds_own, MAX_TEST)

    if mn in ["MLP","CNN"]:
        probs_own = predict_proba_mlp(model, X_own, device)
        probs_foreign = predict_proba_mlp(model, X_foreign, device)
    else:
        probs_own = model.predict_proba(X_own)
        probs_foreign = model.predict_proba(X_foreign)

    le = LabelEncoder()
    le.fit(y_own_raw)
    y_own = le.transform(y_own_raw)

    calibrator = fit_platt_calibrator(probs_own, y_own)
    raw_conf_foreign = probs_foreign.max(axis=1)
    calibrated_conf = calibrator(raw_conf_foreign)

    return float(calibrated_conf.mean()), probs_foreign

def clasp_router_v3(mlp_model, rf_model, xgb_model, tgt_ds,
                     X_foreign, y_foreign_raw,
                     X_tgt_own, y_tgt_own_raw, tgt_classes,
                     device="cpu"):
    n_cls = len(tgt_classes)
    le = LabelEncoder(); le.fit(tgt_classes)
    tgt_set = set(tgt_classes)
    mask = np.array([l in tgt_set for l in y_foreign_raw])
    y_mapped = le.transform(y_foreign_raw[mask])
    X_valid = X_foreign[mask]

    conf_rf, probs_rf = get_calibrated_confidence(
        rf_model, "RF", tgt_ds, X_valid)
    conf_xgb, probs_xgb = get_calibrated_confidence(
        xgb_model, "XGB", tgt_ds, X_valid)
    conf_mlp, probs_mlp = get_calibrated_confidence(
        mlp_model, "MLP", tgt_ds, X_valid, device)

    preds_rf = probs_rf.argmax(1)
    preds_xgb = probs_xgb.argmax(1)

    preds_clasp, mce_b, mce_a, info = clasp_correct(
        mlp_model, "MLP", tgt_ds, X_foreign, y_foreign_raw,
        X_tgt_own, y_tgt_own_raw, tgt_classes, device)

    candidates_conf = {"RF": conf_rf, "XGB": conf_xgb, "MLP+CLASP": conf_mlp}
    candidates_preds = {"RF": preds_rf, "XGB": preds_xgb, "MLP+CLASP": preds_clasp}

    chosen_name = max(candidates_conf, key=candidates_conf.get)
    chosen_preds = candidates_preds[chosen_name]

    mce_router = compute_mce_from_preds(chosen_preds, y_mapped, n_cls)
    mce_per_candidate = {
        name: compute_mce_from_preds(preds, y_mapped, n_cls)
        for name, preds in candidates_preds.items()
    }
    oracle_best = min(mce_per_candidate.values())
    oracle_name = min(mce_per_candidate, key=mce_per_candidate.get)

    return {
        "chosen": chosen_name,
        "calibrated_confidences": {k: round(v,4) for k,v in candidates_conf.items()},
        "mce_router": mce_router,
        "mce_per_candidate": mce_per_candidate,
        "oracle_best_mce": oracle_best,
        "oracle_best_model": oracle_name,
        "router_matches_oracle": chosen_name == oracle_name,
    }

data_cache = {ds: load_test_data(ds, MAX_TEST) for ds in FLOW_DATASETS}
router_results = {}

for tgt_ds in FLOW_DATASETS:
    mlp_model, tgt_classes = load_mlp(tgt_ds)
    rf_model, _ = load_sklearn(tgt_ds, "RF")
    xgb_model, _ = load_sklearn(tgt_ds, "XGB")
    X_tgt_own, y_tgt_own_raw = data_cache[tgt_ds]

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds:
            continue
        X_src, y_src_raw = data_cache[src_ds]

        result = clasp_router_v3(
            mlp_model, rf_model, xgb_model, tgt_ds,
            X_src, y_src_raw, X_tgt_own, y_tgt_own_raw, tgt_classes)

        router_results[f"{src_ds}__{tgt_ds}"] = result

        print(f"{SHORT[src_ds]}->{SHORT[tgt_ds]} "
              f"chose={result['chosen']} "
              f"MCE={result['mce_router']:.4f} "
              f"oracle={result['oracle_best_model']} "
              f"MCE={result['oracle_best_mce']:.4f} "
              f"match={result['router_matches_oracle']}")

    del mlp_model, rf_model, xgb_model
    gc.collect()

with open(RES_DAT/"clasp_router_v3_platt_results.json","w") as f:
    json.dump(router_results, f, indent=2)

mce_router = [v["mce_router"] for v in router_results.values()]
mce_rf     = [v["mce_per_candidate"]["RF"] for v in router_results.values()]
mce_xgb    = [v["mce_per_candidate"]["XGB"] for v in router_results.values()]
mce_clasp  = [v["mce_per_candidate"]["MLP+CLASP"] for v in router_results.values()]
mce_oracle = [v["oracle_best_mce"] for v in router_results.values()]
match_rate = np.mean([v["router_matches_oracle"] for v in router_results.values()])

print(f"Always RF: mean MCE = {np.mean(mce_rf):.4f}")
print(f"Always XGB: mean MCE = {np.mean(mce_xgb):.4f}")
print(f"Always MLP+CLASP: mean MCE = {np.mean(mce_clasp):.4f}")
print(f"Router (calibrated confidence): mean MCE = {np.mean(mce_router):.4f}")
print(f"Oracle (best hindsight): mean MCE = {np.mean(mce_oracle):.4f}")
print(f"Router match rate with oracle: {match_rate*100:.1f}%")

t_s, t_p = stats.ttest_rel(mce_clasp, mce_router)
print(f"Router vs Always-MLP+CLASP: t={t_s:.4f} p={t_p:.4f}")

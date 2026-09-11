import gc, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, pearsonr

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
EPS = 1e-9

ACTION_COLS = [
    "MLP_BASE_error", "MLP_CORAL_error", "MLP_CLASP_error",
    "RF_BASE_error",  "RF_CORAL_error",  "RF_CLASP_error",
    "XGB_BASE_error", "XGB_CORAL_error", "XGB_CLASP_error",
]

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

def mean_ks_shift(X_a, X_b):
    return float(np.mean([ks_2samp(X_a[:,j], X_b[:,j])[0] for j in range(X_a.shape[1])]))

def row_entropy(probs):
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def summarize_probs(probs):
    s = np.sort(probs, axis=1)
    top1 = s[:, -1]
    top2 = s[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs))
    margin = top1 - top2
    ent = row_entropy(probs)
    ent_norm = ent / np.log(max(probs.shape[1], 2))
    return {
        "mean_conf": float(np.mean(top1)),
        "mean_margin": float(np.mean(margin)),
        "mean_entropy": float(np.mean(ent_norm)),
    }

def prob_agreement(probs_a, probs_b):
    return float(np.mean(np.sum(np.minimum(probs_a, probs_b), axis=1)))

def pred_agreement(probs_a, probs_b):
    return float((probs_a.argmax(1) == probs_b.argmax(1)).mean())

gt = pd.read_csv(RES_DAT/"clasp_v7a1_ground_truth.csv")

print("="*100)
print("CLASP-R v7C.1 — Router Feasibility Analysis (фінальна ревізія)")
print("Потік A (features): весь X_foreign, БЕЗ y_foreign")
print("Потік B (ground truth): errors з v7A.1, лише для оцінки")
print("="*100)
print("\n⚠ МЕТОДОЛОГІЧНЕ ЗАСТЕРЕЖЕННЯ (зберігати при інтерпретації):")
print("  features обчислені на ПОВНОМУ X_foreign (весь source test set),")
print("  а ground-truth errors — тільки на підмножині зі спільними класами.")
print("  Це НЕ leakage, але це distribution mismatch між populatiою ознак")
print("  і популяцією, на якій виміряно метрику. Тому цей аналіз відповідає")
print("  лише на питання 'чи є взагалі сигнал?', а НЕ 'чи цей конкретний")
print("  feature є хорошим предиктором для router'. Сильних висновків")
print("  з окремих кореляцій тут робити не можна.\n")

data_cache = {ds: load_test_data(ds, MAX_TEST) for ds in FLOW_DATASETS}
feasibility_records = []

for tgt_ds in FLOW_DATASETS:
    mlp_model, mlp_classes = load_mlp(tgt_ds)
    rf_model, rf_classes = load_sklearn(tgt_ds, "RF")
    xgb_model, xgb_classes = load_sklearn(tgt_ds, "XGB")
    X_tgt_own, y_tgt_own_raw = data_cache[tgt_ds]

    for src_ds in FLOW_DATASETS:
        if src_ds == tgt_ds: continue
        X_src, y_src_raw = data_cache[src_ds]

        X_foreign_all = X_src

        ks_shift = mean_ks_shift(X_tgt_own, X_foreign_all)

        probs_mlp = predict_proba_aligned(
            mlp_model, "MLP", X_foreign_all, mlp_classes, mlp_classes)
        probs_rf  = predict_proba_aligned(
            rf_model, "RF", X_foreign_all, rf_classes, mlp_classes)
        probs_xgb = predict_proba_aligned(
            xgb_model, "XGB", X_foreign_all, xgb_classes, mlp_classes)

        s_mlp = summarize_probs(probs_mlp)
        s_rf  = summarize_probs(probs_rf)
        s_xgb = summarize_probs(probs_xgb)

        feat = {
            "src": SHORT[src_ds], "tgt": SHORT[tgt_ds],
            "n_foreign_total": len(X_foreign_all),
            "ks_shift": round(ks_shift, 4),
            "rf_xgb_pred_agreement": round(pred_agreement(probs_rf, probs_xgb), 4),
            "rf_xgb_prob_agreement": round(prob_agreement(probs_rf, probs_xgb), 4),
            "mlp_rf_pred_agreement": round(pred_agreement(probs_mlp, probs_rf), 4),
            "mlp_xgb_pred_agreement": round(pred_agreement(probs_mlp, probs_xgb), 4),
            "mlp_conf": s_mlp["mean_conf"], "mlp_margin": s_mlp["mean_margin"],
            "mlp_entropy": s_mlp["mean_entropy"],
            "rf_conf": s_rf["mean_conf"], "rf_margin": s_rf["mean_margin"],
            "rf_entropy": s_rf["mean_entropy"],
            "xgb_conf": s_xgb["mean_conf"], "xgb_margin": s_xgb["mean_margin"],
            "xgb_entropy": s_xgb["mean_entropy"],
        }

        pair_gt = gt[(gt["src"]==SHORT[src_ds]) & (gt["tgt"]==SHORT[tgt_ds])]
        for _, r in pair_gt.iterrows():
            mn = r["model"]
            feat[f"{mn}_BASE_error"]  = r["macro_error_base"]
            feat[f"{mn}_CORAL_error"] = r["macro_error_coral"]
            feat[f"{mn}_CLASP_error"] = r["macro_error_coral_bbse"]
        feat["n_shared_classes_for_eval"] = (
            pair_gt["n_shared_classes"].iloc[0] if len(pair_gt) else None)

        all_actions = {
            k: feat[k] for k in ACTION_COLS
            if k in feat and pd.notna(feat[k])
        }
        best_action = min(all_actions, key=all_actions.get)
        feat["winner_action"] = best_action.replace("_error", "")
        feat["winner_error"] = all_actions[best_action]
        sorted_vals = sorted(all_actions.values())
        feat["winner_margin"] = round(sorted_vals[1] - sorted_vals[0], 4)

        feasibility_records.append(feat)

    del mlp_model, rf_model, xgb_model; gc.collect()

feas_df = pd.DataFrame(feasibility_records)
feas_df.to_csv(RES_DAT/"clasp_v7c1_feasibility_noleak.csv", index=False)

print(f"\n{'='*100}")
print("Таблиця label-free ознак (на ПОВНОМУ X_foreign) і winner action")
print(f"{'='*100}")
show_cols = ["src","tgt","n_foreign_total","n_shared_classes_for_eval",
             "ks_shift","rf_xgb_pred_agreement",
             "mlp_conf","rf_conf","xgb_conf","winner_action",
             "winner_error","winner_margin"]
print(feas_df[show_cols].to_string(index=False))

print(f"\n{'='*100}")
print("Кореляція label-free ознак з winner_margin")
print("(exploratory, n=12 — відповідає лише на 'чи є сигнал взагалі')")
print(f"{'='*100}")

feature_cols = ["ks_shift","rf_xgb_pred_agreement","rf_xgb_prob_agreement",
                "mlp_rf_pred_agreement","mlp_xgb_pred_agreement",
                "mlp_conf","mlp_margin","mlp_entropy",
                "rf_conf","rf_margin","rf_entropy",
                "xgb_conf","xgb_margin","xgb_entropy"]

for col in feature_cols:
    r, p = pearsonr(feas_df[col], feas_df["winner_margin"])
    flag = "→ дивитись уважніше" if abs(r) > 0.4 else ""
    print(f"  {col:28s} r={r:+.3f}  p={p:.3f}  {flag}")

print(f"\n{'='*100}")
print("Порівняння: winner = MLP-variant vs winner = Tree(RF/XGB)-variant")
print(f"{'='*100}")

feas_df["winner_family"] = feas_df["winner_action"].apply(
    lambda x: "MLP" if x.startswith("MLP") else "Tree(RF/XGB)")

print(feas_df.groupby("winner_family")[feature_cols].mean().T.to_string())
print(f"\nРозподіл winner_family:")
print(feas_df["winner_family"].value_counts().to_string())

print(f"\nSaved: {RES_DAT}/clasp_v7c1_feasibility_noleak.csv")
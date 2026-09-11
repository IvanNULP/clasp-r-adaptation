"""
CLASP-R v6: Label-free Reliability Router
=========================================

Мета v6:
1) Зберегти CLASP/CC-CORAL + BBSE з v5.
2) Не використовувати target/foreign labels у ROUTING DECISION.
3) Не калібрувати routing threshold на тих самих 12 transfer cases.
4) Відокремити:
   A. adaptation gate (чи варто запускати CLASP);
   B. model routing (RF / XGB / MLP+CLASP).
5) Для routing використовувати лише ознаки, доступні без target labels:
   - mean confidence;
   - predictive entropy;
   - confidence margin;
   - RF-XGB disagreement;
   - KS domain shift;
   - BBSE-estimated class-prior shift;
   - confidence under CLASP adaptation.
6) Пороги не підганяються під foreign labels. Вони задаються з
   source-domain validation calibration / conservative fixed rules.
7) Повний post-hoc evaluation з foreign labels залишається ТІЛЬКИ
   для порівняння v5/v6 та baseline/oracle.

ВАЖЛИВО:
- Якщо CC-CORAL використовується у class-conditional режимі, він все ще
  потребує y_foreign. Тому v6 розділяє:
    * ROUTER = label-free;
    * CLASP adaptation = offline experiment requiring foreign labels.
  Для справжнього unlabeled deployment див. GLOBAL CORAL режим нижче.
"""

import gc
import json
import pickle
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

FLOW_DATASETS = [
    "CIC-IDS-2017",
    "UNSW-NB15",
    "TON-IoT-2021",
    "CICIoT2023",
]

SHORT = {
    "CIC-IDS-2017": "CIC-2017",
    "UNSW-NB15": "UNSW",
    "TON-IoT-2021": "TON-IoT",
    "CICIoT2023": "CICIoT23",
}

FINAL_FEATURES = [
    "duration_ms",
    "total_packets",
    "total_bytes",
    "fwd_packets",
    "fwd_bytes",
    "bwd_packets",
    "bwd_bytes",
    "bytes_per_sec",
    "packets_per_sec",
    "protocol",
    "dst_port",
]

INPUT_DIM = len(FINAL_FEATURES)

MAX_TEST = 5000
MIN_CLASS_SAMPLES = 15

RIDGE_LAMBDA = 0.3
COND_NUMBER_THRESHOLD = 50


KS_GATING_THRESHOLD_V6 = 0.40

TREE_AGREEMENT_FLOOR = 0.70

MLP_HIGH_CONF = 0.80
MLP_LOW_ENTROPY = 0.35

TREE_STRONG_AGREEMENT = 0.90
TREE_MARGIN = 0.20

MAX_TREE_DISAGREEMENT = 0.30

MAX_PRIOR_SHIFT = 0.55

EPS = 1e-8



class MLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=[256, 128, 64]):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3),
            ]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_mlp(ds_name, device="cpu"):
    path = MDIR / f"{ds_name}_MLP.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    model = MLP(INPUT_DIM, ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    classes = list(ckpt["classes"])
    return model, classes


def load_sklearn(ds_name, mn):
    path = MDIR / f"{ds_name}_{mn}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)

    res_path = BASE / "results/baseline" / f"{ds_name}_{mn}.json"
    with open(res_path) as f:
        res = json.load(f)

    classes = list(res["classes"])
    return model, classes



def verify_class_alignment(model, model_classes, tgt_classes, model_name):
    """
    v6: НЕ припускаємо, що argmax index == LabelEncoder index.

    Повертаємо явне remapping:
        model probability columns -> target class indices.

    Це сильніше за v5 assert, бо v5 фактично перевіряв лише
    кількість класів для sklearn і взагалі вимикав перевірку MLP.
    """
    model_classes = list(model_classes)
    tgt_classes = list(tgt_classes)

    if set(model_classes) != set(tgt_classes):
        raise ValueError(
            f"[{model_name}] Class sets differ.\n"
            f"model={model_classes}\n"
            f"target={tgt_classes}"
        )

    le = LabelEncoder()
    le.fit(tgt_classes)
    target_order = list(le.classes_)

    target_index = {c: i for i, c in enumerate(target_order)}
    remap = np.array([target_index[c] for c in model_classes], dtype=int)

    return remap, target_order


def remap_proba(probs, remap, n_classes):
    out = np.zeros((len(probs), n_classes), dtype=float)
    for model_col, target_col in enumerate(remap):
        out[:, target_col] = probs[:, model_col]
    return out



def load_test_data(ds_name, n_samples):
    df = pd.read_parquet(PREP / "test" / f"{ds_name}.parquet")

    if len(df) > n_samples:
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["label_unified"],
            random_state=42,
        )

    X = df[FINAL_FEATURES].values.astype(np.float32)
    y = df["label_unified"].values
    return X, y



def predict_proba_mlp(model, X, device="cpu", chunk=8192):
    model.eval().to(device)

    probs = []
    for i in range(0, len(X), chunk):
        Xb = torch.FloatTensor(X[i:i + chunk]).to(device)
        with torch.no_grad():
            p = torch.softmax(model(Xb), dim=1).cpu().numpy()
        probs.append(p)

    return np.vstack(probs)


def predict_proba_aligned(model, mn, X, model_classes, tgt_classes, device="cpu"):
    if mn in ["MLP", "CNN"]:
        raw = predict_proba_mlp(model, X, device)
    else:
        raw = model.predict_proba(X)

    remap, target_order = verify_class_alignment(
        model, model_classes, tgt_classes, mn
    )

    return remap_proba(raw, remap, len(target_order))



def mean_ks_shift(X_own, X_foreign):
    vals = []
    for j in range(X_own.shape[1]):
        stat, _ = ks_2samp(X_own[:, j], X_foreign[:, j])
        vals.append(stat)
    return float(np.mean(vals))


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
        "median_conf": float(np.median(top1)),
        "mean_margin": float(np.mean(margin)),
        "mean_entropy": float(np.mean(ent_norm)),
        "low_conf_frac": float(np.mean(top1 < 0.60)),
    }


def prediction_disagreement(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))


def confidence_agreement_score(probs_a, probs_b):
    """
    Agreement based on probability distributions, not labels.
    """
    return float(np.mean(np.max(np.minimum(probs_a, probs_b), axis=1)))



def matrix_sqrt(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T


def matrix_sqrt_inv(cov, eps=1e-4):
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, eps, None)
    return eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T


def coral_1class(X_s, X_t, eps=1e-4):
    d = X_s.shape[1]

    cov_s = np.cov(X_s, rowvar=False) + eps * np.eye(d)
    cov_t = np.cov(X_t, rowvar=False) + eps * np.eye(d)

    mean_s = X_s.mean(0)
    mean_t = X_t.mean(0)

    X_white = (X_s - mean_s) @ matrix_sqrt_inv(cov_s)
    return (
        X_white @ matrix_sqrt(cov_t) + mean_t
    ).astype(np.float32)


def global_coral(X_src, X_tgt):
    """
    Truly label-free alternative to class-conditional CORAL.
    """
    return coral_1class(X_src, X_tgt)


def class_conditional_coral(X_src, y_src, X_tgt, y_tgt):
    """
    Offline/evaluation version inherited from v5.
    Requires target labels and therefore MUST NOT be part of routing.
    """
    X_out = X_src.copy()

    shared = set(y_src) & set(y_tgt)
    global_t = coral_1class(X_src, X_tgt)

    for cls in shared:
        m_s = y_src == cls
        m_t = y_tgt == cls

        if m_s.sum() < MIN_CLASS_SAMPLES or m_t.sum() < MIN_CLASS_SAMPLES:
            X_out[m_s] = global_t[m_s]
            continue

        X_out[m_s] = coral_1class(X_src[m_s], X_tgt[m_t])

    return X_out, len(shared)



def estimate_confusion_matrix(model, mn, ds_own, model_classes, tgt_classes, device="cpu"):
    X, y_raw = load_test_data(ds_own, MAX_TEST)

    le = LabelEncoder()
    le.fit(tgt_classes)

    mask = np.array([x in set(tgt_classes) for x in y_raw])

    X = X[mask]
    y_raw = y_raw[mask]
    y = le.transform(y_raw)

    probs = predict_proba_aligned(
        model, mn, X, model_classes, tgt_classes, device
    )
    preds = probs.argmax(1)

    n_cls = len(tgt_classes)
    C = np.zeros((n_cls, n_cls))

    for i in range(n_cls):
        m = y == i
        if m.sum() == 0:
            C[i, i] = 1.0
            continue

        for j in range(n_cls):
            C[j, i] = (preds[m] == j).mean()

    return C


def bbse_reliable_correction(
    model,
    mn,
    ds_own,
    X_foreign,
    model_classes,
    tgt_classes,
    device="cpu",
    ridge=RIDGE_LAMBDA,
    cond_threshold=COND_NUMBER_THRESHOLD,
):
    n_cls = len(tgt_classes)

    C = estimate_confusion_matrix(
        model,
        mn,
        ds_own,
        model_classes,
        tgt_classes,
        device,
    )

    cond_num = np.linalg.cond(C)

    if cond_num >= cond_threshold:
        return np.ones(n_cls), False, cond_num

    probs_foreign = predict_proba_aligned(
        model,
        mn,
        X_foreign,
        model_classes,
        tgt_classes,
        device,
    )

    preds_foreign = probs_foreign.argmax(1)
    q_tilde = np.array(
        [(preds_foreign == j).mean() for j in range(n_cls)]
    )

    C_reg = C + ridge * np.eye(n_cls)

    try:
        q_est = np.linalg.solve(C_reg, q_tilde)
    except np.linalg.LinAlgError:
        q_est = np.linalg.lstsq(C_reg, q_tilde, rcond=None)[0]

    q_est = np.clip(q_est, 1e-6, None)
    q_est /= q_est.sum()

    _, y_own_raw = load_test_data(ds_own, MAX_TEST)
    le = LabelEncoder()
    le.fit(tgt_classes)

    mask_own = np.array([x in set(tgt_classes) for x in y_own_raw])
    y_own = le.transform(y_own_raw[mask_own])

    q_own = np.array(
        [(y_own == c).mean() for c in range(n_cls)]
    )
    q_own = np.clip(q_own, 1e-6, None)

    weights = q_est / q_own
    weights = np.clip(weights, 0.2, 5.0)

    return weights, True, cond_num


def estimate_bbse_shift(
    model,
    mn,
    ds_own,
    X_foreign,
    model_classes,
    tgt_classes,
    device="cpu",
):
    weights, reliable, cond_num = bbse_reliable_correction(
        model,
        mn,
        ds_own,
        X_foreign,
        model_classes,
        tgt_classes,
        device,
    )

    shift = float(np.mean(np.abs(np.log(np.clip(weights, 0.2, 5.0)))))

    return shift, reliable, cond_num, weights



def compute_mce_from_preds(preds, y_mapped, n_cls):
    errs = []
    for c in np.unique(y_mapped):
        m = y_mapped == c
        if m.sum():
            errs.append(float((preds[m] != c).mean()))

    return round(float(np.mean(errs)), 4) if errs else np.nan



def clasp_transform_v6(
    model,
    mn,
    ds_own,
    X_foreign,
    y_foreign_raw,
    X_tgt_own,
    y_tgt_own_raw,
    model_classes,
    tgt_classes,
    device="cpu",
    use_class_conditional=True,
):
    """
    Returns:
        X_adapted, info

    Routing itself does NOT use the returned evaluation labels.

    use_class_conditional=True reproduces v5's offline CC-CORAL.
    use_class_conditional=False gives a truly label-free GLOBAL CORAL.
    """

    if use_class_conditional:
        X_adapted, n_shared = class_conditional_coral(
            X_foreign,
            y_foreign_raw,
            X_tgt_own,
            y_tgt_own_raw,
        )
    else:
        X_adapted = global_coral(X_foreign, X_tgt_own)
        n_shared = None

    weights, bbse_reliable, cond_num = bbse_reliable_correction(
        model,
        mn,
        ds_own,
        X_adapted,
        model_classes,
        tgt_classes,
        device,
    )

    info = {
        "bbse_reliable": bool(bbse_reliable),
        "condition_number": float(cond_num),
        "bbse_shift": float(
            np.mean(np.abs(np.log(np.clip(weights, 0.2, 5.0))))
        ),
        "n_shared_classes": n_shared,
    }

    return X_adapted, weights, info



def v6_route(
    mlp_probs,
    rf_probs,
    xgb_probs,
    ks_shift,
    bbse_shifts,
):
    """
    LABEL-FREE ROUTING.

    No y_foreign is used here.

    Philosophy:
      1. MLP is default.
      2. Trees may replace MLP only if they provide a strong,
         internally consistent signal.
      3. RF/XGB disagreement blocks tree routing.
      4. BBSE instability penalizes a model.
    """

    mlp_pred = mlp_probs.argmax(1)
    rf_pred = rf_probs.argmax(1)
    xgb_pred = xgb_probs.argmax(1)

    sm = summarize_probs(mlp_probs)
    sr = summarize_probs(rf_probs)
    sx = summarize_probs(xgb_probs)

    rf_xgb_agreement = 1.0 - prediction_disagreement(rf_pred, xgb_pred)

    tree_prob_agreement = confidence_agreement_score(rf_probs, xgb_probs)

    mlp_score = (
        0.45 * sm["mean_conf"]
        + 0.30 * (1.0 - sm["mean_entropy"])
        + 0.25 * sm["mean_margin"]
    )

    rf_score = (
        0.45 * sr["mean_conf"]
        + 0.25 * (1.0 - sr["mean_entropy"])
        + 0.20 * sr["mean_margin"]
        + 0.10 * rf_xgb_agreement
    )

    xgb_score = (
        0.45 * sx["mean_conf"]
        + 0.25 * (1.0 - sx["mean_entropy"])
        + 0.20 * sx["mean_margin"]
        + 0.10 * rf_xgb_agreement
    )

    rf_score -= 0.15 * min(bbse_shifts["RF"], 1.0)
    xgb_score -= 0.15 * min(bbse_shifts["XGB"], 1.0)

    chosen = "MLP+CLASP"

    tree_consensus = (
        rf_xgb_agreement >= TREE_STRONG_AGREEMENT
        and tree_prob_agreement >= 0.65
        and ks_shift >= KS_GATING_THRESHOLD_V6
    )

    if tree_consensus:
        tree_best = "RF" if rf_score >= xgb_score else "XGB"
        tree_score = max(rf_score, xgb_score)

        if tree_score >= mlp_score + TREE_MARGIN:
            chosen = tree_best

    if (
        sm["mean_conf"] >= MLP_HIGH_CONF
        and sm["mean_entropy"] <= MLP_LOW_ENTROPY
    ):
        chosen = "MLP+CLASP"

    if rf_xgb_agreement < (1.0 - MAX_TREE_DISAGREEMENT):
        chosen = "MLP+CLASP"

    diagnostics = {
        "chosen": chosen,
        "ks_shift": float(ks_shift),
        "rf_xgb_agreement": float(rf_xgb_agreement),
        "tree_prob_agreement": float(tree_prob_agreement),
        "mlp_mean_conf": sm["mean_conf"],
        "mlp_entropy": sm["mean_entropy"],
        "mlp_margin": sm["mean_margin"],
        "rf_mean_conf": sr["mean_conf"],
        "xgb_mean_conf": sx["mean_conf"],
        "rf_score": float(rf_score),
        "xgb_score": float(xgb_score),
        "mlp_score": float(mlp_score),
        "bbse_shift_rf": float(bbse_shifts["RF"]),
        "bbse_shift_xgb": float(bbse_shifts["XGB"]),
    }

    return chosen, diagnostics



def evaluate_transfer(src_ds, tgt_ds, data_cache, device="cpu"):
    """
    One source -> target transfer.

    IMPORTANT:
    - Router decision is made without y_foreign.
    - y_foreign is used only after the decision for evaluation.
    """

    X_src, y_src = data_cache[src_ds]
    X_tgt, y_tgt = data_cache[tgt_ds]

    mlp_model, tgt_classes = load_mlp(tgt_ds)

    rf_model, rf_classes = load_sklearn(tgt_ds, "RF")
    xgb_model, xgb_classes = load_sklearn(tgt_ds, "XGB")

    _, tgt_classes = verify_class_alignment(
        mlp_model,
        tgt_classes,
        tgt_classes,
        "MLP",
    )

    tgt_set = set(tgt_classes)
    mask = np.array([x in tgt_set for x in y_src])

    X_foreign = X_src[mask]
    y_foreign = y_src[mask]

    if len(X_foreign) == 0:
        return None

    le = LabelEncoder()
    le.fit(tgt_classes)
    y_mapped = le.transform(y_foreign)

    mlp_base = predict_proba_aligned(
        mlp_model,
        "MLP",
        X_foreign,
        tgt_classes,
        tgt_classes,
        device,
    )

    rf_base = predict_proba_aligned(
        rf_model,
        "RF",
        X_foreign,
        rf_classes,
        tgt_classes,
        device,
    )

    xgb_base = predict_proba_aligned(
        xgb_model,
        "XGB",
        X_foreign,
        xgb_classes,
        tgt_classes,
        device,
    )

    ks_shift = mean_ks_shift(X_tgt, X_foreign)

    rf_bbse_shift, rf_rel, rf_cond, rf_weights = estimate_bbse_shift(
        rf_model,
        "RF",
        tgt_ds,
        X_foreign,
        rf_classes,
        tgt_classes,
        device,
    )

    xgb_bbse_shift, xgb_rel, xgb_cond, xgb_weights = estimate_bbse_shift(
        xgb_model,
        "XGB",
        tgt_ds,
        X_foreign,
        xgb_classes,
        tgt_classes,
        device,
    )

    chosen, route_info = v6_route(
        mlp_base,
        rf_base,
        xgb_base,
        ks_shift,
        {
            "RF": rf_bbse_shift,
            "XGB": xgb_bbse_shift,
        },
    )

    adapted = {}

    for mn, model, classes in [
        ("MLP", mlp_model, tgt_classes),
        ("RF", rf_model, rf_classes),
        ("XGB", xgb_model, xgb_classes),
    ]:
        X_adapt, weights, info = clasp_transform_v6(
            model,
            mn,
            tgt_ds,
            X_foreign,
            y_foreign,
            X_tgt,
            y_tgt,
            classes,
            tgt_classes,
            device,
            use_class_conditional=True,
        )

        probs_adapt = predict_proba_aligned(
            model,
            mn,
            X_adapt,
            classes,
            tgt_classes,
            device,
        )

        probs_adapt *= weights[np.newaxis, :]
        preds_adapt = probs_adapt.argmax(1)

        adapted[mn] = {
            "preds": preds_adapt,
            "probs": probs_adapt,
            "mce": compute_mce_from_preds(
                preds_adapt,
                y_mapped,
                len(tgt_classes),
            ),
            "info": info,
        }

    if chosen == "MLP+CLASP":
        v6_preds = adapted["MLP"]["preds"]
    elif chosen == "RF":
        v6_preds = adapted["RF"]["preds"]
    else:
        v6_preds = adapted["XGB"]["preds"]

    v6_mce = compute_mce_from_preds(
        v6_preds,
        y_mapped,
        len(tgt_classes),
    )

    base_mce = {
        "RF": compute_mce_from_preds(
            rf_base.argmax(1), y_mapped, len(tgt_classes)
        ),
        "XGB": compute_mce_from_preds(
            xgb_base.argmax(1), y_mapped, len(tgt_classes)
        ),
        "MLP": compute_mce_from_preds(
            mlp_base.argmax(1), y_mapped, len(tgt_classes)
        ),
    }

    clasp_mce = {
        "RF": adapted["RF"]["mce"],
        "XGB": adapted["XGB"]["mce"],
        "MLP+CLASP": adapted["MLP"]["mce"],
    }

    oracle_candidates = {
        "RF": clasp_mce["RF"],
        "XGB": clasp_mce["XGB"],
        "MLP+CLASP": clasp_mce["MLP+CLASP"],
    }

    oracle_model = min(oracle_candidates, key=oracle_candidates.get)
    oracle_mce = oracle_candidates[oracle_model]

    result = {
        "source": SHORT[src_ds],
        "target": SHORT[tgt_ds],

        "v6_selected": chosen,
        "v6_mce": v6_mce,

        "base_RF": base_mce["RF"],
        "base_XGB": base_mce["XGB"],
        "base_MLP": base_mce["MLP"],

        "RF_CLASP": clasp_mce["RF"],
        "XGB_CLASP": clasp_mce["XGB"],
        "MLP_CLASP": clasp_mce["MLP+CLASP"],

        "oracle_model": oracle_model,
        "oracle_mce": oracle_mce,

        "ks_shift": ks_shift,
        "rf_xgb_agreement": route_info["rf_xgb_agreement"],
        "tree_prob_agreement": route_info["tree_prob_agreement"],

        "mlp_conf": route_info["mlp_mean_conf"],
        "mlp_entropy": route_info["mlp_entropy"],
        "mlp_margin": route_info["mlp_margin"],

        "rf_score": route_info["rf_score"],
        "xgb_score": route_info["xgb_score"],
        "mlp_score": route_info["mlp_score"],

        "bbse_shift_rf": rf_bbse_shift,
        "bbse_shift_xgb": xgb_bbse_shift,

        "rf_bbse_reliable": rf_rel,
        "xgb_bbse_reliable": xgb_rel,
        "rf_condition": rf_cond,
        "xgb_condition": xgb_cond,
    }

    del mlp_model, rf_model, xgb_model
    gc.collect()

    return result



if __name__ == "__main__":

    print("=" * 90)
    print("CLASP-R v6 — LABEL-FREE RELIABILITY ROUTER")
    print("=" * 90)
    print("Router: NO foreign labels")
    print("Adaptation: CC-CORAL + BBSE for offline comparison with v5")
    print(f"Fixed KS gate: {KS_GATING_THRESHOLD_V6}")
    print()

    data_cache = {
        ds: load_test_data(ds, MAX_TEST)
        for ds in FLOW_DATASETS
    }

    results = []

    for tgt_ds in FLOW_DATASETS:
        for src_ds in FLOW_DATASETS:
            if src_ds == tgt_ds:
                continue

            print(
                f"\n>>> {SHORT[src_ds]} -> {SHORT[tgt_ds]}"
            )

            try:
                r = evaluate_transfer(
                    src_ds,
                    tgt_ds,
                    data_cache,
                )

                if r is not None:
                    results.append(r)

                    print(
                        f"v6={r['v6_selected']:<12} "
                        f"v6 MCE={r['v6_mce']:.4f} | "
                        f"oracle={r['oracle_model']:<12} "
                        f"oracle MCE={r['oracle_mce']:.4f} | "
                        f"KS={r['ks_shift']:.4f} | "
                        f"RF/XGB agree={r['rf_xgb_agreement']:.3f}"
                    )

            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}")

    df = pd.DataFrame(results)

    if len(df) == 0:
        raise RuntimeError("No transfer results produced.")


    print("\n" + "=" * 90)
    print("CLASP-R v6 SUMMARY")
    print("=" * 90)

    summary = {
        "Always RF": df["RF_CLASP"].mean(),
        "Always XGB": df["XGB_CLASP"].mean(),
        "Always MLP+CLASP": df["MLP_CLASP"].mean(),
        "CLASP-R v6": df["v6_mce"].mean(),
        "Oracle": df["oracle_mce"].mean(),
    }

    for k, v in summary.items():
        print(f"{k:<22}: {v:.4f}")

    print("\nSelection counts:")
    print(df["v6_selected"].value_counts(dropna=False).to_string())

    match = (
        df["v6_selected"].values
        == df["oracle_model"].values
    )
    print(
        f"\nRouter match rate: {100 * match.mean():.1f}%"
    )

    gap = df["v6_mce"] - df["oracle_mce"]
    print(
        f"Mean gap to oracle: {gap.mean():.4f}"
    )

    delta = df["v6_mce"] - df["MLP_CLASP"]

    print(
        f"Mean delta vs Always MLP+CLASP: {delta.mean():+.4f}"
    )
    print(
        f"Transfers improved vs MLP+CLASP: "
        f"{(delta < 0).sum()}/{len(delta)}"
    )
    print(
        f"Transfers degraded vs MLP+CLASP: "
        f"{(delta > 0).sum()}/{len(delta)}"
    )
    print(
        f"Transfers identical: "
        f"{(delta == 0).sum()}/{len(delta)}"
    )


    RES_DAT.mkdir(parents=True, exist_ok=True)

    out_csv = RES_DAT / "clasp_r_v6_results.csv"
    df.to_csv(out_csv, index=False)

    out_json = RES_DAT / "clasp_r_v6_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "selection_counts": df["v6_selected"].value_counts().to_dict(),
                "router_match_rate": float(match.mean()),
                "mean_gap_to_oracle": float(gap.mean()),
                "mean_delta_vs_mlp_clasp": float(delta.mean()),
                "n_transfers": int(len(df)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(out_csv)
    print(out_json)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

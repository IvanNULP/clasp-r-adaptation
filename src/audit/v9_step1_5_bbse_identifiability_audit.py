import pickle
from pathlib import Path
import numpy as np
import pandas as pd

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

BBSE_RIDGE = 1e-6
BBSE_PINV_RCOND = 1e-8
COND_THRESHOLD = 50

with open(RES_DAT/"v9_frozen_confusion_matrices.pkl", "rb") as f:
    frozen_matrices = pickle.load(f)

print("="*95)
print("CLASP-R v9 — Крок 1.5: BBSE Identifiability/Stability Audit")
print(f"BBSE_RIDGE={BBSE_RIDGE}, BBSE_PINV_RCOND={BBSE_PINV_RCOND}, "
      f"COND_THRESHOLD={COND_THRESHOLD}")
print("="*95)

audit_records = []

for key, entry in frozen_matrices.items():
    C = entry["C"]
    classes = entry["classes"]
    n_cls = len(classes)

    U, S, Vt = np.linalg.svd(C)
    rank = np.linalg.matrix_rank(C)
    min_sv = S.min()
    max_sv = S.max()
    cond_num = max_sv / min_sv if min_sv > 1e-15 else np.inf

    diag = np.diag(C)
    min_diag = diag.min()
    min_diag_class = classes[np.argmin(diag)]

    row_sums_pred = C.sum(axis=1)  # C[predicted, true]
    never_predicted = [classes[i] for i in range(n_cls)
                        if row_sums_pred[i] < 0.01]

    if not np.isfinite(cond_num) or rank < n_cls:
        stability = "SINGULAR_OR_RANK_DEFICIENT"
        recommended_action = "BBSE_DISABLED"
        method_detail = "CLASP = CORAL-only (weights=1)"
    elif cond_num >= COND_THRESHOLD:
        stability = "ILL_CONDITIONED"
        recommended_action = "PSEUDOINVERSE"
        method_detail = f"q_est = pinv(C, rcond={BBSE_PINV_RCOND}) @ q_tilde"
    else:
        stability = "STABLE"
        recommended_action = "RIDGE_SOLVE"
        method_detail = f"q_est = solve(C + {BBSE_RIDGE}*I, q_tilde)"

    audit_records.append({
        "key": key, "dataset": entry["dataset"], "model": entry["model"],
        "n_classes": n_cls, "rank": rank, "rank_deficient": rank < n_cls,
        "min_singular_value": min_sv, "max_singular_value": max_sv,
        "condition_number": cond_num,
        "min_diagonal": min_diag, "min_diagonal_class": min_diag_class,
        "never_predicted_classes": never_predicted,
        "stability_category": stability,
        "recommended_bbse_action": recommended_action,
        "method_detail": method_detail,
    })

    print(f"\n{'─'*80}")
    print(f"{key}  (classes={n_cls})")
    print(f"{'─'*80}")
    print(f"  Rank: {rank}/{n_cls} {'⚠ RANK DEFICIENT' if rank<n_cls else '✓ full rank'}")
    print(f"  Singular values: min={min_sv:.6e}, max={max_sv:.6e}")
    print(f"  Condition number: {cond_num:.4e}" if np.isfinite(cond_num) else "  Condition number: INF")
    print(f"  Min diagonal: {min_diag:.4f} (клас '{min_diag_class}')")
    print(f"  Класи, які модель майже НІКОЛИ не передбачає: "
          f"{never_predicted if never_predicted else 'немає'}")
    print(f"  >>> STABILITY: {stability}  →  ACTION: {recommended_action}")
    print(f"      {method_detail}")

audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(RES_DAT/"v9_bbse_stability_audit.csv", index=False)

print(f"\n{'='*95}")
print("ЗВЕДЕНА ТАБЛИЦЯ")
print(f"{'='*95}")
print(f"\n{'Dataset':12s} {'Model':6s} {'Rank':>8s} {'CondNum':>14s} "
      f"{'Category':28s} {'Action':16s}")
print("─"*90)
for _, r in audit_df.iterrows():
    cond_str = f"{r['condition_number']:.2e}" if np.isfinite(r['condition_number']) else "INF"
    print(f"{r['dataset']:12s} {r['model']:6s} "
          f"{r['rank']}/{r['n_classes']:>4d} {cond_str:>14s} "
          f"{r['stability_category']:28s} {r['recommended_bbse_action']:16s}")

print(f"\nРозподіл категорій:")
print(audit_df["stability_category"].value_counts().to_string())

print(f"\n{'='*95}")
print("ЗАФІКСОВАНА ПОЛІТИКА BBSE (математично однозначна, до Step 2)")
print(f"{'='*95}")
print(f"""
  FULL RANK, cond(C) < {COND_THRESHOLD}  →  RIDGE_SOLVE:
      q_est = solve(C + λI, q_tilde),  λ = {BBSE_RIDGE}

  FULL RANK, cond(C) >= {COND_THRESHOLD}  →  PSEUDOINVERSE:
      q_est = pinv(C, rcond={BBSE_PINV_RCOND}) @ q_tilde

  RANK DEFICIENT / SINGULAR  →  BBSE_DISABLED:
      CLASP = CORAL-only (weights = 1 for all classes)

  Post-processing (для обох STABLE і ILL_CONDITIONED гілок):
      q_est ← clip(q_est, 0, 1)
      q_est ← q_est / Σq_est

  Downstream conversion to class weights (ОКРЕМИЙ крок, НЕ властивість q_est):
      w_c = q_est[c] / p_source[c]
      w_c ← clip(w_c, 0.5, 2.0)

  cond_threshold={COND_THRESHOLD} — pre-specified engineering policy,
  не фундаментальна статистична константа. Зафіксовано ДО Step 2
  і НЕ підлягає зміні після перегляду результатів на 240 instances.
""")

print(f"Saved: {RES_DAT}/v9_bbse_stability_audit.csv")
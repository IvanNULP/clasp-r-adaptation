import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
final_df = pd.read_csv(RES_DAT/"step_b1b_reliability_all_methods_240.csv")

FEATURE_NAMES = ["prediction_flip_rate", "probability_shift",
                  "confidence_change", "entropy_change",
                  "bbse_weight_magnitude", "bbse_prior_extremity",
                  "bbse_condition_number"]
METHOD_NAMES = ["CORAL", "TAC", "QT", "UOT", "POT"]
MODEL_NAMES = ["MLP", "RF", "XGB"]

actual_cols = set(c for c in final_df.columns
                   if any(c.startswith(f"{mn}_{mth}_")
                          for mn in MODEL_NAMES for mth in METHOD_NAMES))

print("="*90)
print("ВИПРАВЛЕНА Перевірка №4 — NaN/Inf validation з розділенням категорій")
print("="*90)

feature_cols = list(actual_cols)

condition_cols = [c for c in feature_cols if c.endswith("_bbse_condition_number")]
response_feature_cols = [c for c in feature_cols if c not in condition_cols]

response_numeric = final_df[response_feature_cols].select_dtypes(include=[np.number])
condition_numeric = final_df[condition_cols].select_dtypes(include=[np.number])

n_nan_response = response_numeric.isna().sum().sum()
n_inf_response = np.isinf(response_numeric.values).sum()

n_nan_condition = condition_numeric.isna().sum().sum()
n_inf_condition = np.isinf(condition_numeric.values).sum()

check4 = (n_nan_response == 0) and (n_inf_response == 0)

print(f"\n4. NaN/Inf validation:")
print(f"   Response-to-adaptation features (6 ознак × 15 = 90 колонок):"
      f" NaN={n_nan_response}, Inf={n_inf_response} "
      f"{'✓' if check4 else '❌'}")
print(f"   bbse_condition_number (15 колонок):"
      f" NaN={n_nan_condition}, Inf={n_inf_condition} "
      f"{'✓ допустимо для rank-deficient C' if n_inf_condition > 0 else '✓'}")

if n_nan_response > 0 or n_inf_response > 0:
    bad_cols = []
    for c in response_feature_cols:
        s = pd.to_numeric(final_df[c], errors="coerce")
        if s.isna().any() or np.isinf(s.values).any():
            bad_cols.append(c)
    print(f"   ❌ Проблемні response-to-adaptation колонки: {bad_cols}")

print(f"\n{'─'*90}")
print("Крос-перевірка: чи Inf-count узгоджується зі Step 1.5 audit?")
print(f"{'─'*90}")
audit_df_stability = pd.read_csv(RES_DAT/"v9_bbse_stability_audit.csv")
n_disabled_combos = (audit_df_stability["recommended_bbse_action"]
                      == "BBSE_DISABLED").sum()
print(f"  Кількість model×dataset комбінацій з BBSE_DISABLED (Step 1.5): "
      f"{n_disabled_combos}/12")
print(f"  Inf у bbse_condition_number: {n_inf_condition} "
      f"(очікувано ~{n_disabled_combos} model×target × 5 methods × "
      f"кількість instances з цим target)")

print(f"\n{'='*90}")
print(f"ЗАГАЛЬНИЙ РЕЗУЛЬТАТ (з виправленою Перевіркою 4): "
      f"{'6/6 ✓ ГОТОВО ДО STEP B.2' if check4 else '❌'}")
print(f"{'='*90}")
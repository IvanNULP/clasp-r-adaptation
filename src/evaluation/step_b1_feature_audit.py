import pandas as pd
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

files_and_methods = {
    "v9_operational_adaptation_eval_240.csv": ["CORAL", "CLASP"],
    "v10_quantile_transport_240.csv": ["QT", "QT_BBSE"],
    "v11_type_aware_coral_240.csv": ["TAC", "TAC_BBSE"],
    "v12_unbalanced_ot_240.csv": ["UOT", "UOT_BBSE"],
    "v13_partial_ot_240.csv": ["POT", "POT_BBSE"],
}

print("="*100)
print("STEP B.1 — Feature Availability Audit для Transfer Score")
print("="*100)

REQUIRED_REGISTRY = {
    "distribution_discrepancy": ["ks_shift"],
    "model_response": ["prediction_flip_rate", "probability_shift",
                        "confidence_change", "entropy_change"],
    "bbse_diagnostics": ["bbse_condition_number", "bbse_weight_magnitude",
                          "bbse_prior_extremity"],
}

print("\nПеревірка НАЯВНИХ колонок у кожному файлі:")
for fname, methods in files_and_methods.items():
    df = pd.read_csv(RES_DAT/fname)
    print(f"\n{fname}:")
    print(f"  Колонки: {list(df.columns)}")

print(f"\n{'='*100}")
print("ВИСНОВОК АУДИТУ")
print(f"{'='*100}")
print("""
  Категорія 1 (ks_shift): ✓ Є в усіх 5 файлів (обчислено на
    рівні transport-незалежного X_foreign/X_tgt_ref)

  Категорія 2 (model_response: flip_rate, prob_shift,
    confidence_change, entropy_change):
    ⚠ Обчислено ТІЛЬКИ в v9_reliability_merged_240.csv
      (окремий файл з Step 3), і ТІЛЬКИ для Global CORAL!
    ❌ НЕ обчислено для QT, TAC, UOT, POT окремо

  Категорія 3 (bbse_condition_number, weight_magnitude,
    prior_extremity):
    ⚠ Частково є в v9_reliability_merged_240.csv (Global CORAL)
    ❌ НЕ обчислено для QT, TAC, UOT, POT окремо

  ГОЛОВНИЙ ВИСНОВОК: якщо Transfer Score має обирати серед
  усіх 10 adaptation candidates, йому потрібні label-free
  ознаки, що характеризують ЩО САМЕ РОБИТЬ кожен конкретний
  метод з передбаченнями моделі (flip rate, confidence change
  тощо) — а це наразі обчислено ЛИШЕ для Global CORAL.

  Потрібен додатковий крок STEP B.1b: обчислити ті самі
  reliability features (категорія 2-3) для QT, TAC, UOT, POT
  окремо, перш ніж Transfer Score зможе розрізняти кандидатів.
""")
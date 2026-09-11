import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
df = pd.read_csv(RES_DAT/"clasp_v7a1_ground_truth.csv")

print("="*100)
print("CLASP-R v7B — Empirical Oracle Upper Bound (n=12 transfer pairs)")
print("Ці числа отримані з доступом до y_foreign і показують емпіричну")
print("верхню межу НА ЦІЙ ВИБІРЦІ — не теоретичну межу методу.")
print("Майбутній label-free router (v8) намагатиметься наблизитись до них.")
print("="*100)

decision_records = []

for (src, tgt), group in df.groupby(["src", "tgt"]):
    row = {"src": src, "tgt": tgt}
    variants = {}
    for _, r in group.iterrows():
        mn = r["model"]
        variants[f"{mn}_BASE"] = r["macro_error_base"]
        variants[f"{mn}_CORAL"] = r["macro_error_coral"]
        variants[f"{mn}_CLASP"] = r["macro_error_coral_bbse"]

    row.update(variants)

    base_options = {k: v for k, v in variants.items() if k.endswith("_BASE")}
    best_base_key = min(base_options, key=base_options.get)
    row["ORACLE_best_base_variant"] = best_base_key
    row["ORACLE_best_base_error"] = base_options[best_base_key]

    clasp_options = {k: v for k, v in variants.items() if k.endswith("_CLASP")}
    best_clasp_key = min(clasp_options, key=clasp_options.get)
    row["ORACLE_3way_model_variant"] = best_clasp_key
    row["ORACLE_3way_model_error"] = clasp_options[best_clasp_key]

    all_options = {k: v for k, v in variants.items()}
    best_overall_key = min(all_options, key=all_options.get)
    row["ORACLE_9way_variant"] = best_overall_key
    row["ORACLE_9way_error"] = all_options[best_overall_key]

    row["oracle_gain_9way_over_base"] = round(
        row["ORACLE_best_base_error"] - row["ORACLE_9way_error"], 4)

    decision_records.append(row)

decision_df = pd.DataFrame(decision_records)

print(f"\n{'Pair':22s} {'ORACLE best-base':22s} "
      f"{'ORACLE 3-way(model)':22s} {'ORACLE 9-way':22s}")
print("─"*95)
for _, r in decision_df.iterrows():
    pair = f"{r['src']}→{r['tgt']}"
    print(f"{pair:22s} "
          f"{r['ORACLE_best_base_variant']:17s}={r['ORACLE_best_base_error']:.4f}  "
          f"{r['ORACLE_3way_model_variant']:17s}={r['ORACLE_3way_model_error']:.4f}  "
          f"{r['ORACLE_9way_variant']:17s}={r['ORACLE_9way_error']:.4f}")

print(f"\n{'='*100}")
print("EMPIRICAL ORACLE UPPER BOUNDS (n=12, обчислено ТІЛЬКИ в цьому скрипті)")
print(f"{'='*100}")

always_mlp_clasp = df[df["model"]=="MLP"]["macro_error_coral_bbse"].mean()
oracle_base_mean  = decision_df["ORACLE_best_base_error"].mean()
oracle_3way_mean  = decision_df["ORACLE_3way_model_error"].mean()
oracle_9way_mean  = decision_df["ORACLE_9way_error"].mean()

print(f"\n  Реальна, вже розгортувана стратегія:")
print(f"    Always MLP+CLASP:            mean = {always_mlp_clasp:.4f}")

print(f"\n  Емпіричний oracle upper bound — 'вибір лише моделі' (n=12):")
print(f"    ORACLE_3way(model):          mean = {oracle_3way_mean:.4f}")
print(f"    Емпіричний розрив над Always-MLP+CLASP: "
      f"{always_mlp_clasp - oracle_3way_mean:+.4f}")
print(f"    → Це найкраще, чого міг би досягти router v8, ЯКЩО він")
print(f"      обирав би лише з {{MLP+CLASP, RF+CLASP, XGB+CLASP}}")
print(f"      і завжди вгадував правильно (на цих 12 парах)")

print(f"\n  Емпіричний oracle upper bound — 'вибір моделі+режиму' (n=12):")
print(f"    ORACLE_9way(model+stage):    mean = {oracle_9way_mean:.4f}")
print(f"    Емпіричний розрив над Always-MLP+CLASP: "
      f"{always_mlp_clasp - oracle_9way_mean:+.4f}")
print(f"    Додатковий розрив понад ORACLE_3way: "
      f"{oracle_3way_mean - oracle_9way_mean:+.4f}")

print(f"\n{'─'*100}")
print("ВИСНОВОК ДЛЯ ПОДАЛЬШОЇ РОБОТИ:")
print(f"{'─'*100}")
print(f"""
  Різниця ORACLE_3way ({oracle_3way_mean:.4f}) vs ORACLE_9way
  ({oracle_9way_mean:.4f}) = {oracle_3way_mean - oracle_9way_mean:+.4f}

  Це емпіричне спостереження на 12 парах, а НЕ статистично
  перевірений результат — n замало для формального тесту
  значущості цієї різниці. Використовувати лише як орієнтир
  для вибору складності майбутнього router v8, не як доведений
  науковий факт.
""")

decision_df.to_csv(RES_DAT/"clasp_v7b_oracle_bounds.csv", index=False)
print(f"Saved: {RES_DAT}/clasp_v7b_oracle_bounds.csv")
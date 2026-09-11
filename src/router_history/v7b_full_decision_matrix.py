import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
df = pd.read_csv(RES_DAT/"clasp_v7a1_ground_truth.csv")

print("="*100)
print("CLASP-R v7B — Повна Decision Matrix (9 варіантів на пару)")
print("="*100)

decision_records = []

for (src, tgt), group in df.groupby(["src", "tgt"]):
    row = {"src": src, "tgt": tgt}

    variants = {}
    for _, r in group.iterrows():
        mn = r["model"]
        variants[f"{mn}_BASE"] = r["macro_error_base"]
        variants[f"{mn}_CORAL"] = r["macro_error_coral"]
        variants[f"{mn}_CLASP"] = r["macro_error_coral_bbse"]  # CORAL+BBSE

    row.update(variants)

    base_options = {k: v for k, v in variants.items() if k.endswith("_BASE")}
    best_base_key = min(base_options, key=base_options.get)
    row["best_base_variant"] = best_base_key
    row["best_base_error"] = base_options[best_base_key]

    clasp_options = {k: v for k, v in variants.items() if k.endswith("_CLASP")}
    best_clasp_key = min(clasp_options, key=clasp_options.get)
    row["best_clasp_variant"] = best_clasp_key
    row["best_clasp_error"] = clasp_options[best_clasp_key]

    all_options = {k: v for k, v in variants.items()}
    best_overall_key = min(all_options, key=all_options.get)
    row["oracle_9way_variant"] = best_overall_key
    row["oracle_9way_error"] = all_options[best_overall_key]

    row["gain_over_best_base"] = round(
        row["best_base_error"] - row["oracle_9way_error"], 4)

    decision_records.append(row)

decision_df = pd.DataFrame(decision_records)

print(f"\n{'Pair':22s} {'Best BASE':20s} {'Best CLASP':20s} "
      f"{'ORACLE (9-way)':20s} {'Gain':>7s}")
print("─"*95)
for _, r in decision_df.iterrows():
    pair = f"{r['src']}→{r['tgt']}"
    print(f"{pair:22s} "
          f"{r['best_base_variant']:15s}={r['best_base_error']:.4f}  "
          f"{r['best_clasp_variant']:15s}={r['best_clasp_error']:.4f}  "
          f"{r['oracle_9way_variant']:15s}={r['oracle_9way_error']:.4f}  "
          f"{r['gain_over_best_base']:>+7.4f}")

print(f"\n{'='*100}")
print("Розподіл переможців серед усіх 9 варіантів (модель × стадія)")
print(f"{'='*100}")
print(decision_df["oracle_9way_variant"].value_counts().to_string())

print(f"\n{'='*100}")
print("ТРИ МОЖЛИВІ ЗАВДАННЯ ДЛЯ ROUTER — порівняння верхньої межі якості")
print(f"{'='*100}")

always_mlp_clasp = df[df["model"]=="MLP"]["macro_error_coral_bbse"].mean()
best_base_mean   = decision_df["best_base_error"].mean()
best_clasp_mean  = decision_df["best_clasp_error"].mean()
oracle_9way_mean = decision_df["oracle_9way_error"].mean()

print(f"\n  Стратегія 1 — Always MLP+CLASP (наш поточний фінальний метод):")
print(f"    mean macro_error = {always_mlp_clasp:.4f}")

print(f"\n  Стратегія 2 — Router вибирає ТІЛЬКИ модель "
      f"(завжди CORAL+BBSE, як у v4/v6):")
print(f"    mean macro_error (oracle серед 3 CLASP-варіантів) = "
      f"{best_clasp_mean:.4f}")
print(f"    Потенційний виграш над Always-MLP+CLASP: "
      f"{always_mlp_clasp - best_clasp_mean:+.4f}")

print(f"\n  Стратегія 3 — Router вибирає МОДЕЛЬ + РЕЖИМ АДАПТАЦІЇ "
      f"(9-way, найскладніша):")
print(f"    mean macro_error (oracle серед усіх 9 варіантів) = "
      f"{oracle_9way_mean:.4f}")
print(f"    Потенційний виграш над Always-MLP+CLASP: "
      f"{always_mlp_clasp - oracle_9way_mean:+.4f}")
print(f"    Додатковий виграш понад Стратегію 2 (лише вибір моделі): "
      f"{best_clasp_mean - oracle_9way_mean:+.4f}")

decision_df.to_csv(RES_DAT/"clasp_v7b_decision_matrix.csv", index=False)
print(f"\nSaved: {RES_DAT}/clasp_v7b_decision_matrix.csv")
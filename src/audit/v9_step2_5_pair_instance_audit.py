import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")

print("="*100)
print("CLASP-R v9 — Крок 2.5 (виправлено): Pair/Instance-Level Audit")
print("="*100)

rng = np.random.default_rng(42)  # ВИПРАВЛЕНО: детермінований, локальний rng
pair_ids = v9_df["pair_id"].unique()

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn}\n{'='*90}")

    v9_df[f"{mn}_delta_coral"] = v9_df[f"{mn}_CORAL_error"] - v9_df[f"{mn}_BASE_error"]
    v9_df[f"{mn}_delta_clasp"] = v9_df[f"{mn}_CLASP_error"] - v9_df[f"{mn}_BASE_error"]
    v9_df[f"{mn}_delta_clasp_vs_coral"] = v9_df[f"{mn}_CLASP_error"] - v9_df[f"{mn}_CORAL_error"]

    for transition, col in [("BASE→CORAL", f"{mn}_delta_coral"),
                             ("BASE→CLASP", f"{mn}_delta_clasp"),
                             ("CORAL→CLASP", f"{mn}_delta_clasp_vs_coral")]:
        d = v9_df[col]
        print(f"\n  {transition} (n=240 instances):")
        print(f"    Mean:     {d.mean():+.4f}")
        print(f"    Median:   {d.median():+.4f}")
        print(f"    Std:      {d.std():.4f}")
        print(f"    % improved (d<0): {(d<0).mean()*100:.1f}%")
        print(f"    % degraded (d>0): {(d>0).mean()*100:.1f}%")

    for transition, col in [("BASE→CORAL", f"{mn}_delta_coral"),
                             ("BASE→CLASP", f"{mn}_delta_clasp"),
                             ("CORAL→CLASP", f"{mn}_delta_clasp_vs_coral")]:
        boot_means = []
        for _ in range(2000):
            sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
            boot_data = pd.concat(
                [v9_df[v9_df["pair_id"]==p] for p in sampled],
                ignore_index=True)
            boot_means.append(boot_data[col].mean())

        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
        observed = v9_df[col].mean()
        robust_direction = ("покращення" if ci_h < 0
                             else ("погіршення" if ci_l > 0 else "невизначено"))

        print(f"\n  {transition} cluster-bootstrap 95% CI: "
              f"[{ci_l:+.4f}, {ci_h:+.4f}] (mean={observed:+.4f}) "
              f"→ {robust_direction}")

    print(f"\n  Pair-level means (BASE → CORAL → CLASP):")
    print(f"  {'Pair':22s} {'BASE':>8s} {'CORAL':>8s} {'CLASP':>8s} "
          f"{'ΔCORAL':>9s} {'ΔCLASP':>9s} {'ΔC→Cl':>9s}")
    for pair_id, group in v9_df.groupby("pair_id"):
        base = group[f"{mn}_BASE_error"].mean()
        coral = group[f"{mn}_CORAL_error"].mean()
        clasp = group[f"{mn}_CLASP_error"].mean()
        print(f"  {pair_id:22s} {base:>8.4f} {coral:>8.4f} {clasp:>8.4f} "
              f"{coral-base:>+9.4f} {clasp-base:>+9.4f} {clasp-coral:>+9.4f}")

    n_pairs_improved_clasp = sum(
        1 for pair_id, group in v9_df.groupby("pair_id")
        if group[f"{mn}_CLASP_error"].mean() < group[f"{mn}_BASE_error"].mean())
    n_pairs_bbse_helps = sum(
        1 for pair_id, group in v9_df.groupby("pair_id")
        if group[f"{mn}_CLASP_error"].mean() < group[f"{mn}_CORAL_error"].mean())
    print(f"\n  Domain pairs де CLASP покращує BASE:        {n_pairs_improved_clasp}/12")
    print(f"  Domain pairs де BBSE покращує над CORAL:     {n_pairs_bbse_helps}/12")

print(f"\n{'='*100}")
print("ПІДСУМОК: яка модель має стабільний unsupervised CLASP-ефект?")
print(f"{'='*100}")
for mn in ["MLP","RF","XGB"]:
    d = v9_df[f"{mn}_delta_clasp"]
    n_pairs_improved = sum(
        1 for pair_id, group in v9_df.groupby("pair_id")
        if group[f"{mn}_CLASP_error"].mean() < group[f"{mn}_BASE_error"].mean())
    print(f"  {mn:5s}: mean_delta={d.mean():+.4f}, "
          f"% instance improved={((d<0).mean()*100):.1f}%, "
          f"pairs improved={n_pairs_improved}/12")

v9_df.to_csv(RES_DAT/"v9_step2_5_audit.csv", index=False)
print(f"\nSaved: {RES_DAT}/v9_step2_5_audit.csv")
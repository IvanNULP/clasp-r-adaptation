import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")
v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")

print("="*100)
print("Крок 4B.5 (виправлено) — Cluster-Bootstrap Audit: Type-aware CORAL")
print("="*100)

rng = np.random.default_rng(42)
pair_ids = tac_df["pair_id"].unique()

def cluster_bootstrap_ci(df, col, pair_ids, rng, n_boot=2000):
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        boot_data = pd.concat(
            [df[df["pair_id"]==p] for p in sampled], ignore_index=True)
        boot_means.append(boot_data[col].mean())
    ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
    direction = ("покращення" if ci_h < 0
                 else ("погіршення" if ci_l > 0 else "невизначено"))
    return ci_l, ci_h, direction

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn}\n{'='*90}")

    tac_df[f"{mn}_delta_tac"] = tac_df[f"{mn}_TAC_error"] - tac_df[f"{mn}_BASE_error"]
    tac_df[f"{mn}_delta_tac_bbse"] = tac_df[f"{mn}_TAC_BBSE_error"] - tac_df[f"{mn}_BASE_error"]
    tac_df[f"{mn}_delta_bbse_vs_tac"] = tac_df[f"{mn}_TAC_BBSE_error"] - tac_df[f"{mn}_TAC_error"]

    for transition, col in [("BASE→TAC", f"{mn}_delta_tac"),
                             ("BASE→TAC+BBSE", f"{mn}_delta_tac_bbse"),
                             ("TAC→TAC+BBSE", f"{mn}_delta_bbse_vs_tac")]:
        d = tac_df[col]
        print(f"\n  {transition} (n=240 instances):")
        print(f"    Mean:     {d.mean():+.4f}")
        print(f"    Median:   {d.median():+.4f}")
        print(f"    Std:      {d.std():.4f}")
        print(f"    % improved (d<0): {(d<0).mean()*100:.1f}%")

        ci_l, ci_h, direction = cluster_bootstrap_ci(tac_df, col, pair_ids, rng)
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}] → {direction}")

    print(f"\n  Pair-level means:")
    print(f"  {'Pair':22s} {'BASE':>8s} {'TAC':>8s} {'TAC+BBSE':>9s} "
          f"{'ΔTAC':>9s} {'ΔTAC+BBSE':>10s}")
    for pair_id, group in tac_df.groupby("pair_id"):
        base = group[f"{mn}_BASE_error"].mean()
        tac = group[f"{mn}_TAC_error"].mean()
        tac_bbse = group[f"{mn}_TAC_BBSE_error"].mean()
        print(f"  {pair_id:22s} {base:>8.4f} {tac:>8.4f} {tac_bbse:>9.4f} "
              f"{tac-base:>+9.4f} {tac_bbse-base:>+10.4f}")

    n_pairs_tac_improve = sum(
        1 for pair_id, group in tac_df.groupby("pair_id")
        if group[f"{mn}_TAC_error"].mean() < group[f"{mn}_BASE_error"].mean())
    n_pairs_tacbbse_improve = sum(
        1 for pair_id, group in tac_df.groupby("pair_id")
        if group[f"{mn}_TAC_BBSE_error"].mean() < group[f"{mn}_BASE_error"].mean())
    print(f"\n  Domain pairs де TAC покращує BASE:        {n_pairs_tac_improve}/12")
    print(f"  Domain pairs де TAC+BBSE покращує BASE:    {n_pairs_tacbbse_improve}/12")

print(f"\n{'='*100}")
print("ПРЯМЕ ПОРІВНЯННЯ: Type-aware CORAL vs Global CORAL")
print(f"{'='*100}")

cmp_df = tac_df[["instance_key", "pair_id"]].copy()

for mn in ["MLP", "RF", "XGB"]:
    v9_sub = v9_df[["instance_key", f"{mn}_CORAL_error", f"{mn}_CLASP_error"]].copy()
    tac_sub = tac_df[["instance_key", f"{mn}_TAC_error", f"{mn}_TAC_BBSE_error"]].copy()

    tmp = (
        cmp_df
        .merge(v9_sub, on="instance_key", how="inner", validate="one_to_one")
        .merge(tac_sub, on="instance_key", how="inner", validate="one_to_one")
    )

    tmp["TAC_minus_CORAL"] = tmp[f"{mn}_TAC_error"] - tmp[f"{mn}_CORAL_error"]
    tmp["TACBBSE_minus_CLASP"] = tmp[f"{mn}_TAC_BBSE_error"] - tmp[f"{mn}_CLASP_error"]

    print(f"\n{'='*80}\n{mn}\n{'='*80}")
    for label, col in [("TAC vs CORAL", "TAC_minus_CORAL"),
                        ("TAC+BBSE vs CORAL+BBSE", "TACBBSE_minus_CLASP")]:
        d = tmp[col]
        ci_l, ci_h, _ = cluster_bootstrap_ci(tmp, col, pair_ids, rng)
        direction = ("TAC КРАЩИЙ" if ci_h < 0
                     else ("TAC ГІРШИЙ" if ci_l > 0 else "НЕВИЗНАЧЕНО"))
        print(f"\n  {label}:")
        print(f"    Mean difference: {d.mean():+.4f}")
        print(f"    Median difference: {d.median():+.4f}")
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}]")
        print(f"    → {direction}")

print(f"\n{'='*100}")
print("ПРЯМЕ ПОРІВНЯННЯ: Type-aware CORAL vs Quantile Transport")
print(f"{'='*100}")

for mn in ["MLP", "RF", "XGB"]:
    qt_sub = qt_df[["instance_key", f"{mn}_QT_error", f"{mn}_QT_BBSE_error"]].copy()
    tac_sub = tac_df[["instance_key", f"{mn}_TAC_error", f"{mn}_TAC_BBSE_error"]].copy()

    tmp = (
        cmp_df
        .merge(qt_sub, on="instance_key", how="inner", validate="one_to_one")
        .merge(tac_sub, on="instance_key", how="inner", validate="one_to_one")
    )

    tmp["TAC_minus_QT"] = tmp[f"{mn}_TAC_error"] - tmp[f"{mn}_QT_error"]
    tmp["TACBBSE_minus_QTBBSE"] = tmp[f"{mn}_TAC_BBSE_error"] - tmp[f"{mn}_QT_BBSE_error"]

    print(f"\n{'='*80}\n{mn}\n{'='*80}")
    for label, col in [("TAC vs QT", "TAC_minus_QT"),
                        ("TAC+BBSE vs QT+BBSE", "TACBBSE_minus_QTBBSE")]:
        d = tmp[col]
        ci_l, ci_h, _ = cluster_bootstrap_ci(tmp, col, pair_ids, rng)
        direction = ("TAC КРАЩИЙ" if ci_h < 0
                     else ("TAC ГІРШИЙ" if ci_l > 0 else "НЕВИЗНАЧЕНО"))
        print(f"\n  {label}:")
        print(f"    Mean difference: {d.mean():+.4f}")
        print(f"    Median difference: {d.median():+.4f}")
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}]")
        print(f"    → {direction}")

tac_df.to_csv(RES_DAT/"v11_tac_step4b5_audit.csv", index=False)
print(f"\nSaved: {RES_DAT}/v11_tac_step4b5_audit.csv")
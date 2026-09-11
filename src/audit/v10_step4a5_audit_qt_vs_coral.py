import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")

print("="*100)
print("Крок 4A.5 — Cluster-Bootstrap Audit: Quantile Transport")
print("="*100)

rng = np.random.default_rng(42)
pair_ids = qt_df["pair_id"].unique()

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn}\n{'='*90}")

    qt_df[f"{mn}_delta_qt"] = qt_df[f"{mn}_QT_error"] - qt_df[f"{mn}_BASE_error"]
    qt_df[f"{mn}_delta_qt_bbse"] = qt_df[f"{mn}_QT_BBSE_error"] - qt_df[f"{mn}_BASE_error"]
    qt_df[f"{mn}_delta_bbse_vs_qt"] = qt_df[f"{mn}_QT_BBSE_error"] - qt_df[f"{mn}_QT_error"]

    for transition, col in [("BASE→QT", f"{mn}_delta_qt"),
                             ("BASE→QT+BBSE", f"{mn}_delta_qt_bbse"),
                             ("QT→QT+BBSE", f"{mn}_delta_bbse_vs_qt")]:
        d = qt_df[col]
        print(f"\n  {transition} (n=240 instances):")
        print(f"    Mean:     {d.mean():+.4f}")
        print(f"    Median:   {d.median():+.4f}")
        print(f"    % improved (d<0): {(d<0).mean()*100:.1f}%")

        boot_means = []
        for _ in range(2000):
            sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
            boot_data = pd.concat(
                [qt_df[qt_df["pair_id"]==p] for p in sampled], ignore_index=True)
            boot_means.append(boot_data[col].mean())
        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
        robust_direction = ("покращення" if ci_h < 0
                             else ("погіршення" if ci_l > 0 else "невизначено"))
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}] "
              f"→ {robust_direction}")

    print(f"\n  Pair-level means:")
    print(f"  {'Pair':22s} {'BASE':>8s} {'QT':>8s} {'QT+BBSE':>9s} "
          f"{'ΔQT':>9s} {'ΔQT+BBSE':>10s}")
    for pair_id, group in qt_df.groupby("pair_id"):
        base = group[f"{mn}_BASE_error"].mean()
        qt = group[f"{mn}_QT_error"].mean()
        qt_bbse = group[f"{mn}_QT_BBSE_error"].mean()
        print(f"  {pair_id:22s} {base:>8.4f} {qt:>8.4f} {qt_bbse:>9.4f} "
              f"{qt-base:>+9.4f} {qt_bbse-base:>+10.4f}")

    n_pairs_qt_improve = sum(
        1 for pair_id, group in qt_df.groupby("pair_id")
        if group[f"{mn}_QT_error"].mean() < group[f"{mn}_BASE_error"].mean())
    n_pairs_qtbbse_improve = sum(
        1 for pair_id, group in qt_df.groupby("pair_id")
        if group[f"{mn}_QT_BBSE_error"].mean() < group[f"{mn}_BASE_error"].mean())
    print(f"\n  Domain pairs де QT покращує BASE:        {n_pairs_qt_improve}/12")
    print(f"  Domain pairs де QT+BBSE покращує BASE:    {n_pairs_qtbbse_improve}/12")

print(f"\n{'='*100}")
print("ПРЯМЕ ПОРІВНЯННЯ: Global CORAL vs Quantile Transport")
print(f"{'='*100}")

cmp_df = qt_df[["instance_key", "pair_id"]].copy()

for mn in ["MLP", "RF", "XGB"]:
    tmp = cmp_df.copy()

    v9_cols = ["instance_key", f"{mn}_CORAL_error", f"{mn}_CLASP_error"]
    v9_sub = v9_df[v9_cols].copy()
    tmp = tmp.merge(v9_sub, on="instance_key", how="inner")

    tmp["QT_minus_CORAL"] = (
        qt_df.loc[qt_df["instance_key"].isin(tmp["instance_key"]),
                  f"{mn}_QT_error"].values - tmp[f"{mn}_CORAL_error"].values
    )
    tmp["QTBBSE_minus_CLASP"] = (
        qt_df.loc[qt_df["instance_key"].isin(tmp["instance_key"]),
                  f"{mn}_QT_BBSE_error"].values - tmp[f"{mn}_CLASP_error"].values
    )

    print(f"\n{'='*80}\n{mn}\n{'='*80}")

    for label, col in [("QT vs CORAL", "QT_minus_CORAL"),
                        ("QT+BBSE vs CORAL+BBSE", "QTBBSE_minus_CLASP")]:
        d = tmp[col]

        boot_means = []
        for _ in range(2000):
            sampled_pairs = rng.choice(pair_ids, len(pair_ids), replace=True)
            boot_data = pd.concat(
                [tmp[tmp["pair_id"] == p] for p in sampled_pairs],
                ignore_index=True)
            boot_means.append(boot_data[col].mean())

        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
        direction = ("QT КРАЩИЙ" if ci_h < 0
                     else ("QT ГІРШИЙ" if ci_l > 0 else "НЕВИЗНАЧЕНО"))

        print(f"\n  {label}:")
        print(f"    Mean difference: {d.mean():+.4f}")
        print(f"    Median difference: {d.median():+.4f}")
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}]")
        print(f"    → {direction}")

qt_df.to_csv(RES_DAT/"v10_qt_step4a5_audit.csv", index=False)
print(f"\nSaved: {RES_DAT}/v10_qt_step4a5_audit.csv")
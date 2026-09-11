import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
pot_df = pd.read_csv(RES_DAT/"v13_partial_ot_240.csv")
v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")

print("="*100)
print("Крок 4D.5 (фінальний) — Cluster-Bootstrap Audit: Partial OT")
print("="*100)

rng = np.random.default_rng(42)
pair_ids = pot_df["pair_id"].unique()

def cluster_bootstrap_ci(df, col, pair_ids, rng, n_boot=2000):
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        boot_data = pd.concat([df[df["pair_id"]==p] for p in sampled], ignore_index=True)
        boot_means.append(boot_data[col].mean())
    ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
    if ci_h < 0: verdict = "ROBUST IMPROVEMENT"
    elif ci_l > 0: verdict = "ROBUST DEGRADATION"
    else: verdict = "UNKNOWN"
    return ci_l, ci_h, verdict

def print_transition(df, label, col, pair_ids, rng):
    d = df[col]
    ci_l, ci_h, verdict = cluster_bootstrap_ci(df, col, pair_ids, rng)
    print(f"\n  {label}:")
    print(f"    Mean: {d.mean():+.4f}  Median: {d.median():+.4f}  "
          f"Std: {d.std():.4f}  %improved: {100*(d<0).mean():.1f}%")
    print(f"    CI: [{ci_l:+.4f}, {ci_h:+.4f}] → {verdict}")
    return verdict

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn}\n{'='*90}")
    pot_df[f"{mn}_delta_pot"] = pot_df[f"{mn}_POT_error"] - pot_df[f"{mn}_BASE_error"]
    pot_df[f"{mn}_delta_pot_bbse"] = pot_df[f"{mn}_POT_BBSE_error"] - pot_df[f"{mn}_BASE_error"]
    pot_df[f"{mn}_delta_bbse_vs_pot"] = pot_df[f"{mn}_POT_BBSE_error"] - pot_df[f"{mn}_POT_error"]

    print_transition(pot_df, "BASE→POT", f"{mn}_delta_pot", pair_ids, rng)
    print_transition(pot_df, "BASE→POT+BBSE", f"{mn}_delta_pot_bbse", pair_ids, rng)
    print_transition(pot_df, "POT→POT+BBSE", f"{mn}_delta_bbse_vs_pot", pair_ids, rng)

    print(f"\n  Pair-level means:")
    for pair_id, group in pot_df.groupby("pair_id"):
        base = group[f"{mn}_BASE_error"].mean()
        pot = group[f"{mn}_POT_error"].mean()
        pot_bbse = group[f"{mn}_POT_BBSE_error"].mean()
        print(f"  {pair_id:22s} BASE={base:.4f} POT={pot:.4f} POT+BBSE={pot_bbse:.4f} "
              f"ΔPOT={pot-base:+.4f} ΔPOT+BBSE={pot_bbse-base:+.4f}")

    n_pairs_pot_improve = sum(
        1 for pair_id, group in pot_df.groupby("pair_id")
        if group[f"{mn}_POT_error"].mean() < group[f"{mn}_BASE_error"].mean())
    n_pairs_potbbse_improve = sum(
        1 for pair_id, group in pot_df.groupby("pair_id")
        if group[f"{mn}_POT_BBSE_error"].mean() < group[f"{mn}_BASE_error"].mean())
    print(f"\n  Domain pairs де POT покращує BASE:        {n_pairs_pot_improve}/12")
    print(f"  Domain pairs де POT+BBSE покращує BASE:    {n_pairs_potbbse_improve}/12")

print(f"\n{'='*100}")
print("ПРЯМІ ПОРІВНЯННЯ: POT vs Global CORAL / TAC / QT / UOT")
print(f"{'='*100}")

cmp_df = pot_df[["instance_key", "pair_id"]].copy()

for mn in ["MLP","RF","XGB"]:
    v9_sub = v9_df[["instance_key", f"{mn}_CORAL_error", f"{mn}_CLASP_error"]].copy()
    qt_sub = qt_df[["instance_key", f"{mn}_QT_error", f"{mn}_QT_BBSE_error"]].copy()
    tac_sub = tac_df[["instance_key", f"{mn}_TAC_error", f"{mn}_TAC_BBSE_error"]].copy()
    uot_sub = uot_df[["instance_key", f"{mn}_UOT_error", f"{mn}_UOT_BBSE_error"]].copy()
    pot_sub = pot_df[["instance_key", f"{mn}_POT_error", f"{mn}_POT_BBSE_error"]].copy()

    tmp = (cmp_df
           .merge(v9_sub, on="instance_key", validate="one_to_one")
           .merge(qt_sub, on="instance_key", validate="one_to_one")
           .merge(tac_sub, on="instance_key", validate="one_to_one")
           .merge(uot_sub, on="instance_key", validate="one_to_one")
           .merge(pot_sub, on="instance_key", validate="one_to_one"))

    tmp["POT_minus_CORAL"] = tmp[f"{mn}_POT_error"] - tmp[f"{mn}_CORAL_error"]
    tmp["POTBBSE_minus_CLASP"] = tmp[f"{mn}_POT_BBSE_error"] - tmp[f"{mn}_CLASP_error"]
    tmp["POT_minus_TAC"] = tmp[f"{mn}_POT_error"] - tmp[f"{mn}_TAC_error"]
    tmp["POTBBSE_minus_TACBBSE"] = tmp[f"{mn}_POT_BBSE_error"] - tmp[f"{mn}_TAC_BBSE_error"]
    tmp["POT_minus_QT"] = tmp[f"{mn}_POT_error"] - tmp[f"{mn}_QT_error"]
    tmp["POTBBSE_minus_QTBBSE"] = tmp[f"{mn}_POT_BBSE_error"] - tmp[f"{mn}_QT_BBSE_error"]
    tmp["POT_minus_UOT"] = tmp[f"{mn}_POT_error"] - tmp[f"{mn}_UOT_error"]
    tmp["POTBBSE_minus_UOTBBSE"] = tmp[f"{mn}_POT_BBSE_error"] - tmp[f"{mn}_UOT_BBSE_error"]

    print(f"\n{'='*80}\n{mn}\n{'='*80}")
    for label, col in [
        ("POT vs Global CORAL", "POT_minus_CORAL"),
        ("POT+BBSE vs CORAL+BBSE", "POTBBSE_minus_CLASP"),
        ("POT vs Type-aware CORAL", "POT_minus_TAC"),
        ("POT+BBSE vs TAC+BBSE", "POTBBSE_minus_TACBBSE"),
        ("POT vs QT", "POT_minus_QT"),
        ("POT+BBSE vs QT+BBSE", "POTBBSE_minus_QTBBSE"),
        ("POT vs UOT", "POT_minus_UOT"),
        ("POT+BBSE vs UOT+BBSE", "POTBBSE_minus_UOTBBSE"),
    ]:
        d = tmp[col]
        ci_l, ci_h, _ = cluster_bootstrap_ci(tmp, col, pair_ids, rng)
        if ci_h < 0: verdict = "POT КРАЩИЙ"
        elif ci_l > 0: verdict = "POT ГІРШИЙ"
        else: verdict = "НЕВИЗНАЧЕНО"
        print(f"\n  {label}:")
        print(f"    Mean: {d.mean():+.4f}  Median: {d.median():+.4f}")
        print(f"    CI: [{ci_l:+.4f}, {ci_h:+.4f}] → {verdict}")

pot_df.to_csv(RES_DAT/"v13_pot_step4d5_audit.csv", index=False)
print(f"\nSaved: {RES_DAT}/v13_pot_step4d5_audit.csv")

print(f"\n{'='*100}")
print("НАГАДУВАННЯ: параметри POT (n_sub=500, m=0.8, seed=5555)")
print("ЗАМОРОЖЕНІ. Це ОСТАННІЙ transport-метод у benchmark.")
print(f"{'='*100}")
import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")
v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")

print("="*100)
print("Крок 4C.5 (повний) — Cluster-Bootstrap Audit: Unbalanced OT")
print("="*100)

rng = np.random.default_rng(42)
pair_ids = uot_df["pair_id"].unique()

def cluster_bootstrap_ci(df, col, pair_ids, rng, n_boot=2000):
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        boot_data = pd.concat([df[df["pair_id"]==p] for p in sampled], ignore_index=True)
        boot_means.append(boot_data[col].mean())
    ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
    if ci_h < 0:
        verdict = "ROBUST IMPROVEMENT"
    elif ci_l > 0:
        verdict = "ROBUST DEGRADATION"
    else:
        verdict = "UNKNOWN"
    return ci_l, ci_h, verdict

def print_transition(df, label, col, pair_ids, rng):
    d = df[col]
    ci_l, ci_h, verdict = cluster_bootstrap_ci(df, col, pair_ids, rng)
    print(f"\n  {label}:")
    print(f"    Mean: {d.mean():+.4f}  Median: {d.median():+.4f}  "
          f"Std: {d.std():.4f}  %improved: {100*(d<0).mean():.1f}%")
    print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}] → {verdict}")
    return verdict

for mn in ["MLP","RF","XGB"]:
    print(f"\n{'='*90}\n{mn}\n{'='*90}")
    uot_df[f"{mn}_delta_uot"] = uot_df[f"{mn}_UOT_error"] - uot_df[f"{mn}_BASE_error"]
    uot_df[f"{mn}_delta_uot_bbse"] = uot_df[f"{mn}_UOT_BBSE_error"] - uot_df[f"{mn}_BASE_error"]
    uot_df[f"{mn}_delta_bbse_vs_uot"] = uot_df[f"{mn}_UOT_BBSE_error"] - uot_df[f"{mn}_UOT_error"]

    print_transition(uot_df, "BASE→UOT", f"{mn}_delta_uot", pair_ids, rng)
    print_transition(uot_df, "BASE→UOT+BBSE", f"{mn}_delta_uot_bbse", pair_ids, rng)
    print_transition(uot_df, "UOT→UOT+BBSE", f"{mn}_delta_bbse_vs_uot", pair_ids, rng)

    print(f"\n  Pair-level means:")
    for pair_id, group in uot_df.groupby("pair_id"):
        base = group[f"{mn}_BASE_error"].mean()
        uot = group[f"{mn}_UOT_error"].mean()
        uot_bbse = group[f"{mn}_UOT_BBSE_error"].mean()
        print(f"  {pair_id:22s} BASE={base:.4f} UOT={uot:.4f} UOT+BBSE={uot_bbse:.4f} "
              f"ΔUOT={uot-base:+.4f} ΔUOT+BBSE={uot_bbse-base:+.4f}")

print(f"\n{'='*100}")
print("ПРЯМІ ПОРІВНЯННЯ: UOT vs Global CORAL / Type-aware CORAL / QT")
print(f"{'='*100}")

cmp_df = uot_df[["instance_key", "pair_id"]].copy()

for mn in ["MLP","RF","XGB"]:
    v9_sub = v9_df[["instance_key", f"{mn}_CORAL_error", f"{mn}_CLASP_error"]].copy()
    qt_sub = qt_df[["instance_key", f"{mn}_QT_error", f"{mn}_QT_BBSE_error"]].copy()
    tac_sub = tac_df[["instance_key", f"{mn}_TAC_error", f"{mn}_TAC_BBSE_error"]].copy()
    uot_sub = uot_df[["instance_key", f"{mn}_UOT_error", f"{mn}_UOT_BBSE_error"]].copy()

    tmp = (cmp_df
           .merge(v9_sub, on="instance_key", validate="one_to_one")
           .merge(qt_sub, on="instance_key", validate="one_to_one")
           .merge(tac_sub, on="instance_key", validate="one_to_one")
           .merge(uot_sub, on="instance_key", validate="one_to_one"))

    tmp["UOT_minus_CORAL"] = tmp[f"{mn}_UOT_error"] - tmp[f"{mn}_CORAL_error"]
    tmp["UOTBBSE_minus_CLASP"] = tmp[f"{mn}_UOT_BBSE_error"] - tmp[f"{mn}_CLASP_error"]
    tmp["UOT_minus_TAC"] = tmp[f"{mn}_UOT_error"] - tmp[f"{mn}_TAC_error"]
    tmp["UOTBBSE_minus_TACBBSE"] = tmp[f"{mn}_UOT_BBSE_error"] - tmp[f"{mn}_TAC_BBSE_error"]
    tmp["UOT_minus_QT"] = tmp[f"{mn}_UOT_error"] - tmp[f"{mn}_QT_error"]
    tmp["UOTBBSE_minus_QTBBSE"] = tmp[f"{mn}_UOT_BBSE_error"] - tmp[f"{mn}_QT_BBSE_error"]

    print(f"\n{'='*80}\n{mn}\n{'='*80}")
    for label, col in [
        ("UOT vs Global CORAL", "UOT_minus_CORAL"),
        ("UOT+BBSE vs CORAL+BBSE", "UOTBBSE_minus_CLASP"),
        ("UOT vs Type-aware CORAL", "UOT_minus_TAC"),
        ("UOT+BBSE vs TAC+BBSE", "UOTBBSE_minus_TACBBSE"),
        ("UOT vs QT", "UOT_minus_QT"),
        ("UOT+BBSE vs QT+BBSE", "UOTBBSE_minus_QTBBSE"),
    ]:
        d = tmp[col]
        ci_l, ci_h, _ = cluster_bootstrap_ci(tmp, col, pair_ids, rng)
        if ci_h < 0:
            verdict = "UOT КРАЩИЙ"
        elif ci_l > 0:
            verdict = "UOT ГІРШИЙ"
        else:
            verdict = "НЕВИЗНАЧЕНО"
        print(f"\n  {label}:")
        print(f"    Mean: {d.mean():+.4f}  Median: {d.median():+.4f}")
        print(f"    CI: [{ci_l:+.4f}, {ci_h:+.4f}] → {verdict}")

uot_df.to_csv(RES_DAT/"v12_uot_step4c5_audit.csv", index=False)
print(f"\nSaved: {RES_DAT}/v12_uot_step4c5_audit.csv")

print(f"\n{'='*100}")
print("НАГАДУВАННЯ: параметри UOT (n_sub=500, reg=0.05, reg_m=1.0, seed=5555)")
print("ЗАМОРОЖЕНІ і НЕ підлягають зміні після цих результатів.")
print(f"{'='*100}")
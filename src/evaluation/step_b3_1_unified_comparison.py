import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")
pot_df = pd.read_csv(RES_DAT/"v13_partial_ot_240.csv")

MODELS = ["MLP", "RF", "XGB"]

print("="*100)
print("STEP B.3.1 (виправлено) — Unified Statistical Comparison")
print("="*100)

print("\nПеревірка узгодженості 5 файлів (локально, незалежно від B.1e):")
reference_keys = set(v9_df["instance_key"])

for name, df in [("v9", v9_df), ("QT", qt_df), ("TAC", tac_df),
                  ("UOT", uot_df), ("POT", pot_df)]:
    assert len(df) == 240, f"{name}: expected 240 rows, got {len(df)}"
    assert set(df["instance_key"]) == reference_keys, \
        f"{name}: instance_key set differs from v9"
    assert df["instance_key"].is_unique, f"{name}: duplicate instance_key"
    print(f"  {name}: 240 rows, instance_key set matches v9, unique ✓")

METHODS = {
    "CORAL":      (v9_df,  "CORAL"),
    "CORAL+BBSE": (v9_df,  "CLASP"),
    "TAC":        (tac_df, "TAC"),
    "TAC+BBSE":   (tac_df, "TAC_BBSE"),
    "QT":         (qt_df,  "QT"),
    "QT+BBSE":    (qt_df,  "QT_BBSE"),
    "UOT":        (uot_df, "UOT"),
    "UOT+BBSE":   (uot_df, "UOT_BBSE"),
    "POT":        (pot_df, "POT"),
    "POT+BBSE":   (pot_df, "POT_BBSE"),
}

rng = np.random.default_rng(42)

def cluster_bootstrap_ci(df, col, pair_ids, rng, n_boot=2000):
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        boot_data = pd.concat([df[df["pair_id"]==p] for p in sampled], ignore_index=True)
        boot_means.append(boot_data[col].mean())
    ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
    if ci_h < 0: verdict = "IMPROVE"
    elif ci_l > 0: verdict = "DEGRADE"
    else: verdict = "UNKNOWN"
    return ci_l, ci_h, verdict

records = []

for candidate_name, (df, col_suffix) in METHODS.items():
    for mn in MODELS:
        base_col = f"{mn}_BASE_error"
        cand_col = f"{mn}_{col_suffix}_error"

        assert len(df) == 240, \
            f"{candidate_name}/{mn}: expected 240 rows, got {len(df)}"
        assert df["instance_key"].is_unique, \
            f"{candidate_name}: instance_key is not unique"
        assert df["instance_key"].notna().all(), \
            f"{candidate_name}: missing instance_key"
        assert df[cand_col].notna().all(), \
            f"{candidate_name}/{mn}: NaN in candidate error"
        assert df[base_col].notna().all(), \
            f"{candidate_name}/{mn}: NaN in BASE error"

        d_df = df[["instance_key", "pair_id", cand_col, base_col]].copy()
        d_df = d_df.sort_values("instance_key").reset_index(drop=True)

        d = d_df[cand_col] - d_df[base_col]
        pair_ids = sorted(d_df["pair_id"].unique())

        mean_error = d_df[cand_col].mean()
        median_error = d_df[cand_col].median()
        std_error = d_df[cand_col].std()

        mean_delta = d.mean()
        median_delta = d.median()

        ci_l, ci_h, verdict = cluster_bootstrap_ci(
            d_df.assign(_delta=d), "_delta", pair_ids, rng)

        frac_improved_instance = (d < 0).mean()
        frac_degraded_instance = (d > 0).mean()

        pair_means = d_df.assign(_delta=d).groupby("pair_id")["_delta"].mean()
        n_pairs_improved = (pair_means < 0).sum()
        n_pairs_degraded = (pair_means > 0).sum()

        worst_pair = pair_means.idxmax()
        worst_pair_delta = pair_means.max()
        best_pair = pair_means.idxmin()
        best_pair_delta = pair_means.min()

        degraded_deltas = d[d > 0]
        severity = degraded_deltas.mean() if len(degraded_deltas) > 0 else 0.0

        records.append({
            "candidate": candidate_name, "model": mn,
            "mean_error": round(mean_error, 4),
            "median_error": round(median_error, 4),
            "std_error": round(std_error, 4),
            "mean_delta": round(mean_delta, 4),
            "median_delta": round(median_delta, 4),
            "ci_low": round(ci_l, 4), "ci_high": round(ci_h, 4),
            "verdict": verdict,
            "frac_improved_instance": round(frac_improved_instance, 3),
            "frac_degraded_instance": round(frac_degraded_instance, 3),
            "n_pairs_improved": n_pairs_improved,
            "n_pairs_degraded": n_pairs_degraded,
            "worst_pair": worst_pair, "worst_pair_delta": round(worst_pair_delta, 4),
            "best_pair": best_pair, "best_pair_delta": round(best_pair_delta, 4),
            "negative_transfer_severity": round(severity, 4),
        })

b31_df = pd.DataFrame(records)
b31_df.to_csv(RES_DAT/"step_b3_1_unified_comparison.csv", index=False)

for mn in MODELS:
    print(f"\n{'='*100}\n{mn}\n{'='*100}")
    mn_df = b31_df[b31_df["model"]==mn].sort_values("mean_delta")
    print(f"\n{'Candidate':14s} {'MeanΔ':>8s} {'Verdict':9s} {'PairsImp':>9s} "
          f"{'PairsDeg':>9s} {'Severity':>9s} {'WorstPair':22s}")
    print("─"*100)
    for _, r in mn_df.iterrows():
        print(f"{r['candidate']:14s} {r['mean_delta']:>+8.4f} {r['verdict']:9s} "
              f"{r['n_pairs_improved']:>9d} {r['n_pairs_degraded']:>9d} "
              f"{r['negative_transfer_severity']:>9.4f} {r['worst_pair']:22s}")

print(f"\n{'='*100}")
print("Saved: step_b3_1_unified_comparison.csv")
print(f"{'='*100}")
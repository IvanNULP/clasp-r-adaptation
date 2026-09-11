import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

results_df = pd.read_csv(RES_DAT/"step_b2_lodo_results.csv")

MODELS = ["MLP", "RF", "XGB"]

print("="*100)
print("STEP B.2.6 (виправлено) — LODO Candidate-Ranking Decomposition")
print("="*100)
print("\nПРИМІТКА: mode() при ties бере лексикографічно перший candidate.")
print("Це прийнятно для описової таблиці, не впливає на regret-обчислення.")

decomposition_rows = []

for mn in MODELS:
    print(f"\n{'='*100}\nMODEL: {mn}\n{'='*100}")

    mn_df = results_df[results_df["model"] == mn]
    pair_ids = sorted(mn_df["pair_id"].unique())
    assert len(pair_ids) == 12, f"{mn}: {len(pair_ids)} pairs, очікувалось 12"

    print(f"\n{'Pair':22s} {'BestFixed':14s} {'TS_mode':14s} {'Oracle_mode':14s} "
          f"{'TS_regret':>10s} {'Fixed_regret':>13s} {'BASE_regret':>12s} "
          f"{'TS<BASE':>8s} {'TS≡Fixed':>9s}")
    print("─"*128)

    for pair in pair_ids:
        pair_df = mn_df[mn_df["pair_id"] == pair]

        ts_rows = pair_df[pair_df["strategy"] == "TransferScore"].sort_values("instance_key")
        fixed_rows = pair_df[pair_df["strategy"] == "BestFixedPerFold"].sort_values("instance_key")
        base_rows = pair_df[pair_df["strategy"] == "AlwaysBASE"].sort_values("instance_key")
        oracle_rows = pair_df[pair_df["strategy"] == "Oracle"].sort_values("instance_key")

        assert len(ts_rows) == 20, f"{mn}/{pair}: TS has {len(ts_rows)} rows, expected 20"
        assert len(fixed_rows) == 20, f"{mn}/{pair}: Fixed has {len(fixed_rows)} rows, expected 20"
        assert len(base_rows) == 20, f"{mn}/{pair}: Base has {len(base_rows)} rows, expected 20"
        assert len(oracle_rows) == 20, f"{mn}/{pair}: Oracle has {len(oracle_rows)} rows, expected 20"

        assert fixed_rows["chosen_candidate"].nunique() == 1, \
            f"{mn}/{pair}: BestFixedPerFold is not fixed within fold " \
            f"(found {fixed_rows['chosen_candidate'].unique()})"

        best_fixed_candidate = fixed_rows["chosen_candidate"].mode()[0]
        ts_mode_candidate = ts_rows["chosen_candidate"].mode()[0]
        oracle_mode_candidate = oracle_rows["chosen_candidate"].mode()[0]

        ts_mean_regret = ts_rows["regret"].mean()
        fixed_mean_regret = fixed_rows["regret"].mean()
        base_mean_regret = base_rows["regret"].mean()

        ts_better_than_base = ts_mean_regret < base_mean_regret
        ts_better_than_fixed = ts_mean_regret < fixed_mean_regret

        assert (ts_rows["instance_key"].values == fixed_rows["instance_key"].values).all(), \
            f"{mn}/{pair}: instance_key order mismatch between TS and Fixed rows"
        ts_matches_fixed = (
            ts_rows["chosen_candidate"].values ==
            fixed_rows["chosen_candidate"].values
        ).mean()

        print(f"{pair:22s} {best_fixed_candidate:14s} {ts_mode_candidate:14s} "
              f"{oracle_mode_candidate:14s} {ts_mean_regret:>10.4f} "
              f"{fixed_mean_regret:>13.4f} {base_mean_regret:>12.4f} "
              f"{'✓' if ts_better_than_base else '✗':>8s} "
              f"{ts_matches_fixed*100:>8.1f}%")

        decomposition_rows.append({
            "model": mn, "pair_id": pair,
            "best_fixed_candidate": best_fixed_candidate,
            "ts_mode_candidate": ts_mode_candidate,
            "oracle_mode_candidate": oracle_mode_candidate,
            "ts_mean_regret": ts_mean_regret,
            "fixed_mean_regret": fixed_mean_regret,
            "base_mean_regret": base_mean_regret,
            "ts_better_than_base": ts_better_than_base,
            "ts_better_than_fixed": ts_better_than_fixed,
            "ts_matches_fixed": ts_matches_fixed,
        })

    n_ts_better_base = sum(1 for r in decomposition_rows
                             if r["model"]==mn and r["ts_better_than_base"])
    n_ts_better_fixed = sum(1 for r in decomposition_rows
                              if r["model"]==mn and r["ts_better_than_fixed"])
    mean_agreement = np.mean([r["ts_matches_fixed"] for r in decomposition_rows
                                if r["model"]==mn])
    print(f"\n  TS краще за BASE у {n_ts_better_base}/12 pairs")
    print(f"  TS краще за BestFixed у {n_ts_better_fixed}/12 pairs")
    print(f"  Середній ts_matches_fixed rate: {mean_agreement*100:.1f}% "
          f"(instance-level agreement з global best-fixed candidate)")

decomp_df = pd.DataFrame(decomposition_rows)
decomp_df.to_csv(RES_DAT/"step_b2_6_lodo_decomposition.csv", index=False)

print(f"\n{'='*100}")
print("REGRET CONCENTRATION ACROSS DOMAIN PAIRS")
print("(share of total instance-weighted regret attributable to")
print(" the three worst domain pairs — коректно, бо кожен pair має")
print(" рівно 20 instances, тож pair-mean sum ≡ instance-weighted sum)")
print(f"{'='*100}")

for mn in MODELS:
    mn_decomp = decomp_df[decomp_df["model"]==mn].sort_values(
        "ts_mean_regret", ascending=False)
    total_regret = mn_decomp["ts_mean_regret"].sum()
    top3_regret = mn_decomp.head(3)["ts_mean_regret"].sum()
    concentration_pct = (top3_regret / total_regret * 100) if total_regret > 0 else 0
    
    print(f"\n{mn}:")
    print(f"  Топ-3 найгірших pairs: {mn_decomp.head(3)['pair_id'].tolist()}")
    print(f"  Regret concentration (топ-3 з 12 pairs): {concentration_pct:.1f}%")
    print("  (25% = рівномірний reference для 3 із 12 pairs;")
    print("   істотно вище 25% вказує на концентрацію regret)")
print(f"\n{'='*100}")
print("Saved: step_b2_6_lodo_decomposition.csv")
print("STEP B.2.6 завершено")
print(f"{'='*100}")
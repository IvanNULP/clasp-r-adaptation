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
print("STEP B.3.2 — Negative-Transfer Stability Analysis")
print("="*100)

reference_keys = set(v9_df["instance_key"])
assert len(reference_keys) == 240, \
    f"reference_keys (v9) має {len(reference_keys)} унікальних instance_key, очікувалось 240"

for name, df in [("v9", v9_df), ("QT", qt_df), ("TAC", tac_df),
                  ("UOT", uot_df), ("POT", pot_df)]:
    assert len(df) == 240, f"{name}: expected 240 rows, got {len(df)}"
    assert df["instance_key"].notna().all(), f"{name}: missing instance_key"
    assert set(df["instance_key"]) == reference_keys, f"{name}: instance_key mismatch"
    assert df["instance_key"].is_unique, f"{name}: duplicate instance_key"
print("✓ Усі 5 файлів узгоджені (240 rows, identical instance_key sets, no NaN)")

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
    return np.percentile(boot_means, [2.5, 97.5])

summary_records = []
pairwise_records = []

for candidate_name, (df, col_suffix) in METHODS.items():
    for mn in MODELS:
        base_col = f"{mn}_BASE_error"
        cand_col = f"{mn}_{col_suffix}_error"

        assert df[cand_col].notna().all(), f"{candidate_name}/{mn}: NaN in candidate error"
        assert df[base_col].notna().all(), f"{candidate_name}/{mn}: NaN in BASE error"

        d_df = df[["instance_key", "pair_id", cand_col, base_col]].copy()
        d_df = d_df.sort_values("instance_key").reset_index(drop=True)
        d_df["delta"] = d_df[cand_col] - d_df[base_col]

        d = d_df["delta"]
        pair_ids = sorted(d_df["pair_id"].unique())

        mean_d, median_d, sd_d = d.mean(), d.median(), d.std()
        p_improved = (d < 0).mean()
        p_degraded = (d > 0).mean()
        p_unchanged = (d == 0).mean()
        n_improved = (d < 0).sum()
        n_degraded = (d > 0).sum()
        n_unchanged = (d == 0).sum()

        pair_means = d_df.groupby("pair_id")["delta"].mean()
        n_pairs_improved = (pair_means < 0).sum()
        n_pairs_degraded = (pair_means > 0).sum()
        n_pairs_unchanged = (pair_means == 0).sum()

        worst_pair = pair_means.idxmax()
        worst_pair_delta = pair_means.max()
        best_pair = pair_means.idxmin()
        best_pair_delta = pair_means.min()

        degraded_deltas = d[d > 0]
        mean_positive_delta = degraded_deltas.mean() if len(degraded_deltas) > 0 else 0.0
        median_positive_delta = degraded_deltas.median() if len(degraded_deltas) > 0 else 0.0
        max_instance_positive_delta = degraded_deltas.max() if len(degraded_deltas) > 0 else 0.0
        max_pair_positive_delta = pair_means[pair_means > 0].max() if (pair_means > 0).any() else 0.0

        ci_l, ci_h = cluster_bootstrap_ci(d_df, "delta", pair_ids, rng)

        summary_records.append({
            "candidate": candidate_name, "model": mn,
            "mean_delta": round(mean_d, 4), "median_delta": round(median_d, 4),
            "sd_delta": round(sd_d, 4),
            "p_improved": round(p_improved, 3), "p_degraded": round(p_degraded, 3),
            "p_unchanged": round(p_unchanged, 3),
            "n_improved": n_improved, "n_degraded": n_degraded, "n_unchanged": n_unchanged,
            "n_pairs_improved": n_pairs_improved, "n_pairs_degraded": n_pairs_degraded,
            "n_pairs_unchanged": n_pairs_unchanged,
            "best_pair": best_pair, "best_pair_delta": round(best_pair_delta, 4),
            "worst_pair": worst_pair, "worst_pair_delta": round(worst_pair_delta, 4),
            "mean_positive_delta": round(mean_positive_delta, 4),
            "median_positive_delta": round(median_positive_delta, 4),
            "max_instance_positive_delta": round(max_instance_positive_delta, 4),
            "max_pair_positive_delta": round(max_pair_positive_delta, 4),
            "ci_low": round(ci_l, 4), "ci_high": round(ci_h, 4),
        })

        for pair, mean_delta_pair in pair_means.items():
            pairwise_records.append({
                "candidate": candidate_name, "model": mn,
                "pair_id": pair, "mean_delta": round(mean_delta_pair, 4),
            })

summary_df = pd.DataFrame(summary_records)
pairwise_df = pd.DataFrame(pairwise_records)

summary_df.to_csv(RES_DAT/"step_b3_2_negative_transfer.csv", index=False)
pairwise_df.to_csv(RES_DAT/"step_b3_2_pairwise_delta.csv", index=False)

for mn in MODELS:
    print(f"\n{'='*100}\n{mn} — Negative-Transfer Risk Profile\n{'='*100}")
    mn_df = summary_df[summary_df["model"]==mn].sort_values("mean_delta")

    print(f"\n{'Candidate':14s} {'MeanΔ':>8s} {'P(deg)':>7s} {'PairsDeg':>9s} "
          f"{'MeanPosΔ':>9s} {'MaxInstΔ':>9s} {'MaxPairΔ':>9s} {'WorstPair':22s}")
    print("─"*105)
    for _, r in mn_df.iterrows():
        print(f"{r['candidate']:14s} {r['mean_delta']:>+8.4f} {r['p_degraded']:>7.3f} "
              f"{r['n_pairs_degraded']:>9d} {r['mean_positive_delta']:>9.4f} "
              f"{r['max_instance_positive_delta']:>9.4f} {r['max_pair_positive_delta']:>9.4f} "
              f"{r['worst_pair']:22s}")

print(f"\n{'='*100}")
print("Saved: step_b3_2_negative_transfer.csv")
print("Saved: step_b3_2_pairwise_delta.csv")
print(f"{'='*100}")
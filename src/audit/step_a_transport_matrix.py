import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")
pot_df = pd.read_csv(RES_DAT/"v13_partial_ot_240.csv")

print("="*100)
print("STEP A (виправлено) — Transport Matrix")
print("="*100)

expected_pairs = set(v9_df["pair_id"].unique())
for name, df in [("QT", qt_df), ("TAC", tac_df), ("UOT", uot_df), ("POT", pot_df)]:
    actual_pairs = set(df["pair_id"].unique())
    assert actual_pairs == expected_pairs, (
        f"{name}: pair_id mismatch. "
        f"Missing={expected_pairs - actual_pairs}, "
        f"Extra={actual_pairs - expected_pairs}")
print(f"✓ Перевірка pair_id: усі 5 файлів мають однакові "
      f"{len(expected_pairs)} pair_id")

expected_instances = set(v9_df["instance_key"])
for name, df in [("QT", qt_df), ("TAC", tac_df), ("UOT", uot_df), ("POT", pot_df)]:
    actual_instances = set(df["instance_key"])
    assert actual_instances == expected_instances, (
        f"{name}: instance_key mismatch. "
        f"n_missing={len(expected_instances - actual_instances)}, "
        f"n_extra={len(actual_instances - expected_instances)}")
print(f"✓ Перевірка instance_key: усі 5 файлів мають однакові "
      f"{len(expected_instances)} instance_key")

rng = np.random.default_rng(42)
pair_ids = v9_df["pair_id"].unique()

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

METHODS = {
    "Global CORAL":       (v9_df,  "CORAL", "CLASP",   "CORAL+BBSE"),
    "Type-aware CORAL":   (tac_df, "TAC",   "TAC_BBSE","TAC+BBSE"),
    "Quantile Transport": (qt_df,  "QT",    "QT_BBSE", "QT+BBSE"),
    "Unbalanced OT":      (uot_df, "UOT",   "UOT_BBSE","UOT+BBSE"),
    "Partial OT":         (pot_df, "POT",   "POT_BBSE","POT+BBSE"),
}

matrix_records = []

for method_name, (df, col_no_bbse, col_bbse, bbse_variant_label) in METHODS.items():
    for mn in ["MLP","RF","XGB"]:
        base_col = f"{mn}_BASE_error"
        no_bbse_col = f"{mn}_{col_no_bbse}_error"
        bbse_col = f"{mn}_{col_bbse}_error"

        d_no_bbse = df[no_bbse_col] - df[base_col]
        d_bbse = df[bbse_col] - df[base_col]

        tmp_nb = df.assign(_delta=d_no_bbse)
        tmp_b = df.assign(_delta=d_bbse)

        ci_l_nb, ci_h_nb, verdict_nb = cluster_bootstrap_ci(tmp_nb, "_delta", pair_ids, rng)
        ci_l_b, ci_h_b, verdict_b = cluster_bootstrap_ci(tmp_b, "_delta", pair_ids, rng)

        n_pairs_improve_nb = sum(
            1 for p, g in df.groupby("pair_id")
            if g[no_bbse_col].mean() < g[base_col].mean())
        n_pairs_improve_b = sum(
            1 for p, g in df.groupby("pair_id")
            if g[bbse_col].mean() < g[base_col].mean())

        pair_deltas_nb = tmp_nb.groupby("pair_id")["_delta"].mean()
        pair_deltas_b = tmp_b.groupby("pair_id")["_delta"].mean()

        matrix_records.append({
            "method": method_name, "model": mn,
            "bbse_variant_label": bbse_variant_label,

            "mean_delta_no_bbse": round(d_no_bbse.mean(), 4),
            "median_delta_no_bbse": round(d_no_bbse.median(), 4),
            "pct_improved_no_bbse": round((d_no_bbse<0).mean()*100, 1),
            "ci_low_no_bbse": round(ci_l_nb, 4),
            "ci_high_no_bbse": round(ci_h_nb, 4),
            "verdict_no_bbse": verdict_nb,
            "pairs_improved_no_bbse": f"{n_pairs_improve_nb}/12",
            "pair_delta_min_no_bbse": round(pair_deltas_nb.min(), 4),
            "pair_delta_max_no_bbse": round(pair_deltas_nb.max(), 4),

            "mean_delta_bbse": round(d_bbse.mean(), 4),
            "median_delta_bbse": round(d_bbse.median(), 4),
            "pct_improved_bbse": round((d_bbse<0).mean()*100, 1),
            "ci_low_bbse": round(ci_l_b, 4),
            "ci_high_bbse": round(ci_h_b, 4),
            "verdict_bbse": verdict_b,
            "pairs_improved_bbse": f"{n_pairs_improve_b}/12",
            "pair_delta_min_bbse": round(pair_deltas_b.min(), 4),
            "pair_delta_max_bbse": round(pair_deltas_b.max(), 4),
        })

matrix_df = pd.DataFrame(matrix_records)
matrix_df.to_csv(RES_DAT/"step_a_transport_matrix.csv", index=False)

print(f"\n{'='*100}")
print("ТАБЛИЦЯ 1 — Adaptation БЕЗ BBSE")
print(f"{'='*100}")
print(f"\n{'Method':22s} {'Model':6s} {'MeanΔ':>9s} {'Verdict':10s} "
      f"{'Pairs':>7s} {'RangeΔ (min..max)':>22s}")
print("─"*85)
for _, r in matrix_df.iterrows():
    print(f"{r['method']:22s} {r['model']:6s} {r['mean_delta_no_bbse']:>+9.4f} "
          f"{r['verdict_no_bbse']:10s} {r['pairs_improved_no_bbse']:>7s} "
          f"[{r['pair_delta_min_no_bbse']:+.4f}, {r['pair_delta_max_no_bbse']:+.4f}]")

print(f"\n{'='*100}")
print("ТАБЛИЦЯ 2 — Adaptation + BBSE")
print(f"{'='*100}")
print(f"\n{'Method':22s} {'Model':6s} {'MeanΔ':>9s} {'Verdict':10s} "
      f"{'Pairs':>7s} {'RangeΔ (min..max)':>22s}")
print("─"*85)
for _, r in matrix_df.iterrows():
    print(f"{r['method']:22s} {r['model']:6s} {r['mean_delta_bbse']:>+9.4f} "
          f"{r['verdict_bbse']:10s} {r['pairs_improved_bbse']:>7s} "
          f"[{r['pair_delta_min_bbse']:+.4f}, {r['pair_delta_max_bbse']:+.4f}]")

print(f"\n{'='*100}")
print("PIVOT: Verdict Matrix (+BBSE)")
print(f"{'='*100}")
pivot = matrix_df.pivot(index="method", columns="model", values="verdict_bbse")
pivot = pivot.reindex(["Global CORAL","Type-aware CORAL",
                        "Quantile Transport","Unbalanced OT","Partial OT"])
print(pivot.to_string())

print(f"\n{'='*100}")
print("FACTUAL SUMMARY (не paper conclusion — лише підрахунок)")
print(f"{'='*100}")
n_improve = (matrix_df["verdict_bbse"]=="IMPROVE").sum()
n_degrade = (matrix_df["verdict_bbse"]=="DEGRADE").sum()
n_unknown = (matrix_df["verdict_bbse"]=="UNKNOWN").sum()
print(f"""
  З {len(matrix_df)} комбінацій (5 методів × 3 моделі), +BBSE:
    ROBUST IMPROVEMENT: {n_improve}/{len(matrix_df)}
    ROBUST DEGRADATION: {n_degrade}/{len(matrix_df)}
    UNKNOWN:            {n_unknown}/{len(matrix_df)}

  Max heterogeneity (найширший pair-level розкид, +BBSE):
""")
widest = matrix_df.assign(
    range_width=matrix_df["pair_delta_max_bbse"] - matrix_df["pair_delta_min_bbse"]
).sort_values("range_width", ascending=False).head(3)
for _, r in widest.iterrows():
    print(f"    {r['method']} / {r['model']}: "
          f"range=[{r['pair_delta_min_bbse']:+.4f}, {r['pair_delta_max_bbse']:+.4f}] "
          f"(width={r['range_width']:.4f})")

print(f"\nSaved: {RES_DAT}/step_a_transport_matrix.csv "
      f"(з окремими numeric ci_low/ci_high колонками)")
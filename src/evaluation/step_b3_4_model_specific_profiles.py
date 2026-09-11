import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

b31_df = pd.read_csv(RES_DAT/"step_b3_1_unified_comparison.csv")
b32_df = pd.read_csv(RES_DAT/"step_b3_2_negative_transfer.csv")
asym_df = pd.read_csv(RES_DAT/"step_b3_3_directional_asymmetry.csv")

MODELS = ["MLP", "RF", "XGB"]

print("="*100)
print("STEP B.3.4 (виправлено) — Model-Specific Adaptation Profiles")
print("="*100)

assert len(b31_df) == 30, f"B.3.1 expected 30 rows, got {len(b31_df)}"
assert len(b32_df) == 30, f"B.3.2 expected 30 rows, got {len(b32_df)}"
assert set(b31_df["model"]) == set(MODELS)
assert set(b32_df["model"]) == set(MODELS)
assert b31_df[["model", "candidate"]].duplicated().sum() == 0
assert b32_df[["model", "candidate"]].duplicated().sum() == 0
assert asym_df["sign_flip"].notna().all()

print("\n✓ B.3.1: 30 rows, no duplicates")
print("✓ B.3.2: 30 rows, no duplicates")
print("✓ B.3.3 asymmetry: sign_flip has no NaN")

print()
for mn in MODELS:
    mn_b31_check = b31_df[b31_df["model"] == mn]
    mn_asym_check = asym_df[asym_df["model"] == mn]
    all_b31_candidates = set(mn_b31_check["candidate"])
    observed_candidates_check = set(mn_asym_check["candidate"])

    assert observed_candidates_check.issubset(all_b31_candidates), (
        f"{mn}: B.3.3 contains candidates NOT present in B.3.1! "
        f"Unexpected={observed_candidates_check - all_b31_candidates}")

    expected_rows_check = len(observed_candidates_check) * 6
    assert len(mn_asym_check) == expected_rows_check, (
        f"{mn}: expected {expected_rows_check} rows, got {len(mn_asym_check)}")

    pair_counts_check = mn_asym_check.groupby("candidate").size()
    assert (pair_counts_check == 6).all(), f"{mn}: pair counts wrong"

    print(f"  {mn}: B.3.3 focus candidates = {sorted(observed_candidates_check)} ✓")

print("\n✓ B.3.3: focus-candidate subset перевірена коректно")

sample_merge = b31_df[b31_df["model"]=="MLP"].merge(
    b32_df[b32_df["model"]=="MLP"], on=["candidate","model"], suffixes=("_b31","_b32"))
print(f"\nКолонки після merge (для перевірки): {list(sample_merge.columns)}")

profile_records = []

for mn in MODELS:
    print(f"\n{'='*100}\n{mn}\n{'='*100}")

    mn_b31 = b31_df[b31_df["model"]==mn].copy()
    mn_b32 = b32_df[b32_df["model"]==mn].copy()
    merged = mn_b31.merge(mn_b32, on=["candidate","model"], suffixes=("_b31","_b32"))

    expected_candidates = set(mn_b31["candidate"])
    assert set(merged["candidate"]) == expected_candidates
    assert len(merged) == len(expected_candidates)

    most_promising = merged.loc[merged["mean_delta_b31"].idxmin(), "candidate"]
    most_promising_val = merged["mean_delta_b31"].min()

    non_worsening = merged[merged["mean_delta_b31"] <= 0]
    stability_pool = non_worsening if len(non_worsening) > 0 else merged
    most_stable = stability_pool.loc[stability_pool["sd_delta"].idxmin(), "candidate"]
    most_stable_sd = stability_pool["sd_delta"].min()

    highest_risk = merged.loc[merged["p_degraded"].idxmax(), "candidate"]
    highest_risk_val = merged["p_degraded"].max()

    best_worst_case = merged.loc[merged["max_pair_positive_delta"].idxmin(), "candidate"]
    best_worst_case_val = merged["max_pair_positive_delta"].min()

    mn_asym = asym_df[asym_df["model"]==mn]
    n_sign_flips = mn_asym["sign_flip"].sum()
    n_total_asym = len(mn_asym)
    mean_abs_asymmetry = mn_asym["directional_diff"].abs().mean()
    max_abs_asymmetry = mn_asym["directional_diff"].abs().max()

    print(f"\nA. Most promising (lowest mean Δ): {most_promising} ({most_promising_val:+.4f})")
    print(f"B. Lowest variability among non-worsening candidates: "
          f"{most_stable} (SD={most_stable_sd:.4f})")
    print(f"C. Highest negative-transfer risk (highest P(deg)): {highest_risk} ({highest_risk_val:.3f})")
    print(f"D. Best worst-case pair-level negative transfer: "
          f"{best_worst_case} ({best_worst_case_val:+.4f})")
    print(f"\nDirectional asymmetry ({', '.join(mn_asym['candidate'].unique())}):")
    print(f"  sign-flip pairs: {n_sign_flips}/{n_total_asym}")
    print(f"  mean |directional_diff|: {mean_abs_asymmetry:.4f}")
    print(f"  max |directional_diff|: {max_abs_asymmetry:.4f}")

    print(f"\nПовна таблиця (сортовано за mean Δ):")
    display_cols = ["candidate","mean_delta_b31","sd_delta","p_degraded",
                      "n_pairs_degraded_b32","max_pair_positive_delta"]
    print(merged[display_cols].sort_values("mean_delta_b31").to_string(index=False))

    profile_records.append({
        "model": mn,
        "A_most_promising": most_promising, "A_value": round(most_promising_val,4),
        "B_most_stable_nonworsening": most_stable, "B_value": round(most_stable_sd,4),
        "C_highest_risk": highest_risk, "C_value": round(highest_risk_val,3),
        "D_best_worst_case": best_worst_case, "D_value": round(best_worst_case_val,4),
        "n_sign_flips": int(n_sign_flips), "n_asymmetry_tested": int(n_total_asym),
        "mean_abs_directional_diff": round(mean_abs_asymmetry, 4),
        "max_abs_directional_diff": round(max_abs_asymmetry, 4),
    })

profile_df = pd.DataFrame(profile_records)
profile_df.to_csv(RES_DAT/"step_b3_4_model_specific_profiles.csv", index=False)

print(f"\n{'='*100}")
print("ЗВЕДЕНА ТАБЛИЦЯ ПРОФІЛІВ (A-D)")
print(f"{'='*100}")
print(profile_df.to_string(index=False))
print(f"\nSaved: step_b3_4_model_specific_profiles.csv")
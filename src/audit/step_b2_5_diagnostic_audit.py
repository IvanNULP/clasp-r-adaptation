import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

results_df = pd.read_csv(RES_DAT/"step_b2_lodo_results.csv")
b1d_df = pd.read_csv(RES_DAT/"step_b1d_ground_truth_long.csv")

MODELS = ["MLP", "RF", "XGB"]
ALL_CANDIDATES = ["BASE", "CORAL", "CORAL+BBSE", "TAC", "TAC+BBSE",
                   "QT", "QT+BBSE", "UOT", "UOT+BBSE", "POT", "POT+BBSE"]

print("="*90)
print("STEP B.2.5 (фінальна версія) — Diagnostic Audit")
print("="*90)

b1d_pivot = b1d_df.pivot_table(
    index=["instance_key","model"], columns="candidate", values="macro_error")

expected_n = 240 * 3
assert len(b1d_pivot) == expected_n, \
    f"Expected {expected_n} instance×model rows, got {len(b1d_pivot)}"

assert set(ALL_CANDIDATES).issubset(set(b1d_pivot.columns)), \
    "B.1d pivot does not contain all 11 candidates"

assert b1d_pivot[ALL_CANDIDATES].notna().all().all(), \
    "B.1d contains NaN macro_error for some candidate"

print(f"\n✓ b1d_pivot перевірено: {len(b1d_pivot)} instance×model rows, "
      f"усі {len(ALL_CANDIDATES)} candidates присутні, без NaN")

for mn in MODELS:
    print(f"\n{'='*90}\nMODEL: {mn}\n{'='*90}")

    ts_df = results_df[(results_df["model"]==mn) & (results_df["strategy"]=="TransferScore")]
    oracle_df = results_df[(results_df["model"]==mn) & (results_df["strategy"]=="Oracle")]

    assert len(ts_df) == 240, f"{mn}: ts_df має {len(ts_df)} рядків, очікувалось 240"
    assert len(oracle_df) == 240, f"{mn}: oracle_df має {len(oracle_df)} рядків, очікувалось 240"

    ts_merged = ts_df.merge(
        oracle_df[["instance_key","chosen_candidate"]],
        on="instance_key", suffixes=("", "_oracle"), validate="one_to_one")
    ts_merged = ts_merged.rename(columns={"chosen_candidate_oracle": "oracle_candidate"})

    assert len(ts_merged) == 240, \
        f"{mn}: ts_merged має {len(ts_merged)} рядків, очікувалось 240"

    n_top1_match = (ts_merged["chosen_candidate"] == ts_merged["oracle_candidate"]).sum()
    top1_accuracy = n_top1_match / len(ts_merged)
    print(f"\n1-2. Top-1 selection accuracy: {n_top1_match}/{len(ts_merged)} "
          f"= {top1_accuracy*100:.1f}%")
    print(f"     (Random reference: 1/11 ≈ 9.1%)")

    n_top2_match = 0
    for _, row in ts_merged.iterrows():
        inst_key = row["instance_key"]
        model_key = mn
        if (inst_key, model_key) not in b1d_pivot.index:
            continue
        errors_row = b1d_pivot.loc[(inst_key, model_key)]
        sorted_candidates = errors_row.sort_values()
        top2_actual = set(sorted_candidates.index[:2])
        if row["chosen_candidate"] in top2_actual:
            n_top2_match += 1
    top2_rate = n_top2_match / len(ts_merged)
    print(f"\n3. Oracle-proximity: chosen candidate входить "
          f"до 2 НАЙКРАЩИХ actual candidates: "
          f"{n_top2_match}/{len(ts_merged)} = {top2_rate*100:.1f}%")
    print(f"   (Це НЕ ranking accuracy Transfer Score — це частка випадків,")
    print(f"    коли обраний candidate фактично був серед двох найкращих")
    print(f"    за ground truth error, незалежно від того, як TS його ранжував)")

    print(f"\n4. Розподіл обраних candidates:")
    print(f"\n   Transfer Score choices:")
    print(ts_merged["chosen_candidate"].value_counts().to_string())
    print(f"\n   Oracle choices (справжній найкращий):")
    print(ts_merged["oracle_candidate"].value_counts().to_string())

    print(f"\n5-6. Pair-level regret (Transfer Score):")
    pair_regret = ts_merged.groupby("pair_id")["regret"].agg(["mean","std","count"])
    pair_regret = pair_regret.sort_values("mean")
    print(pair_regret.to_string())

    best_pairs = pair_regret.head(3).index.tolist()
    worst_pairs = pair_regret.tail(3).index.tolist()
    print(f"\n   Найкращі 3 pairs (низький regret): {best_pairs}")
    print(f"   Найгірші 3 pairs (високий regret): {worst_pairs}")

    base_df = results_df[(results_df["model"]==mn) & (results_df["strategy"]=="AlwaysBASE")]
    assert len(base_df) == 240, f"{mn}: base_df має {len(base_df)} рядків, очікувалось 240"

    norm_merged = ts_merged.merge(
        base_df[["instance_key","macro_error"]],
        on="instance_key", suffixes=("", "_base"), validate="one_to_one")
    assert len(norm_merged) == 240, \
        f"{mn}: norm_merged має {len(norm_merged)} рядків, очікувалось 240"

    norm_merged = norm_merged.rename(columns={"macro_error_base": "base_error"})

    oracle_error_lookup = oracle_df.set_index("instance_key")["macro_error"]
    norm_merged["oracle_error"] = norm_merged["instance_key"].map(oracle_error_lookup)

    denom = norm_merged["base_error"] - norm_merged["oracle_error"]
    valid_denom = denom.abs() > 1e-9
    norm_merged["normalized_regret"] = np.nan
    norm_merged.loc[valid_denom, "normalized_regret"] = (
        norm_merged.loc[valid_denom, "regret"] / denom[valid_denom])

    n_valid = valid_denom.sum()
    mean_norm_regret = norm_merged.loc[valid_denom, "normalized_regret"].mean()
    median_norm_regret = norm_merged.loc[valid_denom, "normalized_regret"].median()

    print(f"\n   SUPPLEMENTARY: Normalized regret "
          f"(n_valid={n_valid}/{len(norm_merged)}, де BASE≠Oracle):")
    print(f"     mean={mean_norm_regret:.4f}, median={median_norm_regret:.4f}")
    print(f"     Інтерпретація: 0=Oracle-рівень, 1=BASE-рівень, "
          f"<1=краще за BASE, >1=гірше за BASE (може бути >>1)")

print(f"\n{'='*90}\n7. BestFixedPerFold VARIABILITY ПО 12 FOLDS\n{'='*90}")

for mn in MODELS:
    fixed_df = results_df[(results_df["model"]==mn) &
                            (results_df["strategy"]=="BestFixedPerFold")]
    fold_choices = fixed_df.groupby("fold_test_pair")["chosen_candidate"].first()
    n_unique_fixed = fold_choices.nunique()
    print(f"\n  {mn}: {n_unique_fixed}/12 унікальних best_fixed_candidate по фолдах")
    print(f"    Розподіл: {fold_choices.value_counts().to_dict()}")
    if n_unique_fixed == 1:
        print(f"    ℹ best_fixed_candidate однаковий у всіх folds; "
              f"це допустимо, але варто перевірити train-fold means як sanity check.")

print(f"\n{'='*90}")
print("STEP B.2.5 завершено")
print(f"{'='*90}")
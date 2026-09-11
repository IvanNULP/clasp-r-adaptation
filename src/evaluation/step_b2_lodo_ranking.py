import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

b1c_df = pd.read_csv(RES_DAT/"step_b1c_candidate_matrix.csv")
b1d_df = pd.read_csv(RES_DAT/"step_b1d_ground_truth_long.csv")

FEATURE_COLS = ["ks_shift","prediction_flip_rate","probability_shift",
                 "confidence_change","entropy_change","bbse_weight_magnitude",
                 "bbse_prior_extremity","bbse_condition_log","bbse_disabled",
                 "candidate_is_base"]
ALL_CANDIDATES = ["BASE", "CORAL", "CORAL+BBSE", "TAC", "TAC+BBSE",
                   "QT", "QT+BBSE", "UOT", "UOT+BBSE", "POT", "POT+BBSE"]
MODELS = ["MLP", "RF", "XGB"]
N_RANDOM_SEEDS = 50

print("="*90)
print("STEP B.2 (фінальна версія) — LODO Ranking + Regret Evaluation")
print("="*90)

merged = b1c_df.merge(
    b1d_df[["instance_key","model","candidate","macro_error"]],
    on=["instance_key","model","candidate"],
    how="inner", validate="one_to_one")

assert len(merged) == 7920, f"Очікувалось 7920, отримано {len(merged)}"
assert "macro_error" not in FEATURE_COLS, "macro_error випадково в FEATURE_COLS!"
print(f"\n✓ Merged X+y: {len(merged)} рядків (очікувано 7920)")

pair_ids = sorted(merged["pair_id"].unique())
assert len(pair_ids) == 12, f"Очікувалось 12 pair_id, отримано {len(pair_ids)}"
print(f"✓ 12 unique pair_id підтверджено")

def deterministic_random_regrets(inst_rows_by_candidate_error, n_seeds=N_RANDOM_SEEDS):
    errors = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        chosen_idx = rng.randint(0, len(ALL_CANDIDATES))
        chosen_cand = ALL_CANDIDATES[chosen_idx]
        errors.append(inst_rows_by_candidate_error.loc[chosen_cand])
    return np.array(errors)

all_results = []

for mn in MODELS:
    print(f"\n{'='*90}\nMODEL: {mn}\n{'='*90}")

    model_df = merged[merged["model"] == mn].copy()

    for fold_idx, test_pair in enumerate(pair_ids):
        train_df = model_df[model_df["pair_id"] != test_pair].copy()
        test_df = model_df[model_df["pair_id"] == test_pair].copy()

        assert train_df["pair_id"].nunique() == 11, \
            f"Fold {test_pair}: train має {train_df['pair_id'].nunique()} pairs, очікувалось 11"
        assert set(train_df["pair_id"]).isdisjoint(set(test_df["pair_id"])), \
            f"Fold {test_pair}: train і test pairs перетинаються!"

        assert len(train_df) == 2420, \
            f"Fold {test_pair}/{mn}: train={len(train_df)}, очікувалось 2420 (11×20×11)"
        assert len(test_df) == 220, \
            f"Fold {test_pair}/{mn}: test={len(test_df)}, очікувалось 220 (1×20×11)"

        test_instance_keys = test_df["instance_key"].unique()
        assert len(test_instance_keys) == 20, \
            f"Fold {test_pair}: {len(test_instance_keys)} unique instances, очікувалось 20"

        for inst_key in test_instance_keys:
            n_cand_test = test_df[test_df["instance_key"]==inst_key]["candidate"].nunique()
            assert n_cand_test == 11, \
                f"Fold {test_pair}, instance {inst_key}: {n_cand_test} candidates, очікувалось 11"

        encoder = OneHotEncoder(categories=[ALL_CANDIDATES], sparse_output=False,
                                  handle_unknown="error")
        train_cand_encoded = encoder.fit_transform(train_df[["candidate"]])
        test_cand_encoded = encoder.transform(test_df[["candidate"]])

        X_train = np.hstack([train_df[FEATURE_COLS].values, train_cand_encoded])
        y_train = train_df["macro_error"].values

        assert X_train.shape[1] == 21, \
            f"Expected 21 X columns, got {X_train.shape[1]}"

        rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                     random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        X_test = np.hstack([test_df[FEATURE_COLS].values, test_cand_encoded])
        assert X_test.shape[1] == 21, \
            f"Expected 21 X columns, got {X_test.shape[1]}"

        y_pred = rf.predict(X_test)
        test_df = test_df.copy()
        test_df["ts_pred_error"] = y_pred

        train_mean_error_per_candidate = train_df.groupby("candidate")["macro_error"].mean()
        best_fixed_candidate = train_mean_error_per_candidate.idxmin()

        for inst_key in test_instance_keys:
            inst_rows = test_df[test_df["instance_key"] == inst_key].set_index("candidate")

            assert set(inst_rows.index) == set(ALL_CANDIDATES), \
                f"{inst_key}: candidate set mismatch"
            assert len(inst_rows) == 11, \
                f"{inst_key}: expected 11 candidate rows, got {len(inst_rows)}"
            assert np.isfinite(inst_rows["macro_error"]).all(), \
                f"{inst_key}: non-finite macro_error"

            oracle_error = inst_rows["macro_error"].min()
            oracle_candidate = inst_rows["macro_error"].idxmin()

            ts_chosen = inst_rows["ts_pred_error"].idxmin()
            ts_error = inst_rows.loc[ts_chosen, "macro_error"]

            base_error = inst_rows.loc["BASE", "macro_error"]

            fixed_error = inst_rows.loc[best_fixed_candidate, "macro_error"]

            random_errors_array = deterministic_random_regrets(
                inst_rows["macro_error"])
            random_error_mean = random_errors_array.mean()
            random_error_std = random_errors_array.std()

            for strategy, chosen_cand, chosen_error, extra in [
                ("TransferScore", ts_chosen, ts_error, {}),
                ("AlwaysBASE", "BASE", base_error, {}),
                ("BestFixedPerFold", best_fixed_candidate, fixed_error, {}),
                ("Random", "N/A_averaged", random_error_mean,
                 {"random_error_std": random_error_std,
                  "random_n_seeds": N_RANDOM_SEEDS}),
                ("Oracle", oracle_candidate, oracle_error, {}),
            ]:
                row = {
                    "model": mn, "fold_test_pair": test_pair,
                    "instance_key": inst_key, "pair_id": test_pair,
                    "strategy": strategy, "chosen_candidate": chosen_cand,
                    "macro_error": chosen_error,
                    "regret": chosen_error - oracle_error,
                }
                row.update(extra)
                all_results.append(row)

    print(f"  {mn}: 12 folds завершено")

results_df = pd.DataFrame(all_results)
del all_results

n_expected = 3 * 240 * 5
assert len(results_df) == n_expected, \
    f"Очікувалось {n_expected} рядків результатів, отримано {len(results_df)}"
print(f"\n✓ Загальна кількість результатів: {len(results_df)} "
      f"(3 моделі × 240 instances × 5 стратегій = {n_expected})")

for mn in MODELS:
    for strat in ["TransferScore","AlwaysBASE","BestFixedPerFold","Random","Oracle"]:
        n = len(results_df[(results_df["model"]==mn) & (results_df["strategy"]==strat)])
        assert n == 240, f"{mn}/{strat}: {n} rows, очікувалось 240"

print(f"✓ Кожна (model, strategy) комбінація має рівно 240 рядків")

random_std_sample = results_df[results_df["strategy"]=="Random"]["random_error_std"]
print(f"\n✓ Random baseline variability: mean_std={random_std_sample.mean():.4f}")

results_df.to_csv(RES_DAT/"step_b2_lodo_results.csv", index=False)
print(f"\nSaved: {RES_DAT}/step_b2_lodo_results.csv")

print(f"\n{'='*90}\nREGRET ANALYSIS (усі baseline порівняння окремо)\n{'='*90}")

rng_boot = np.random.default_rng(42)

def cluster_bootstrap_ci(df, col, pair_ids, rng, n_boot=2000):
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, len(pair_ids), replace=True)
        boot_data = pd.concat([df[df["pair_id"]==p] for p in sampled], ignore_index=True)
        boot_means.append(boot_data[col].mean())
    return np.percentile(boot_means, [2.5, 97.5])

for mn in MODELS:
    print(f"\n{'='*80}\n{mn}\n{'='*80}")
    mn_df = results_df[results_df["model"] == mn]

    print(f"\n{'Strategy':20s} {'MeanRegret':>11s} {'MedianRegret':>13s} "
          f"{'P95Regret':>10s}")
    print("─"*60)
    for strat in ["TransferScore","AlwaysBASE","BestFixedPerFold","Random","Oracle"]:
        strat_df = mn_df[mn_df["strategy"] == strat]
        mean_r = strat_df["regret"].mean()
        median_r = strat_df["regret"].median()
        p95_r = strat_df["regret"].quantile(0.95)
        print(f"{strat:20s} {mean_r:>+11.4f} {median_r:>+13.4f} {p95_r:>10.4f}")

    print(f"\n  Paired comparisons (Transfer Score vs кожен baseline окремо):")
    for baseline_name in ["AlwaysBASE", "BestFixedPerFold", "Random"]:
        ts_df = mn_df[mn_df["strategy"] == "TransferScore"][
            ["pair_id", "instance_key", "regret"]]
        bl_df = mn_df[mn_df["strategy"] == baseline_name][
            ["pair_id", "instance_key", "regret"]]

        diff_df = ts_df.merge(
            bl_df, on=["pair_id", "instance_key"],
            suffixes=("_ts", "_baseline"), validate="one_to_one")
        assert len(diff_df) == 240, \
            f"{mn}/{baseline_name}: paired comparison has {len(diff_df)} rows, expected 240"

        diff_df["regret_diff"] = diff_df["regret_ts"] - diff_df["regret_baseline"]
        diff_df["regret_diff"] = diff_df["regret_ts"] - diff_df["regret_baseline"]

        ci_l, ci_h = cluster_bootstrap_ci(diff_df, "regret_diff", pair_ids, rng_boot)
        mean_diff = diff_df["regret_diff"].mean()

        if ci_h < 0:
            verdict = "✓ Transfer Score РОБАСТНО КРАЩИЙ"
        elif ci_l > 0:
            verdict = "❌ Transfer Score РОБАСТНО ГІРШИЙ"
        else:
            verdict = "⚠ НЕВИЗНАЧЕНО (CI перетинає нуль)"

        print(f"\n  Transfer Score vs {baseline_name}:")
        print(f"    mean_diff={mean_diff:+.4f}")
        print(f"    Cluster-bootstrap 95% CI: [{ci_l:+.4f}, {ci_h:+.4f}]")
        print(f"    Вердикт: {verdict}")

print(f"\n{'='*90}")
print("STEP B.2 завершено")
print(f"{'='*90}")
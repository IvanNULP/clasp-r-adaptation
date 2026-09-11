import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")
COND_MAX = 1e16
sentinel_val = np.log10(COND_MAX)

FLOW_DATASETS = ["CIC-IDS-2017","UNSW-NB15","TON-IoT-2021","CICIoT2023"]
SHORT = {"CIC-IDS-2017":"CIC-2017","UNSW-NB15":"UNSW",
         "TON-IoT-2021":"TON-IoT","CICIoT2023":"CICIoT23"}
ALL_CANDIDATES = ["BASE", "CORAL", "CORAL+BBSE", "TAC", "TAC+BBSE",
                   "QT", "QT+BBSE", "UOT", "UOT+BBSE", "POT", "POT+BBSE"]
FEATURE_COLS = ["ks_shift","prediction_flip_rate","probability_shift",
                 "confidence_change","entropy_change","bbse_weight_magnitude",
                 "bbse_prior_extremity","bbse_condition_log","bbse_disabled",
                 "candidate_is_base"]

def safe_log_condition(cond_num, cond_max=COND_MAX):
    if not np.isfinite(cond_num):
        return np.log10(cond_max)
    return np.log10(max(cond_num, 1.0))

df = pd.read_csv(RES_DAT/"step_b1c_candidate_matrix.csv")

with open(RES_DAT/"v9_frozen_confusion_matrices.pkl", "rb") as f:
    frozen_matrices = pickle.load(f)
audit_df_stability = pd.read_csv(RES_DAT/"v9_bbse_stability_audit.csv")
audit_lookup = {row["key"]: row["recommended_bbse_action"]
                for _, row in audit_df_stability.iterrows()}

print("="*90)
print("STEP B.1c — ФІНАЛЬНИЙ VALIDATION BLOCK (23 перевірки)")
print("="*90)

checks = {}

checks["n_rows_7920"] = len(df) == 7920
print(f"1. Total rows: {len(df)} {'✓' if checks['n_rows_7920'] else '❌ очікувалось 7920'}")

n_instances = df["instance_key"].nunique()
checks["n_instances_240"] = n_instances == 240
print(f"2. Unique instance_key: {n_instances} {'✓' if checks['n_instances_240'] else '❌'}")

n_models = df["model"].nunique()
checks["n_models_3"] = n_models == 3
print(f"3. Unique models: {n_models} {'✓' if checks['n_models_3'] else '❌'}")

cand_counts = df.groupby(["instance_key","model"])["candidate"].nunique()
checks["all_11_candidates"] = (cand_counts == 11).all()
print(f"4. Всі (instance,model) мають 11 candidates: "
      f"{'✓' if checks['all_11_candidates'] else f'❌ {(cand_counts!=11).sum()} задач неповні'}")

n_tasks = len(cand_counts)
checks["n_tasks_720"] = n_tasks == 720
print(f"5. Unique (instance,model) tasks: {n_tasks} {'✓' if checks['n_tasks_720'] else '❌'}")

dup_check = df.groupby(["instance_key","model","candidate"]).size()
checks["no_duplicates"] = (dup_check == 1).all()
print(f"6. Без дублікатів candidate у задачі: "
      f"{'✓' if checks['no_duplicates'] else f'❌ {(dup_check>1).sum()} дублікатів'}")

checks["10_feature_cols"] = all(c in df.columns for c in FEATURE_COLS)
print(f"7. Усі 10 feature columns присутні: {'✓' if checks['10_feature_cols'] else '❌'}")

response_cols = ["prediction_flip_rate","probability_shift",
                  "confidence_change","entropy_change"]
resp_finite = df[response_cols].apply(lambda s: np.isfinite(s)).all().all()
checks["response_finite"] = resp_finite
print(f"8. Response features finite: {'✓' if resp_finite else '❌'}")

cond_finite = np.isfinite(df["bbse_condition_log"]).all()
checks["cond_log_finite"] = cond_finite
print(f"9. bbse_condition_log finite: {'✓' if cond_finite else '❌'}")

checks["bbse_disabled_binary"] = set(df["bbse_disabled"].unique()) <= {0,1}
print(f"10. bbse_disabled ∈ {{0,1}}: {'✓' if checks['bbse_disabled_binary'] else '❌'}")

checks["is_base_binary"] = set(df["candidate_is_base"].unique()) <= {0,1}
print(f"11. candidate_is_base ∈ {{0,1}}: {'✓' if checks['is_base_binary'] else '❌'}")

base_rows = df[df["candidate"]=="BASE"]
base_zero_check = (
    (base_rows[response_cols] == 0).all().all() and
    (base_rows["bbse_weight_magnitude"] == 0).all() and
    (base_rows["bbse_prior_extremity"] == 0).all())
checks["base_all_zero"] = base_zero_check
print(f"12. BASE rows мають response+BBSE magnitude = 0: "
      f"{'✓' if base_zero_check else '❌'}")

non_bbse_rows = df[(~df["candidate"].str.contains("\\+BBSE")) & (df["candidate"]!="BASE")]
non_bbse_zero_check = (
    (non_bbse_rows["bbse_weight_magnitude"] == 0).all() and
    (non_bbse_rows["bbse_prior_extremity"] == 0).all())
checks["non_bbse_zero"] = non_bbse_zero_check
print(f"13. non-BBSE candidates мають bbse_weight/prior=0: "
      f"{'✓' if non_bbse_zero_check else '❌'}")

bbse_rows = df[df["candidate"].str.contains("\\+BBSE")]
checks["bbse_rows_count"] = len(bbse_rows) == 3600
print(f"14. +BBSE candidates count: {len(bbse_rows)} "
      f"(очікується 3600) {'✓' if checks['bbse_rows_count'] else '❌'}")

print(f"\n15. Sanity check: response(non-BBSE) == response(+BBSE) "
      f"коли bbse_disabled=1")
mismatch_count = 0
for method in ["CORAL","TAC","QT","UOT","POT"]:
    nb = df[df["candidate"]==method][
        ["instance_key","model","prediction_flip_rate","probability_shift",
         "confidence_change","entropy_change"]].set_index(["instance_key","model"])
    b = df[(df["candidate"]==f"{method}+BBSE") & (df["bbse_disabled"]==1)][
        ["instance_key","model","prediction_flip_rate","probability_shift",
         "confidence_change","entropy_change"]].set_index(["instance_key","model"])
    if len(b) == 0:
        continue
    common_idx = nb.index.intersection(b.index)
    diff = (nb.loc[common_idx] - b.loc[common_idx]).abs()
    n_mismatch = (diff.max(axis=1) > 1e-6).sum()
    mismatch_count += n_mismatch
checks["disabled_consistency"] = mismatch_count == 0
print(f"    Mismatch count: {mismatch_count} "
      f"{'✓' if checks['disabled_consistency'] else '❌'}")

print(f"\n16. bbse_condition_log ↔ frozen BBSE policy "
      f"(COND_MAX={COND_MAX}, sentinel={sentinel_val}):")

policy_mismatch = 0
cond_mismatch = 0

for mn in ["MLP", "RF", "XGB"]:
    for tgt_ds in FLOW_DATASETS:
        frozen_key = f"{tgt_ds}__{mn}"
        expected_action = audit_lookup[frozen_key]
        tgt_short = SHORT[tgt_ds]

        target_rows = df[(df["model"] == mn) &
                          (df["pair_id"].str.endswith(f"__{tgt_short}"))]
        if len(target_rows) == 0:
            continue

        bbse_target_rows = target_rows[target_rows["candidate"].str.endswith("+BBSE")]
        expected_disabled = int(expected_action == "BBSE_DISABLED")

        if not (bbse_target_rows["bbse_disabled"] == expected_disabled).all():
            policy_mismatch += 1
            print(f"    ❌ policy mismatch: {mn}/{tgt_short}: "
                  f"expected bbse_disabled={expected_disabled}")

        expected_cond = float(np.linalg.cond(frozen_matrices[frozen_key]["C"]))
        expected_log = safe_log_condition(expected_cond)
        actual_logs = bbse_target_rows["bbse_condition_log"].values

        if not np.allclose(actual_logs, expected_log, rtol=0, atol=1e-6):
            cond_mismatch += 1
            print(f"    ❌ cond_log mismatch: {mn}/{tgt_short}: "
                  f"expected={expected_log}")

checks["bbse_policy_consistency"] = (policy_mismatch == 0 and cond_mismatch == 0)
print(f"    Policy mismatches: {policy_mismatch}")
print(f"    Condition-log mismatches: {cond_mismatch}")
print(f"    {'✓' if checks['bbse_policy_consistency'] else '❌'}")

checks["ks_shift_no_nan"] = df["ks_shift"].notna().all()
print(f"\n17. ks_shift без NaN: {'✓' if checks['ks_shift_no_nan'] else '❌'}")

forbidden_patterns = ["error", "y_foreign", "y_target", "macro_error",
                        "oracle", "margin", "_true_"]
leaked_cols = [c for c in df.columns
               if any(p in c.lower() for p in forbidden_patterns)]
checks["no_leaked_cols"] = len(leaked_cols) == 0
print(f"18. Відсутність label/error-derived колонок (pattern-based): "
      f"{'✓' if checks['no_leaked_cols'] else f'❌ Знайдено: {leaked_cols}'}")

pair_counts = df.drop_duplicates("instance_key").groupby("pair_id").size()
checks["pair_structure"] = (len(pair_counts)==12) and (pair_counts==20).all()
print(f"19. pair_id структура (12×20): "
      f"{'✓' if checks['pair_structure'] else f'❌ {dict(pair_counts)}'}")

actual_candidates = set(df["candidate"].unique())
expected_candidates = set(ALL_CANDIDATES)
checks["candidate_set_exact"] = actual_candidates == expected_candidates
print(f"20. Candidate set exact match: "
      f"{'✓' if checks['candidate_set_exact'] else f'❌ diff={actual_candidates ^ expected_candidates}'}")

candidate_global_counts = df["candidate"].value_counts()
checks["candidate_balance"] = (
    set(candidate_global_counts.index) == set(ALL_CANDIDATES)
    and (candidate_global_counts == 720).all())
print(f"\n21. Кожен candidate має рівно 720 rows: "
      f"{'✓' if checks['candidate_balance'] else '❌'}")
if not checks["candidate_balance"]:
    print(candidate_global_counts.sort_index())

ks_counts = df.groupby("instance_key")["ks_shift"].nunique()
checks["ks_shift_consistent"] = (ks_counts == 1).all()
print(f"\n22. ks_shift однаковий для всіх model×candidate в межах instance: "
      f"{'✓' if checks['ks_shift_consistent'] else '❌'}")

EXPECTED_COLUMNS = {"instance_key", "pair_id", "model", "candidate", *FEATURE_COLS}
checks["exact_schema"] = set(df.columns) == EXPECTED_COLUMNS
print(f"\n23. Exact schema — тільки IDs + 10 features: "
      f"{'✓' if checks['exact_schema'] else '❌'}")
if not checks["exact_schema"]:
    print("    Extra:", sorted(set(df.columns) - EXPECTED_COLUMNS))
    print("    Missing:", sorted(EXPECTED_COLUMNS - set(df.columns)))

print(f"\n{'='*90}")
n_pass = sum(checks.values())
n_total = len(checks)
print(f"ЗАГАЛЬНИЙ РЕЗУЛЬТАТ: {n_pass}/{n_total} перевірок пройдено "
      f"{'✓ ГОТОВО ДО B.2 — FREEZE B.1c' if n_pass==n_total else '❌ ПОТРЕБУЄ ВИПРАВЛЕННЯ'}")
print(f"{'='*90}")
if n_pass < n_total:
    failed = [k for k,v in checks.items() if not v]
    print(f"Провалені перевірки: {failed}")
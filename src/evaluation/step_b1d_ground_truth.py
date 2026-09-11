import pandas as pd
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
tac_df = pd.read_csv(RES_DAT/"v11_type_aware_coral_240.csv")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")
pot_df = pd.read_csv(RES_DAT/"v13_partial_ot_240.csv")

CANDIDATE_COLUMN_MAP = {
    "BASE":       ("v9",  "{mn}_BASE_error"),
    "CORAL":      ("v9",  "{mn}_CORAL_error"),
    "CORAL+BBSE": ("v9",  "{mn}_CLASP_error"),
    "TAC":        ("tac", "{mn}_TAC_error"),
    "TAC+BBSE":   ("tac", "{mn}_TAC_BBSE_error"),
    "QT":         ("qt",  "{mn}_QT_error"),
    "QT+BBSE":    ("qt",  "{mn}_QT_BBSE_error"),
    "UOT":        ("uot", "{mn}_UOT_error"),
    "UOT+BBSE":   ("uot", "{mn}_UOT_BBSE_error"),
    "POT":        ("pot", "{mn}_POT_error"),
    "POT+BBSE":   ("pot", "{mn}_POT_BBSE_error"),
}
SOURCE_DFS = {"v9": v9_df, "tac": tac_df, "qt": qt_df, "uot": uot_df, "pot": pot_df}

print("="*90)
print("STEP B.1d — Ground Truth Long Format")
print("="*90)

ground_truth_rows = []
for mn in ["MLP", "RF", "XGB"]:
    for candidate, (src_key, col_template) in CANDIDATE_COLUMN_MAP.items():
        src_df = SOURCE_DFS[src_key]
        col_name = col_template.format(mn=mn)
        if col_name not in src_df.columns:
            print(f"  ❌ Колонка {col_name} відсутня в {src_key} "
                  f"(candidate={candidate}, model={mn})")
            continue

        sub = src_df[["instance_key", "pair_id", col_name]].copy()
        sub["model"] = mn
        sub["candidate"] = candidate
        sub = sub.rename(columns={col_name: "macro_error"})
        ground_truth_rows.append(sub)

gt_long = pd.concat(ground_truth_rows, ignore_index=True)
gt_long = gt_long[["instance_key", "pair_id", "model", "candidate", "macro_error"]]

print(f"\n{'─'*90}")
print("ДІАГНОСТИКА (перед hard assert)")
print(f"{'─'*90}")

n_rows = len(gt_long)
print(f"Всього рядків у gt_long: {n_rows} (очікувано 7920)")

n_dup = gt_long.duplicated(subset=["instance_key","model","candidate"]).sum()
print(f"Дублікатів (instance_key,model,candidate): {n_dup}")

gt_task_counts = gt_long.groupby(["instance_key","model"])["candidate"].nunique()
n_incomplete_tasks = (gt_task_counts != 11).sum()
print(f"Задач з неповним набором candidates (≠11): {n_incomplete_tasks}")
if n_incomplete_tasks > 0:
    print(f"  Приклади: {gt_task_counts[gt_task_counts!=11].head()}")

n_nan_error = gt_long["macro_error"].isna().sum()
print(f"NaN у macro_error: {n_nan_error}")

b1c_df = pd.read_csv(RES_DAT/"step_b1c_candidate_matrix.csv")

b1c_keys = set(zip(b1c_df["instance_key"], b1c_df["model"], b1c_df["candidate"]))
gt_keys = set(zip(gt_long["instance_key"], gt_long["model"], gt_long["candidate"]))

only_in_b1c = b1c_keys - gt_keys
only_in_gt = gt_keys - b1c_keys

print(f"\nКлючі (instance_key, model, candidate) присутні ТІЛЬКИ в B.1c: "
      f"{len(only_in_b1c)}")
if only_in_b1c:
    print(f"  Приклади: {list(only_in_b1c)[:5]}")

print(f"Ключі присутні ТІЛЬКИ в gt_long: {len(only_in_gt)}")
if only_in_gt:
    print(f"  Приклади: {list(only_in_gt)[:5]}")

keys_match = (len(only_in_b1c) == 0) and (len(only_in_gt) == 0)
print(f"\nПовна ідентичність множин ключів: "
      f"{'✓' if keys_match else '❌ РОЗБІЖНІСТЬ'}")

b1c_candidates = set(b1c_df["candidate"].unique())
gt_candidates = set(gt_long["candidate"].unique())
candidate_sets_match = b1c_candidates == gt_candidates
print(f"Candidate set B.1c: {sorted(b1c_candidates)}")
print(f"Candidate set gt_long: {sorted(gt_candidates)}")
print(f"Candidate sets ідентичні: "
      f"{'✓' if candidate_sets_match else '❌ РОЗБІЖНІСТЬ'}")

print(f"\n{'─'*90}")
print("HARD ASSERTIONS (тільки якщо діагностика чиста)")
print(f"{'─'*90}")

assert n_rows == 7920, f"Очікувалось 7920 рядків, отримано {n_rows}"
assert n_dup == 0, f"Знайдено {n_dup} дублікатів"
assert n_incomplete_tasks == 0, f"{n_incomplete_tasks} задач з неповним набором candidates"
assert n_nan_error == 0, f"Знайдено {n_nan_error} NaN у macro_error"
assert keys_match, "Множини ключів (instance_key,model,candidate) НЕ ідентичні між B.1c і gt_long"
assert candidate_sets_match, "Candidate sets НЕ ідентичні між B.1c і gt_long"

print("✓ Усі assertions пройдені")

merged_check = b1c_df.merge(
    gt_long, on=["instance_key","model","candidate"],
    how="inner", validate="one_to_one")
assert len(merged_check) == 7920
print(f"✓ Merge з B.1c підтверджено: {len(merged_check)} рядків")

gt_long.to_csv(RES_DAT/"step_b1d_ground_truth_long.csv", index=False)
print(f"\nSaved: {RES_DAT}/step_b1d_ground_truth_long.csv")

print(f"\nMean macro_error по candidate (сирий, для sanity check):")
print(gt_long.groupby("candidate")["macro_error"].mean().sort_values().to_string())
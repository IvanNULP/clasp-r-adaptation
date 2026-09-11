import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

FILES = {
    "v9":  "v9_operational_adaptation_eval_240.csv",
    "v10": "v10_quantile_transport_240.csv",
    "v11": "v11_type_aware_coral_240.csv",
    "v12": "v12_unbalanced_ot_240.csv",
    "v13": "v13_partial_ot_240.csv",
    "b1b": "step_b1b_reliability_all_methods_240.csv",
    "b1c": "step_b1c_candidate_matrix.csv",
    "b1d": "step_b1d_ground_truth_long.csv",
}

dfs = {name: pd.read_csv(RES_DAT/fname) for name, fname in FILES.items()}

print("="*90)
print("STEP B.1e — Cross-Artifact Consistency Audit (7 пунктів)")
print("="*90)

print(f"\n{'='*90}\n1. INSTANCE-LEVEL IDENTITY\n{'='*90}")

instance_sets = {name: set(df["instance_key"].unique())
                  for name, df in dfs.items() if "instance_key" in df.columns}
reference_set = instance_sets["v9"]
all_identical = all(s == reference_set for s in instance_sets.values())

for name, s in instance_sets.items():
    status = "✓" if s == reference_set else f"❌ diff={len(s ^ reference_set)}"
    print(f"  {name:5s}: {len(s)} instance_key  {status}")

print(f"\n  Усі множини ідентичні: {'✓' if all_identical else '❌'}")

print(f"\n{'='*90}\n2. SEED PROVENANCE IDENTITY\n{'='*90}")

seed_files = {k: v for k, v in dfs.items()
              if "foreign_seed" in v.columns and "target_ref_seed" in v.columns}
ref_seeds = dfs["v9"].set_index("instance_key")[["foreign_seed","target_ref_seed"]]
seed_mismatch_total = 0

for name, df in seed_files.items():
    if name == "v9":
        continue
    cmp_seeds = df.set_index("instance_key")[["foreign_seed","target_ref_seed"]]
    common_idx = ref_seeds.index.intersection(cmp_seeds.index)
    diff = (ref_seeds.loc[common_idx] != cmp_seeds.loc[common_idx]).any(axis=1)
    n_mismatch = diff.sum()
    seed_mismatch_total += n_mismatch
    print(f"  {name:5s} vs v9: {'✓' if n_mismatch==0 else f'❌ {n_mismatch}'}")

print(f"\n  Загальна кількість seed mismatches: {seed_mismatch_total} "
      f"{'✓' if seed_mismatch_total==0 else '❌'}")

print(f"\n{'='*90}\n3. STRUCTURAL instance_key ↔ pair_id ↔ replication_idx\n{'='*90}")

def decompose_instance_key(inst_key):
    parts = inst_key.split("__")
    assert len(parts) == 3, f"Неочікуваний формат: {inst_key}"
    src, tgt, k_part = parts
    assert k_part.startswith("k"), f"Очікувався префікс 'k': {k_part}"
    return src, tgt, int(k_part[1:])

structural_mismatch_total = 0
for name, df in dfs.items():
    if "instance_key" not in df.columns or "pair_id" not in df.columns:
        continue
    check_df = df.drop_duplicates("instance_key")[["instance_key", "pair_id"]].copy()
    decomposed = check_df["instance_key"].apply(decompose_instance_key)
    check_df["src_from_key"] = decomposed.apply(lambda x: x[0])
    check_df["tgt_from_key"] = decomposed.apply(lambda x: x[1])
    check_df["rep_idx_from_key"] = decomposed.apply(lambda x: x[2])
    check_df["expected_pair_id"] = check_df["src_from_key"] + "__" + check_df["tgt_from_key"]
    mismatch = check_df["pair_id"] != check_df["expected_pair_id"]
    n_mismatch = mismatch.sum()
    structural_mismatch_total += n_mismatch
    rep_idx_ok = check_df["rep_idx_from_key"].between(0, 19).all()
    print(f"  {name:5s}: pair_id match={n_mismatch==0}, rep_idx∈[0,19]={rep_idx_ok} "
          f"{'✓' if (n_mismatch==0 and rep_idx_ok) else '❌'}")
    if n_mismatch > 0:
        print(f"    Приклади розбіжностей:\n"
              f"{check_df[mismatch][['instance_key','pair_id','expected_pair_id']].head()}")

print(f"\n  Загальна кількість structural mismatches: {structural_mismatch_total} "
      f"{'✓' if structural_mismatch_total==0 else '❌'}")

print(f"\n{'='*90}\n4. B.1c ↔ B.1d COVERAGE\n{'='*90}")

b1c_keys = set(zip(dfs["b1c"]["instance_key"], dfs["b1c"]["model"], dfs["b1c"]["candidate"]))
b1d_keys = set(zip(dfs["b1d"]["instance_key"], dfs["b1d"]["model"], dfs["b1d"]["candidate"]))
coverage_match = b1c_keys == b1d_keys

print(f"  B.1c: {len(b1c_keys)}, B.1d: {len(b1d_keys)}, "
      f"Match: {'✓' if coverage_match else '❌'}")
if not coverage_match:
    only_b1c = b1c_keys - b1d_keys
    only_b1d = b1d_keys - b1c_keys
    print(f"    only_in_b1c={len(only_b1c)}, only_in_b1d={len(only_b1d)}")

print(f"\n{'='*90}\n5. DETERMINISTIC X_FOREIGN RECONSTRUCTION CHECK\n{'='*90}")
print("(Підтверджує лише: seed→X_foreign функція детермінована САМА")
print(" ІЗ СОБОЮ. НЕ підтверджує узгодженість з історичними запусками")
print(" v9-v13/b1c — для цього дивись Пункт 6 нижче.)")

BASE = Path("/srv/ids_research")
PREP = BASE / "data/preprocessed"
FINAL_FEATURES = [
    "duration_ms","total_packets","total_bytes",
    "fwd_packets","fwd_bytes","bwd_packets","bwd_bytes",
    "bytes_per_sec","packets_per_sec","protocol","dst_port",
]
manifest_df = pd.read_csv(RES_DAT/"v8_instance_manifest.csv")
SHORT = {"CIC-IDS-2017":"CIC-2017","UNSW-NB15":"UNSW",
         "TON-IoT-2021":"TON-IoT","CICIoT2023":"CICIoT23"}

from sklearn.model_selection import train_test_split

def recompute_X_foreign(src_ds, foreign_seed, max_sample=5000):
    df_src_full = pd.read_parquet(PREP/"test"/f"{src_ds}.parquet").reset_index(drop=True)
    if len(df_src_full) > max_sample:
        df_foreign, _ = train_test_split(
            df_src_full, train_size=max_sample,
            stratify=df_src_full["label_unified"], random_state=foreign_seed)
    else:
        df_foreign = df_src_full
    return df_foreign[FINAL_FEATURES].values.astype(np.float32)

def array_hash(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()

np.random.seed(999)
all_instance_keys = list(reference_set)
spot_check_instances = np.random.choice(all_instance_keys, 5, replace=False)

determinism_ok = True
for inst_key in spot_check_instances:
    manifest_row = None
    for _, row in manifest_df.iterrows():
        src_short, tgt_short = SHORT[row["src"]], SHORT[row["tgt"]]
        if f"{src_short}__{tgt_short}__k{row['replication_idx']}" == inst_key:
            manifest_row = row
            break
    if manifest_row is None:
        print(f"  ❌ {inst_key}: не знайдено в manifest")
        determinism_ok = False
        continue

    src_ds = manifest_row["src"]
    foreign_seed = int(manifest_row["foreign_seed"])

    X_call_1 = recompute_X_foreign(src_ds, foreign_seed)
    X_call_2 = recompute_X_foreign(src_ds, foreign_seed)
    match = array_hash(X_call_1) == array_hash(X_call_2)
    determinism_ok = determinism_ok and match
    print(f"  {inst_key}: self-consistency {'✓' if match else '❌'}")

print(f"\n  X_foreign reconstruction детермінована: "
      f"{'✓' if determinism_ok else '❌'}")

print(f"\n{'='*90}\n6. BASE ERROR NUMERICAL CONSISTENCY: v9 ↔ v10 ↔ v11 ↔ v12 ↔ v13\n{'='*90}")
print("(Найсильніший доступний непрямий доказ узгодженості:")
print(" BASE_error ідентичний у 5 незалежно згенерованих артефактах")
print(" для кожного instance×model. Це підтверджує відсутність")
print(" числових розбіжностей у BASE evaluation; разом із")
print(" seed/structural checks підтримує узгодженість backbone.)")

v9_rows_ok = len(dfs["v9"]) == 240
if not v9_rows_ok:
    print(f"\n❌ v9 (reference) rows = {len(dfs['v9'])}, expected 240 — "
          f"усі подальші порівняння в цьому пункті будуть НЕНАДІЙНІ")
else:
    print(f"\n✓ v9 (reference) rows = 240, як очікувалось")

BASE_COL_MAP = {
    "v9": {"MLP":"MLP_BASE_error", "RF":"RF_BASE_error", "XGB":"XGB_BASE_error"},
    "v10":{"MLP":"MLP_BASE_error", "RF":"RF_BASE_error", "XGB":"XGB_BASE_error"},
    "v11":{"MLP":"MLP_BASE_error", "RF":"RF_BASE_error", "XGB":"XGB_BASE_error"},
    "v12":{"MLP":"MLP_BASE_error", "RF":"RF_BASE_error", "XGB":"XGB_BASE_error"},
    "v13":{"MLP":"MLP_BASE_error", "RF":"RF_BASE_error", "XGB":"XGB_BASE_error"},
}

base_consistency_total_mismatch = 0
base_coverage_incomplete = False

for mn in ["MLP", "RF", "XGB"]:
    ref_base = dfs["v9"].set_index("instance_key")[BASE_COL_MAP["v9"][mn]]
    print(f"\n  Model: {mn}")
    for name in ["v10", "v11", "v12", "v13"]:
        cmp_base = dfs[name].set_index("instance_key")[BASE_COL_MAP[name][mn]]
        common_idx = ref_base.index.intersection(cmp_base.index)

        if len(common_idx) != 240:
            print(f"    ❌ {name}: common instances = {len(common_idx)}, expected 240")
            base_coverage_incomplete = True
        else:
            print(f"    {name}: common instances = {len(common_idx)}/240 ✓")

        diff = np.abs(ref_base.loc[common_idx] - cmp_base.loc[common_idx])
        n_mismatch = (diff > 1e-9).sum()
        max_diff = diff.max() if len(diff) > 0 else float("nan")
        base_consistency_total_mismatch += n_mismatch
        status = "✓" if n_mismatch == 0 else f"❌ {n_mismatch} mismatches, max_diff={max_diff:.2e}"
        print(f"      v9 vs {name} value match: {status}")

print(f"\n  Загальна кількість BASE_error mismatches: "
      f"{base_consistency_total_mismatch} "
      f"{'✓' if base_consistency_total_mismatch==0 else '❌'}")
print(f"  Повне покриття (240/240) у всіх порівняннях: "
      f"{'✓' if not base_coverage_incomplete else '❌ ДЕЯКІ ПОРІВНЯННЯ НЕПОВНІ — див. вище'}")

print(f"\n{'='*90}\n7. UOT/POT DIAGNOSTICS INTEGRITY (explicit schema + row count audit)\n{'='*90}")

uot_df = dfs["v12"]
pot_df = dfs["v13"]

uot_rows_ok = len(uot_df) == 240
pot_rows_ok = len(pot_df) == 240

print(f"  UOT rows: {len(uot_df)} {'✓' if uot_rows_ok else '❌ expected 240'}")
print(f"  POT rows: {len(pot_df)} {'✓' if pot_rows_ok else '❌ expected 240'}")

required_v12 = [
    "sinkhorn_converged",
    "uot_total_mass",
    "uot_mean_row_mass",
    "uot_min_row_mass",
    "uot_max_row_mass",
]
required_v13 = [
    "pot_converged",
    "pot_total_mass",
    "pot_mean_row_mass",
    "pot_n_unmatched_source",
]

print("\n  UOT (v12) schema check:")
v12_schema_ok = True
for col in required_v12:
    present = col in uot_df.columns
    v12_schema_ok = v12_schema_ok and present
    print(f"    {col}: {'✓' if present else '❌ MISSING'}")

print("\n  POT (v13) schema check:")
v13_schema_ok = True
for col in required_v13:
    present = col in pot_df.columns
    v13_schema_ok = v13_schema_ok and present
    print(f"    {col}: {'✓' if present else '❌ MISSING'}")

print(f"\n  Schema audit: UOT {'✓' if v12_schema_ok else '❌'}, "
      f"POT {'✓' if v13_schema_ok else '❌'}")

if not v12_schema_ok:
    print(f"\n  ⚠ Реальні колонки v12: {list(uot_df.columns)}")
if not v13_schema_ok:
    print(f"  ⚠ Реальні колонки v13: {list(pot_df.columns)}")

if v12_schema_ok and uot_rows_ok:
    n_fallback_uot = (~uot_df["sinkhorn_converged"]).sum()
    mean_mass = uot_df["uot_total_mass"].mean()
    print(f"\n  UOT: sinkhorn_converged=False count: {n_fallback_uot} "
          f"{'✓' if n_fallback_uot==0 else '⚠ перевір ці instances окремо'}")
    print(f"  UOT: mean total_mass={mean_mass:.4f} (reference run ≈0.9886)")
elif not v12_schema_ok:
    print(f"\n  ⚠ UOT diagnostics ПРОПУЩЕНО через розбіжність схеми")
else:
    print(f"\n  ⚠ UOT diagnostics ПРОПУЩЕНО через неповний row count")

if v13_schema_ok and pot_rows_ok:
    n_fallback_pot = (~pot_df["pot_converged"]).sum()
    mean_mass_pot = pot_df["pot_total_mass"].mean()
    print(f"\n  POT: pot_converged=False count: {n_fallback_pot} "
          f"{'✓' if n_fallback_pot==0 else '⚠ перевір ці instances окремо'}")
    print(f"  POT: mean total_mass={mean_mass_pot:.4f} (target m=0.8)")
elif not v13_schema_ok:
    print(f"\n  ⚠ POT diagnostics ПРОПУЩЕНО через розбіжність схеми")
else:
    print(f"\n  ⚠ POT diagnostics ПРОПУЩЕНО через неповний row count")

print(f"\n{'='*90}")
print("ЗВЕДЕНИЙ ПІДСУМОК АУДИТУ (7 пунктів)")
print(f"{'='*90}")

summary_checks = {
    "1. Instance identity": all_identical,
    "2. Seed provenance": seed_mismatch_total == 0,
    "3. Structural mapping": structural_mismatch_total == 0,
    "4. B.1c/B.1d coverage": coverage_match,
    "5. Deterministic reconstruction": determinism_ok,
    "6. BASE consistency": (
        v9_rows_ok
        and base_consistency_total_mismatch == 0
        and not base_coverage_incomplete
    ),
    "7. UOT/POT diagnostics": (
        v12_schema_ok
        and v13_schema_ok
        and uot_rows_ok
        and pot_rows_ok
    ),
}

for name, passed in summary_checks.items():
    print(f"  {name}: {'✓' if passed else '❌'}")

n_pass = sum(summary_checks.values())
n_total = len(summary_checks)
print(f"\nЗАГАЛЬНИЙ РЕЗУЛЬТАТ: {n_pass}/{n_total} "
      f"{'✓ ГОТОВО ДО STEP B.2' if n_pass==n_total else '❌ ПОТРЕБУЄ ВИПРАВЛЕННЯ'}")
print(f"{'='*90}")
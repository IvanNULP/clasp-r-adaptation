import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

b31_df = pd.read_csv(RES_DAT/"step_b3_1_unified_comparison.csv")
b32_df = pd.read_csv(RES_DAT/"step_b3_2_negative_transfer.csv")
b34_df = pd.read_csv(RES_DAT/"step_b3_4_model_specific_profiles.csv")

MODELS = ["MLP", "RF", "XGB"]

print("="*100)
print("STEP B.3.5 (фінальна версія) — Robustness Decision")
print("="*100)

assert len(b31_df) == 30, f"B.3.1 expected 30 rows, got {len(b31_df)}"
assert len(b32_df) == 30, f"B.3.2 expected 30 rows, got {len(b32_df)}"
assert len(b34_df) == 3, f"B.3.4 expected 3 rows (audit only), got {len(b34_df)}"

assert set(b31_df["model"]) == set(MODELS)
assert set(b32_df["model"]) == set(MODELS)
assert set(b34_df["model"]) == set(MODELS)

assert b31_df[["model", "candidate"]].duplicated().sum() == 0
assert b32_df[["model", "candidate"]].duplicated().sum() == 0
assert b34_df[["model"]].duplicated().sum() == 0

assert b31_df["mean_delta"].notna().all()
assert b32_df["p_degraded"].notna().all()
assert b32_df["max_pair_positive_delta"].notna().all()
assert b32_df["sd_delta"].notna().all()

print("\n✓ Всі identity/completeness checks пройдені")
print("✓ B.3.4 завантажено лише як audit context (не для рішення)")

print("""
ЗАФІКСОВАНІ КРИТЕРІЇ (визначені ДО перегляду того, як вони спрацюють):

  R1. mean_delta < 0          (у середньому покращує відносно BASE)
  R2. p_degraded < 0.50       (шкодить менш ніж половині instances)
  R3. НЕ найгірший за жодним risk-критерієм (P(deg) чи max_pair_positive_delta)

  0 candidates задовольняють  → Сценарій C: жоден метод недостатньо надійний
  1 candidate задовольняє     → Сценарій A: стабільний model-specific fixed policy
  >1 candidates задовольняють → Сценарій B: детермінований tie-break:
                                  lowest SD → lowest max_pair_positive_delta
                                  → lowest mean_delta.
                                  Якщо ПОВНА нічия за всіма трьома —
                                  fail-loud (жодного четвертого критерію
                                  не вигадуємо post-hoc).
""")

screening_records = []
decision_records = []

for mn in MODELS:
    print(f"\n{'='*100}\n{mn}\n{'='*100}")

    mn_b31 = b31_df[b31_df["model"]==mn].copy()
    mn_b32 = b32_df[b32_df["model"]==mn].copy()

    assert set(mn_b31["candidate"]) == set(mn_b32["candidate"]), \
        f"{mn}: B.3.1/B.3.2 candidate coverage mismatch"

    merged = mn_b31.merge(mn_b32, on=["candidate","model"], suffixes=("_b31","_b32"))

    assert len(merged) == 10, f"{mn}: очікувалось 10 candidates, отримано {len(merged)}"
    assert not merged.isna().any().any(), f"{mn}: NaN появився після merge"

    required_cols = ["mean_delta_b31", "sd_delta", "p_degraded", "max_pair_positive_delta"]
    assert all(c in merged.columns for c in required_cols), (
        f"{mn}: missing required columns after merge: "
        f"{[c for c in required_cols if c not in merged.columns]}"
    )

    worst_p_degraded = merged["p_degraded"].max()
    worst_max_pair_delta = merged["max_pair_positive_delta"].max()

    merged["R1"] = merged["mean_delta_b31"] < 0
    merged["R2"] = merged["p_degraded"] < 0.50
    merged["R3"] = (
        (merged["p_degraded"] < worst_p_degraded) &
        (merged["max_pair_positive_delta"] < worst_max_pair_delta)
    )

    merged["robust"] = merged["R1"] & merged["R2"] & merged["R3"]

    for _, r in merged.iterrows():
        screening_records.append({
            "model": mn,
            "candidate": r["candidate"],
            "mean_delta": r["mean_delta_b31"],
            "p_degraded": r["p_degraded"],
            "max_pair_positive_delta": r["max_pair_positive_delta"],
            "sd_delta": r["sd_delta"],
            "R1_mean_improvement": bool(r["R1"]),
            "R2_low_degradation_probability": bool(r["R2"]),
            "R3_not_worst_on_any_risk": bool(r["R3"]),
            "robust": bool(r["robust"]),
        })

    print(f"\n{'Candidate':14s} {'MeanΔ':>8s} {'R1':>4s} {'P(deg)':>7s} "
          f"{'R2':>4s} {'MaxPairΔ':>9s} {'R3':>4s} {'ROBUST':>7s}")
    print("─"*70)
    for _, r in merged.sort_values("mean_delta_b31").iterrows():
        print(f"{r['candidate']:14s} {r['mean_delta_b31']:>+8.4f} "
              f"{'✓' if r['R1'] else '✗':>4s} {r['p_degraded']:>7.3f} "
              f"{'✓' if r['R2'] else '✗':>4s} {r['max_pair_positive_delta']:>9.4f} "
              f"{'✓' if r['R3'] else '✗':>4s} {'✓✓✓' if r['robust'] else '':>7s}")

    robust_candidates = merged[merged["robust"]]["candidate"].tolist()
    n_robust = len(robust_candidates)

    if n_robust == 0:
        scenario = "C"
        chosen = None
        rationale = "Жоден candidate не задовольняє всі три критерії R1-R3"
    elif n_robust == 1:
        scenario = "A"
        chosen = robust_candidates[0]
        rationale = f"Єдиний candidate задовольняє R1-R3: {chosen}"
    else:
        scenario = "B"
        robust_df = merged[merged["robust"]].copy()
        robust_df = robust_df.sort_values(
            by=["sd_delta", "max_pair_positive_delta", "mean_delta_b31"],
            ascending=[True, True, True]
        )

        tie_cols = ["sd_delta", "max_pair_positive_delta", "mean_delta_b31"]
        if robust_df.duplicated(subset=tie_cols, keep=False).any():
            tied = robust_df.loc[
                robust_df.duplicated(subset=tie_cols, keep=False),
                ["candidate"] + tie_cols
            ]
            raise AssertionError(
                f"{mn}: unresolved tie among robust candidates under the "
                f"predefined tie-break criteria:\n{tied.to_string(index=False)}"
            )

        chosen = robust_df.iloc[0]["candidate"]
        rationale = (
            f"Декілька robust candidates ({robust_candidates}); "
            f"tie-break: lowest SD → lowest max_pair_positive_delta "
            f"→ lowest mean_delta: {chosen}"
        )

    print(f"\nSCENARIO: {scenario}")
    print(f"  Robust candidates: {robust_candidates if robust_candidates else 'НЕМАЄ'}")
    print(f"  Recommended: {chosen if chosen else 'N/A — потрібен додатковий mechanism'}")
    print(f"  Rationale: {rationale}")

    decision_records.append({
        "model": mn, "scenario": scenario,
        "n_robust_candidates": n_robust,
        "robust_candidates": ", ".join(robust_candidates) if robust_candidates else "none",
        "recommended_candidate": chosen if chosen else "N/A",
        "rationale": rationale,
    })

screening_df = pd.DataFrame(screening_records)
assert len(screening_df) == 30, f"B.3.5 screening expected 30 rows, got {len(screening_df)}"
assert screening_df[["model", "candidate"]].duplicated().sum() == 0

screening_df.to_csv(RES_DAT/"step_b3_5_robustness_screening.csv", index=False)
print(f"\nSaved: step_b3_5_robustness_screening.csv")

decision_df = pd.DataFrame(decision_records)
decision_df.to_csv(RES_DAT/"step_b3_5_robustness_decision.csv", index=False)

print(f"\n{'='*100}")
print("ЗВЕДЕНЕ РІШЕННЯ ПО ВСІХ МОДЕЛЯХ")
print(f"{'='*100}")
print(decision_df.to_string(index=False))

print(f"\n{'='*100}")
print("Saved: step_b3_5_robustness_decision.csv")
print(f"{'='*100}")
import pandas as pd
import numpy as np
from pathlib import Path

RES_DAT = Path("/srv/ids_research/results/domain_adaptation")

v9_df = pd.read_csv(RES_DAT/"v9_operational_adaptation_eval_240.csv")
qt_df = pd.read_csv(RES_DAT/"v10_quantile_transport_240.csv")
uot_df = pd.read_csv(RES_DAT/"v12_unbalanced_ot_240.csv")

DOMAINS = ["CIC-2017", "UNSW", "TON-IoT", "CICIoT23"]

FOCUS_CANDIDATES = {
    "MLP": [
        ("CORAL+BBSE", v9_df, "CLASP"),
        ("CORAL",      v9_df, "CORAL"),
        ("QT+BBSE",    qt_df, "QT_BBSE"),
        ("UOT+BBSE",   uot_df, "UOT_BBSE"),
    ],
    "RF": [
        ("QT",         qt_df, "QT"),
        ("QT+BBSE",    qt_df, "QT_BBSE"),
    ],
    "XGB": [
        ("UOT+BBSE",   uot_df, "UOT_BBSE"),
        ("QT",         qt_df, "QT"),
        ("QT+BBSE",    qt_df, "QT_BBSE"),
    ],
}

print("="*100)
print("STEP B.3.3 (фінальна версія) — Asymmetric Transfer Matrices")
print("(тільки перспективні candidates за B.3.1/B.3.2)")
print("="*100)

matrix_records = []
asymmetry_records = []

for mn, candidates_list in FOCUS_CANDIDATES.items():
    for candidate_name, df, col_suffix in candidates_list:

        assert len(df) == 240, (
            f"{candidate_name}/{mn}: expected 240 rows, got {len(df)}"
        )
        assert df["instance_key"].is_unique, (
            f"{candidate_name}/{mn}: duplicate instance_key"
        )
        assert set(df["instance_key"]) == set(v9_df["instance_key"]), (
            f"{candidate_name}/{mn}: instance_key mismatch vs v9"
        )

        base_col = f"{mn}_BASE_error"
        cand_col = f"{mn}_{col_suffix}_error"

        assert df[cand_col].notna().all(), f"{candidate_name}/{mn}: NaN in candidate error"
        assert df[base_col].notna().all(), f"{candidate_name}/{mn}: NaN in BASE error"

        d_df = df[["instance_key","pair_id",cand_col,base_col]].copy()
        d_df = d_df.sort_values("instance_key").reset_index(drop=True)
        d_df["delta"] = d_df[cand_col] - d_df[base_col]

        pair_parts = d_df["pair_id"].str.split("__", expand=True)
        assert pair_parts.shape[1] == 2, \
            f"{candidate_name}/{mn}: malformed pair_id detected"

        d_df["src"] = pair_parts[0]
        d_df["tgt"] = pair_parts[1]

        assert d_df["src"].isin(DOMAINS).all(), \
            f"{candidate_name}/{mn}: unknown source domain"
        assert d_df["tgt"].isin(DOMAINS).all(), \
            f"{candidate_name}/{mn}: unknown target domain"
        assert (d_df["src"] != d_df["tgt"]).all(), \
            f"{candidate_name}/{mn}: self-transfer pair detected"

        expected_pairs = {(src, tgt) for src in DOMAINS for tgt in DOMAINS if src != tgt}
        observed_pairs = set(zip(d_df["src"], d_df["tgt"]))
        assert observed_pairs == expected_pairs, (
            f"{candidate_name}/{mn}: directed-pair mismatch. "
            f"Expected 12 pairs, got {len(observed_pairs)}. "
            f"Missing={expected_pairs - observed_pairs}; "
            f"Extra={observed_pairs - expected_pairs}")

        pair_counts = d_df.groupby(["src", "tgt"]).size()
        assert (pair_counts == 20).all(), (
            f"{candidate_name}/{mn}: each directed pair must have exactly 20 replications. "
            f"Counts:\n{pair_counts}")

        pair_means = d_df.groupby(["src","tgt"])["delta"].mean()

        header_label = "SOURCE\\TARGET"
        print(f"\n{'='*90}\n{mn} — {candidate_name}\n{'='*90}")
        print(f"\n{header_label:14s}" + "".join(f"{t:>12s}" for t in DOMAINS))
        for src in DOMAINS:
            row_str = f"{src:14s}"
            for tgt in DOMAINS:
                if src == tgt:
                    row_str += f"{'NA':>12s}"
                    matrix_records.append({
                        "model": mn, "candidate": candidate_name,
                        "source": src, "target": tgt, "mean_delta": np.nan})
                else:
                    val = pair_means[(src, tgt)]
                    row_str += f"{val:>+12.4f}"
                    matrix_records.append({
                        "model": mn, "candidate": candidate_name,
                        "source": src, "target": tgt, "mean_delta": val})
            print(row_str)

        print(f"\n  Directional asymmetry (A→B vs B→A):")
        checked_pairs = set()
        for src in DOMAINS:
            for tgt in DOMAINS:
                if src == tgt: continue
                pair_key = tuple(sorted([src, tgt]))
                if pair_key in checked_pairs: continue
                checked_pairs.add(pair_key)

                fwd = pair_means[(src, tgt)]
                bwd = pair_means[(tgt, src)]

                directional_diff = fwd - bwd
                sign_flip = (fwd > 0) != (bwd > 0)

                print(f"    {src}↔{tgt}: {src}→{tgt}={fwd:+.4f}, "
                      f"{tgt}→{src}={bwd:+.4f}, diff={directional_diff:+.4f}, "
                      f"sign_flip={'YES' if sign_flip else 'no'}")

                asymmetry_records.append({
                    "model": mn, "candidate": candidate_name,
                    "pair": f"{src}__{tgt}",
                    "forward": round(fwd, 4), "backward": round(bwd, 4),
                    "directional_diff": round(directional_diff, 4),
                    "sign_flip": sign_flip,
                })

matrix_df = pd.DataFrame(matrix_records)
asymmetry_df = pd.DataFrame(asymmetry_records)

matrix_df.to_csv(RES_DAT/"step_b3_3_asymmetric_transfer_matrices.csv", index=False)
asymmetry_df.to_csv(RES_DAT/"step_b3_3_directional_asymmetry.csv", index=False)

print(f"\n{'='*100}")
print("Saved: step_b3_3_asymmetric_transfer_matrices.csv")
print("Saved: step_b3_3_directional_asymmetry.csv")
print(f"{'='*100}")

print(f"\n{'='*100}")
print("ЗВЕДЕННЯ: Sign-flip cases (описовий індикатор asymmetric transfer)")
print(f"{'='*100}")
n_sign_flips = asymmetry_df["sign_flip"].sum()
n_total = len(asymmetry_df)
print(f"\nЗагалом sign_flip=YES: {n_sign_flips}/{n_total}")
if n_sign_flips > 0:
    print(f"\n{asymmetry_df[asymmetry_df['sign_flip']].to_string(index=False)}")
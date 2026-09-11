# Cross-Dataset Adaptation for Intrusion Detection — Experiment Code

Code for reproducing the adaptation experiments: dataset unification,
adaptation operators (CORAL, class-conditional CORAL, BBSE, CLASP,
quantile/unbalanced/partial transport), the confidence-based router,
and the evaluation protocol (leave-one-domain-pair-out selector,
transport benchmark, negative-transfer and directional-asymmetry analysis).

## Datasets
Not included — see `DATA.md`.

## Environment
See `ENVIRONMENT.md` and `requirements.txt`.

## Structure
- `src/data_prep/` — feature unification, train/test partitioning
- `src/adaptation/` — CORAL, BBSE, CLASP, transport operators
- `src/router_history/` — confidence-based router (v6-v7)
- `src/audit/` — identifiability, consistency, and robustness audits
- `src/evaluation/` — LODO selector, unified comparison, final policy

Result artefacts (per-instance CSVs, confusion matrices) are archived
separately on Zenodo (Data and result artefacts: https://doi.org/10.5281/zenodo.22708682); see the manuscript's data availability statement.

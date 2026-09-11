# Cross-Dataset Adaptation for Intrusion Detection — Experiment Code

Code and result artefacts for reproducing the experiments: dataset
unification, adaptation operators, frozen classifiers, and the evaluation
protocol (leave-one-domain-pair-out selector, transport benchmark,
negative-transfer and directional-asymmetry analysis).

## Datasets

Not included. Each of the four datasets is obtained from its original
authors under their own terms; see `DATA.md` for exact sources and the
preprocessing needed to reach the unified feature representation this code
expects as input.

## Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Exact versions used to produce the committed results are pinned in
`requirements.txt`. Hardware and OS details are in `ENVIRONMENT.md`.

## Running

```bash
python src/data_prep/unify_features.py --dataset all
python src/data_prep/partition_deployment_domain.py --dataset all
python src/data_prep/construct_transfer_instances.py
python src/evaluation/train_classifiers.py --dataset all
python src/audit/confusion_matrix_audit.py
python src/adaptation/oracle_study.py
python src/adaptation/run_operator.py --operator <coral|coral_bbse|tac|tac_bbse|qt|qt_bbse|uot|uot_bbse|pot|pot_bbse>
python src/audit/consistency_audit.py
python src/evaluation/build_response_features.py
python src/evaluation/lodo_selector.py
python src/evaluation/unified_comparison.py
python src/evaluation/negative_transfer_profile.py
python src/evaluation/directional_asymmetry.py
python src/evaluation/derive_policy.py
```

Seeds, expected output shapes, and consistency checks for each step are
documented in the docstring of the corresponding script.

## Results

`results/` holds the frozen artefacts these scripts produce: twelve
confusion matrices, per-instance error tables for ten adaptation
candidates, and aggregate comparison tables. These are the numbers
reported in the associated manuscript.

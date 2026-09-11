# Data Sources

Four public network intrusion detection datasets, none redistributed here.

| Dataset | Source | Obtained from |
|---|---|---|
| CIC-IDS2017 | Canadian Institute for Cybersecurity, UNB | [insert URL / access date] |
| UNSW-NB15 | Australian Centre for Cyber Security | [insert URL / access date] |
| TON_IoT | UNSW Canberra Cyber | [insert URL / access date] |
| CICIoT2023 | Canadian Institute for Cybersecurity | [insert URL / access date] |

## Unified representation

`src/data_prep/unify_features.py` reprocesses each dataset into eleven
common flow-level features: flow duration, total packet count, total byte
count, forward/backward packet counts, forward/backward byte counts, byte
rate, packet rate, transport protocol, destination port.

Expected input layout: `data/raw/<dataset_name>/` (native downloaded
format). Output: `data/unified/<dataset_name>.parquet`.

Attack labels are mapped onto a shared taxonomy via
`src/data_prep/taxonomy_mapping.csv`; categories absent from a given
dataset are left unrepresented rather than merged into a residual class.

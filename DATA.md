# Data Sources

Four public network intrusion detection datasets, none redistributed here.
All four were downloaded on 2026-08-11.

| Dataset | Source | Official page |
|---|---|---|
| CIC-IDS2017 | Canadian Institute for Cybersecurity, UNB | https://www.unb.ca/cic/datasets/ids-2017.html |
| UNSW-NB15 | Australian Centre for Cyber Security, UNSW Canberra | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| TON_IoT | UNSW Canberra Cyber | https://research.unsw.edu.au/projects/toniot-datasets |
| CICIoT2023 | Canadian Institute for Cybersecurity, UNB | https://www.unb.ca/cic/datasets/iotdataset-2023.html |

Citations for each dataset are given in the manuscript, Section 4.1.

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

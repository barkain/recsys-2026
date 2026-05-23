# R80 Phase 0A — top-300 listwise dataset audit

Elapsed: 57s
Cases (fold-0): 1600
h7 cases: 200

## Pool coverage

- GT in top-300: **967/1600 = 0.6044**
- GT in top-300 (h7): **123/200 = 0.6150**

## Baseline (OOF R54c top-20)

| Subset | n | nDCG@20 |
|---|---:|---:|
| all_fold0 | 1600 | 0.2110 |
| h7 | 200 | 0.2213 |
| same_artist | 531 | 0.4447 |
| diff_artist | 1069 | 0.0949 |
| h7_same | 83 | 0.4262 |
| h7_diff | 117 | 0.0760 |

## Phase 0B gates

- h7 nDCG Δ ≥ +0.005 vs baseline above
- same-artist Δ ≥ -0.002
- recovered > lost on h7
- top-1 churn /80 ≤ 25
- top-20 overlap ≥ 14/20

## Per-candidate feature schema

47 numeric + 1024 BGE-large track embedding + 1024 BGE-large query embedding (broadcast)
= 2095 dim per candidate. Project to 256 in model.

## Files

- Dataset: `/Users/nadavbarkai/dev/recsys-2026/cache/r80/listwise_dataset_fold0.pkl` (0.11 GB)
- Baseline: `/Users/nadavbarkai/dev/recsys-2026/cache/r80/eval_baseline.json`

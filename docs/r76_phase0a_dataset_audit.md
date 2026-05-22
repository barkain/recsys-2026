# R76 Phase 0A — Top-30 OOF dataset for neural residual ranker

Elapsed: 189s
Fold: 0

## Dataset shape

- Cases (fold-0): **1600**
- h7 cases: 200
- same-artist cases: 531
- Candidate rows (cases × top-30): **48000**
- Features per row: **47** (37 LR + 3 R68 + 5 semantic + 1 R54c score + 1 R54c rank_inv)

## GT-in-top-30 (the candidate ceiling)

- all_fold0: **679/1600 = 0.4244**
- h7: **87/200 = 0.4350**

If h7 GT-in-top-30 < 0.50, the ceiling for a residual reranker is hard.

## Baseline OOF R54c top-20 nDCG (reproduces well from R71)

| Subset | n | nDCG@20 |
|---|---:|---:|
| all_fold0 | 1600 | 0.2110 |
| h7 | 200 | 0.2213 |
| same_artist | 531 | 0.4447 |
| diff_artist | 1069 | 0.0949 |

## Files

- Dataset: `/Users/nadavbarkai/dev/recsys-2026/cache/r76/top30_fold0_dataset.pkl`
- OOF R54c model: `/Users/nadavbarkai/dev/recsys-2026/cache/r76/oof_r54c_fold0.txt`
- Stats JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR76_phase0a_dataset_stats.json`

## Next: Phase 0B

Train a small MLP residual model on this dataset.
Score = zscore(oof_r54c_score) + beta * neural_delta(features).
Listwise CE or pairwise softplus loss. Fold-0 CV-within.
Gate: h7 Δ ≥ +0.005 vs baseline, same-artist Δ ≥ -0.002, recovered > lost.

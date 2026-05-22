# R71 — Stacker on OOF R54c + R68 features (fold-0)

HEAD: `54b991b6b1`  Elapsed: 185s

## Verdict: **STACKER_FAIL**

## Design

- OOF R54c-style LR trained on folds 1-4 (6400 cases). Scores fold-0 candidates.
- Take OOF R54c top-30 per fold-0 case.
- For each, compute features: r54c_score, r54c_rank_inv, r68_rank_inv, r68_presence, r68_cosine.
- Stacker: LightGBM LambdaRank (20 rounds), 5-fold CV within fold-0.

## Metrics (5-fold inner CV)

| Subset | n | OOF R54c top-20 | Stacker top-20 | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2110 | 0.2075 | -0.0035 |
| h7 | 200 | 0.2213 | 0.2163 | -0.0050 |
| same_artist | 531 | 0.4447 | 0.4410 | -0.0037 |
| diff_artist | 1069 | 0.0949 | 0.0916 | -0.0034 |
| h7_same | 83 | 0.4262 | 0.4203 | -0.0060 |
| h7_diff | 117 | 0.0760 | 0.0716 | -0.0044 |

- h7 recovered=0, lost=3, net=-3
- top-1 churn /80 = 17.85
- top-20 overlap mean = 18.45/20

## Gates
- h7 Δ ≥ +0.005:        **False** (-0.0050)
- same-artist Δ ≥ -0.002: **False** (-0.0037)
- h7 net > 0:           **False** (-3)
- churn /80 ≤ 25:       **True** (17.85)

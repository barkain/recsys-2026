# R76 Phase 0B — MLP residual ranker (fold-0 inner CV, CPU)

Elapsed: 47s

## Verdict: **ARCHIVE**

## Hyperparams

hidden=64 layers=3 dropout=0.2 lr=0.001 epochs=30 wd=0.0001

## Metrics

| Subset | n | OOF R54c top-20 | R76 top-20 | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2110 | 0.1913 | -0.0197 |
| h7 | 200 | 0.2213 | 0.2079 | -0.0134 |
| same_artist | 531 | 0.4447 | 0.3992 | -0.0455 |
| diff_artist | 1069 | 0.0949 | 0.0880 | -0.0069 |
| h7_same | 83 | 0.4262 | 0.4128 | -0.0134 |
| h7_diff | 117 | 0.0760 | 0.0626 | -0.0134 |

- h7 recovered=0, lost=7, net=-7
- top-1 churn /80 = 46.50
- top-20 overlap mean = 16.42/20

## Gates
- h7 Δ ≥ +0.005: **False** (-0.0134)
- same-artist Δ ≥ -0.002: **False** (-0.0455)
- diff-artist Δ ≥ 0: **False** (-0.0069)
- h7 net > 0: **False** (-7)
- top-20 overlap ≥ 14: **True** (16.42)
- top-1 churn /80 ≤ 25: **False** (46.50)

# R70b 5-fold OOF — true OOF R54c-equivalent baseline

HEAD: `0ff1d51a90`  Elapsed: 304s

## Interpretation: **UNCLEAR**

5-fold OOF result doesn't cleanly match either prediction. Investigate fold variance.

## 5-fold aggregate

| Subset | n | R54c in-sample | R70b 5-fold OOF | Δ |
|---|---:|---:|---:|---:|
| all_dev | 8000 | 0.3159 | 0.2236 | -0.0922 |
| h7 | 1000 | 0.3484 | 0.2442 | -0.1042 |
| same_artist | 2857 | 0.6282 | 0.4498 | -0.1785 |
| diff_artist | 5143 | 0.1424 | 0.0980 | -0.0443 |

## Per-fold deltas

| Fold | n | Δ all | Δ h7 | Δ same-artist |
|---|---:|---:|---:|---:|
| 0 | 1600 | -0.0756 | -0.0818 | -0.1509 |
| 1 | 1600 | -0.0897 | -0.1127 | -0.1726 |
| 2 | 1600 | -0.0913 | -0.0997 | -0.1680 |
| 3 | 1600 | -0.1027 | -0.1125 | -0.1850 |
| 4 | 1600 | -0.1019 | -0.1142 | -0.2119 |

- h7 recovered=2, lost=79, net=-77
- top-1 churn /80 = 22.02

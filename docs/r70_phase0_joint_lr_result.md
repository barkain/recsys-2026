# R70 Phase 0 — joint R54+R68 LR (addition form), fold-0

Branch: `r70-joint-r54-r68-features`  HEAD: `1c399fe1b5`  Elapsed: 336s

## Verdict: **ARCHIVE_PHASE_0**

## Design

**Pool**: R54-stacked RRF top-300 (`SW_BASELINE`), bitwise identical to R54c production.

**Features (40)**: FEAT_R39_ALL (34) + FEAT_R54 (3) + FEAT_R68 (3)

**Sibling LR**: LightGBM LambdaRank, 300 rounds, same hyperparams as R54c, retrained on fold-0 train cases.

Distinct from prior failed paths:
- R68/R68.1 substitution: pool changed (R54→R68) AND r54_* features dropped.
- R60 matched-pool: pool changed (C3 admission).
- R70 addition: pool unchanged, r68_* features added.

## Metrics

| Subset | n | Baseline (R54c frozen LR) | R70 sibling LR | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2879 | 0.2092 | -0.0787 |
| h7 | 200 | 0.3043 | 0.2245 | -0.0798 |
| same_artist | 531 | 0.5984 | 0.4388 | -0.1595 |
| diff_artist | 1069 | 0.1337 | 0.0951 | -0.0386 |
| h7_same | 83 | 0.5698 | 0.4405 | -0.1292 |
| h7_diff | 117 | 0.1160 | 0.0713 | -0.0447 |

- h7 recovered=0, lost=17, net=-17
- top-1 churn /80=35.30  top-20 overlap mean=15.33/20
- dev cases missing R68 list: 0  train cases missing R68: 0

## Gates (predeclared)

- h7 Δ ≥ +0.005: **False** (-0.0798)
- same-artist Δ ≥ −0.002: **False** (-0.1595)
- all_fold0 Δ ≥ 0: **False** (-0.0787)
- h7 net > 0: **False** (-17)

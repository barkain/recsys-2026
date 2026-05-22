# R70b Phase 0 — Sibling LR Control (37 feats, NO r68)

HEAD: `997fb2a2ef`  Elapsed: 185s

## Interpretation: **ARTIFACT_LOCKED**

Retraining 37 features regresses ~similarly to R70's 40-feature regression. → R54c is artifact-locked. Sibling LR cannot reproduce R54c from the same feature schema. Sprint pivots to frozen-ranker-compatible interfaces (stacker, candidate injection, residual rerank).

## Metrics

| Subset | n | R54c frozen | R70b sibling (37f) | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2879 | 0.2110 | -0.0769 |
| h7 | 200 | 0.3043 | 0.2213 | -0.0830 |
| same_artist | 531 | 0.5984 | 0.4447 | -0.1537 |
| diff_artist | 1069 | 0.1337 | 0.0949 | -0.0387 |
| h7_same | 83 | 0.5698 | 0.4262 | -0.1436 |
| h7_diff | 117 | 0.1160 | 0.0760 | -0.0401 |

- h7 recovered=1, lost=15, net=-14
- top-1 churn /80=32.65  top-20 overlap mean=15.99/20

## R70b vs R70 (with r68) — the discriminator

| | R70 (40 feats, +r68) | R70b (37 feats, no r68) | diff |
|---|---:|---:|---:|
| h7 nDCG Δ | -0.0798 | -0.0830 | -0.0032 |
| same-artist Δ | -0.1595 | -0.1537 | +0.0058 |
| all_fold0 Δ | -0.0787 | -0.0769 | +0.0018 |

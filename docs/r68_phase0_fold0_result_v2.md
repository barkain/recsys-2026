# R68 Phase 0 fold-0 result

Created: 2026-05-21T18:09:02.256523
Branch: `r68-large-scale-retrieval`
HEAD: `1735349f13396049182b597c2a48639b21e9d4a9`
Fold: 0  n_fold0=1600  h7=200

## Verdict: **ARCHIVE_PHASE_0**

## Gate results

| Gate | Rule | Value | Pass |
|---|---|---|:---:|
| 1 | recovery | unique_h7=15, Δsingle_h7=+0.0350 | PASS |
| 2 | pool_hit | Δstacked_all=-0.0025 | FAIL |
| 3 | nDCG | Δh7=-0.0811, Δsame_artist=-0.1557 | FAIL |

## Single-source pool_hit @300 (fold-0)

| Subset | R54 | R68 | Δ |
|---|---:|---:|---:|
| h7 (n=200) | 0.5350 | 0.5700 | +0.0350 |
| all (n=1600) | 0.5756 | 0.5806 | +0.0050 |

Unique h7 recoveries (R68 only): **15**, lost h7 (R54 only): **8**, net **+7**.

## Stacked-RRF pool_hit @300 (fold-0)

| Subset | Baseline (R54-stacked) | R68-stacked | Δ |
|---|---:|---:|---:|
| all | 0.6044 | 0.6019 | -0.0025 |
| h7 | 0.6150 | 0.6300 | +0.0150 |

## nDCG@20 (fold-0)

| Subset | n | Baseline | R68 stacked + sibling LR | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2879 | 0.2084 | -0.0795 |
| h7 | 200 | 0.3043 | 0.2233 | -0.0811 |
| same_artist | 531 | 0.5984 | 0.4427 | -0.1557 |
| diff_artist | 1069 | 0.1337 | 0.0920 | -0.0417 |
| h7_same | 83 | 0.5698 | 0.4384 | -0.1313 |
| h7_diff | 117 | 0.1160 | 0.0706 | -0.0454 |

## Notes

- Pool admission unchanged; only 3 LR feature columns swapped (r54_* -> r68_*).
- Sibling LR trained on fold-0 TRAIN cases (R68 features zero-stubbed for TRAIN; Phase 1 produces clean OOF features for full 5-fold).
- This is feature substitution, NOT matched-pool retraining.
- Sibling LR: `cache/r68_phase0_sibling_lr.txt`
- Elapsed: 216.9s

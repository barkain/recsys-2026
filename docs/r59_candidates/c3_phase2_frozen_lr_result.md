# R59 C3 Phase 2 Frozen-LR Conversion

Created: 2026-05-16T19:47:35.287439

Verdict: **ARCHIVE**

## Baseline Reproduction

| Metric | R55 decomp | Recomputed OOF | Abs delta | Pass |
|---|---:|---:|---:|---:|
| all-dev nDCG@20 | 0.22460 | 0.22460 | 0.000000 | True |
| h7 nDCG@20 | 0.24511 | 0.24511 | 0.000000 | True |

## Headline

| Metric | RRF pool | Learned pool | Delta |
|---|---:|---:|---:|
| h7 nDCG@20 | 0.34838 | 0.34684 | -0.00154 |
| all-dev nDCG@20 | 0.31588 | 0.31551 | -0.00036 |
| all-dev nDCG@10 | 0.29958 | 0.29873 | -0.00085 |
| all-dev nDCG@7 | 0.29048 | 0.28983 | -0.00066 |
| same-artist nDCG@20 | 0.62821 | 0.62706 | -0.00116 |
| diff-artist nDCG@20 | 0.14237 | 0.14245 | +0.00008 |
| pool_hit@300 | 0.62200 | 0.68163 | +0.05962 |

## Conversion

| Metric | Value |
|---|---:|
| top-20 recovered | 58 |
| top-20 lost | 43 |
| top-20 net | 15 |
| POOL_MISS admitted to learned pool | 580 |
| POOL_MISS converted to frozen-LR top-20 | 48 |
| POOL_MISS admitted but buried 21-300 | 532 |
| convert rate | 48 / 580 (8.28%) |

## History Depth

| Bucket | n | RRF nDCG@20 | Learned nDCG@20 | Delta |
|---|---:|---:|---:|---:|
| h0 | 1000 | 0.22995 | 0.22984 | -0.00011 |
| h1 | 1000 | 0.36364 | 0.36410 | +0.00046 |
| h2 | 1000 | 0.32658 | 0.32358 | -0.00300 |
| h3 | 1000 | 0.30967 | 0.30798 | -0.00168 |
| h4 | 1000 | 0.31419 | 0.31569 | +0.00150 |
| h5 | 1000 | 0.31642 | 0.31724 | +0.00082 |
| h6 | 1000 | 0.31818 | 0.31883 | +0.00065 |
| h7 | 1000 | 0.34838 | 0.34684 | -0.00154 |

## Churn

| Split | top-1 changed | top-1 equiv /80 | top-20 overlap mean | top-20 overlap median |
|---|---:|---:|---:|---:|
| all-dev | 298/8000 | 3.0/80 | 18.27/20 | 19.00/20 |
| h7 | 70/1000 | 5.6/80 | 17.05/20 | 18.00/20 |

## Gate Checks

| Gate | Pass |
|---|---:|
| h7_delta_ge_010 | False |
| h7_delta_ge_005 | False |
| same_artist_nonnegative | False |
| diff_artist_nonnegative | True |
| top1_churn_ok | True |
| top20_overlap_ok | True |
| churn_controlled | True |
| recovered_gt_exceeds_lost | True |

## Notes

- LR model is loaded from `cache/r54_phase3_lr_model.txt` and is never retrained.
- Learned admission is rebuilt OOF using the same fixed Phase 1 LambdaRank admission setup.
- `admission_score` is not used as an LR feature.
- Learned-pool `rrf_rank_inv` is pinned to full weighted-RRF source-union rank, not admission rank.
- Dev R54 source/cosine features use the preserved Phase 2 OOF proxy.

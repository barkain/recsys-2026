# R59 C3 Pool-Admission Diagnostic Result

Created: 2026-05-16T19:08:29.015094

## Verdict

**PROCEED**

Recommendation: proceed only to a later dev-only frozen-LR conversion check.

## Baseline Reproduction

| Metric | Value |
|---|---:|
| R55 weighted_rrf pool_hit@300 | 0.6220 |
| This script weighted_rrf pool_hit@300 | 0.6220 |
| Abs delta | 0.000000 |
| Pass epsilon 0.0005 | True |

## Headline Metrics

| Metric | Value |
|---|---:|
| weighted_rrf pool_hit@300 | 0.6220 |
| learned admission pool_hit@300 | 0.6816 |
| delta pool_hit@300 | +0.0596 |
| POOL_MISS recovered | 580 / 1163 |
| Previously-covered GT lost | 103 / 4976 |
| Net pool recovery | 477 |

## Breakdowns

| Split | n | RRF hit | Learned hit | Delta | Recovered POOL_MISS | Lost covered | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_dev | 8000 | 0.6220 | 0.6816 | +0.0596 | 580 | 103 | 477 |
| h7 | 1000 | 0.6130 | 0.6830 | +0.0700 | 80 | 10 | 70 |
| same_artist | 2857 | 0.9720 | 0.9765 | +0.0046 | 26 | 13 | 13 |
| diff_artist | 5143 | 0.4276 | 0.5178 | +0.0902 | 554 | 90 | 464 |

## Bucket Of Origin

| Bucket | n | RRF hit | Learned hit | Delta | Recovered POOL_MISS | Lost covered |
|---|---:|---:|---:|---:|---:|---:|
| HIT | 3348 | 1.0000 | 0.9982 | -0.0018 | 0 | 6 |
| DEMOTED | 1628 | 1.0000 | 0.9404 | -0.0596 | 0 | 97 |
| POOL_MISS | 1163 | 0.0000 | 0.4987 | +0.4987 | 580 | 0 |
| UNREACHABLE | 1861 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |

## Top-300 Overlap With Weighted RRF

| Statistic | Overlap fraction |
|---|---:|
| mean | 0.5978 |
| median | 0.6067 |
| p10 | 0.4700 |
| p25 | 0.5400 |
| p75 | 0.6633 |
| p90 | 0.7167 |

## Source Coverage Of Recovered POOL_MISS Cases

| Source | Recovered GT count |
|---|---:|
| A | 9 |
| ALS | 136 |
| B | 124 |
| C | 155 |
| D | 9 |
| F | 18 |
| R21 | 256 |
| R54 | 287 |

Top source patterns:

| Pattern | Count |
|---|---:|
| R21+R54 | 121 |
| ALS | 75 |
| R54 | 65 |
| R21 | 45 |
| C | 38 |
| B+C | 24 |
| B+R21+R54 | 21 |
| B | 21 |
| C+R21+R54 | 20 |
| C+R21 | 15 |
| C+R54 | 15 |
| C+ALS | 15 |
| B+R54 | 14 |
| F+ALS | 11 |
| ALS+R54 | 9 |
| B+ALS | 9 |
| B+R21 | 8 |
| B+C+R21 | 7 |
| B+C+R54 | 6 |
| B+C+R21+R54 | 6 |

## Gate Checks

| Gate | Pass |
|---|---:|
| learned_pool_hit_beats_weighted_rrf | True |
| net_pool_recovery_positive | True |
| recovered_pool_miss_at_least_100 | True |
| loss_lte_25pct_gained | True |
| gains_not_concentrated_single_source | True |

## LightGBM Feature Importances

| Rank | Feature | Mean gain |
|---:|---|---:|
| 1 | weighted_rrf_score | 183912.41 |
| 2 | als_score | 46152.16 |
| 3 | n_sources_present | 32439.43 |
| 4 | r54_cosine | 25572.81 |
| 5 | weighted_rrf_rank_inv | 17800.13 |
| 6 | n_unique_artists | 16415.71 |
| 7 | rank_dispersion | 11126.85 |
| 8 | r54_rank_inv | 8082.98 |
| 9 | best_dense_rank | 7549.07 |
| 10 | min_rank | 6744.10 |

## Recommendation For Next Phase

Run a separate dev-only frozen-LR conversion phase. Keep LR frozen and do not add admission_score as an LR feature.

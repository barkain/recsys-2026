| Kill-shot metric | Fold-0 value | Target / stop | Status |
|---|---:|---:|---:|
| Baseline reproduction | True | epsilon <= 0.0005 on all 4 metrics | PASS |
| h7 nDCG@20 delta | -0.08854 | archive if < 0; proceed if >= +0.003 | FAIL |
| POOL_MISS conversion | 11 / 132 (8.33%) | archive if <= 10%; proceed if >= 15% | FAIL |
| Top-20 recovered / lost / net | 18 / 111 / -93 | archive if net < 0; proceed if net > 0 | FAIL |
| Same-artist canary delta | -0.14156 | archive if < -0.002 | FAIL |
| Top-1 churn blind-eq | 32.20/80 | archive if any tracked split > 35/80 | PASS |
| Top-20 overlap mean | 15.04/20 | archive if any tracked split < 14/20 | PASS |
| Verdict | **ARCHIVE** | see §8/§9 gates | ARCHIVE |

# R60 Variant A Fold-0 Result

Created: 2026-05-16T23:33:05.544698

Next step recommendation: **archive R60 Variant A matched-pool fold-0 path**

## Baseline Reproduction

| Split | n | Target | R59 artifact | Recomputed | Abs delta vs artifact | Pass |
|---|---:|---:|---:|---:|---:|---:|
| all | 8000 | 0.31588 | 0.31588 | 0.31588 | 0.000000 | True |
| h7 | 1000 | 0.34838 | 0.34838 | 0.34838 | 0.000000 | True |
| same_artist | 2857 | 0.62821 | 0.62821 | 0.62821 | 0.000000 | True |
| diff_artist | 5143 | 0.14237 | 0.14237 | 0.14237 | 0.000000 | True |
| pool_hit@300 | 8000 | 0.62200 | 0.62200 | 0.62200 | 0.000000 | True |

## Fold-0 Metrics

| Split | n | RRF frozen-LR baseline | Variant A | Delta |
|---|---:|---:|---:|---:|
| all | 1600 | 0.28788 | 0.21474 | -0.07315 |
| h7 | 200 | 0.30434 | 0.21581 | -0.08854 |
| same_artist | 531 | 0.59837 | 0.45681 | -0.14156 |
| diff_artist | 1069 | 0.13366 | 0.09449 | -0.03916 |

## Conversion

| Metric | Value |
|---|---:|
| Fold-0 cases | 1600 |
| Fold-0 h7 cases | 200 |
| Fold-0 same-artist cases | 531 |
| Fold-0 diff-artist cases | 1069 |
| Global learned-pool POOL_MISS admitted | 580 |
| Fold-0 POOL_MISS admitted | 132 |
| Fold-0 POOL_MISS converted top-20 | 11 |
| Fold-0 POOL_MISS buried 21-300 | 121 |
| Fold-0 conversion rate | 8.33% |
| Top-20 recovered | 18 |
| Top-20 lost | 111 |
| Top-20 net | -93 |

## Churn

| Split | n | Top-1 changed | Top-1 blind-eq | Top-20 overlap mean | Top-20 overlap median |
|---|---:|---:|---:|---:|---:|
| all | 1600 | 644 | 32.20/80 | 15.04/20 | 15.00/20 |
| h7 | 200 | 83 | 33.20/80 | 14.22/20 | 14.00/20 |
| same_artist | 531 | 228 | 34.35/80 | 15.76/20 | 16.00/20 |
| diff_artist | 1069 | 416 | 31.13/80 | 14.68/20 | 15.00/20 |

## Feature Contract

- Variant A LR features: 37
- C3 admission features: 99
- C3 admission artifact features: 99
- `admission_score` and `admission_rank_inv` are not used.
- Learned-pool `rrf_rank_inv` is pinned to full weighted-RRF source-union rank.

## Top Feature Importances

| Rank | Feature | Gain |
|---:|---|---:|
| 1 | rrf_rank_inv | 117584.372 |
| 2 | popularity | 47244.025 |
| 3 | recency_score | 20187.148 |
| 4 | r54_cosine | 16386.002 |
| 5 | src_c_rank_inv | 14402.485 |
| 6 | als_dot | 13611.951 |
| 7 | r54_rank_inv | 8501.004 |
| 8 | r21_rank_inv | 7924.609 |
| 9 | same_album_any | 7827.435 |
| 10 | src_als_rank_inv | 7487.256 |
| 11 | src_b_rank_inv | 6878.955 |
| 12 | pool_artist_frac | 6608.251 |
| 13 | last_tag_jaccard | 6510.741 |
| 14 | pool_same_album_count | 6204.439 |
| 15 | album_history_count | 5934.808 |

## Reproducibility

- `dev_only`: True
- `blind_access`: False
- Fold split: `grouped_session_folds(seed=0)`, fold 0 n=1600
- LR training cases: folds 1-4 only; no production initialization.
- Admission artifact loaded: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR59_c3_pool_admission.json`
- Admission scores source: rebuilt with c3.train_cv5; expR59_c3_pool_admission.json has aggregate metrics only

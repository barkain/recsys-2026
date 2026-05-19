# R68 Baseline Reproduction (Wave 0)

Created: 2026-05-19T23:26:34.522629
Branch: `r68-large-scale-retrieval`
HEAD: `451bfd1ecd212501b2155f0e42e18562ca315e2c`
Reference: `exp/eval/expR59_c3_phase2_frozen_lr.json`
Epsilon gate: 0.0005

## Verdict: **PASS**
max |delta| = 0.000000

## Metrics

| Metric | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| all_dev_ndcg20 | 0.315875 | 0.315875 | +0.000000 |
| h7_ndcg20 | 0.348378 | 0.348378 | -0.000000 |
| same_artist_ndcg20 | 0.628214 | 0.628214 | +0.000000 |
| diff_artist_ndcg20 | 0.142367 | 0.142367 | +0.000000 |
| pool_hit_all | 0.622000 | 0.622000 | +0.000000 |
| pool_hit_h7 | 0.613000 | 0.613000 | +0.000000 |

## R54 Reference Stats

- R54 single-source pool_hit @300 (all): 0.5909 (4727/8000)
- R54 single-source pool_hit @300 (h7): 0.5430 (543/1000)

### Feature distributions (per-case, GT row of frozen LR input)

| Feature | Mean | Std |
|---|---:|---:|
| r54_rank_inv | 0.093831 | 0.219771 |
| r54_presence | 0.590875 | 0.491672 |
| r54_cosine | 0.389314 | 0.332085 |

## Fold Assignment

Source: `cache/r54/phase2_full/oof_manifest.json` (function=scripts.expS2_lambdarank_grouped.grouped_session_folds, seed=0, k=5)

| Fold | Count |
|---|---:|
| 0 | 1600 |
| 1 | 1600 |
| 2 | 1600 |
| 3 | 1600 |
| 4 | 1600 |

## Artifacts

- `exp/eval/expR68_baseline_repro.json`
- `exp/eval/expR68_r54_reference_stats.pkl`
- `exp/eval/expR68_r54_aggregate.json`

## Notes

- Elapsed: 164.8s
- Wave 0 reuses cached intermediates and frozen production LR.
- Fold split derived via `grouped_session_folds(sessions, seed=0, k=5)`; matches manifest `val_indices_sample` for all 5 folds.

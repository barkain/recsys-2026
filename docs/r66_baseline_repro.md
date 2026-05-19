# R66 Baseline Reproduction (Wave 0)

Created: 2026-05-19T19:57:14.792878
Branch: `r66-learned-depth-source-router`
HEAD: `6bba86d64cf99744f3c179f167aae0b11d9639b6`
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

## weighted_rrf Signature

Supports per-source weights: True

`weighted_rrf(sources: dict[str, list[str]], weights: dict[str, float], topk: int, k: int = 20) -> list[str] in scripts/expF1_cfbpr_retrieval.py:158`

## Notes

- Elapsed: 164.1s
- Wave 0 reuses cached intermediates and frozen production LR.
- Admission LR training and learned-pool scoring skipped (RRF baseline only).

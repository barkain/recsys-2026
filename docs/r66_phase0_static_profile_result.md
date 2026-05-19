# R66 Phase 0 — Static Profile Conversion Result

Created: 2026-05-19T20:21:56.442257
HEAD before: `cf1684c`
HEAD after: `cf1684cdd955fd50bef1abb9003f350c6347c58d`
Elapsed: 1186.1s

## Verdict: **ARCHIVE_PHASE_0**

Passers: `none`

## P0 Bitwise Sanity Check (vs reference)

| Metric | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| all_dev_ndcg20 | 0.315875 | 0.315875 | -0.000000 |
| h7_ndcg20 | 0.348378 | 0.348378 | +0.000000 |
| same_artist_ndcg20 | 0.628214 | 0.628214 | +0.000000 |
| diff_artist_ndcg20 | 0.142367 | 0.142367 | -0.000000 |
| pool_hit_all | 0.622000 | 0.622000 | +0.000000 |
| pool_hit_h7 | 0.613000 | 0.613000 | +0.000000 |

P0 sanity verdict: **PASS**  (h7 |delta|=0.000000, gate eps=0.0005)

## Phase 0 Kill Gate (per profile, vs P0)

All 4 conditions must hold (non-P0):
1. `pool_hit_h7` lift >= +0.010
2. `h7_ndcg20` delta >= 0
3. `same_artist_ndcg20` delta >= -0.002
4. `recovered_h7 > lost_h7`

## Profile Weights

| Profile | Label | A | B | C | D | F | ALS | R21 | R54 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | R54c baseline | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| P1 | text-heavy | 1.5 | 1.5 | 1.5 | 0.5 | 1.5 | 0.5 | 0.7 | 0.7 |
| P2 | collaborative-heavy | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 1.5 | 1.5 | 1.5 |
| P3 | R54-heavy | 0.7 | 0.7 | 0.7 | 0.3 | 0.7 | 0.7 | 0.7 | 2.0 |
| P4 | R21/R54 pair | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 0.5 | 1.5 | 1.5 |
| P5 | BM25-only | 1.5 | 1.5 | 1.5 | 0.5 | 1.5 | 0.3 | 0.3 | 0.3 |
| P6 | C+R54 dominant | 0.5 | 0.5 | 2.0 | 0.3 | 0.5 | 0.5 | 0.7 | 2.0 |
| P7 | ALS+R54 dominant | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 2.0 | 0.7 | 2.0 |

## Profile Metrics

| Profile | pool_hit_h7 | Δpool_hit_h7 | h7_ndcg20 | Δh7_ndcg | Δsame_artist | rec | lost | net | churn/80 | overlap@20 | passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| P0 | 0.6130 | +0.0000 | 0.348378 | +0.000000 | +0.000000 | 0 | 0 | +0 | 0 | 20.00 | — |
| P1 | 0.5960 | -0.0170 | 0.342472 | -0.005905 | -0.004226 | 1 | 14 | -13 | 13 | 17.32 | ✗ |
| P2 | 0.6200 | +0.0070 | 0.326991 | -0.021387 | -0.017745 | 4 | 21 | -17 | 15 | 17.41 | ✗ |
| P3 | 0.6190 | +0.0060 | 0.335061 | -0.013317 | -0.011446 | 4 | 17 | -13 | 12 | 17.67 | ✗ |
| P4 | 0.6190 | +0.0060 | 0.330228 | -0.018150 | -0.019361 | 10 | 22 | -12 | 15 | 17.04 | ✗ |
| P5 | 0.5830 | -0.0300 | 0.338396 | -0.009982 | -0.010273 | 6 | 28 | -22 | 14 | 15.78 | ✗ |
| P6 | 0.6270 | +0.0140 | 0.333694 | -0.014684 | -0.014717 | 10 | 24 | -14 | 15 | 17.44 | ✗ |
| P7 | 0.6190 | +0.0060 | 0.326830 | -0.021548 | -0.017300 | 6 | 24 | -18 | 18 | 17.30 | ✗ |

## Per-Profile Gate Breakdown

| Profile | pool_hit_lift≥+0.010 | h7_ndcg_Δ≥0 | same_artist_Δ≥-0.002 | rec>lost | passes |
|---|:---:|:---:|:---:|:---:|:---:|
| P0 | — | — | — | — | — |
| P1 | ✗ | ✗ | ✗ | ✗ | ✗ |
| P2 | ✗ | ✗ | ✗ | ✗ | ✗ |
| P3 | ✗ | ✗ | ✗ | ✗ | ✗ |
| P4 | ✗ | ✗ | ✗ | ✗ | ✗ |
| P5 | ✗ | ✗ | ✗ | ✗ | ✗ |
| P6 | ✓ | ✗ | ✗ | ✗ | ✗ |
| P7 | ✗ | ✗ | ✗ | ✗ | ✗ |

## Conclusion

No profile cleared all 4 Phase 0 conditions; sprint archives at Phase 0. Static RRF re-weighting (within the menu) does not unlock learned routing.

## Notes

- Frozen LR: `cache/r54_phase3_lr_model.txt` (read-only)
- Churn sample: 80 cases, seed=0
- Pool depth: 300; top-K: 20
- Per-profile elapsed:
  - P0: 146.5s
  - P1: 146.7s
  - P2: 144.7s
  - P3: 144.5s
  - P4: 143.1s
  - P5: 147.4s
  - P6: 146.5s
  - P7: 144.6s

# R79 Phase 0A — data + baseline audit

Elapsed: 302s
Total cases: 8000  Folds: 5

## Training pair stats

- Cases: 8000
- GT in OOF R54c top-20: **3297/8000 = 0.4121**
- Avg hard negs per case: ~19.6

## Phase 0B baseline (OOF R54c standalone top-20)

This is what R79's standalone retriever must beat to clear gates.

| Subset | n | nDCG@20 | in_top20 |
|---|---:|---:|---:|
| all | 8000 | 0.2236 | 3297 (0.412) |
| h7 | 1000 | 0.2442 | 423 (0.423) |
| same_artist | 2857 | 0.4498 | 2243 (0.785) |
| diff_artist | 5143 | 0.0980 | 1054 (0.205) |
| h7_same | 467 | 0.4537 | 344 (0.737) |
| h7_diff | 533 | 0.0606 | 79 (0.148) |
| fold_0 | 1600 | 0.2123 | 619 (0.387) |
| fold_0_h7 | 200 | 0.2226 | 83 (0.415) |

## Phase 0B gates

Use fold-0 subset specifically (n=1600, h7=200).

- h7 nDCG Δ ≥ +0.005 vs baseline (h7 = 0.2213 above)
- same-artist Δ ≥ -0.002 (subset)
- recovered (R79 in top-20, R54c not) > lost (R54c in top-20, R79 not) on h7
- top-1 churn /80 ≤ 25 on fold-0

## Files

- Training pairs: `/Users/nadavbarkai/dev/recsys-2026/cache/r79/training_pairs.pkl`
- Baseline: `/Users/nadavbarkai/dev/recsys-2026/cache/r79/eval_baseline.json`
- Stats: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR79_phase0a_stats.json`

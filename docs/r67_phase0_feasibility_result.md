# R67 Phase 0 Feasibility — ARCHIVE_PHASE_0

- HEAD: `987eb88a1f843795766a6402192db4d29757e5d6`
- model: `claude-opus-4-7`
- sample size: 150
- strata: {'S1': 30, 'S2': 30, 'S3': 30, 'S4': 30, 'S5': 30}
- winner: **NONE**

## Telemetry
- n_calls (live): 300
- cache_hits: 0, cache_misses: 300
- retries: 6
- malformed: 1
- total input tokens: 1369839
- total output tokens: 39990
- elapsed_s: 352.2

## Metrics

| metric | Style A | Style B |
|---|---|---|
| n_valid | 149 | 150 |
| ndcg20_sample | 0.1513 | 0.1425 |
| ndcg20_sample_lr | 0.2265 | 0.2250 |
| delta_sample_vs_lr | -0.0752 | -0.0825 |
| ndcg20_h7 | 0.1957 | 0.1755 |
| ndcg20_h7_lr | 0.4128 | 0.4128 |
| delta_h7_vs_lr | -0.2171 | -0.2373 |
| ndcg20_same_artist | 0.3261 | 0.2925 |
| delta_same_artist | -0.1793 | -0.2129 |
| ndcg20_diff_artist | 0.0707 | 0.0740 |
| delta_diff_artist | -0.0273 | -0.0230 |
| recovered | 2 | 3 |
| lost | 9 | 13 |
| net | -7 | -10 |
| top1_changed_count | 130 | 134 |
| churn_per_80 | 69.7987 | 71.4667 |
| validity_rate | 0.9933 | 1.0000 |
| gate_h7_delta_ge_0005 | FAIL | FAIL |
| gate_sample_delta_ge_0 | FAIL | FAIL |
| gate_recovered_gt_lost | FAIL | FAIL |
| gate_churn_le_25_80 | FAIL | FAIL |
| gate_validity_ge_095 | PASS | PASS |
| passes_phase0_gate | FAIL | FAIL |

## Verdict

ARCHIVE Phase 0 — neither prompt style cleared the gate.
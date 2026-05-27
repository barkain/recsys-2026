# R90 Phase 1 LR Conversion Test

Date: 2026-05-27T17:25:28.728024  
Head: aaac7b8050  
Elapsed: 490s  

## Verdict: **PROCEED_TO_BLIND_B**

This is the LR scoring + selective routing test that R84c won
Blind-A with. Source-alone retrieval gains (see
`docs/r90_phase1_5fold_compare.md`) do NOT automatically convert
through the LR layer per `feedback_lr_conversion_wall_confirmed`.

### Per-condition aggregate (8000 OOF cases)

| condition | h7 nDCG@20 | same-art h7 | diff-art h7 | h7 hit@20 | all nDCG@20 |
|---|---:|---:|---:|---:|---:|
| R84_oof_alone | 0.2421 | 0.4420 | 0.0670 | 0.428 | 0.2261 |
| R90_oof_alone | 0.2447 | 0.4470 | 0.0675 | 0.431 | 0.2283 |
| R84_routed_0.5_2.0 | 0.2681 | 0.4890 | 0.0745 | 0.451 | 0.2462 |
| R90_routed_0.5_2.0 | 0.2718 | 0.4924 | 0.0784 | 0.455 | 0.2489 |
| R84_routed_0.25_2.0 | 0.2920 | 0.5254 | 0.0875 | 0.458 | 0.2699 |
| R90_routed_0.25_2.0 | 0.2952 | 0.5275 | 0.0917 | 0.463 | 0.2716 |

### Blind gate: R90 routed vs R84 routed (apples-to-apples)

Blind-readiness gate: R90-routed h7 nDCG@20 > R84-routed h7 nDCG@20
AND same-artist Δ ≥ -0.005 (canary safe).

| thresholds | h7 Δ | same-art Δ | diff-art Δ | gate |
|---|---:|---:|---:|---|
| 0.5/2.0 | +0.0037 | +0.0034 | +0.0039 | PASS |
| 0.25/2.0 | +0.0032 | +0.0021 | +0.0041 | PASS |

### Recovered / lost top-20 (R90 vs R84 paired comparisons)

| comparison | h7 rec | h7 lost | net | top-1 churn | top-20 overlap |
|---|---:|---:|---:|---:|---:|
| R90_oof_alone | 28 | 25 | +3 | 44.1% | 15.49 |
| R90_routed_0.5_2.0 | 22 | 18 | +4 | 36.8% | 16.81 |
| R90_routed_0.25_2.0 | 18 | 13 | +5 | 26.8% | 17.78 |

### n_prior buckets (h7 nDCG@20 by condition)

| n_prior | R84_oof_alone | R90_oof_alone | R84_routed_0.5_2.0 | R90_routed_0.5_2.0 | R84_routed_0.25_2.0 | R90_routed_0.25_2.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2070 | 0.2171 | 0.2113 | 0.2208 | 0.2192 | 0.2261 |
| 1 | 0.2585 | 0.2588 | 0.2904 | 0.2897 | 0.3169 | 0.3149 |
| 2 | 0.2248 | 0.2301 | 0.2473 | 0.2520 | 0.2837 | 0.2854 |
| 3 | 0.2009 | 0.2084 | 0.2175 | 0.2254 | 0.2495 | 0.2511 |
| 4 | 0.2209 | 0.2236 | 0.2443 | 0.2488 | 0.2607 | 0.2676 |
| 5 | 0.2285 | 0.2275 | 0.2446 | 0.2430 | 0.2678 | 0.2691 |
| 6 | 0.2258 | 0.2163 | 0.2465 | 0.2399 | 0.2692 | 0.2637 |
| 7 | 0.2421 | 0.2447 | 0.2681 | 0.2718 | 0.2920 | 0.2952 |

Files: `exp/eval/expR90_phase1_lr_conversion.json` (this JSON), `docs/r90_phase1_lr_conversion.md` (this report).
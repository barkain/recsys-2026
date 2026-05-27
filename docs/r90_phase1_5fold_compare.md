# R90 Phase 1 Variant A — 5-fold OOF compare

Date: 2026-05-27T17:13:14.800704  
Baseline: **R84_5fold** (R84 5-fold OOF)  
Candidate: **R90_varA_5fold** (R90 Variant A 5-fold OOF)  
n cases: 8000  h7: 1000

## Verdict: **PROCEED_TO_LR_CONVERSION_TEST**

### Gate summary (aggregate across all 5 folds)
| gate | result | detail |
|---|---|---|
| G1 same-artist canary | PASS | h7-same Δ = +0.0005 (threshold ≥ -0.005) |
| G2 h7 aggregate | PASS | h7 Δ = +0.0011, rec/lost = 47/35 (net +12) |
| G3 history buckets (n_prior 2-6) | PASS | see breakdown below |
| G4 per-fold sanity (each fold's h7 Δ ≥ -0.005) | PASS | see per-fold table below |

### h7 nDCG@20 detail (aggregate)
| segment | n | R84_5fold | R90_varA_5fold | Δ |
|---|---:|---:|---:|---:|
| h7 (all) | 1000 | 0.1100 | 0.1111 | +0.0011 |
| h7 same-artist | 467 | 0.2092 | 0.2097 | +0.0005 |
| h7 diff-artist | 533 | 0.0231 | 0.0248 | +0.0017 |

### History buckets (n_prior 0-7; gated: 2-6)
| n_prior | n | R84_5fold | R90_varA_5fold | Δ | gate | result |
|---:|---:|---:|---:|---:|---|---|
| 0 | 1000 | 0.1460 | 0.1653 | +0.0194 | no | — |
| 1 | 1000 | 0.1732 | 0.1850 | +0.0118 | no | — |
| 2 | 1000 | 0.1472 | 0.1552 | +0.0080 | yes | PASS |
| 3 | 1000 | 0.1219 | 0.1256 | +0.0036 | yes | PASS |
| 4 | 1000 | 0.1219 | 0.1273 | +0.0054 | yes | PASS |
| 5 | 1000 | 0.1046 | 0.1163 | +0.0117 | yes | PASS |
| 6 | 1000 | 0.1004 | 0.1110 | +0.0106 | yes | PASS |
| 7 | 1000 | 0.1100 | 0.1111 | +0.0011 | no | — |

### Per-fold breakdown
| fold | n_h7 | h7 R84_5fold | h7 R90_varA_5fold | h7 Δ | same Δ | diff Δ | gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 200 | 0.0963 | 0.1038 | +0.0074 | +0.0230 | -0.0036 | PASS |
| 1 | 200 | 0.0976 | 0.0929 | -0.0046 | -0.0048 | -0.0045 | PASS |
| 2 | 200 | 0.1143 | 0.1168 | +0.0025 | -0.0018 | +0.0068 | PASS |
| 3 | 200 | 0.1023 | 0.1039 | +0.0016 | -0.0080 | +0.0102 | PASS |
| 4 | 200 | 0.1395 | 0.1382 | -0.0013 | -0.0030 | +0.0007 | PASS |

Files: `exp/eval/expR90_phase1_5fold_compare.json` (this JSON), `docs/r90_phase1_5fold_compare.md` (this report).
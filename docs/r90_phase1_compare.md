# R90 Phase 1 Variant A compare

Date: 2026-05-27T06:54:31.586406  
Baseline: **R84_fold0** (cache/r84/phase0b_fold0/)  
Candidate: **R90_varA_fold0** (cache/r90/phase1_fold0_varA/)

## Verdict: **PROCEED_TO_5FOLD**

### Gate summary
| gate | result | detail |
|---|---|---|
| G1 same-artist canary | PASS | h7-same Δ = +0.0230 (threshold ≥ -0.005) |
| G2 h7 aggregate | PASS | h7 Δ = +0.0074, rec/lost = 9/9 (net +0) |
| G3 history buckets (n_prior 2-6) | PASS | see breakdown below |
| G4 fold-0 sanity | PASS | fold-0 h7 Δ = +0.0074 (threshold ≥ -0.005) |

### h7 nDCG@20 detail
| segment | n | R84_fold0 | R90_varA_fold0 | Δ |
|---|---:|---:|---:|---:|
| h7 (all) | 200 | 0.0963 | 0.1038 | +0.0074 |
| h7 same-artist | 83 | 0.2027 | 0.2256 | +0.0230 |
| h7 diff-artist | 117 | 0.0209 | 0.0173 | -0.0036 |

### History buckets (n_prior 2-6)
| n_prior | n | R84_fold0 | R90_varA_fold0 | Δ | gate |
|---:|---:|---:|---:|---:|---|
| 2 | 200 | 0.1599 | 0.1704 | +0.0105 | PASS |
| 3 | 200 | 0.1174 | 0.1264 | +0.0090 | PASS |
| 4 | 200 | 0.1252 | 0.1339 | +0.0087 | PASS |
| 5 | 200 | 0.1043 | 0.1111 | +0.0068 | PASS |
| 6 | 200 | 0.0964 | 0.1125 | +0.0162 | PASS |

Files: `exp/eval/expR90_phase1_compare.json` (this JSON), `docs/r90_phase1_compare.md` (this report).
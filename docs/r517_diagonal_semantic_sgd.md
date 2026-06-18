# R517 - Diagonal Semantic SGD

**Date:** 2026-06-18
**Verdict:** **NO_GO**

## Best Offline Result

- policy: `blend_bw8_keep0`
- nDCG@20: `0.315221` vs base `0.315875`
- dNDCG: `-0.000655`
- same/diff/h7 delta: `-0.002152` / `0.000177` / `-0.002170`
- churn top1 per 80: `0.02`
- overlap@20: `18.427`

## Interpretation

This tests whether a learned diagonal semantic metric over q*track embeddings can convert beyond R516's scalar cosine features. It remains all-dev OOF and deployment-faithful; no Blind-A submission is justified unless it clears the +0.010 dNDCG gate.

Full JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR517_diagonal_semantic_sgd.json`

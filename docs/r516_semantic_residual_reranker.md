# R516 - Semantic Residual Reranker

**Date:** 2026-06-18
**Verdict:** **NO_GO**

## Best Offline Result

- depth: `80`
- policy: `blend_bw0.8_keep0`
- nDCG@20: `0.320671` vs base `0.315875`
- dNDCG: `0.004795`
- same/diff/h7 delta: `0.003659` / `0.005427` / `0.005656`
- churn top1 per 80: `3.78`
- overlap@20: `17.256`

## Interpretation

This is an all-dev OOF test over production top20 plus natural R480 insertion candidates, with R21 query-track and history-track semantic features. It is deployment-faithful: no GT injection and no miss-only selection. A blind build requires roughly +0.010 dNDCG and sane churn/overlap; otherwise this path is not strong enough for the 0.55 target.

Full JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR516_semantic_residual_reranker.json`

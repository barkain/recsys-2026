# R518 - Wide Semantic Residual

**Date:** 2026-06-18
**Verdict:** **NO_GO**

## Best Offline Result

- config: `r48080_r54r84_100`
- policy: `blend_bw0.02_keep0`
- nDCG@20: `0.325566` vs base `0.315875`
- dNDCG: `0.009691`
- same/diff/h7 delta: `0.003001` / `0.013407` / `0.005129`
- churn top1 per 80: `7.97`
- overlap@20: `14.501`

## Interpretation

This tests whether the positive R516 semantic residual can exploit the extra reachability from R54/R84 retrieval candidates. A blind build is justified only if this clears about +0.010 all-dev dNDCG with sane churn.

Full JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR518_wide_semantic_residual.json`

# R519 - Selective Wide Semantic

**Date:** 2026-06-18
**Verdict:** **NO_GO**

## Best Offline Result

- policy: `blend_bw0.02_keep0`
- selected: `8000` / `8000` (`1.000`)
- nDCG@20: `0.325566` vs base `0.315875`
- dNDCG: `0.009691`
- churn top1 per 80: `7.97`
- overlap@20: `14.501`

## Interpretation

This searches simple GT-independent selectors on R518 rankings using fold-held-out rule choice. It is designed to recover R518's dNDCG while restoring overlap/churn safety.

Full JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR519_selective_wide_semantic.json`

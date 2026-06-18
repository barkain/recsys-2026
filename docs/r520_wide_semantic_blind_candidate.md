# R520 - Wide Semantic Blind Candidate

**Date:** 2026-06-18
**Intent:** nDCG-first deployment of the R518 wide semantic residual reranker on top of R510.

## Candidate

- zip: `/Users/nadavbarkai/dev/recsys-2026/exp/inference/blind_a/r520_wide_semantic_r510/r520_r510_wide_semantic_blend002_submission.zip`
- sha256: `8abf05b3c566c7938d819fe9fa807356e1499f550845aa599718aef7bec70b65`
- policy: `blend_bw0.02_keep0`, R480 depth `80`, R54/R84 depth `100`

## Preflight

- rows: `80`
- changed rows: `80`
- top-1 churn: `12/80`
- mean overlap@20 vs R510: `12.863`
- response changes: `0`

This candidate is not composite-safe by design; it keeps R510 responses while changing rankings.
Full audit JSON: `/Users/nadavbarkai/dev/recsys-2026/exp/eval/expR520_wide_semantic_blind_candidate.json`

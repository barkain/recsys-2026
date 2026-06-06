# R431 User-CF Fast Policy Eval

Baseline internal dev nDCG@20: `0.3159`

## Injection Precision

| max user-cf rank | considered | GT hits | precision |
|---:|---:|---:|---:|
| 1 | 6177 | 4 | 0.0006 |
| 3 | 6344 | 4 | 0.0006 |
| 5 | 6367 | 4 | 0.0006 |
| 10 | 6368 | 4 | 0.0006 |
| 20 | 6368 | 4 | 0.0006 |
| 50 | 6368 | 4 | 0.0006 |
| 100 | 6368 | 4 | 0.0006 |

## Best Policies

| policy | params | dNDCG | dH7 | dSame | dDiff | churn/80 | overlap@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| blend_top20 | alpha=0.2000 | +0.0000 | -0.0002 | +0.0001 | -0.0000 | 0.0 | 20.00 |
| inject_score_thr | max_ucf_rank=20, insert_pos=20, score_thr=0.6200, changed=319 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0 | 19.96 |
| blend_top20 | alpha=0.0500 | -0.0000 | -0.0000 | +0.0000 | -0.0000 | 0.0 | 20.00 |
| blend_top20 | alpha=0.1000 | -0.0000 | -0.0001 | +0.0001 | -0.0000 | 0.0 | 20.00 |
| inject_score_thr | max_ucf_rank=20, insert_pos=20, score_thr=0.5909, changed=642 | -0.0000 | +0.0000 | +0.0000 | -0.0000 | 0.0 | 19.92 |
| inject_score_thr | max_ucf_rank=20, insert_pos=20, score_thr=0.5244, changed=1955 | -0.0001 | +0.0000 | -0.0001 | -0.0001 | 0.0 | 19.76 |
| inject_score_thr | max_ucf_rank=20, insert_pos=20, score_thr=0.5409, changed=1277 | -0.0001 | +0.0000 | -0.0001 | -0.0001 | 0.0 | 19.84 |
| inject_score_thr | max_ucf_rank=20, insert_pos=20, score_thr=0.4653, changed=3190 | -0.0001 | +0.0000 | -0.0001 | -0.0002 | 0.0 | 19.60 |
| blend_top20 | alpha=0.5000 | -0.0002 | -0.0005 | -0.0003 | -0.0001 | 0.0 | 20.00 |
| inject | max_ucf_rank=1, insert_pos=20, changed=6177 | -0.0005 | +0.0000 | -0.0003 | -0.0006 | 0.0 | 19.23 |

## Interpretation

R180 found real recall: official user `cf-bpr` recovers 257 GTs that are absent
from the current union@300. R431 shows why that is not directly deployable:

- The first user-cf candidate outside production top-20 is the GT only 4 times
  across 6,177-6,368 injection opportunities (`~0.06%` precision).
- Reordering production top-20 by user-cf rank gives essentially zero lift.
- Injecting user-cf candidates at rank 20 is either zero-lift or slightly
  negative, even with high score thresholds.

Conclusion: public user-cf is a **recall-positive, precision-poor** signal. It
can justify a full LR/source-feature integration if run on a faster machine, but
it is not safe as a direct blind candidate and is unlikely to explain the
leaders' nDCG by itself.

# R514 - R500 Top20 Branch + Official-Positive Rows

**Recommended upload:**

`exp/inference/blind_a/r514_r500_plus_official_rows/r514_r500_top5_plus_r510_positive_rows_submission.zip`

`sha256: 9d53717064d59303cfbfd34f71d0ceb3233ebdbb8551569e550d3d581d3c04a2`

**2026-06-19 status:** still unspent in the local official-score archive; zip structure and hash revalidated.

## Rationale

R513 proved that borrowing R500 order inside the R510/R498 candidate pool is neutral. R514 tests the opposite composition: use the R500 top20-only GPT-4.1 branch as the base, then copy the four rows already banked in R510 from independent official-positive probes.

The recommended variant uses `r500_top5_keep_top1`, because that was the R500 policy with the best risk profile: no response changes, no top-20 membership changes relative to the R432s base, and strong dev lift on admitted-hit rows.

## Validation

- copied rows from R510: `[4, 9, 40, 65]`
- changed rows vs R500 top5 base: `[4, 9, 40, 65]`
- response rows changed vs R500 top5 base: `[4, 40]`
- top1 churn rows vs R500 top5 base: `[4, 40]`
- validation issues: `[]`

## Risk

This is a distinct branch, not an incremental R510 tweak. It may underperform R510 if the R500 top20-only blind transfer is weak. It is still the cleanest unspent mechanism because R500 had strong fold-positive dev evidence and was not found in the official submission archive.

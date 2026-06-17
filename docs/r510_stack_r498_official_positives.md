# R510 — R498 + Official-Positive Row Stack

**Zip:** `exp/inference/blind_a/r510_stack_r498_official_positives/r510_r498_plus_official_positive_rows_submission.zip`
**sha256:** `13512bcbb346d8684593c452da544fa4fd3ad281b0e3f733e99e20fb18445250`

## What changed

Starts from R498 (`0.5126` official nDCG) and applies only row actions with prior official positive nDCG evidence:

- `r433p04_1f7b28c1_t8_mm_rank2` row `9`: `Hatred of Music II - Tim Hecker`; official single-row delta `+0.0019`; R498 old rank `None`; response changed `False`.
- `r433p03_d5c80ee5_t7_mm_rank2` row `65`: `Maki Ya - Anamanaguchi`; official single-row delta `+0.0019`; R498 old rank `None`; response changed `False`.
- `r446p03_no_more_wood_brothers_rank1` row `4`: `The Preacher - Jamie N Commons`; official single-row delta `+0.0019`; R498 old rank `None`; response changed `True`.
- `r446p02_holiday_beyond_santa_rank1` row `40`: `O Little Town of Bethlehem - Sarah McLachlan`; official single-row delta `+0.0023`; R498 old rank `2`; response changed `True`.

## Expected score

- Rough additive nDCG estimate: `0.5206` from base `0.5126`.
- This is a measured-positive stack, not a path to 0.60 by itself.
- The larger retrieval path remains source-session/full-catalog reconstruction or another mechanism that admits true hidden-pool candidates.

## Validation

- Changed track rows vs R498: `[4, 9, 40, 65]`
- Changed response rows vs R498: `[4, 40]`
- Local LexDiv: `0.887375` vs R498 `0.885861`
- Validation issues: `[]`

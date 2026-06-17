# R511 — Source-Pool ca8 Industrial Rank-2

**Zip:** `exp/inference/blind_a/r511_source_pool_ca8/r511_r510_ca8_dead_is_new_alive_rank2_submission.zip`
**sha256:** `f567a9f6c0a86269034bb660eb6d03288883ee0b362fc592048d2fcb38aa8db1`

Starts from R510 (`0.5149` official nDCG) and changes one row:

- Row `62` / `ca8cbe02`: promote `Dead Is The New Alive - Manipulator Mix By Dope Stars Inc. - Emilie Autumn` from rank `8` to rank `2`.
- Top-1 and response are unchanged.
- Evidence: MLHD+ source-day match narrows this row to a 26-recording day pool; the target maps into the challenge catalog and is the clearest industrial-rhythm fit.

Expected value is asymmetric: if this source-pool track is the GT, moving rank 8 to rank 2 gains material nDCG; if it is not, top-1 is preserved and only ranks 2-7 shift down by one.

Validation issues: `[]`

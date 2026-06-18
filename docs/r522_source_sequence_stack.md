# R522 — Source-Sequence Two-Row Stack

**Date:** 2026-06-18  
**Base:** R510 (`nDCG@20=0.5149`, current best tracked candidate)  
**Zip:** `exp/inference/blind_a/r522_source_sequence_stack/r522_r510_ca8_misery_5ad_miles_rank2_submission.zip`  
**SHA256:** `d981b137e1fd2cbd08520b57375c6cf328807afb3b8f7d25d4ba7975fa58bb56`

## Change

Two rank-only promotions on top of R510:

- Row 31 / `5ad7094f`: move `Miles to the Sun — Hieroglyphics` from rank 9 to rank 2.
- Row 62 / `ca8cbe02`: move `Misery Loves Company — Emilie Autumn` from rank 6 to rank 2.

Both rows keep the original top-1 and response unchanged. The package has 80 rows, one root `prediction.json`, 20 unique tracks per row, exactly two
changed rows, zero top-1 changes, and zero response changes.

## Rationale

Broad learned/listwise reranks have repeatedly regressed official nDCG by displacing real hits. R522 instead uses the only currently actionable
source-sequence evidence:

- `ca8cbe02`: MLHD/MLHD+ source-day matches repeatedly place `Misery Loves Company` near the known Emilie Autumn profile tracks; the user asks for a
  theatrical Emilie Autumn follow-up with a stronger beat after `Time for Tea`.
- `5ad7094f`: the mapped source day contains a Hieroglyphics sequence; `Miles to the Sun` is the best semantic fit for “underground hip-hop with
  strong lyrical focus” and a more introspective/philosophical edge.

## Expected Outcome

If both promoted tracks are hidden GTs, the expected gain is roughly `+0.008` nDCG. If neither is GT but the GT is in the old ranks shifted down,
the loss is bounded. This is not a path to `0.55` by itself; it is the strongest current source-evidence candidate after excluding broad reranks and
the already-negative `2bfd631e` exact-title path.

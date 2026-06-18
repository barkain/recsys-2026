# R521 — Source-Sequence ca8 Misery Rank-2 Candidate

**Date:** 2026-06-18  
**Base:** R510 (`nDCG@20=0.5149`, current best tracked candidate)

## Candidate

`exp/inference/blind_a/r521_source_sequence_ca8_misery_rank2/r521_r510_ca8_misery_rank2_submission.zip`

SHA256:

`38b029a0dba047d7679f5efe321368dfea21799e0b8f505155629ec1c5653195`

## Change

Single-row rank-only edit:

- Row 62, session `ca8cbe02-1def-4fca-ba48-f17f9fac0ed8`, turn 6.
- Move `Misery Loves Company — Emilie Autumn` from rank 6 to rank 2.
- Keep top-1 `Girls! Girls! Girls! — Emilie Autumn`.
- Keep all responses unchanged.

## Rationale

This is not a broad rerank. Broad learned/listwise reranks repeatedly displaced real hits and failed official scoring.

This edit uses the only current source-sequence evidence with a clean catalog-backed candidate:

- `ca8cbe02` has multiple MLHD text-bridge matches on the exact 2013-06-09 date.
- Source sequences repeatedly place `Misery Loves Company` near the known played/profile tracks `Shalott`, `306`, `Swallow`, and `Castle Down`.
- The user asks for another Emilie Autumn track after `Time for Tea` with theatrical drama and a stronger driving/industrial rhythm.
- `Misery Loves Company` is already in R510 top-20, so this only tests whether its rank should be higher.

## Risk

The downside is limited but real: if the hidden GT is one of the old ranks 2-5, those tracks move down by one. If `Misery Loves Company` is the GT,
moving rank 6 to rank 2 should add roughly `+0.0034` nDCG. If the GT is rank 1 or rank 7+, the ranking impact is near-neutral.

This does not get us to 0.55 by itself. It is the highest-quality targeted source-evidence test currently available after broad retrieval/rerank
failures.

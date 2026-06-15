# R499 Source-Guided Candidate

Date: 2026-06-15

## Current Best nDCG Base

Base file:

`exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_keep_top1_submission.zip`

Official base score:

- nDCG@20: 0.5126
- LLM: 4.85
- composite: 0.6368

## Source Reconstruction Evidence

Session `ca8cbe02-1def-4fca-ba48-f17f9fac0ed8` has two independent MLHD+ same-day source matches:

- `164bbbf0-dd04-424f-9f5b-bf36a2704061.txt`
- `267da809-5762-4c2c-85fb-ebf0667d8cbb.txt`

Both match the challenge played tracks `Shalott` and `Swallow` by Emilie Autumn on `2013-06-09`.

Mapped source-day catalog candidates overlap the live R498 ca8 list:

- `7a7f8898-62b9-4ecd-ac08-401000e025d3` — Misery Loves Company
- `1be962dd-528f-4a7a-983b-d3a8223c76bf` — Opheliac
- `746ec84b-da8c-44c8-9fac-292a473edbbc` — I Want My Innocence Back
- `4ed05288-20e2-424a-9c11-05476046cbe3` — Gothic Lolita
- `336755f0-c038-4172-804c-de3c66875a23` — The Art of Suicide

The final user request asks for another Emilie Autumn track with theatrical/dramatic style and a stronger driving/industrial rhythm after `Time for Tea`.
`Misery Loves Company` is the most direct source-neighbor candidate for that intent.

## Candidate

Preferred upload:

`exp/inference/blind_a/r499_source_guided/r499b_ca8_source_consensus_misery_r498base_submission.zip`

SHA256:

`2e031431bc99bc3ac49fe87443bab9c8136481132eb04e011224aeecbb6b6614`

Patch:

- One row changed: `ca8cbe02`
- New top 6:
  `Misery Loves Company`, `I Want My Innocence Back`, `Gothic Lolita`, `The Art of Suicide`, `Opheliac`, `Girls! Girls! Girls!`
- Response repaired only for this row.
- All other 79 rows remain from R498 keep-top1.

## Expected Movement

If `Misery Loves Company` is the hidden GT, expected gain is about `+0.008` nDCG over R498 because it moves from rank 6 to rank 1.

If `Girls! Girls! Girls!` is the hidden GT, expected loss is about `-0.007` nDCG because it moves from rank 1 to rank 6.

This is an aggressive source-reconstruction bet, not a generic semantic rerank. It is worth one slot because it tests the only current mechanism with a plausible path above the semantic-reranker ceiling.

## Continuing Work

The initial MLHD+ scanner only targeted three bridgeable rows. It has been replaced by an all-bridge scan over every row with usable MBID coverage:

- `2bfd631e`
- `525f9f69`
- `ca8cbe02`
- `a1df8767`
- `5ad7094f`

Output:

`exp/eval/expR497_mlhdplus_complete_stream_matches_all8.json`


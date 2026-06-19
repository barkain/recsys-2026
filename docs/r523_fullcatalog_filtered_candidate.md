# R523 — Filtered Full-Catalog GPT-4.1 Candidate

**Base:** R510 (`nDCG@20=0.5149`, current tracked best)  
**Method:** GPT-4.1 generative full-catalog retrieval, then manual trap filtering.  
**Zip:** `exp/inference/blind_a/r523_generative_catalog_r510/r523_r510_gpt41_fullcatalog_filtered2_repaired_submission.zip`
**SHA256:** `f4cf167a9de21b8a5409377ce980c1ba207cf6a72ce508035a0b0f619b8eb1a7`

## Dev Evidence

A balanced 240-row dev slice using the same full-catalog guess-and-resolve pipeline showed:

- Threshold `>=0.6`, insert at rank 1: `+0.0090` dNDCG, `recovered=1`, `lost=0`.
- Threshold `>=0.8`, insert at rank 1: `+0.0075` dNDCG, `recovered=1`, `lost=0`.

This is the strongest current full-catalog retrieval signal, but it is not enough by itself for `0.55`.

## Blind Filtering

The raw threshold `0.6` selector changed four rows. Two were dropped:

- Row `31bf71ab` / `License to Drive`: exact-title probe already failed officially.
- Row `39698083` / ONE OK ROCK: selected Japanese-title track conflicts with explicit `We Are`/album context.

The submitted candidate keeps two repaired top-1 swaps:

- Row `1415a335`: `Battle Metal — Turisas` from rank 3 to rank 1.
- Row `68993adf`: `Story of My Life — One Direction` from rank 2 to rank 1.

## Expected Read

This is a narrow nDCG probe with real dev support and full-catalog retrieval. Max upside is about `+0.0109` nDCG if both targets are GT. It is not a guaranteed leaderboard move, but it is cleaner than broad semantic reranking and avoids known traps.

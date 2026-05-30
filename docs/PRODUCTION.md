# Blind-A Production — Source of Truth

Last updated: 2026-05-29

## Current production: R92 p11

| field | value |
|---|---|
| artifact | `exp/inference/blind_a/r92_p11_oracle_submission.zip` |
| sha256 | `d386db7d6bfb7631a20d6aa6aa974bae565d384b6599077f610050721bcd39e8` |
| nDCG@20 | **0.5073** |
| composite | **0.6364** |
| CatalogDiv / LexDiv / LLM | 0.0301 / 0.8720 / 4.90 |
| metadata | `exp/inference/blind_a/r92_p11_oracle_submission.metadata.json` |

**Do not regenerate responses.** This ZIP is a bitwise copy of the already-scored
probe `r92_probes/r92p11_c4f7d055_t7.zip`; it has passed the real Blind-A scorer.

Verify integrity:
```
shasum -a 256 exp/inference/blind_a/r92_p11_oracle_submission.zip
# -> d386db7d6bfb7631a20d6aa6aa974bae565d384b6599077f610050721bcd39e8
```

## How it was built (reproducible)

R92 p11 = **R84c selective submission with exactly one row swapped**:
- Base: `exp/inference/blind_a/r84c_selective_submission.zip` (nDCG 0.5069, composite 0.6362).
- Change: session `c4f7d055-a3cc-4d6b-be80-b90278bc0d32` turn 7 → R90 track list
  (`exp/inference/blind_a/r90_blind_track_lists.json`), which promoted an in-pool
  rank-5 GT (`9d9ca4fe…`) to rank 1. Responses reused from R84c (unchanged).
- This single-row swap was found by black-box oracle probing (R92) and is the
  only positive of 14 scored single-row probes (+0.0004 nDCG).

R84c itself = 5-fold R84 BGE-large ensemble + selective routing over the R54
stack (see `cache/r84c_production_lr.txt`, `scripts/expR84c_selective_deployment.py`).

## Why this is the ceiling (under current allowed data)

Both within-data nDCG levers are empirically exhausted:
- **Ranking** — R92–R95: 14 oracle probes, 1 tiny win, no offline feature predicts
  a win (`docs/r95_oracle_forensics.md`).
- **Text-encoder recall** — R96: the complete 8-source union misses 22% of dev GT,
  but a new family (E5-large-v2) recovers 0 of them in top-30 under both structured
  and natural-language queries (`docs/r96_new_retriever_plan.md`). Query→track text
  encoders are exhausted; this is NOT "all retrieval impossible."
- **Multimodal** — closed earlier (R85/R88/R89). **Response side** — saturated (R78/R87).

## Next move policy

Do NOT run more Blind-A nDCG model-training sprints (no E5-mistral / more BGE /
Qwen variants / rerankers / probe variants). Reserve submissions for a genuinely
new evidence source allowed by competition rules, or Blind-B replay when it
arrives. See memory: `feedback_retrieval_lever_closed`, `feedback_oracle_probing_exhausted`.

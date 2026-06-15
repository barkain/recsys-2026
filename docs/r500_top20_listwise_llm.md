# R500 - Top20-Only Listwise LLM Reranker

**Date:** 2026-06-15
**Status:** Candidate staged. Recommended first upload: `top5_keep_top1`.

## Why this exists

R498 proved that GPT-4.1 listwise reranking transfers a small Blind-A nDCG gain, but the 60-track slate mixed production top-20 tracks with deeper challengers. That introduced false challenger promotions and made the signal noisy.

R500 removes that failure mode: the model ranks only the existing production/R432s top-20. This cannot recover absent ground truth tracks, but it can improve nDCG by moving already-admitted GTs upward while preserving recall.

## Offline calibration

Prompt pack:

- `exp/eval/r500_top20_llm/r500_dev_foldbalanced500_top20_prompts.jsonl`
- 500 dev hit rows only, because top20-only reranking is defined only for cases where GT is already in the base top-20.
- Model: `gpt-4.1`

Best policies:

| policy | rows | delta nDCG | fold deltas | notes |
| --- | ---: | ---: | --- | --- |
| `full20` | 500 | +0.0209 | +0.0089 / +0.0343 / +0.0269 / +0.0084 / +0.0259 | Highest offline, but changes top-1 and needs response repair. Not the first upload. |
| `top5_keep_top1` | 500 | +0.0167 | +0.0119 / +0.0207 / +0.0248 / +0.0225 / +0.0036 | Recommended. Preserves top-1/responses, only reorders visible head. |
| `full20_keep_top1` | 500 | +0.0163 | +0.0144 / +0.0143 / +0.0241 / +0.0220 / +0.0064 | Backup. Preserves top-1 but has more positional churn. |

## Blind-A artifacts

Recommended upload:

- `exp/inference/blind_a/r500_top20_llm/r500_gpt41_top20_top5_keep_top1_submission.zip`
- SHA256: `573daa50dda5314ed9f0f4e1c309db0f7b180b3997fdebe9381b9c17dbda81ed`
- 80 valid rows
- 77 rows changed
- 0 top-1 changes
- 0 top-20 membership changes
- 0 response changes

Backup:

- `exp/inference/blind_a/r500_top20_llm/r500_gpt41_top20_full20_keep_top1_submission.zip`
- SHA256: `ff85844b82c2404c985597845b01bdd36304ee109246adc0d9d6b44cfe0f00c8`
- 80 valid rows
- 80 rows changed
- 0 top-1 changes
- 0 top-20 membership changes
- 0 response changes

## Expected readout

If the dev hit-row lift transfers, `top5_keep_top1` should beat the R432s/R106-family nDCG `0.5092` and may beat R498's `0.5126`, because it removes challenger insertion noise while keeping the rank-lift signal.

If it fails, the likely conclusion is that the hit-row rank ordering learned from dev does not transfer to Blind-A, not that the source/retrieval recall is worse.

# R498 — Aggressive Listwise LLM Reranker

**Status:** **GO candidate built.** `gpt-4.1` listwise reranking produced the first large,
fold-positive offline nDCG signal since the Blind-A scorer reset. A rank-preserving
Blind-A candidate is ready for upload.

## Why this is different

Prior reranker failures mostly tested one of two weak forms:

- pointwise scoring: score `(conversation, candidate)` independently, then sort;
- safe micro-edits: preserve top-1 and only promote rank-2/rank-3 candidates.

Those cannot plausibly move nDCG from `0.5092` to `>0.60`. R498 is intentionally more aggressive:

- The model sees the **current top-20 and challenger pool together**.
- It is asked to predict the **hidden GT**, not the best user-facing recommendation.
- It may replace top-1 during offline evaluation, but the recommended Blind-A package
  preserves top-1 to avoid response mismatch.
- It is evaluated against current-hit displacement, not just recoverable misses.

## Built Artifacts

- Script: `scripts/expR498_listwise_llm_reranker.py`
- Dev smoke prompts: `exp/eval/r498_listwise_llm/r498_dev_smoke_prompts.jsonl`
- Dev balanced benchmark prompts: `exp/eval/r498_listwise_llm/r498_dev_balanced400_prompts.jsonl`
- Dev fold-balanced benchmark prompts: `exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_prompts.jsonl`
- Blind-A prompts: `exp/eval/r498_listwise_llm/r498_blind_prompts.jsonl`
- Recommended Blind-A candidate: `exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_keep_top1_submission.zip`

The first `balanced400` pack was useful but skewed toward folds 0/1. The decision-grade
pack is `foldbalanced500`: exactly `50` recoverable misses and `50` current hits from
each of the five folds.

## Result Summary

Decision-grade offline benchmark:

```text
pack:    r498_dev_foldbalanced500_prompts.jsonl
model:   gpt-4.1
rows:    500 = 250 recoverable misses + 250 current hits, fold-balanced
```

Best aggressive policy:

```text
policy=full20
delta_nDCG = +0.078599
recovered/lost = 125 / 36
fold deltas = +0.0463 / +0.1181 / +0.0756 / +0.0393 / +0.1137
```

Recommended submission policy:

```text
policy=full20_keep_top1
delta_nDCG = +0.067562
recovered/lost = 125 / 35
fold deltas = +0.0551 / +0.0834 / +0.0670 / +0.0449 / +0.0873
```

The `keep_top1` policy deliberately gives up about `0.011` offline nDCG versus raw
`full20`, but it keeps every top recommendation and response unchanged. This removes
the obvious failure mode from earlier R432-style variants: changing the top track while
leaving a response that describes the old track.

Blind-A candidate audit:

```text
file:    exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_keep_top1_submission.zip
sha256:  2f041f248bffab28202d3e203b13b353250f2d6cf3541040a2395bdd36ce700d
rows:    80
changed rows: 80
top-1 churn: 0
response changes: 0
bad lengths / duplicate tracks: 0 / 0
mean top20 overlap vs R432s: 9.82
```

## Exact Run Protocol

Use a normal networked shell with `OPENAI_API_KEY` or `ANTHROPIC_RECSYS_API_KEY` available.

Smoke run:

```bash
cd /Users/nadavbarkai/dev/recsys-2026
MCRS_LLM_CACHE_DIR=cache/r498_llm_api \
  .venv/bin/python scripts/expR498_listwise_llm_reranker.py run \
  --prompts exp/eval/r498_listwise_llm/r498_dev_smoke_prompts.jsonl \
  --out exp/eval/r498_listwise_llm/r498_dev_smoke_gpt41_outputs.jsonl \
  --model gpt-4.1 \
  --limit 20 \
  --max-tokens 2048 \
  --overwrite
```

Evaluate smoke:

```bash
.venv/bin/python scripts/expR498_listwise_llm_reranker.py eval-dev \
  --prompts exp/eval/r498_listwise_llm/r498_dev_smoke_prompts.jsonl \
  --outputs exp/eval/r498_listwise_llm/r498_dev_smoke_gpt41_outputs.jsonl \
  --out exp/eval/r498_listwise_llm/r498_dev_smoke_gpt41_eval.json
```

Fold-balanced 500-row run:

```bash
MCRS_LLM_CACHE_DIR=cache/r498_llm_api \
  .venv/bin/python scripts/expR498_listwise_llm_reranker.py run \
  --prompts exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_prompts.jsonl \
  --out exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_gpt41_outputs.jsonl \
  --model gpt-4.1 \
  --max-tokens 2048 \
  --overwrite
```

Fold-balanced evaluation:

```bash
.venv/bin/python scripts/expR498_listwise_llm_reranker.py eval-dev \
  --prompts exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_prompts.jsonl \
  --outputs exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_gpt41_outputs.jsonl \
  --out exp/eval/r498_listwise_llm/r498_dev_foldbalanced500_gpt41_keep_top1_eval.json \
  --policies top3_keep_top1,top5_keep_top1,full20_keep_top1 \
  --thresholds 0.0,0.55,0.65,0.75,0.85
```

If the balanced run is strong, build a Blind-A output file:

```bash
MCRS_LLM_CACHE_DIR=cache/r498_llm_api \
  .venv/bin/python scripts/expR498_listwise_llm_reranker.py run \
  --prompts exp/eval/r498_listwise_llm/r498_blind_prompts.jsonl \
  --out exp/eval/r498_listwise_llm/r498_blind_gpt41_outputs.jsonl \
  --model gpt-4.1 \
  --max-tokens 2048 \
  --overwrite

.venv/bin/python scripts/expR498_listwise_llm_reranker.py build-blind \
  --prompts exp/eval/r498_listwise_llm/r498_blind_prompts.jsonl \
  --outputs exp/eval/r498_listwise_llm/r498_blind_gpt41_outputs.jsonl \
  --policy full20_keep_top1 \
  --threshold 0.0 \
  --out exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_keep_top1_submission.zip
```

## Decision Rule

This is intentionally less conservative, but it is not blind gambling.

- `GO`: fold-balanced `delta_ndcg >= +0.025`, all folds positive, recovered/lost at least `2:1`.
- `WEAK_GO`: fold-balanced `delta_ndcg >= +0.010` and no catastrophic hit displacement.
- `NO_GO`: fold-balanced near zero or negative.

R498 passes the `GO` gate. The recommended first upload is
`r498_gpt41_full20_keep_top1_submission.zip`, not raw `full20`, because the rank-preserving
variant keeps response coherence while retaining most of the offline nDCG gain.

## Notes

- Identity-output sanity checks passed: if the model returns `C01..C20`, all policies produce exactly zero nDCG delta.
- The prompt explicitly tells the model to predict the hidden benchmark GT, not to behave as a normal recommender.
- This is the first R498 test that allows enough churn to matter. If it fails, the failure is informative.

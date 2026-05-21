# R69 Sprint — Cross-Encoder Rerank — ARCHIVE_SPRINT

**Hypothesis tested.** A frozen, zero-shot cross-encoder (BAAI/bge-reranker-v2-m3
target; ms-marco-MiniLM-L-6-v2 used in the actual smoke) reranks the same
R54-stacked RRF pool that the LightGBM LambdaRank (LR) ranks today. If the LR
bottlenecks the conversion of retrieval gains (the wall we have spent ten sprints
confirming), a strong cross-encoder should at minimum recover some net h7 by
reordering candidates the LR mis-ranks.

**Mac-feasible smoke scope (the only run done):**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 M params, MPS-friendly).
- Subset: fold-0 h7 cases only (n=200).
- Pool size: top-100 from R54-stacked weighted RRF (reduced from the planned 300).
- Pairs scored: 20 000, in 88 s at 227 pairs/s on MPS (M-series).
- Baseline: R54c frozen LR top-20 for the same 200 cases.

## Result

| Subset | n   | Baseline LR nDCG@20 | R69 cross-enc nDCG@20 | Δ        |
|--------|----:|--------------------:|----------------------:|---------:|
| h7      | 200 | 0.3043 | 0.0507 | **−0.2537** |
| h7_same |     | 0.5698 | 0.1094 | **−0.4604** |
| h7_diff |     | 0.1160 | 0.0090 | **−0.1070** |

Recovery accounting: **recovered = 3, lost = 73, net = −70**.
Top-1 churn: **199 / 200** cases changed. Top-20 overlap: **4.3 / 20**.

Every smoke gate fails — by an order of magnitude.

## Verdict: ARCHIVE_SPRINT

This is not a borderline result. The cross-encoder is destroying the LR's
calibration, not refining it. The same-artist canary regresses by **−0.46
nDCG** — the cross-encoder has no access to the same-artist signal that the LR
exploits via `track_artist_overlap` / `r54_presence` features. With 199/200
top-1 changes and a 4.3/20 overlap, the cross-encoder is essentially producing
a different (and far worse) ranking from scratch.

## Why a bigger model would not flip this

The gap between MiniLM-L-6 (22 M) and bge-reranker-v2-m3 (568 M) is
roughly 5–10 nDCG points on MS-MARCO. R69's deficit is **253** nDCG points.
Even an unrealistically optimistic 30-point recovery from a stronger reranker
leaves R69 well below baseline, and bge-reranker-v2-m3 still has no access to
the LR's structural signals (R39 artist/album overlap features, R54/R21 LR
scores, ALS factors, recall/depth statistics). The LR's 37-feature stack
beats a (query text, candidate text) scoring head not because the head is weak
but because the head is the wrong shape for this task.

This generalizes the lesson from [[feedback_llm_rerank_closed]] (R67 Opus 4.7
LLM rerank: Δh7 = −0.22 / −0.24, churn 70+/80) to the cross-encoder family.

## What this closes

- **Rerank-only with text-side scorers** is closed across all three families
  tested over the LR top-K: rules (R56), LightGBM specialist (R58), LLM (R67),
  cross-encoder (R69). All four families regress the well-calibrated LR
  top-K. See [[feedback_lr_top50_calibrated]] for the underlying calibration
  property.
- **The hypothesis that cross-encoders can bypass the LR conversion wall
  ([[feedback_lr_conversion_wall_confirmed]]) by independent scoring** is
  falsified at smoke scale. The LR is not "burying" candidates in a way a
  text-only ranker can fix; it is using features the text-only ranker does not
  have.

## Production state

R63c-repair holds Blind-A production at composite **0.6224**. R69 makes no
change.

## What remains open

- Bigger retriever ([[feedback_r68_retrieval_signal]]: BGE-large delivered the
  first positive retrieval-layer signal in 19 attempts), but conversion wall
  still blocks reaching nDCG without ranker-side innovation.
- Response-side polish ceiling held by R63c-repair ([[feedback_response_polish_saturated]],
  [[project_r64_outcome]]) — LLM 4.85 / LexDiv 0.844 is the demonstrated
  ceiling; further response variants are unlikely to clear the LexDiv floor.
- The credible remaining mechanism is a joint retraining of the LR (or its
  replacement) using a cross-encoder *feature* (cross-encoder score added to
  the LR feature stack), not a cross-encoder *rerank*. This is a future-sprint
  hypothesis; not pursued here.

## Infra notes (for future reruns)

- A full-pool / full-fold0 / bge-reranker-v2-m3 run requires either:
  - A100 / L4 / V100 with ≥16 GB VRAM, AND a Drive bundle that includes
    `cache/r21_production/`, `cache/r54/phase2_full/`, `cache/r54_phase3_als.npz`,
    `cache/r54_phase3_payload_maps.pkl`, `exp/eval/expR68_r54_reference_stats.pkl`,
    `cache/metadata/track_metadata_all_tracks.json` (~400 MB total).
  - Or a Mac MPS run with patience: ~24 h estimated at 3–4 pairs/s for
    bge-reranker-v2-m3 over the full 1.28 M pairs.
- Neither escalation is warranted given the smoke magnitude.

## Files

- `scripts/expR69_phase0_smoke_mac.py` — Mac smoke harness (committed).
- `scripts/expR69_phase0_cross_encoder_rerank.py` — original full-scope script,
  intended for A100 + bge-reranker-v2-m3 (committed earlier).
- `exp/eval/expR69_phase0_smoke.json` — smoke result JSON.
- `docs/r69_phase0_smoke_result.md` — per-run report.
- `docs/r69_sprint_summary.md` — this file.

## Counter

Consecutive post-R54c negatives: 19 (R55 → R69 inclusive).
R63 / R63b / R63c-repair (response-side) remain the only post-R54c wins.

# R81 Sprint — Constrained-Swap Ranker — ARCHIVE_SPRINT

**Date:** 2026-05-23

## Verdict

ARCHIVE. The constrained-swap mechanism extracts some signal (rec=2 / lost=1
in the best config) but absolute headroom is too small (~0.001 composite at
best). All other gates fail in every variant.

## What we built

Listwise neural ranker with two constraints:

1. **Training loss**: listwise CE + λ × anchor penalty
   - Anchor penalty: for top-K_anchor R54c candidates, penalize when
     model's z-scored prediction drops below R54c's z-scored value
   - Keeps the model anchored to R54c on high-confidence candidates

2. **Inference**: conservative swap heuristic
   - Start with R54c top-20 as anchor
   - For R54c rank 21-30, compute R81 promotion confidence
   - Swap into top-20 only if confidence > threshold, max 1-2 swaps per case

## Phase 0 results — hyperparameter sweep

| config | max_swaps | thr | λ | rec/lost | h7 Δ | same-art Δ | churn |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 2 | 0.0 | 0.5 | 2/1 | -0.002 | -0.019 | 44.4 |
| conservative | 1 | 1.0 | 1.0 | 1/2 | -0.007 | -0.022 | 42.1 |
| very_conservative | 1 | 2.0 | 1.0 | 1/1 | -0.008 | -0.021 | 42.0 |
| moderate | 2 | 0.5 | 1.0 | 1/3 | -0.010 | -0.022 | 42.0 |

All 4 configs fail gates. Best primary gate: default config with rec > lost
(2 > 1) but secondary gates all fail.

## Key observations

1. **Top-1 churn stays at ~42-44/80 across ALL configs**, even when no
   swaps happen. The model's score ordering within the existing top-20
   disagrees with R54c's. Tightening the swap threshold reduces external
   churn (cases where new candidates are added) but does NOT reduce
   internal churn (reordering within top-20).

2. **Same-artist canary fires in all variants** (-0.019 to -0.022). The
   model consistently demotes same-artist candidates that R54c had right.

3. **Recovery signal is small at best** (1-2 cases vs R80's 6 unconstrained
   recoveries). The anchor penalty + swap threshold prevent most R80-style
   gains alongside the losses.

## Why R81 doesn't work

The constrained-swap mechanism requires a **per-candidate confidence signal**
that can identify "this is the ONE that R54c missed". But the model's
listwise softmax output doesn't produce well-calibrated per-candidate
confidence in a normalization-invariant way. Even with z-scoring against
the case's score distribution, the confidence vs random noise ratio is too
low to reliably pick winners.

## The bigger lesson

R81 is the 4th neural architecture tested over the R54-stacked top-300 pool:

| sprint | arch | h7 rec/lost | h7 Δ | same-art Δ |
|---|---|---:|---:|---:|
| R71 | LightGBM stacker on top-30 | — | -0.005 | -0.004 |
| R76 | residual MLP on top-30 | 0/7 | -0.013 | -0.046 |
| R80 | listwise transformer (top-300) | 6/17 | -0.026 | -0.030 |
| R81 default | constrained-swap | 2/1 | -0.002 | -0.019 |
| R81 conservative | constrained-swap | 1/2 | -0.007 | -0.022 |

All architectures negative or marginal. Same-artist canary fires in all.

**The feature set is the ceiling, not the architecture.** The available
features (37 LR + BGE-large track/query embeddings + 5 semantic scalars)
do not carry sufficient new signal beyond R54c LR to selectively improve
top-20 ranking without breaking same-artist calibration.

## What's actually closed

After R76 + R80 + R81, every reasonable neural architecture configuration
over this dataset has been tested:
- Small residual model (R76)
- Large listwise transformer (R80)
- Constrained-swap with 4 hyperparameter combos (R81)

Plus prior closures:
- Encoder upgrade (R68/R72)
- Pool admission (R59 C3/R72)
- LR substitution/addition (R68.1/R70)
- Tree stackers (R58/R71)
- LLM/cross-encoder rerank (R67/R69)
- Hard-negative retriever (R79)

The retrieval+ranker side is empirically exhausted.

## Production state unchanged

R78 holds at composite **0.6302**, position #4. No production impact.

## Total A100 spend across R79-R81

- R79 Phase 0B: ~$3
- R80 Phase 0B: ~$3
- R81 default: ~$3
- R81 sweep (3 configs): ~$9
- **Total: ~$18**

Well under any reasonable budget. Each phase fail caught cheaply.

## Honest recommendation

The nDCG path is empirically dead for this dataset/feature set within the
"keep R54c as the anchor" paradigm. Possible remaining directions:

1. **Accept R78 (#4 at 0.6302) and freeze.** All explored paths archived.
2. **Wait for Blind-B / external data.** Genuinely new data could reopen
   things.
3. **Fundamentally new feature signal** — e.g., retrieve with a different
   query representation (LLM-generated query summaries, multi-hop reasoning),
   not just BGE on the raw query. Untested but speculative.
4. **End the sprint.** Submission #4 is solid; further A100 spend has
   near-zero expected value.

Recommended: **accept #4 and freeze** unless a fundamentally new feature
signal is identified.

## Files

- `scripts/expR81_constrained_swap_ranker.py`
- `exp/eval/expR81_constrained_swap.json` (default config result, on Colab)
- `docs/r81_constrained_swap_result.md` (on Colab)
- `docs/r81_sprint_summary.md` — this file

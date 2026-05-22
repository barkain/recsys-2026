# R76 Neural Residual Ranker — ARCHIVE_SPRINT (Phase 0B fail)

**Date:** 2026-05-22

## Verdict

ARCHIVE at Phase 0B. Per the predeclared hard rule from Codex consult:
"If this fails, stop. No A100-heavy cross-encoder."

Phase 0C (A100 semantic upgrade) **not executed**.

## What was tested

Phase 0A (dataset build) — passed:
- 48,000 candidate rows for 1600 fold-0 cases (top-30 from OOF R54c)
- 46 features per candidate: 37 LR + 3 R68 + 5 semantic + R54c rank/score
- GT-in-top-30 ceiling: 42.4% all_fold0 / 43.5% h7

Phase 0B (MLP residual model) — FAIL:
- 3-layer MLP, hidden=64, dropout=0.2, AdamW, listwise CE loss
- Score = zscore(R54c_score) + β · neural_delta(features)
- β learned via gradient descent, converged to ~0.82
- 5-way inner CV within fold-0 (1280 train / 320 test per inner fold)
- CPU only, deterministic

## Results

| subset | n | OOF R54c | R76 | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2110 | 0.1913 | **−0.0197** |
| h7 | 200 | 0.2213 | 0.2079 | **−0.0134** |
| same_artist | 531 | 0.4447 | 0.3992 | **−0.0455** |
| diff_artist | 1069 | 0.0949 | 0.0880 | −0.0069 |
| h7_same | 83 | 0.4262 | 0.4128 | −0.0134 |
| h7_diff | 117 | 0.0760 | 0.0626 | −0.0134 |

Recovery: h7 recovered=0, lost=7, net=−7
Churn: top-1 changed 930/1600 (46.5 / 80, far above 25 cap)
Overlap: 16.4 / 20

ALL gates fail.

## Diagnosis

**β converged to ~0.82.** The "residual" is supposed to be a small correction
on top of R54c's calibrated score. Instead the model gave the neural delta
weight nearly equal to R54c's own. This is over-fitting to per-candidate
features that don't generalize to ranking better than R54c.

**Same-artist −0.0455 is the canary.** R54c's strength is calibrated
same-artist preference (it explicitly captures `r54_rank_inv`, `r54_cosine`,
`last_artist_match`, etc.). The MLP could see those features but learned to
weight semantic similarity (BGE query-candidate cosine, max-sim-to-played)
more heavily. Semantic similarity rewards diff-artist candidates that share
genre/mood but aren't the actual continuation. Result: diff-artist promoted,
same-artist demoted, net loss.

**46.5/80 top-1 churn** confirms the model is not "polishing" R54c's
ranking — it's reordering substantially. Even if some moves were
improvements, the net is negative.

## Why this failed (the meta-insight)

Codex's design correctly anticipated this risk: "naked text rerankers are
dangerous (R51/R67/R69 proven negative)... the large model must enter as
features, not as the sole scorer." This Phase 0B did enter the semantic
features as features (not naked scorer), but the MLP head still over-relied
on them. The structural LR features (37 of them) were drowned out by 5
strong-signal semantic features whose discriminative power for the WRONG
quantity (semantic similarity rather than ranking calibration) was higher
than the structural features' discriminative power for the RIGHT quantity.

This generalizes: **adding any sufficiently expressive head on top of R54c's
features tends to chase the most easily-fit signal, which is semantic
similarity, which is not what optimizes nDCG on this task.**

## What R76 rules out

- Neural residual ranker over R54c top-K with current feature set is closed.
- Phase 0C (heavier semantic model as a feature) was conditional on Phase 0B
  showing non-negative signal. Phase 0B shows clearly negative signal.
  Phase 0C would amplify the same failure mode at higher cost.
- More feature engineering (e.g., adding 800-dim BGE-large embeddings as
  features) would likely worsen the problem by giving the MLP more semantic
  signal to over-fit on.

## What remains open

1. **Different loss / architecture**: monotone constraint on R54c score
   (force model to never decrease its weight below 1.0), penalty on
   same-artist regression in the loss, explicit feature gating. Could
   try, but each adds complexity and the underlying signal is weak.
2. **A direct ranking architecture that doesn't try to "residualize"**: a
   single LightGBM model trained on the full 46-feature set with explicit
   monotone constraints on R54c_score. Similar to what R58 tested at lower
   feature dimension — and R58 also failed.
3. **Pool admission** — already closed by R72 (only 2 h7 cases rescuable).
4. **Stop and freeze R74.**

## Recommendation

Freeze R74 as production (composite 0.6252, position #5). Sprint pivots
back to either a new LexDiv push (controlled, low risk) or accepts #5 and
defends.

The neural-ranker direction is closed on this dataset/feature set.

R63c-repair held production prior to R73. R73 superseded by R74. R76
artifacts: `scripts/expR76_*`, `cache/r76/`, `exp/eval/expR76_*`,
`docs/r76_*`.

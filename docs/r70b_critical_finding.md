# R70b — CRITICAL FINDING — "LR Conversion Wall" is Largely an Artifact

## Summary

After 20 consecutive post-R54c "LR conversion wall" experiments (R55→R70), the
discriminator we never ran reveals the wall is **largely a measurement
artifact**, not a structural property of LightGBM consuming new features.

The frozen `R54c LR` (`cache/r54_phase3_lr_model.txt`) — used as the baseline
in every sprint since R54 — was trained on **all 8000 dev cases** with no fold
split (`scripts/expR54_phase3_blind_submission.py:205` "Phase TRAIN:
Production LR on all dev"). Every sprint's "sibling LR" was trained on 6400
fold-0 train cases (OOF) and evaluated on 1600 fold-0 dev cases where R54c
had **already seen the labels**.

## R70 + R70b head-to-head (the experiment we should have run first)

Both siblings trained on fold-0 train (6400 cases), evaluated on fold-0 dev
(1600 cases). Pool unchanged (R54-stacked RRF top-300). Same hyperparams as
R54c (lambdarank, num_leaves=31, lr=0.05, min_data=10, seed=0,
num_boost_round=300).

|              | Features | h7 Δ vs frozen R54c | same-artist Δ |
|---           |---       |---:                  |---:           |
| R70          | 40 (+r68)| **−0.0798**          | **−0.1595**   |
| **R70b**     | 37 (no r68) | **−0.0830**       | **−0.1537**   |
| R70 − R70b   | (OOF vs OOF) | **+0.0032**     | **−0.0058**   |

R70b's regression vs frozen R54c (**−0.083 h7, −0.154 same-artist**) is
essentially identical to R70's (**−0.080 h7, −0.160 same-artist**). The
~−0.08 h7 gap is the **train/dev memorization gap**, not the "LR conversion
wall". Adding r68 features makes essentially no difference to this gap.

## What this implies — past experiments

The −0.08 h7 / −0.16 same-artist deltas we documented across R56/R58/R60/R66/
R67/R68/R68.1/R70 are not evidence that:
- LightGBM cannot consume new retrieval features.
- The R54c LR is artifact-locked in a deep sense.
- Cross-encoder / LLM rerank can't extract retrieval signal.

They are evidence that:
- The dev-OOF sprint evaluation framework systematically penalizes OOF
  siblings vs the in-sample frozen R54c by ~0.08 h7 / ~0.16 same-artist on
  fold-0.
- This penalty exists *regardless of feature set*. Any OOF retrained ranker
  loses by this amount on this measurement framework.

**Specific re-interpretations:**

- **R68 retrieval signal preserved:** +0.035 h7 single-source pool_hit, 15
  unique GT recoveries — these are pool-level metrics independent of the LR.
  They remain valid. ([[feedback_r68_retrieval_signal]] still holds.)
- **R68/R68.1 sibling LR Δh7=−0.08:** likely an artifact (or near-artifact)
  of the OOF-vs-in-sample comparison.
- **R69 cross-encoder rerank Δh7=−0.25:** even with the artifact, R69's
  magnitude (−0.25) is far worse. The cross-encoder result remains genuinely
  negative.
- **R60/R66/R67:** their failure mechanisms (pool change, source-rank
  reweighting, semantic rerank) are independent of OOF-LR comparison
  artifact; their negative results remain valid in their own right.
- **R70 (this sprint) +R70b:** when measured as OOF-vs-OOF (R70 vs R70b),
  adding r68 features yields **+0.0032 h7, −0.0058 same-artist** on fold-0
  — small, noisy, fold-0-only.

## What this implies — future experiments

The blind-track production loop is:
1. Train R54c-style LR on all 8000 dev (no fold split, in-sample).
2. Score blind test (truly held-out) → leaderboard.

The OOF sprint loop is the proxy. Until now the proxy was implicitly
"sibling-OOF vs frozen-R54c-in-sample", which is biased by ~−0.08 h7. The
**correct OOF proxy is sibling-OOF vs sibling-OOF** with one variable
changed (e.g., +r68 features).

To decide whether adding R68 features helps **production blind nDCG**, the
right test is 5-fold OOF R70 vs 5-fold OOF R70b. Fold-0 alone gave R70 a
+0.0032 h7 edge (noisy on 200 h7 cases).

## Production state

R63c-repair holds Blind-A at composite 0.6224. Nothing changes about
production. The strategic question is whether to invest in adding r68 features
to a new production LR (and possibly 5-fold BGE-large training on A100).

## Counter

The "20 consecutive post-R54c negatives" tally needs reframing:
- Response-side wins remain (R63, R63b, R63c-repair).
- Pool-level signal wins remain (R68 +0.035 h7 single-source pool_hit).
- Of the 20 "negatives", most relied on the biased OOF-vs-in-sample comparison
  for their archive decision. They aren't necessarily wrong, but the magnitude
  of their "failure" is overstated by ~0.08 h7.

## What I'm doing next

1. Commit R70b + this document.
2. Update [[feedback_lr_conversion_wall_confirmed]] to reflect the artifact
   finding.
3. Run 5-fold OOF R70b (37 features, no r68) — establishes the true
   OOF-vs-frozen-R54c baseline across all 5 folds. Mac-feasible (~30 min).
4. Decide path forward based on whether the fold-0 R70 vs R70b +0.003 h7
   edge persists across 5 folds. If it does, A100 escalation to generate
   R68 lists for folds 1–4 is justified.

# R70 Sprint — Joint R54+R68 LR (Addition Form) — ARCHIVE_PHASE_0

## Question

Can the ranker consume R68 (BGE-large) as **added** features while preserving
R54's features? This is the gating question for the A100 retrieval escalation
([[feedback_r68_retrieval_signal]] — R68 surfaced the first positive
retrieval-layer signal in 19 attempts but its substitution failed conversion).

- R68/R68.1 tested **substitution** (drop r54_*, replace with r68_*). Failed:
  Δh7 = −0.081, same-artist = −0.156.
- R60 tested **matched-pool retrain** (pool changed). Failed.
- R70 tests **addition** (pool unchanged, r68_* features added). This is the
  third — and last untested — form.

## Design

| Property | R68.1 substitution | **R70 addition** |
|---|---|---|
| Pool | R68-stacked (R54 dropped) | **R54-stacked (unchanged from R54c)** |
| Features | 37 (R39 + R68) | **40 (R39 + R54 + R68)** |
| Sibling LR | LightGBM, 300 rounds | **LightGBM, 300 rounds** |
| Train data | fold-0 train | **fold-0 train (same)** |

R54-stacked RRF pool is **bitwise identical to R54c production**. The only
change vs the frozen R54c LR is the LR retrain with 3 added feature columns
(`r68_rank_inv`, `r68_presence`, `r68_cosine`). All R68 artifacts already
existed in `cache/r68/phase0_fold0/` (full coverage: 6400 train + 1600 dev).

## Result — ARCHIVE_PHASE_0

| Subset | n | R54c baseline | R70 sibling LR | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2879 | 0.2092 | **−0.0787** |
| h7 | 200 | 0.3043 | 0.2245 | **−0.0798** |
| same_artist | 531 | 0.5984 | 0.4388 | **−0.1595** |
| diff_artist | 1069 | — | — | — |
| h7_same | 83 | — | — | — |
| h7_diff | 117 | — | — | — |

- h7 recovered = **0**, lost = **17**, net = **−17**
- top-1 churn /80 = **35.3** (above the 25 hard stop)
- top-20 overlap mean = **15.33 / 20**
- elapsed = **336 s** total (train feats 205s, LR train 26s, dev eval 80s)

Predeclared gates: **all four FAIL**.

## The damning comparison

| Metric | R68.1 substitution Δ | **R70 addition Δ** |
|---|---:|---:|
| all_fold0 nDCG | −0.0795 | **−0.0787** |
| h7 nDCG | −0.0811 | **−0.0798** |
| same_artist nDCG | −0.1557 | **−0.1595** |

**Substitution and addition produce essentially identical regressions** (within
0.001-0.004 of each other on every subset). Keeping the r54_* feature columns
alongside r68_* did not protect the ranker. LightGBM is learning to weight the
r68 features so strongly that the r54 calibration is overwritten.

## What this rules out

This is the third independent retraining form falsified:

1. R60: retrain on *changed pool* (C3 admission) → fails.
2. R68.1: retrain with *substituted features* on R68-stacked pool → fails.
3. **R70: retrain with *added features* on R54-stacked pool → fails identically.**

The conversion failure is **not** a pool-change artifact. It is **not** a
feature-substitution artifact. It is a **ranker-calibration / retraining
property**: any LightGBM LambdaRank retrained on this fold-0 train data with
the BGE-large `r68_cosine` feature available converges to the same bad ranking
relative to R54c.

## Implication for the A100 retrieval direction

Per the user's predeclared conditional:

> "If R70 Phase 0 fails, bigger encoders are less likely to help because the
> bottleneck is feature consumption/ranker calibration, not source quality."

Confirmed. **Do not spend A100 hours on BGE-large 5-fold or alternate large
encoders (E5-large-v2, GTE-large-en-v1.5) until the ranker-side question is
resolved.**

The ranker-side question is: *why does LightGBM, given a strictly larger
feature set including the R54c-equivalent stack, fail to recover R54c-level
ranking?* Candidate hypotheses for the next investigation:

1. **Hyperparameter mismatch.** R54c's frozen LR was trained with different
   `num_boost_round`/`min_data_in_leaf`/regularization than this sibling. A
   sibling control trained with FEAT_R39_ALL + FEAT_R54 (37 features, no
   r68_*) on the same fold-0 train data would isolate this — if it also
   regresses by ~similar magnitude, the issue is "retraining ANY LR loses
   calibration", not "R68 features specifically". 3-minute experiment.

2. **r68_cosine dynamic range dominates.** Normalized BGE-large dot products
   may have a higher signal-to-split-purity ratio than r54_cosine, causing
   LightGBM to over-weight them. Feature scaling, monotone constraints, or
   regularization on r68_* might mitigate.

3. **Ranker family mismatch.** Maybe LightGBM cannot consume an arbitrary
   additional dense-retriever signal cleanly. A linear stacker on top of
   R54c's top-K LR scores + r68_* features could be a less brittle interface.

## Production state

R63c-repair holds Blind-A at composite **0.6224**. No change.

## Next-decision summary

**Do not escalate to A100 retrieval.** Either:
- Run the sibling-control diagnostic (hyp 1 above) to clarify whether R70's
  regression is from retraining itself or from R68 features specifically. If
  retraining alone regresses, the ranker-side investigation is broader than
  "consuming R68". If retraining alone matches R54c, then R68 features are
  specifically toxic — investigate scaling/constraints (hyp 2).
- OR pivot the ranker-side work entirely: try a linear stacker (hyp 3), or
  declare LR conversion permanently closed and look at end-to-end approaches
  (e.g., LLM listwise rerank with R54c top-K + R68 top-K as joint candidate
  pool, scored by a model that can see both feature stacks). The R67 result
  already closed naked LLM rerank, so this would need to be feature-aware.

## Files

- `scripts/expR70_phase0_joint_lr.py` — eval harness (committed).
- `exp/eval/expR70_phase0_joint_lr.json` — full per-case metrics + gates.
- `docs/r70_phase0_joint_lr_result.md` — Phase 0 result report.
- `cache/r70_phase0_sibling_lr.txt` — trained sibling LR (40-feature, fold-0).
- `docs/r70_sprint_summary.md` — this file.

## Counter

Consecutive post-R54c negatives: **20** (R55 → R70 inclusive).
R63 / R63b / R63c-repair remain the only post-R54c wins (all response-side).

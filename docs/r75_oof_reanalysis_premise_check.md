# R75 Premise Check — OOF Reanalysis is NOT Needed for R56-R60

## TL;DR

The premise that R56-R60 archive verdicts would flip under a corrected OOF
framework is **wrong**. Inspection of their saved baseline metrics shows they
were already evaluated against an OOF CV5 baseline that is ≈ R70b 5-fold OOF.
No reanalysis matrix is justified.

## What R70b proved (recap)

Today's R70b finding identified that the "LR conversion wall" was largely a
measurement artifact: **frozen R54c** (`cache/r54_phase3_lr_model.txt`) was
trained on all 8000 dev cases. Comparing OOF siblings to it on dev-fold-0
introduced a systematic ~−0.08 h7 / ~−0.16 same-artist bias. The artifact bias
applies only to experiments that used the frozen R54c LR as the baseline.

## The check I ran

Read baseline metrics from the saved JSON outputs of R56, R58, R59 C3, R60
and compared to R70b 5-fold OOF.

| sprint | baseline all_nDCG | baseline h7_nDCG | baseline same_artist | baseline source |
|---|---:|---:|---:|---|
| R56 | 0.2246 | 0.2451 | 0.4485 | "R39+R54 Phase 2 OOF, CV5 LambdaRank" |
| R58 | 0.2246 | 0.2451 | 0.4485 | same |
| R59 C3 | 0.2246 | 0.2451 | — | same |
| R60 | (same reproduction values) | | | |
| **R70b 5-fold OOF (today)** | **0.2236** | **0.2442** | **0.4498** | sibling LR, 37 feats, current code |

These are essentially identical. The 0.001 gap reflects mild recipe drift
between R56-era code (May 2026) and current code (R70b run today), not a
methodological difference.

## Implication

The artifact bias **did NOT contaminate** R56/R58/R59 C3/R60 archive decisions
because those sprints already used a fold-aware OOF CV5 LambdaRank as their
baseline. Their negative magnitudes were real OOF-vs-OOF measurements, not
inflated by the train/dev memorization gap.

The experiments that **did** use frozen R54c as the baseline (and were
therefore biased) are:

- **R67** (frontier LLM rerank): Δh7=−0.22 to −0.24 — magnitude too large to
  flip with a +0.08 artifact correction.
- **R68/R68.1** (substitution form): Δh7=−0.08, same-artist=−0.16. Today's
  R70b discriminator showed this matches the OOF penalty almost exactly.
- **R70** (joint addition form): Δh7=−0.08. Same artifact pattern.
- **R71/R71b** (stacker on OOF R54c): R71 used a TRULY OOF R54c-style sibling
  (trained on folds 1-4, scoring fold-0) as the input. So R71's negative
  result was already a fair test.

None of these flip under reanalysis:
- R67 is too negative.
- R68/R68.1/R70 were re-tested today (R70b + R71): R68 features genuinely
  don't add ranker-stage signal.
- R71 is already a fair OOF-vs-OOF gate.

## Decision

**No R75 matrix script.** The reanalysis sprint is ceremony if the saved JSONs
are trusted. The numbers are unambiguous; the verdicts hold.

Next paths (per Codex consult after this finding):

1. Drift forensics, time-boxed 45 min. PYTHONHASHSEED, set/dict iteration in
   `_featurize_row`. If we can fix the 3 drifted features
   (`query_meta_tok_overlap`, `als_dot`, `pool_same_album_count`) and
   reproduce R54c bitwise, future ranker experiments become trustworthy.
2. Wait for R74 blind result before any further work.
3. New ranker architecture (pairwise/listwise neural reranker on top-K) if we
   spend A100 time.
4. Freeze R74 if it lifts enough and Blind-B is near.

R63c-repair holds production at composite 0.6224. R73 in production at 0.6234.
R74 zip pending blind verdict.

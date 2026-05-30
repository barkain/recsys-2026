# R84 Phase 0B — Fold-0 Result (BGE-large, full corpus)

**Date:** 2026-05-23
**Branch:** `r84-full-corpus-retriever`
**Trained:** Colab A100 (27.9 min, loss 1.85 → 1.16 smooth convergence)
**Eval:** Mac local compare against R70b OOF sibling LR (apples-to-apples per [[feedback_lr_wall_was_artifact]])

## Verdict

**ARCHIVE_SPRINT** at Phase 0B per declared gate. Gate B1 (same-artist canary) fails by 0.006; A1/A2 fail. A3 (≥10 unique h7 top-30 recoveries) **PASSES** at **21 recoveries**.

This is the first sprint since R63c-repair where the retrieval layer produced **meaningful unique signal** through to the compare layer. The conversion gap is the same structural pattern as R68/R76/R80/R81 (encoder over-prioritizes semantic similarity → same-artist nDCG regresses).

## Training summary

| param | value |
|---|---|
| model | `BAAI/bge-large-en-v1.5` |
| pairs | **127,992** (vs R54's 26,400) = **4.85×** more |
| max_seq_query / track | 384 / 256 (R54 used 256/256) |
| batch / lr / epochs | 32 / 1e-5 / 1 |
| negatives | in-batch + 64 random catalog (`hard_neg_weight=0.0`) |
| precision | bf16 autocast |
| wall | 27.9 min A100 |
| loss | 1.85 → 1.16 |
| train query truncation @256 | **24.7%** (R54's silent loss) |
| train query truncation @384 | 1.4% |

Cost: ~$1.50 A100 + ~$0 Mac compare.

## Retrieval-layer wins (real signal)

| metric | R54 | R84 | Δ |
|---|---:|---:|---:|
| source-alone h7 hit@20 | 0.2150 | **0.2200** | +0.0050 |
| source-alone h7 hit@30 | 0.2450 | **0.3100** | **+0.0650** |
| RRF pool_hit (8-source) | 0.6044 | n/a | — |
| RRF pool_hit (R84-add 9-source) | — | **0.6362** | **+0.0319** |
| RRF pool_hit (R84-replace 8-src) | — | **0.6306** | +0.0262 |
| **h7 top-30 unique recoveries** (R84 surfaces GT, R54 misses) | — | **21** | A3 PASS |

The retrieval layer is unambiguously upgraded by full-corpus BGE-large training. R84 finds 21 h7 GTs that R54 never surfaces in its top-30 — that's a 10.5% lift in h7 candidate set quality.

## Conversion-layer regression (gates fail)

Apples-to-apples: R84 sibling LR vs R70b sibling LR (same 37-feat schema, same R54-stacked pool, same LR_PARAMS, only `r54_*` ↔ `r84_*` swap):

| subset | n | frozen R54c (in-sample) | sibling R54 (R70b) | sibling R84 | Δ R84 vs R54 |
|---|---:|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2879 | 0.2110 | 0.2084 | **−0.0026** |
| **h7** | 200 | 0.3043 | 0.2213 | 0.2154 | **−0.0059** ❌ A1 |
| **same_artist** | 531 | 0.5984 | 0.4447 | 0.4335 | **−0.0112** ❌ B1 |
| diff_artist | 1069 | 0.1337 | 0.0949 | 0.0966 | +0.0017 ✓ B2 |
| h7_same | 83 | 0.5698 | 0.4262 | 0.4107 | −0.0155 |
| h7_diff | 117 | 0.1160 | 0.0760 | 0.0768 | +0.0009 |

- h7 recovered=3, lost=5 (net −2) ❌ A2
- top-1 churn (R84 sibling vs frozen R54c): 38/80
- top-20 overlap: 14.9/20 ✓ B3
- Compare elapsed: 7 min Mac

### Gates

| gate | value | pass |
|---|---|---|
| A1: h7 Δ ≥ +0.005 | −0.0059 | ❌ |
| A2: h7 rec > lost | 3 > 5 | ❌ |
| **A3: ≥10 h7 top-30 unique** | **21** | **✓** |
| A4: ambig-positive | n/a | ❌ |
| B1 (canary): same-artist Δ ≥ −0.005 | −0.0112 | ❌ |
| B2: diff-artist Δ ≥ −0.005 | +0.0017 | ✓ |
| B3: overlap ≥ 8/20 | 14.9 | ✓ |

A passes, B fails → **ARCHIVE_SPRINT** per declared rule.

## Important caveat: this is a partial-OOF probe

The sibling LR was trained on fold-{1..4} with **R54 features** (not R84) because R84 only ran on fold-0. At fold-0 eval, R84 features are surgically swapped into a model whose weights were never tuned to R84's distribution.

**A full-fairness Phase 1** would train R84 5-fold and retrain the sibling LR with R84 features on train folds. That could partially close the conversion gap **if** the LR can re-calibrate to R84's score scale.

But the same-artist regression (−0.0112) is structural at the encoder layer:
- R84 trained with longer queries (max_seq 384) → emphasizes user-text matches over artist continuation
- BGE-large over BGE-base may amplify semantic similarity over collaborative/artist signals
- Mirrors R76/R80/R81 same-artist canary failure pattern

A Phase 1 retrained sibling LR would need to recover **+0.007** on same-artist to clear B1. That's plausible but not certain.

## Cost-benefit for Phase 1

| | cost | upside | downside |
|---|---|---|---|
| Phase 1 (5-fold R84 + 5-fold sibling LR retrain) | ~$10–15 A100 + ~30 min Mac compare | Same-artist may recover via LR retraining (+0.007 to clear B1); h7 nDCG could go +0.000 to +0.010 | Same-artist may still regress (structural); $15 burned with no actionable result |

**Expected value: borderline.** Phase 1 PROCEEDS only if user explicitly approves the additional spend on a probabilistic recovery of same-artist via LR retraining.

## What R84 proved positively

1. **Full-corpus training works**: 4.85× more pairs + BGE-large + max_seq 384 produces measurable retrieval lift (h7 hit@30 +6.5pt, 21 unique top-30 recoveries vs R54).
2. **Truncation @256 was a real cost**: R54 was silently dropping 24.7% of training context — R84 recovers most of this.
3. **Pool admission via RRF works**: +3.2% pool_hit when R84 added as 9th source. This is the largest retrieval-layer pool gain since R59 C3 (+0.0596 pool_hit, also blocked by frozen LR).

## What R84 didn't change

- Same-artist canary fires (B1 gate): an encoder-side bias that recurs across R76/R80/R81/R68.
- LR conversion wall: same-artist + h7 nDCG still don't convert at the swap-only Phase 0B test (Phase 1 could partially close but not guaranteed).

## Production

R78 holds Blind-A at composite 0.6302, position #4. **Untouched.**

## Next decisions for user

1. **Archive R84 cleanly** (Phase 1 not justified) — most defensible: declared B1 gate fails, structural same-artist regression matches R76/R80/R81 pattern, $1.50 caught it cheaply.
2. **Approve Phase 1** (~$15) to test whether sibling LR can re-calibrate to R84 features and close the −0.011 same-artist gap. ~50/50 odds based on R68/R70 precedent.
3. **Pivot to R85 multimodal fusion** (BGE-large text + audio CLAP + lyrics Qwen) — was the documented fallback if Phase 0B failed without unique recovery. R84 DID show unique recovery, so this is "additive on top of R84" rather than fallback.

## Files

- `scripts/expR84_phase0a_census.py` (Phase 0A, shipped)
- `scripts/expR84_phase0b_train.py` (Colab A100)
- `scripts/expR84_phase0b_eval.py` (Colab A100)
- `scripts/expR84_phase0b_compare.py` (Mac)
- `cache/r84/phase0a/pair_manifest.parquet` (18.9 MB, in git)
- `cache/r84/phase0b_fold0/oof_r84_lists.json` (29.8 MB — too large for git)
- `cache/r84/phase0b_fold0/training_summary.json`, `eval_summary.json`, logs
- `exp/eval/expR84_phase0a_census.json` (in git)
- `exp/eval/expR84_phase0b.json` (in git, full gate report)

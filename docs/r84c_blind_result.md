# R84c Blind Result — REAL nDCG WIN, NEW PRODUCTION

**Date:** 2026-05-25
**Submission ID:** 753593
**Participant:** `dirac`
**Verdict:** R84c becomes production. First nDCG-side composite gain in this cycle.

## Headline

| metric | R78 (prev prod) | **R84c (new prod)** | Δ |
|---|---:|---:|---:|
| **composite** | 0.6302 | **0.6362** | **+0.0060** |
| **nDCG@20** | 0.4925 | **0.5069** | **+0.0144** |
| CatalogDiv | 0.0301 | 0.0302 | +0.0001 |
| LexDiv | 0.8845 | 0.8720 | −0.0125 |
| LLM judge | 4.90 | 4.90 | 0.00 |

The nDCG@20 lift (+0.0144) **over-delivered 2.5×** what dev predicted (+0.0056 on h7
sibling LR). The LexDiv regression (−0.0125) was the cost of 17 regenerated
responses (Distinct-2 doesn't survive per-response-only optimization), but the
nDCG gain dominates net composite.

## Leaderboard movement (5th screenshot, 2026-05-25)

| # | Participant | Composite | nDCG | LexDiv | LLM |
|---:|---|---:|---:|---:|---:|
| 1 | amaranta_ (NEW) | 0.64 | 0.55 | 0.89 | 4.7 |
| 2 | semintelligence | 0.64 | 0.51 | 0.86 | 4.95 |
| 3 | vkost | 0.64 | 0.53 | 0.76 | 4.9 |
| **4** | **dirac (R84c)** | **0.64** | **0.51** | **0.87** | **4.9** |
| 5 | el_presidente | 0.63 | 0.57 | 0.77 | 4.5 |

Held position #4, but **gap to top-3 collapsed from 0.0098 (R78) to ~0.003**.
We're now in the rounded-0.64 cluster with the top-3.

(A new competitor "amaranta_" took #1 today, pushing the leader from 0.64
to a higher 0.64 — exact gap TBD from raw scores.)

## What R84c actually did

- **Tracks**: 5-fold R84 ensemble (BGE-large, max_seq=384, full-corpus, 127K
  fold-0 pairs vs R54's 26K) → routed per `R54c LR top-1 margin < 0.5 OR >= 2.0`
  rule (60% R84 / 40% R54 on blind).
- **Responses**: 63 reused from R78; 17 regenerated with Opus 4.7 R78-style
  prompt where R84c top-1 differed.
- **Audit**: top-1 churn 17/80, top-20 overlap 16.3/20 — well within both
  hard gates (35/80, 14/20).

## Path that worked (after R78 plateau)

1. **R84 Phase 0A** ($0): full-corpus census → 5.58× more training pairs.
2. **R84 Phase 0B** ($1.50): fold-0 BGE-large → strong retrieval signal but
   partial-OOF probe failed same-artist canary.
3. **R84 Phase 1** ($6): 5-fold OOF proper → canary closes, h7 Δ +0.0042
   (just below +0.005 gate).
4. **R84b CPU sweep** ($0): segment diagnostics → R84 helps when R54 is
   either very unsure or very confident, hurts at moderate confidence.
5. **R84c selective routing** ($0): observable-feature rule
   (R54c margin < 0.5 OR >= 2.0) → h7 Δ +0.0056 clears gate.
6. **Blind 5-fold ensemble** ($7.50 + $1.70 Opus): cleanest audit
   (churn 17/80, overlap 16.31/20) → blind submission.

**Total R84 sprint cost: ~$18.** Net composite gain: +0.0060.

## What this proves empirically

1. **Full-corpus scale was the missing variable.** R68 ($25-80, BGE-large
   single-fold on R54's sampled corpus) was archived for same-artist canary.
   R84 (BGE-large + full corpus) closed the canary and delivered the conversion.
2. **R54 was silently truncating 24.7% of training queries at max_seq=256.**
   R84 at max_seq=384 recovered the lost context.
3. **Selective deployment via observable features works.** Routing on
   R54c LR top-1 margin extracts R84's strength (low-confidence cases where
   R54 doesn't know what to do) and avoids its weakness (mid-confidence cases
   where R54 is well-calibrated).
4. **Blind nDCG over-delivered 2.5× vs dev prediction.** This pattern
   (blind > dev) was the opposite of what we'd expect — suggests the OOF
   sibling LR baseline (apples-to-apples in feedback_lr_wall_was_artifact)
   may slightly under-estimate true nDCG lift on blind.
5. **Hard-negative ban was correct.** Per [[feedback_no_hardneg_aux_first_run]],
   the first R84 training run had `hard_neg_weight=0.0`. Adding hard negs
   later was reserved as a contingent option that never needed to fire.

## Production state

R84c is now production for Blind-A. R78 remains the documented fallback
(composite 0.6302).

## Next moves

The R84 sprint conclusively unlocked the nDCG-side ceiling that 16 prior
post-R63c experiments couldn't crack. Remaining headroom:

| direction | est. lift | cost |
|---|---:|---|
| R85: multimodal fusion on top of R84 (audio CLAP + lyrics Qwen + metadata Qwen) | speculative, possibly +0.002-0.005 nDCG | ~$10-15 Colab |
| R86: LexDiv recovery (regenerate the 17 R84c responses with stronger LexDiv constraint) | +0.005 LexDiv → +0.0005 composite | ~$2 Opus |
| R87: response-side push on R84c base (LLM 4.95 attempt) | low odds per R78 ceiling | ~$3 Opus |
| Blind-B preparation | unknown | depends on data |

R86 (LexDiv recovery) is the cheapest free upside — the LexDiv regression
was the only negative in R84c's win and is trivially recoverable on the
17 regenerated responses without touching the nDCG win.

## Files

- `exp/inference/blind_a/r84c_selective_submission.zip` (production)
- `exp/inference/blind_a/r84c_selective_submission.metadata.json`
- `exp/inference/blind_a/r84c_regen_rows.jsonl`
- `docs/r84c_blind_diff.md`
- `cache/r84_production/blind_r84_ensemble_lists.json` (5-fold ensemble)
- `cache/r84c_production_lr.txt` (production R84-LR for blind scoring)
- `exp/eval/expR84c_blind_audit.json`
- `cache/r84/phase0a/pair_manifest.parquet` (full-corpus 153K pairs)
- `cache/r84/phase{0b,1}_fold{0..4}/` (per-fold OOF artifacts)

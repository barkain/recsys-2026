# R84 Phase 1 — 5-Fold OOF (Proper Sibling vs Sibling)

**Date:** 2026-05-24
**Branch:** `r84-full-corpus-retriever`
**Verdict:** **INVESTIGATE** (not archive — significantly better than Phase 0B)

## Headline

Phase 0B's partial-OOF probe pointed to ARCHIVE_SPRINT because:
- h7 Δ = −0.0059 (regressed)
- same_artist Δ = −0.0112 (canary FAILED)

Phase 1 (proper 5-fold OOF — sibling LR retrained per fold with R84 features available throughout) **reverses** this:

| metric | Phase 0B fold-0 partial-OOF | Phase 1 5-fold OOF | Δ |
|---|---:|---:|---:|
| h7 Δ vs sibling R54 | **−0.0059** | **+0.0042** | +0.010 |
| same_artist Δ vs sibling R54 | **−0.0112** ❌ | **−0.0028** ✓ | +0.0084 |
| diff_artist Δ | +0.0017 | +0.0031 | +0.001 |
| all_fold0 Δ | −0.0026 | +0.0010 | +0.004 |
| h7 recovered / lost | 3 / 5 | 28 / 31 | proportional |

**Conclusion:** the same-artist canary that triggered Phase 0B archive was a partial-OOF artifact. Properly OOF-trained sibling LR re-calibrates to R84's feature distribution and the canary closes.

## Aggregate metrics (8000 cases, 1000 h7)

| subset | n | frozen R54c (in-sample) | sibling R54 (apples) | sibling R84 | Δ R84 vs R54 |
|---|---:|---:|---:|---:|---:|
| all | 8000 | 0.3159 | 0.2246 | 0.2256 | **+0.0010** |
| **h7** | 1000 | 0.3484 | 0.2451 | 0.2493 | **+0.0042** |
| same_artist | 2857 | 0.6282 | 0.4485 | 0.4458 | −0.0028 |
| diff_artist | 5143 | 0.1424 | 0.1002 | 0.1033 | +0.0031 |
| h7_same | 467 | 0.6316 | 0.4562 | 0.4612 | **+0.0050** |
| h7_diff | 533 | 0.1002 | 0.0601 | 0.0636 | +0.0035 |

Both h7_same and h7_diff are POSITIVE. The all-aggregate same_artist regression (−0.0028) comes from non-h7 same-artist cases.

## Per-fold h7 (the story)

| fold | sib_r54 | sib_r84 | Δ |
|---|---:|---:|---:|
| 0 | 0.2213 | 0.2256 | +0.0043 |
| 1 | 0.2294 | 0.2310 | +0.0017 |
| 2 | 0.2802 | 0.2819 | +0.0017 |
| 3 | 0.2106 | 0.2093 | −0.0013 |
| **4** | 0.2840 | **0.2986** | **+0.0146** |

4 of 5 folds positive. Fold 3 essentially flat. Fold 4 strongly positive.

## Gates

| | value | pass |
|---|---|---|
| A1: h7 Δ ≥ +0.005 | +0.0042 | ❌ miss by 0.0008 |
| A2: h7 recov > lost | 28 > 31 | ❌ miss by 4 |
| **B1 (canary): same-artist Δ ≥ −0.005** | **−0.0028** | **✓** |
| B2: diff-artist Δ ≥ −0.005 | +0.0031 | ✓ |
| B3: overlap ≥ 8/20 | 14.5 | ✓ |

A passes none, all B canaries pass → **INVESTIGATE** (not archive).

## Spend

| phase | cost | wall |
|---|---|---|
| Phase 0A | $0 | 9 sec |
| Phase 0B (fold-0) | $1.50 | 28 min train + 1 min eval |
| Phase 1 (folds 1-4) | $6 | 117 min train + 4 min eval |
| Compare (5-fold OOF) | $0 | 8 min Mac |
| **Total** | **~$7.50** | — |

## Production

R78 holds Blind-A at composite 0.6302 (#4). **Untouched.**

## What R84 has proven

1. **Retrieval signal**: +6.5pt source-alone h7 hit@30, 21+ unique h7 top-30 recoveries (Phase 0B confirmed).
2. **Conversion (proper OOF)**: h7 +0.0042, same-artist canary passes, no structural regression. Just barely below PROCEED gate.
3. **Phase 0B failure was an artifact**: same-artist canary closed completely when sibling LR is trained with R84 features in train folds (not just at eval).

## Next step: R84b calibration sweep (CPU only)

Per [[user direction 2026-05-24]]: explore the **feature interface, LR hyperparams, and conservative blend** to see if existing R84 artifacts can be tuned past the +0.005 gate without more A100 spend.

Sweeps planned:
1. **Feature interface**: R84-replace-R54 vs R84-added-to-R54 vs R84-source-in-RRF; R84 weight sweep 0.25–1.5.
2. **LR hyperparams**: num_leaves {15,31,63}, min_data_in_leaf {10,20,50}, lambda_l2 {0,1,5}.
3. **Score blend**: zscore(R54 LR) + β·zscore(R84 LR), β ∈ [0.05, 0.50].
4. **Segment diagnostics**: per-case h7 lift by same-artist, history-depth, R54 margin, R84 unique-top-30. Find a deployable observable gate (not fold-id).

PROCEED gate for R84b: h7 Δ ≥ +0.005, all Δ ≥ 0, same Δ ≥ −0.005, diff Δ ≥ 0, recov ≥ lost, overlap ≥ 14/20.

## Files

- `scripts/expR84_phase1_compare.py` — 5-fold OOF compare (Mac, ~10 min)
- `cache/r84/phase1_fold{1,2,3,4}/oof_r84_lists.json` (29.8 MB each, gitignored — too large)
- `cache/r84/phase1_fold{1,2,3,4}/{training,eval}_summary.json`, `r84_features.npy`, logs (committed)
- `exp/eval/expR84_phase1.json` (full gate report, committed)
- `docs/r84_phase1_result.md` (this file)

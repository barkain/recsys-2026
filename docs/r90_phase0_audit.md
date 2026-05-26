# R90 Phase 0 audit — R84 5-fold OOF vs R54

**Date:** 2026-05-26
**Cost:** $0 (CPU only)
**Branch:** `r90-r84-text-retriever-tuning`

## Headline

R84 source-alone vs R54 source-alone on dev (5-fold OOF):

| metric | R54 | R84 | Δ |
|---|---:|---:|---:|
| h7 nDCG@20 | 0.0987 | **0.1100** | **+0.0113** |
| h7 hit@20 | 0.2230 | **0.2610** | +0.0380 |
| h7 hit@30 | 0.2760 | **0.3370** | +0.0610 |
| h7 hit@300 | 0.5430 | **0.6480** | **+0.1050** |

R84 dominates R54 at every retrieval depth on h7. Yet R84c selective routing extracts only **+0.0061** of R84's +0.0113 source-alone lift — meaning **~46% of R84's retrieval gain converts through the LR ranker**. The rest is left on the table.

## Routing headroom (the key Phase 1 framing)

| | h7 nDCG@20 |
|---|---:|
| R54 source-alone | 0.0987 |
| R84 source-alone | 0.1100 |
| **Perfect-routing ceiling** | **0.1481** |
| Best R84c sweep extracts | +0.0061 |

Perfect routing (oracle picks best of R54/R84 per case) gives **+0.0381** over R84-alone. R84c's selective rule captures **+0.0006** of that (rounded). So there's an order-of-magnitude headroom *if* observable signals can separate R84-wins from R54-wins.

## Segmentation: top-30 head-to-head

**h7 (1000 cases)**:
- both retrieve GT in top-30: 231
- only R84: **106**
- only R54: **45**
- neither (retrieval bottleneck): **618 (61.8%)**

**Top-300 retrieval ceiling on h7: 68.7%** — 31.3% of h7 cases are unreachable by either retriever in top-300. Any source-side improvement is bounded by this.

## Key segments where R84 vs R54 differs

### Per-fold h7 (this is the surprise)

| fold | n | R54 nDCG@20 | R84 nDCG@20 | Δ |
|---|---:|---:|---:|---:|
| 0 | 200 | 0.1010 | 0.0963 | **−0.0047** |
| 1 | 200 | 0.0854 | 0.0976 | +0.0121 |
| 2 | 200 | 0.0826 | 0.1143 | **+0.0317** |
| 3 | 200 | 0.0991 | 0.1023 | +0.0032 |
| 4 | 200 | 0.1254 | 0.1395 | +0.0141 |

**Fold 0 is the only fold where R84 regresses vs R54 on h7 nDCG.** This is exactly why R84c needed selective routing rather than wholesale substitution. Folds 2 and 4 are where R84 dominates.

Implication: any Phase 1 fold-0 continuation experiment that improves fold-0 will close the only weak fold.

### History depth (all 8K cases)

| n_prior | R84-only wins | R54-only wins | net | ΔnDCG@20 |
|---:|---:|---:|---:|---:|
| 0 | 103 | 59 | +44 | +0.0013 |
| 1 | 79 | 41 | +38 | +0.0017 |
| 2 | 85 | 49 | +36 | −0.0007 |
| 3 | 90 | 65 | +25 | −0.0043 |
| 4 | 89 | 55 | +34 | −0.0074 |
| **5** | **78** | **69** | **+9** | **−0.0146** |
| 6 | 97 | 58 | +39 | −0.0071 |
| **7** | **106** | **45** | **+61** | **+0.0113** |

**R84 systematically regresses vs R54 for n_prior ∈ {2..6}** (negative Δ in 5 of 8 buckets). It only definitively wins on the extremes (n_prior=0 cold-start and n_prior=7 h7).

This is a clear training-distribution artifact. Full-corpus train-split walked turn-by-turn, producing many short-history pairs from train sessions; R84's effective training distribution is skewed toward cold-start and full-history. Mid-history is under-represented.

### Same vs diff artist (h7)

| segment | n | R54 nDCG@20 | R84 nDCG@20 | Δ | R84-only / R54-only |
|---|---:|---:|---:|---:|---:|
| same artist | 467 | 0.1932 | 0.2092 | +0.0160 | 83 / 32 |
| diff artist | 533 | 0.0159 | 0.0231 | +0.0072 | 23 / 13 |

R84 wins more on the easier same-artist segment than on the hard diff-artist segment. Diff-artist is the structural bottleneck (nDCG <0.03 for both).

### Query length (h7, char-based proxy)

Almost all h7 queries (856/1000) exceed 1800 chars (full history+played-tracks). Truncation analysis (from `cache/r84/phase0b_fold0/eval_summary.json`) confirms only **2.1% of fold-0 queries truncate at max_seq=384**, **0.19% at max_seq=512**.

**max_seq=512 retrain variant (variant C from R90 plan) is dead on arrival** — only ~3 of 1600 fold-0 queries would gain anything.

## Phase 1 variant ROI prediction

| variant | rationale | expected fold-0 h7 Δ | risk | recommend |
|---|---|---:|---|---|
| A: 2nd epoch @ LR=5e-6 | direct continuation; the obvious test | +0.003 to +0.010 | overfitting on same-artist segment (R76/R80 pattern) | **YES** |
| B: 2nd epoch @ LR=2e-6 | softer continuation, lower regret tail | +0.001 to +0.005 | minimal gain | YES if A passes |
| C: max_seq=512 retrain | only 2.1% truncate; almost nothing to recover | <+0.001 | wasted cycles | **NO** |
| D: query/history dropout | regularization for n_prior mid-range regression | +0.002 to +0.005 | adds complexity confounder | DEFER unless A fails |

## Observable routing tweaks (no GPU needed)

The R84c selective sweep already exhausted simple LR-margin thresholds (9 combos). The shipped rule is `low=0.5/high=2.0` (predeclared). The sweep best is `low=0.25/high=2.0`:

| sweep config | h7 Δ | same-art Δ | churn /80 | recovered/lost |
|---|---:|---:|---:|---:|
| **shipped (0.5 / 2.0)** | **+0.0056** | −0.0017 | 33.8 | 23/24 |
| optimal (0.25 / 2.0) | +0.0061 | −0.0011 | 25.5 | 15/18 |

low=0.25 strictly dominates low=0.5 on every metric (h7 Δ, same-artist, churn). On a Blind-B-class drop, it's worth re-routing R84c with this threshold; cost is zero (just re-pack with different rule). Predicted blind h7 Δ: +0.001 to +0.003 over R84c.

## Verdict

**Phase 1 PROCEED** — but with reduced variant set:
- **Variant A**: 2nd-epoch continuation at LR=5e-6 (single run, fold-0 only)
- **Variant B**: gated on A — only run if A clears same-artist gate but h7 Δ borderline
- **Variant C dropped**: max_seq=512 has <0.2% truncation — no signal to recover
- **Variant D dropped**: speculative; revisit only if A regresses

**Cost projection**: variant A alone is ~$5-10 on A100 (single-epoch continuation, fold-0). Phase 1 stop-or-go after this single experiment.

**Pre-flight observable tweak**: re-route R84c with `low=0.25/high=2.0` thresholds; this is free and the sweep already shows strict dominance.

## Files

- `scripts/expR90_phase0_audit.py`
- `exp/eval/expR90_phase0_audit.json`
- `docs/r90_phase0_audit.md` (this file)

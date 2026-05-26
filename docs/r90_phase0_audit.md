# R90 Phase 0 audit — R84 5-fold OOF vs R54

**Date:** 2026-05-26
**Cost:** $0 (CPU only)
**Branch:** `r90-r84-text-retriever-tuning`

> **Scale note.** Every nDCG number on this page is **retrieval-layer** —
> GT rank within the retriever's top-K, *no* LR rerank. The R84c selective
> sweep (`exp/eval/expR84c_selective.json`) reports **LR-scored top-20 nDCG**
> on a different scale (~2.3× larger because the LR concentrates GT into
> top-20 of the RRF pool). The two cannot be compared by absolute delta.
> This audit answers "where does the retriever leave room?", not "what
> would R84c-routed LR have produced?"

## Headline (retrieval-layer)

R84 source-alone vs R54 source-alone on dev (5-fold OOF), h7 subset:

| metric | R54 | R84 | Δ |
|---|---:|---:|---:|
| h7 nDCG@20 (retrieval) | 0.0987 | **0.1100** | **+0.0113** |
| h7 hit@20 | 0.2230 | **0.2610** | +0.0380 |
| h7 hit@30 | 0.2760 | **0.3370** | +0.0610 |
| h7 hit@300 | 0.5430 | **0.6480** | **+0.1050** |

R84 dominates R54 at every retrieval depth on h7.

## Retrieval-layer routing headroom

This is the *upper bound* on what any per-case binary R54-vs-R84 routing rule
could achieve at the retrieval layer (oracle picks max(R54, R84) GT rank per case):

| | retrieval h7 nDCG@20 |
|---|---:|
| R54 source-alone | 0.0987 |
| R84 source-alone | 0.1100 |
| Perfect-routing ceiling | 0.1481 |
| Retrieval-layer headroom | +0.0381 over R84-alone |

**This is NOT comparable to R84c selective sweep's +0.0061 dev h7 Δ** — that
delta lives on the LR-scored axis. The retrieval ceiling tells us *the upstream
signal exists*; whether any observable LR-margin rule can extract it through
the frozen ranker is a different question that the R84c selective sweep
already explored on the LR scale.

## Segmentation: top-30 head-to-head (retrieval)

**h7 (1000 cases)**:
- both retrieve GT in top-30: 231
- only R84: **106**
- only R54: **45**
- neither (retrieval bottleneck): **618 (61.8%)**

**Top-300 retrieval ceiling on h7: 68.7%** — 31.3% of h7 cases are unreachable
by either retriever in top-300. Any source-side improvement is bounded by this.

## Key segments where R84 vs R54 differs

### Per-fold h7 (retrieval-layer surprise)

| fold | n | R54 nDCG@20 | R84 nDCG@20 | Δ |
|---|---:|---:|---:|---:|
| 0 | 200 | 0.1010 | 0.0963 | **−0.0047** |
| 1 | 200 | 0.0854 | 0.0976 | +0.0121 |
| 2 | 200 | 0.0826 | 0.1143 | **+0.0317** |
| 3 | 200 | 0.0991 | 0.1023 | +0.0032 |
| 4 | 200 | 0.1254 | 0.1395 | +0.0141 |

**Fold 0 is the only fold where R84 regresses vs R54 on h7 retrieval.** This
is precisely why R84c needed selective routing rather than wholesale substitution.

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

**R84 regresses vs R54 for n_prior ∈ {2..6}** (negative Δ in 5 of 8 history
buckets). It only definitively wins on the extremes (cold-start and h7). This
is consistent with a training-distribution artifact: walking the full train
split turn-by-turn produces many short-history pairs from train sessions;
R84's effective training distribution skews toward cold-start and full-history.
Mid-history is under-represented.

**Phase 1 must verify n_prior buckets, not just h7 aggregate**, or a second
epoch may reinforce this skew rather than correct it.

### Same vs diff artist (h7)

| segment | n | R54 nDCG@20 | R84 nDCG@20 | Δ | R84-only / R54-only |
|---|---:|---:|---:|---:|---:|
| same artist | 467 | 0.1932 | 0.2092 | +0.0160 | 83 / 32 |
| diff artist | 533 | 0.0159 | 0.0231 | +0.0072 | 23 / 13 |

R84 wins more on the easier same-artist segment than on diff-artist. Diff-artist
is the structural bottleneck (nDCG <0.03 for both).

### max_seq=512 retrain is dead on arrival

From `cache/r84/phase0b_fold0/eval_summary.json`: only **2.1% of fold-0 queries
truncate at max_seq=384, 0.19% at max_seq=512**. Variant C of the original R90
plan would gain ~3 of 1600 queries; drop it.

## Routing-rule comparison: shipped vs sweep-best (LR-scored layer)

Source: `exp/eval/expR84c_selective.json`. These numbers are LR-scored top-20
nDCG deltas vs R54c-alone baseline on dev (5-fold OOF).

| config | h7 Δ | same-art Δ | churn /80 | top-20 overlap | rec / lost |
|---|---:|---:|---:|---:|---:|
| **shipped (0.5 / 2.0)** | **+0.0056** | −0.0017 | 33.8 | 16.08 | **23 / 24** |
| optimal (0.25 / 2.0) | +0.0061 | −0.0011 | 25.5 | (higher) | 15 / 18 |

**Tradeoff, not strict dominance.** `low=0.25/high=2.0` improves h7, same-artist,
churn, and overlap — but rec/lost shrinks from 23/24 to 15/18 (it's more
conservative about routing to R84). Whether this trade is favorable depends on
weighting: by h7 nDCG and same-artist canary it's better; by raw "how many
h7 GTs did the new rule recover vs lose" it's smaller in both directions.

For Blind-B, the case for `0.25/2.0` is the lower churn (25.5 vs 33.8/80, ~25%
fewer top-1 changes from R78 baseline) and better same-artist Δ. The smaller
rec/lost count reflects more conservative routing, which is what we want on
a fresh-blind set without R87-style overconfidence.

## Phase 1 variant ROI prediction

The original R90 plan called for a continuation from R84 fold-0 checkpoint. **The
checkpoint is not available locally** (only eval artifacts, not model weights,
were downloaded from Colab). Variant A is therefore reformulated as a clean
2-epoch retrain from HF base, which tests the same "more training" hypothesis
with no checkpoint-restore dependency.

| variant | rationale | expected fold-0 h7 retrieval Δ | risk | recommend |
|---|---|---:|---|---|
| A: **2-epoch retrain @ LR=1e-5 (cosine over both)** | clean test of "more training"; no checkpoint needed | +0.003 to +0.010 | over-fit to same training-distribution skew → mid-history regression | YES with n_prior gate |
| B: 2-epoch retrain with **balanced-by-history sampler** | corrects the mid-history regression | +0.001 to +0.005 (broader) | adds two confounders (more training + new sampler); not a clean A/B | DEFER unless A passes h7 but fails n_prior |
| ~~C: max_seq=512 retrain~~ | only 0.19% truncate | <+0.001 | wasted cycles | **DROP** |
| ~~D: query/history dropout~~ | speculative regularization | unknown | adds confounder | **DROP** |

## Phase 1 gate set (raised per review)

A passing Variant A must clear **all four** gates:

1. **Same-artist canary**: h7 same-artist nDCG@20 Δ ≥ −0.005 vs R84 fold-0
   (hard stop, R76/R80/R81 pattern)
2. **h7 aggregate** (any of): nDCG@20 Δ ≥ +0.003, OR ≥5 net unique top-30
   recoveries (recovered − lost ≥ 5)
3. **History-bucket safety**: for n_prior ∈ {2, 3, 4, 5, 6}, mean nDCG@20
   bucket Δ ≥ −0.005. Even if h7 passes, mid-history regression aborts.
4. **Per-fold sanity**: fold-0 was R84's weakest (Δ=−0.0047 vs R54);
   continuation must not make any *other* fold worse than the current R84
   fold OOF baseline (≥ −0.005 each).

Failing #3 or #4 with #1 + #2 passing → INVESTIGATE (report numbers, no
auto-escalate to 5-fold).

## Files

- `scripts/expR90_phase0_audit.py`
- `exp/eval/expR90_phase0_audit.json`
- `docs/r90_phase0_audit.md` (this file)

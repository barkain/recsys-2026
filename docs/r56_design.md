# R56: Source-Rank Protection — Design Doc

**Status: design only. No implementation, no blind submission until reviewed.**

## 1. Context

Current production: R54c (composite 0.6106 on Blind-A, leaderboard #5).
R55 (single all-data retriever) and R55h (manual hybrid) both archived as
negative or flat. Refreshed dev decomposition (`expR55_post_refresh_decomp.py`,
R39+R54 baseline via Phase 2 OOF) gives this fixable-bucket view of the 8000
dev cases:

| Bucket | Count | % of all | Where the fix lives |
|---|---|---|---|
| HIT (LR top-20 has GT) | 3348 | 41.9% | already correct |
| **DEMOTED** (GT in pool@300 but LR ranks >20) | **1628** | **20.3%** | **LR side — this doc** |
| POOL_MISS (GT in source union but RRF drops it) | 1163 | 14.5% | RRF / pool — R56b later |
| UNREACHABLE (GT in no source) | 1861 | 23.3% | new retriever — R57 territory |

DEMOTED is the largest *LR-fixable* bucket. Sub-analysis of which sources
already surfaced GT before LR demoted it:

| Source had GT in top-20 (DEMOTED cases only) | Count |
|---|---|
| R54 (Phase 2 OOF proxy) | 237 |
| R21 | 207 |
| src_c (BM25, full-query) | 195 |
| src_b (BM25, last-music-meta) | 184 |

These overlap (some DEMOTED cases have GT in two or three of these
simultaneously) but the union of "top-20 in at least one strong source" is
the actionable population.

## 2. Hypothesis

LambdaRank can be overly aggressive in demoting candidates that strong
sources surfaced near the top. A **post-LR protection layer** that gives a
rank floor to candidates with high source-rank can recover some DEMOTED
cases without retraining LR.

We treat this as *structural calibration*, not a new learned-feature hunt.
The protection rule is deterministic and ablatable.

## 3. Approach: post-LR protection layer

Pipeline change (final ranking stage only):

1. RRF builds pool@300 — **unchanged**
2. LR predicts scores per candidate — **unchanged**
3. Sort by LR score — **unchanged** (this is the baseline ordering)
4. **NEW: apply protection rule** — promote candidates that hit the source
   criterion, subject to per-variant scope and a per-session cap
5. Final top-20 = first 20 items after protection re-order

Why a post-LR layer and not new LR features:

- **Bounded**: only modifies ranking, not learned scores. Easy to revert.
- **Fast**: no 5-fold LR retraining per variant. Tens of seconds vs minutes.
- **Falsifiable**: ablation is trivial (turn protection off, re-run).
- **Safer for blind**: if protection regresses on dev, we never run it
  on blind. We don't risk modifying LR weights that already work.

### Exact protection mechanics

Define `qualifies(c)` per variant (see §4) and `K` = per-session protection
cap (swept; see §5).

```
baseline_order = candidates sorted by LR score descending
protected_set  = { c in baseline_order : qualifies(c) and baseline_rank(c) > 20 }

# Cap: take at most K protected candidates per session, ordered by LR score
protected_insert = first K of protected_set sorted by LR score descending

unprotected_top20 = first 20 of baseline_order that are NOT in protected_insert
unprotected_tail  = baseline_order minus (unprotected_top20 ∪ protected_insert)
                    preserving baseline order

# Final order: protected candidates go into the top band; displacement comes
# from the lowest-LR-score members of unprotected_top20.
new_top20 = (unprotected_top20 minus the K lowest-LR-score members)
            ∪ protected_insert
new_top20 = stable-sorted by LR score descending  # preserves LR's
                                                  # relative preference

final_order = new_top20 + unprotected_top20_lowest_K_displaced
              + unprotected_tail
```

Properties:

- Unprotected candidates keep their LR-induced relative order.
- Protected candidates are *inserted* into the top-20 band at the cost of
  the K lowest-LR-score unprotected candidates that were in the top-20.
- Within the new top-20, ordering is still by LR score, so protection
  doesn't artificially elevate a protected candidate above a higher-LR
  unprotected one — it only ensures the protected candidate is *in* the
  top-20.
- The cap K bounds blast radius. One noisy session can't insert 5 wrong
  candidates and erase a correct LR top-5.

Sweep K = {1, 2, 3}. Default K = 1. Higher K only considered if K=1 shows
clean signal.

## 4. Variants

All five variants share the same execution path. They differ only in the
*criterion* (which candidates qualify for protection) and *scope* (which
sessions the protection applies to).

For all variants, when a candidate qualifies for protection AND its LR rank
is > 20, we promote it to the lowest unfilled rank in the protected band
(20 by default). Ties among multiple protected candidates are broken by
LR score (preserving LR's relative preference among them).

| Variant | Criterion (what qualifies a candidate) | Scope (session-level routing) |
|---|---|---|
| **A** | top-20 in any of {R54, R21, src_b, src_c} | all 8000 cases |
| **B** | top-10 in any of {R54, R21, src_b, src_c} | all 8000 cases |
| **C** | top-20 in R54 OR R21 (dense only) | all 8000 cases |
| **D** | top-20 in src_b OR src_c (lexical only) | all 8000 cases |
| **E** | top-20 in any of {R54, R21, src_b, src_c} | **observable-only gated** (see below) |
| **O (ORACLE, diagnostic only)** | top-20 in any of {R54, R21, src_b, src_c} | gated by true `diff_artist` (GT-side label) |

Rationale per variant:

- **A** — broadest. Establishes ceiling for protection-style fix.
- **B** — tighter signal, less recall but higher precision.
- **C** — tests whether the dense retrievers (R54/R21) are specifically
  the ones being over-demoted by LR.
- **D** — symmetric test for lexical (BM25) retrievers.
- **E** — applies protection only on **observable** session features that
  *correlate with* the "diff_artist / hard case" regime, since the true
  diff_artist label is GT-side and not available at inference. Deployable.
- **O (ORACLE)** — tells us the *ceiling* of perfect gating. Uses true
  diff_artist (artist of GT ∉ played artists). **Diagnostic only. Never
  a blind candidate.** If O materially beats E, we know the gating signal
  is weaker than the ideal and observable proxies could be improved.

### Variant E observable gates

Variant E applies protection on a session iff at least one of the following
holds (all computable at inference time without GT):

| Gate | Definition | Why |
|---|---|---|
| `n_prior_music < 3` | played history is short | LR has little signal to work with; protection lifts safe high-source candidates |
| `n_unique_artists >= 5` | played history spans 5+ distinct artists | Session is exploration-mode, not artist-deep-dive; LR's same-artist bias is less reliable |
| `pool_top20_same_artist_share <= 0.30` | of the LR top-20, ≤30% share the played-artist set | If LR's top-20 is already dominated by same-artist tracks, protection adds little; if it's NOT, the case is structurally diff-artist and protection has more lift potential |

We OR the three gates. E qualifies a session if ANY observable gate fires.

True same/diff artist is **reported, not routed** — used in metric splits
only to detect regressions and characterise the lift. The ORACLE variant
(O) is the only place GT-side gating is permitted, and it stays in the
diagnostic table.

We DO NOT include a "top-50 in any source" variant: at that depth the
signal-to-noise ratio is too low, and we'd be lifting tracks the LR
correctly demoted.

## 5. Evaluation methodology

### Data

- Dev: all 8000 cases. Same R39+R54 Phase 2 OOF features as the refreshed
  decomposition. CV5 LambdaRank scores already computed by
  `expR55_post_refresh_decomp.py` — we can cache `case_lr_rank` + the
  full ordered candidate lists per case and reuse them across variants.
  No LR retraining.

### Per-variant report

For each variant (and each K ∈ {1, 2, 3}) we report:

| Metric | Reason |
|---|---|
| h7 nDCG@20 | Primary production target |
| all-dev nDCG@20 | Aggregate consistency check |
| Same-artist nDCG@20 | Regression detector (split is for reporting only, not routing) |
| Diff-artist nDCG@20 | Where the lift should appear |
| Hist-depth breakdown (h0..h7+) | Spot if protection helps short or long history more |
| Top-1 churn (all-dev) | Stability — count of all-8000 sessions where top-1 changed |
| Top-1 churn (h7-only) | Stability on the production-target slice |
| Top-20 overlap median vs baseline | Stability |
| Sessions affected (≥1 candidate inserted) | How wide the intervention reaches |
| Recovered (DEMOTED → HIT) | Direct benefit count |
| Lost (HIT → DEMOTED) | Direct cost count |
| Net recovery = recovered − lost | Headline number |

### Gates

The earlier draft used dev top-1 churn thresholds of <25/8000 and <10/8000
— those are ≤0.3% and ≤0.125% of cases, which is effectively zero and
would reject any method that does anything. Re-scaled below.

- **Production candidate** (eligible for direct blind submission):
  - h7 nDCG **+0.010** or better vs baseline (per `[[blind-gate]]` memory)
  - Same-artist nDCG **does not regress** (Δ ≥ 0)
  - **Top-1 churn (all-dev) ≤ 240/8000 (3.0%)**
  - **Top-1 churn (h7-only) ≤ 30/1000 (3.0%)** reported separately
- **Exploratory blind candidate** (slot for measurement only):
  - h7 nDCG **+0.005** or better
  - Same-artist nDCG does not regress
  - **Top-1 churn (all-dev) ≤ 120/8000 (1.5%)**
  - **Top-1 churn (h7-only) ≤ 15/1000 (1.5%)** reported separately
- **Stop** any variant where:
  - Net recovery (recovered − lost) ≤ 0 on dev
  - Same-artist nDCG regresses by more than 0.002
  - All-dev nDCG worse than baseline

### Selection rule

Pick the *best variant by h7 nDCG* among variants that pass both
"no regression" gates. Tiebreak by:
1. Lower top-1 churn
2. Higher net recovery
3. Variant ID alphabetical order (A > B > C > D > E)

## 6. What this does NOT cover

- **LR retraining with new features** — that's a different experiment
  (would be R56b if protection shows signal). We're explicitly testing a
  cheaper structural alternative first.
- **RRF weight tuning** — separate axis (R56c). Targets POOL_MISS bucket,
  not DEMOTED.
- **Pool expansion** — same family as RRF tuning. Out of scope here.
- **New retriever / structural features** — that's R57. Targets
  UNREACHABLE bucket.

## 7. Implementation outline (only after design review)

```
scripts/expR56_source_rank_protection.py
  - Loads cached features + CV5 LR ranks from the refreshed decomp
  - Computes baseline (no protection) metrics for reference
  - Iterates over 5 variants, applies protection rule, recomputes ranks
  - Reports per-variant metrics + gate verdicts in one table
  - Saves per-variant case-level diagnostics to
    exp/eval/expR56_protection_variants.json
```

Estimated runtime: ~5-10 minutes total on CPU (no training, just ranking
re-orders + nDCG recomputation).

## 8. If protection shows signal, downstream plan

1. Run on dev (this script). Pick best variant per selection rule.
2. If h7 nDCG gain ≥ +0.010 with no same-artist regression: prepare blind
   submission. The protection rule is applied to R54b's blind LR output
   (we still have the cached LR top-20 + R54 ensemble blind lists).
3. If h7 nDCG gain in [+0.005, +0.010) with very low churn: queue an
   exploratory blind slot.
4. If no variant shows signal: archive R56 as negative. Move to R56b
   (RRF/pool tuning) or pause for R57 structural feature design.

## 9. If protection does NOT show signal

This is informative — tells us LR's demotion is *learned correctly* from
training data and the DEMOTED cases are genuinely lower-ranked. Then the
focus shifts to:

- Why LR thinks these are bad despite strong source signal? Feature
  importance audit. Might point to a feature that's miscalibrated.
- Or: the 1628 DEMOTED cases are genuinely hard and the gain ceiling for
  pure ranker-side fixes is < +0.005. Move to RRF or retriever-side work.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Protection promotes popular non-GT tracks → regresses same-artist | Always report same-artist nDCG. Stop on regression. |
| Phase 2 OOF drift from Phase 3 production | Treat dev results as relative-direction signal. Validate on blind churn before submitting. |
| Variant E gating may over-fit to "diff_artist + short_history" | Report E's behavior on the cases it skips too — should be unchanged from baseline there. |
| Multiple protected candidates per case → which gets the rank-20 slot | Deterministic tiebreak by LR score (preserves LR's preference among protected candidates). |
| Source-rank "top-20" threshold is arbitrary | Variants A and B span two values (20 and 10). Treat as a small sweep. No more thresholds without first evidence one works. |

## 11. Open questions — resolved

1. **No stacking in first pass.** Test single-variant signal first
   (A, B, C, D, E, O). Stacking experiments (e.g. A ∩ D) only after a
   single-variant winner is established.
2. **Scan full pool@300.** Pool of 300 is small, the compute is
   negligible, and a few DEMOTED cases have GT at rank 50-300.
3. **Top-1 eligible for protection swap.** Not pinned to LR's top-1.
   Protection-driven top-1 changes are part of the test — and are
   bounded by the per-session K cap and the churn gate.

---

**Awaiting design review v2.** No code, no blind submission until approved.

## 12. Patch history

**v2 (post-review):**

- §3 Approach — added "Exact protection mechanics" subsection with
  precise insertion algorithm, displacement rule, and per-session cap K
  swept over {1, 2, 3} (default 1). Prevents one noisy session from
  flooding the top-20.
- §4 Variants — Variant E is now **observable-only**: gated by
  `n_prior_music<3` OR `n_unique_artists>=5` OR
  `pool_top20_same_artist_share<=0.30`, none of which use GT. True
  `diff_artist` is reporting-only.
- §4 Variants — Added Variant O (ORACLE), gated by true `diff_artist`.
  Diagnostic only, never a blind candidate. Quantifies the ceiling of
  perfect gating so we know whether observable E is leaving signal on
  the table.
- §5 Gates — Top-1 churn thresholds rescaled from <25/<10 per 8000
  (impossible) to **≤240/8000 (3%)** for production and **≤120/8000
  (1.5%)** for exploratory. Now also report h7-only top-1 churn at
  ≤30/1000 (3%) and ≤15/1000 (1.5%) separately.
- §11 Open questions — collapsed into "resolved" answers since the
  three calls didn't change in review.


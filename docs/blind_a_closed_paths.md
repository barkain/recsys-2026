# Blind-A Closed Paths

## TL;DR (read only this if you're in a hurry)

R54c sits at composite **0.6106** (#5 Blind-A) with a 37-feature LambdaRank over an 8-source RRF pool@300. Eleven consecutive post-R54c experiments attempted to improve it. All failed.

The failures cluster into **6 closed mechanism classes** (pool broadening, structural features, rule-based reranks, retriever swaps, manual hybrids, response polish) plus one pre-R54 lesson (hard-negative mining). Each class has been tested from multiple angles with consistent negative outcomes.

**Why the optimum is durable:** The 37 source-rank features carry most signal. The LR converged on the distribution RRF produces. Pool-broadening hits a calibration wall (8% conversion ceiling). Feature-broadening overfits. Specialists can't outperform LR's top-50 ordering.

**What could break it:** External data (C4, rules-gated), a non-LR scoring family with fundamentally different features, or Blind-B distribution shift.

**Rule of thumb:** If a new proposal's mechanism matches any of the 6 closed classes below, it needs an explicit argument for why it's structurally different from the experiments that closed that class. "But this time it'll be different" is not an argument.

---

## 1. Production State

| Field | Value |
|---|---|
| Submission | R54c (response polish on R54b track IDs) |
| Composite | 0.6106 |
| nDCG@20 | 0.4925 |
| LexDiv (Distinct-2) | 0.8381 |
| LLM Judge | 4.70 |
| Leaderboard | #5 (2026-05-16) |
| Architecture | 37-feature LambdaRank on 8-source weighted RRF pool@300 |
| Retriever | 5-fold BGE-base-en-v1.5 ensemble (R54) + R21 + A/B/C/D/F/ALS |
| Response | Haiku-generated, audit-polished, strip_tag_prefix |

The gap to top-4 (#2-#4 at composite ~0.62-0.63) is dominated by nDCG: they have nDCG 0.51-0.57 vs our 0.4925. LexDiv and LLM judge are at parity or near-ceiling.

---

## 2. The 11 Closed Experiments

| # | Label | Mechanism Class | Key Metric | Diagnosis |
|---|---|---|---|---|
| 1 | R55 | Retriever swap | nDCG -0.0067 (blind) | Single all-data model loses ensemble smoothing; variance reduction is real signal. |
| 2 | R55h | Manual hybrid | Composite -0.0015 (blind) | Human-defensible 2-row swaps regressed; intuition without GT-side signal doesn't predict judge or nDCG. |
| 3 | R56 | Rule-based rerank | All 18 variants + ORACLE FAIL on dev | LR already weighs source-rank; overriding it with rules loses more HITs than it recovers DEMOTEDs. |
| 4 | R57/R57b (×3) | Structural features | h7 Δ -0.0085 / +0.0003 / -0.0060 | High LightGBM gain but CV5 nDCG regresses via per-fold overfitting; same-artist canary fires every time. |
| 5 | R58 | Specialist reranker | 28 configs all FAIL | Stage-2 LightGBM on LR top-50 cannot beat LR's own ordering; architecture path exhausted. |
| 6 | R59-C3 | Pool broadening | h7 Δ -0.0015 (Phase 2) | Pool admission real (+0.0596 recall) but frozen LR buries 91.7% of recovered candidates (48/580 convert). |
| 7 | R60 | Pool broadening + retrain | h7 Δ -0.0885 (fold-0) | Matched-pool LR training 57× worse than frozen LR. Destroys same-artist calibration (-0.142). |
| 8 | R61-C1 | Pool broadening | h7 Δ +0.00016, 4.6% churn | Count-based transitions add candidates but LR scores them near-randomly. 20 unique hits (target ≥30). |
| 9-11 | (R57b counts as 3) | (see row 4) | | Three separate feature configs tested independently. |

**Consecutive negatives:** 11 (R55 → R55h → R56 → R57b×3 → R58 → R59-C3 → R60 → R61-C1).

---

## 3. Mechanism Classes That Are CLOSED

### 3.1 Pool Broadening

**Experiments:** R59-C3 (learned admission), R60 (matched-pool retrain), R61-C1 (train-split transitions).

**The wall:** Any pool-broadening that introduces candidates without producing the 37 source-rank-style features the LR was trained on results in:
1. Near-random LR scores for new candidates (no source presence in A/B/C/D/F/ALS/R21/R54).
2. Top-1 churn driven by noise, not signal.
3. nDCG flat or negative — well-calibrated existing candidates displaced by mis-scored newcomers.

**Evidence chain:**
- C3 Phase 1: +0.0596 pool_hit (580 POOL_MISS recovered). Phase 2: only 48/580 (8.3%) convert via frozen LR. h7 Δ = -0.0015. (`docs/r59_candidates/c3_phase2_frozen_lr_result.md`)
- R60: Retraining LR on the broadened pool is WORSE. h7 Δ = -0.0885. Same-artist -0.142. Conversion rate still 8.3%. (`docs/r60_variant_a_fold0_result.md`)
- R61: Completely different candidate source (count-based transitions). Same outcome: 20/30 novelty, +0.00016 nDCG, 4.6% churn. (`docs/r61_c1_transition_probe_result.md`)

**Why retraining doesn't fix it:** The 8% conversion ceiling is a feature-space limitation, not a training-distribution issue. Candidates admitted without source-rank features are unscoreable by any LR in the 37-feature family. Matched training breaks same-artist calibration (a hard-won asset) without improving conversion.

**Cross-ref:** `feedback_pool_broadening_wall.md`, `feedback_matched_pool_training_falsified.md`

### 3.2 Structural Feature Additions

**Experiments:** R49C (duration, release_year), R57b (ISRC country, ISRC registrant, artist_id_match_history — 3 configs).

**Pattern:** Forensic analysis finds 30+ cases where "GT has pattern, top-1 doesn't." LightGBM reports high gain importance (3000+). But aggregate CV5 nDCG regresses via per-fold overfitting. Same-artist canary fires on every config.

| Feature | h7 Δ | net recovered/lost |
|---|---:|---:|
| +duration | -0.0035 | -3 |
| +release_year | +0.0002 | -4 |
| +ISRC country+registrant | -0.0085 | -16 |
| +artist_id_match_history | +0.0003 | -44 |
| +ISRC+artist_id combined | -0.0060 | -43 |

**Why it doesn't transfer:** Forensic count (pattern exists) ≠ predictive power (pattern reliably identifies GT over alternatives). The LR already captures artist/album/tag signal through existing features; categorical structural additions overfit to train-set patterns that don't generalize.

**Cross-ref:** `feedback_structural_features_exhausted.md`

### 3.3 Rule-Based Reranks / Specialist Rerankers

**Experiments:** R56 (source-rank protection, 18 variants + ORACLE), R58 (stage-2 LightGBM specialist, 4 configs × 7 betas = 28 evaluations).

**The lesson:** LR's top-50 ordering is well-calibrated. Any intervention that reshuffles the LR's top-20 — whether rule-based (R56) or learned (R58) — loses more HIT cases than it recovers DEMOTEDs.

**R56 ORACLE result:** Even with perfect knowledge of which candidates are true diff-artist GT (never deployable), the oracle rule at K=1 recovered 15 DEMOTEDs but lost 26 HITs. Net = -11. The LR is correctly ranking its top-20 candidates; the DEMOTED bucket cannot be recovered by reordering within the existing pool.

**R58:** All 28 specialist configs regress. Beta-weighted ensembles between LR and specialist uniformly lose. Architecture path (post-LR learned reranker) is closed.

**Cross-ref:** `feedback_lr_top50_calibrated.md`

### 3.4 Retriever Architecture Swaps

**Experiments:** R55 (single all-data BGE), R51 (cross-encoder, design-only), R31 (sparse sequence model, design-only).

**The lesson:** The 5-fold ensemble's smoothing is real signal. A single all-data model with identical architecture and training data produces nDCG -0.0067 on blind. Ensemble variance reduction > any single-model's point estimate.

**R55 numbers:**
| Metric | R54b (ensemble) | R55 (single) | Δ |
|---|---:|---:|---:|
| nDCG@20 | 0.4925 | 0.4858 | -0.0067 |
| Composite | 0.6106 | 0.6108 | +0.0002 |

Composite was flat only because LLM judge was +0.05 (noise range). nDCG regression is the real signal.

**Cross-ref:** `feedback_ensemble_smoothing.md`

### 3.5 Manual Hybrids

**Experiment:** R55h — 2 rows manually swapped where R55's picks "obviously" better aligned with user intent.

**Result:** Both swaps regressed. Composite -0.0015.

**Why:** Human intuition about "better recommendation" doesn't predict (a) the Gemini judge's scoring (it reads response in isolation), or (b) nDCG (no GT-side signal available to human). Manual row edits are systematically unreliable.

**Cross-ref:** `feedback_no_manual_row_edits.md`

### 3.6 Response Polish

**Experiment:** R54c targeted polish on R54b.

**Result:** Flat (0.6106 = 0.6106). LLM 4.70 held. LexDiv 0.8381 vs 0.8387 (noise).

**Ceiling diagnosis:** Track IDs are the nDCG bottleneck. LLM judge at 4.70 is at or near its discrimination threshold for response-quality variation. Further response work has near-zero expected lift.

**Cross-ref:** `feedback_response_polish_saturated.md`

### 3.7 Hard-Negative Mining (Pre-R54c, Confirmed Closed)

**Experiments:** R23, R23a.

**Result:** Hurt scores. Music neighbors are plausible recommendations (same artist/era/genre); pushing them away from the query embedding damages the local manifold the retriever relies on.

**Cross-ref:** `feedback_no_hard_negatives.md`

---

## 4. Why the Local Optimum Is Durable

The R54c system has converged to a local optimum that resists incremental perturbation:

1. **Source-rank features dominate.** The LR's top gain features are `rrf_rank_inv`, `popularity`, `recency_score`, `r54_cosine`, and per-source rank inversions. These features encode "how strongly do the retrieval sources agree this is a good candidate?" The LR has learned this signal precisely for the distribution that 8-source RRF produces.

2. **The LR is calibrated within its training distribution.** When the pool is the same shape as training (RRF@300 with the same 8 sources), the LR's relative ordering within top-50 is empirically correct — cheap rules lose and specialists lose (R56, R58). The calibration breaks only when you change the pool distribution (R59-C3, R60, R61).

3. **Pool-broadening hits a feature-space wall.** New candidates admitted from outside the 8 sources lack the features the LR needs to score them. The ~8% conversion ceiling holds across both frozen and retrained LR variants. The wall is in the feature space, not the training.

4. **Feature-broadening overfits on 8000 cases.** The dev set is 8000 cases with 1000 unique sessions. Per-fold structural-metadata splits are enough to cause overfitting even with deep-tree regularization (num_leaves=31). Five categorical features have failed this way.

5. **Ensemble smoothing is non-trivial to beat.** The 5-fold ensemble provides variance reduction that a single model cannot match on 80 blind queries. Any retriever swap must clear this smoothing benefit plus its own noise.

**The implication:** Incremental modifications within the same architecture (same pool, same features, same LR family, same ensemble structure) have exhausted their return. The system is at its basin's floor.

---

## 5. What COULD Break the Freeze

### 5.1 External Data (C4)

MusicBrainz / Spotify / Last.fm metadata could provide features the current 37 cannot capture (listening counts, genre taxonomies, release metadata richer than ISRC). This is the one remaining mechanism that hasn't been falsified.

**Status:** Exploratory research allowed (Phase 1-2 under `cache/r59/c4_experimental/`). Phase 3 (LR integration) requires explicit approval + organizer rules clarity.

**Why it might work:** External metadata could provide candidate-side features that differentiate POOL_MISS admits from noise, breaking the 8% conversion ceiling. It's a genuinely new feature class, not a rearrangement of existing source-rank signals.

### 5.2 Non-LR Scoring Family

The 37-feature LR is a gradient-boosted tree ranker. A fundamentally different scoring model — e.g., a cross-attention model that ingests raw query+candidate embeddings, or a session-conditional model with utterance-level features — could learn scoring functions the tree cannot represent.

**Caveat:** R51 cross-encoder was designed but the precedent of R58 (learned stage-2 specialist) failing suggests that anything operating on the same pool + same candidate features will face similar calibration issues. A new model class needs genuinely new input features (utterance text, raw embeddings, external metadata) to have a different operating point.

### 5.3 Blind-B Distribution Shift

Blind-B's data distribution may differ from Blind-A in ways that reopen closed paths:
- **POOL_MISS/UNREACHABLE fractions** might shift — if more GT falls within reach of the existing pool, the system performs better out-of-the-box; if less, pool-broadening becomes more important.
- **Session characteristics** might differ — longer histories, different genres, different user personas could make structural features more or less predictive.
- **The LR's calibration** is fitted to Blind-A's implicit distribution. Blind-B might have a sufficiently different candidate-quality landscape that specialists or features that overfit on Blind-A's dev actually generalize on Blind-B.

**Action:** Do NOT auto-port R54c. First reproduce the baseline on Blind-B dev, check bucket distributions, then decide which (if any) archived directions to revive. See `docs/blind_a_final_state.md §5`.

---

## 6. Anti-Patterns to Avoid

| # | Don't | Why | Cross-ref |
|---|---|---|---|
| 1 | Add categorical LR features because forensic count shows a pattern | Forensic count ≠ predictive. 5/5 have regressed via overfitting. | `feedback_structural_features_exhausted.md` |
| 2 | Build a post-LR rule that promotes source-strong candidates | LR already weighs source-rank. Overriding it loses HITs. | `feedback_lr_top50_calibrated.md` |
| 3 | Propose pool-broadening without answering "how will LR score new candidates?" | If frozen LR → 8% conversion. If retrain → destroys calibration. Feature-space wall. | `feedback_pool_broadening_wall.md` |
| 4 | Swap to a single all-data retriever "for simplicity" | Ensemble smoothing is real. nDCG -0.0067 on blind. | `feedback_ensemble_smoothing.md` |
| 5 | Manually edit 2-5 rows that "look wrong" | Human intuition doesn't predict judge or nDCG without GT signal. | `feedback_no_manual_row_edits.md` |
| 6 | Iterate on response text (style, opener, length) | Saturated at LLM 4.70 / LexDiv 0.84. Zero expected lift. | `feedback_response_polish_saturated.md` |
| 7 | Use hard-negative mining for retriever training | Music neighbors are plausible; pushing them away damages retrieval manifold. | `feedback_no_hard_negatives.md` |
| 8 | Submit to blind with dev h7 Δ < +0.010 (non-structural) | R41a passed +0.005 gate, dropped 0.023 on blind. Tag/noise features don't transfer. | `feedback_blind_gate.md` |
| 9 | Retrain LR on a changed pool distribution | R60: 57× worse than frozen LR. Breaks same-artist. Don't chase this. | `feedback_matched_pool_training_falsified.md` |
| 10 | Frame a new experiment as "one more conservative try" | The 11 negatives ARE the conservative tries. The mechanism space is explored. | `project_blind_a_freeze.md` |

---

## 7. Summary Statistics

| Metric | Value |
|---|---|
| Experiments run post-R54c | 11 |
| Experiments positive | 0 |
| Mechanism classes tested | 6 (+ hard-neg from pre-R54c) |
| Mechanism classes closed | All 6 (+ hard-neg) |
| Blind submissions burned | 2 (R55, R55h) — both regressed |
| Dev-only experiments (no blind burn) | 9 |
| Total configs evaluated | 18 (R56) + 3 (R57b) + 28 (R58) + 1 (C3) + 1 (R60) + 3 (R61) = 54+ |
| Best dev result of any post-R54c experiment | R57b artist_id: h7 Δ +0.0003 (with same-artist regression — FAIL) |
| Conversion ceiling for pool-broadening | ~8% under any LR family variant |
| Same-artist canary fires | Every structural feature, every pool-broadening retrain |

**The evidence base is not thin.** These are not 11 variants of the same idea — they span retriever architecture, feature engineering, rule-based reranking, learned reranking, pool admission (3 independent mechanisms), response polish, and manual overrides. The convergence of failure across orthogonal approaches is what makes the freeze decision durable.

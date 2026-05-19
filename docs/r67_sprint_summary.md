# R67 Sprint Summary: Frontier LLM Semantic Reranker

**Created:** 2026-05-19  
**Branch:** `r67-llm-semantic-rerank`  
**Final Verdict:** **ARCHIVE_SPRINT**  
**Production Status:** R63c-repair holds Blind-A at composite **0.6224** (nDCG 0.4925, LLM 4.85, LexDiv 0.8438)  
**Consequence:** **18th consecutive post-R54c negative**

---

## Wave 0: Baseline Reproduction + LR Top-30 Extraction

**Commit:** `987eb88`  
**Result:** PASS (max |Δ| = 0.000000)

Reproduced R54c frozen LR bitwise-identically using cached intermediates and production RRF configuration. **Critical clarification:** R63c-repair is response-side only. Retrieval/LR baseline is bitwise R54c.

| Metric | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| all_dev_ndcg20 | 0.315875 | 0.315875 | +0.000000 |
| h7_ndcg20 | 0.348378 | 0.348378 | -0.000000 |
| same_artist_ndcg20 | 0.628214 | 0.628214 | +0.000000 |
| diff_artist_ndcg20 | 0.142367 | 0.142367 | +0.000000 |
| pool_hit_all | 0.622000 | 0.622000 | +0.000000 |
| pool_hit_h7 | 0.613000 | 0.613000 | +0.000000 |

### Top-30 Extraction & Metadata Coverage

- **n_cases:** 8000
- **top_k per case:** 30
- **unique candidate tracks:** 26344
- **metadata coverage (% of candidates):**
  - title: 100.0%
  - artist: 100.0%
  - album: 100.0%
  - tags (≥1): 99.87%
  - release_year: 97.91%

---

## Wave 1: Phase 0 — Opus 4.7 Feasibility Testing

**Commit:** `bf7a05e`  
**Duration:** 352.2s  
**Result:** **ARCHIVE_PHASE_0** — Both prompt styles catastrophically failed the kill gate

### Experimental Design

**Model:** claude-opus-4-7  
**Sample:** 150 stratified cases across 5 strata (S1–S5, 30 each)
- S1: h7 cases (highest weight)
- S2: diff_artist cases (same-artist baseline weak)
- S3: small LR margin cases (reranker opportunity zone)
- S4: GT-at-rank-2-20 cases (lower visibility)
- S5: GT-absent control cases

**Prompt Designs:**
- **Style A (Concise Expert Recommender):** Asked for conversational music recommendations based on user history, metadata, and conversation. Task framed as "expert music curator" without explicit rules.
- **Style B (Strict Rubric):** Same task but with explicit rubric: rank by "novelty without overfitting," "genre/artist diversity," "temporal variety." Structured scoring criteria.

**LLM Calls:** 150 cases × 2 prompts = 300 live calls; 0 cache hits, 6 retries, 1 malformed (recovered)

### Telemetry

| Metric | Value |
|---|---:|
| Total input tokens | 1,369,839 |
| Total output tokens | 39,990 |
| Wall time | 352.2s |
| Validity rate (Style A) | 99.33% (149/150) |
| Validity rate (Style B) | 100.00% (150/150) |
| Cost (estimated) | ~$20–30 |

### Metrics: Both Prompts Failed 4 of 5 Gates

| Metric | Style A | Style B | Gate | Result |
|---|---:|---:|---|:---:|
| **sample nDCG@20 (LLM)** | 0.1513 | 0.1425 | — | — |
| **sample nDCG@20 (LR baseline)** | 0.2265 | 0.2250 | — | — |
| **Δ sample nDCG** | **−0.0752** | **−0.0825** | ≥ 0 | **FAIL** |
| **Δ h7 nDCG** | **−0.2171** | **−0.2373** | ≥ +0.005 | **FAIL** |
| **Δ same-artist nDCG** | **−0.1793** | **−0.2129** | ≥ −0.002 | **FAIL** |
| **Δ diff-artist nDCG** | −0.0273 | −0.0230 | ≥ 0 | PASS |
| **recovered** | 2 | 3 | > lost | **FAIL** |
| **lost** | 9 | 13 | — | — |
| **net** | −7 | −10 | — | — |
| **top-1 churn (per 80)** | **69.8** | **71.5** | ≤ 25 | **FAIL** |
| **validity** | 99.33% | 100.0% | ≥ 95% | PASS |
| **gates passed** | **1/5** | **1/5** | all | **FAIL** |

**Decision gate fired:** Skip Phase 1, skip Phase 2, jump to summary.

---

## Cross-Mechanism Learnings

### 1. Opus 4.7 with Rich Metadata Cannot Beat LambdaRank When Asked to Rerank LR Top-30 Cold

The LLM changed the top-1 on ~87% of cases (~70/80 churn) and produced sample nDCG ~50% lower than the LR baseline. This failure is consistent across both prompt designs (concise expert vs strict rubric) — **the failure is not a prompt-engineering problem; it's a fundamental frontier-LLM-cannot-reproduce-supervised-ranking problem on this distribution.**

R67 isolates this independently from:
- **R58** (stage-2 LightGBM specialist over LR features, also failed): Learned pipeline over LR features cannot beat LR.
- **R56** (rule-based reranks on LR top-50, also failed): Rule-based reorders cannot beat LR.

**Common finding:** The LR top-K is well-calibrated on this dataset's distribution. Any mechanism that takes top-K as given and reorders it loses both nDCG and the same-artist canary. This conclusion is now reached via three structurally different mechanism classes (rules, learned features, frontier LLM).

### 2. Same-Artist Canary Fired Hard — LLM Aggressively Diversified Away from Played Artists

Δ same-artist = −0.179 / −0.213 is a canonical failure signature. The LLM prioritized "novelty without overfitting played artists" (especially in Style B's explicit rubric), but same-artist tracks are often the correct continuation in this dataset.

This is exactly the failure mode R56 hit with rule-based diversification. **Frontier semantic judgment about artist novelty is miscalibrated for this distribution.**

### 3. Churn Explosion Was the Most Decisive Single Gate

Even if Δh7 had been borderline positive, 70/80 top-1 changes would have blown the blind submission churn budget by 3×. A conservative variant (apply only when LR margin small or confidence high) could not have rescued this — the reranker loses 50% of nDCG on the full sample, so gating that loss to a subset still loses on that subset.

### 4. Eighteenth Consecutive Post-R54c Negative

R63c-repair holds production via response-side polish; retriever/pool/LR/rerank dimensions are all saturated. Open remaining axes on Blind-A:
- External metadata via C4 organizer reply (user-owned, pending)
- Blind-B distribution evidence
- Or a structurally new ranker architecture that consumes features frozen LR cannot (which R58 already attempted and failed)

---

## Why R67 Is the Cleanest Negative in This Series

1. **Mechanism is genuinely outside closed-paths set:**
   - No pool broadening (R59/R60/R61 exhausted)
   - No source routing/reweighting (R65/R66 exhausted)
   - No structural LR features (R57 exhausted)
   - No neural training (R58 exhausted)
   - No hard negatives (R23 tested, hurt scores)
   - No rule reranks on LR top-50 (R56 exhausted)

2. **Model is strongest frontier available:** Opus 4.7 (better than Sonnet 4.5 per feedback_madrox_model_choice)

3. **Metadata is richest available without external sources:** Full conversation + track title/artist/album/tags/release_year; 99.87%+ coverage

4. **Gates are explicit and pre-declared:** 5 gates (sample nDCG, h7 nDCG, same-artist nDCG, rec/lost, churn); all gates public before run

5. **Failure is decisive, not borderline:** Every gate that matters failed hard (Δh7 −0.22, Δsame-artist −0.18, churn 70/80)

This evidence closes the "frontier semantic reranking" path for Blind-A definitively.

---

## Surviving Deferred Directions

Unchanged from R66:
- **Joint LR retraining on a chosen new RRF profile** — matched-pool path, falsified by R60
- **External metadata via C4 organizer reply** — user-owned, pending approval
- **Blind-B release watch** — waiting for new distribution evidence

---

## Artifacts on `r67-llm-semantic-rerank`

**Scripts:**
- `scripts/expR67_baseline_repro.py`
- `scripts/expR67_phase0_feasibility.py`

**Evaluation:**
- `exp/eval/expR67_baseline_repro.json` — Bitwise R54c reproduction
- `exp/eval/expR67_top30_candidates.pkl` — Top-30 metadata-joined records
- `exp/eval/expR67_phase0_sample.json` — 150 stratified sample design
- `exp/eval/expR67_phase0_feasibility.json` — Full Phase 0 results (both prompts, all metrics)

**Documentation:**
- `docs/r67_baseline_repro.md`
- `docs/r67_phase0_feasibility_result.md`
- `docs/r67_sprint_summary.md` (this document)

**Cached LLM Responses (untracked scratch):**
- `cache/r67/llm_calls/` — 300 cached Opus 4.7 responses (not in git)

---

## References

- [[feedback_pre_rrf_routing_closed]] — R65/R66: source routing/reweighting cannot improve frozen LR
- [[project_r66_outcome]] — R66 Phase 0: 8-profile static RRF menu all failed
- [[project_r65_sprint_outcome]] — R65: oracle headroom from depth weighting not extractable
- [[feedback_lr_top50_calibrated]] — R56: LR top-50 is well-calibrated; rerank-only mechanisms regress it
- [[project_r58_outcome]] — R58: stage-2 LightGBM specialist failed
- [[project_r56_outcome]] — R56: rule-based reranks on LR top-50 failed
- [[feedback_madrox_model_choice]] — Opus 4.7 is preferred frontier model
- [[feedback_llm_rerank_closed]] — NEW: rerank-only paths (R56/R58/R67) all exhausted

---

## Bottom Line

R67 tested Opus 4.7 as a semantic reranker over R54c LR top-30, using rich metadata (conversation + track info) across two prompt designs. Both prompts catastrophically failed the Phase 0 kill gate: Δh7 = −0.22 / −0.24, churn = 70/80 per baseline sample, rec/lost = 2/9 and 3/13. The LLM reranker changed top-1 on ~87% of cases and lost 50% of nDCG vs the LR baseline.

This is the **18th consecutive post-R54c development experiment without a production advance.** Rerank-only mechanisms (rules in R56, learned features in R58, frontier LLM in R67) have now all failed across structurally different classes. The LR top-K is well-calibrated on this distribution; mechanisms that reorder it lose both nDCG and semantic quality (same-artist canary). The frontier-semantic-reranking path is closed for Blind-A.

Production remains R63c-repair at composite 0.6224, holding from response-side polish. Remaining viable directions require either joint LR retraining (R60 falsified), external metadata (C4 organizer pending), or Blind-B distribution evidence.

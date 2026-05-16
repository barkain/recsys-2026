# Blind-A — Final State (frozen)

**Frozen 2026-05-16.** No further submissions or experiments on this task
until the competition changes, Blind-B arrives, or a new architectural
idea materializes.

## 1. Production submission

| Field | Value |
|---|---|
| Submission label | **R54c** (response polish on top of R54b track IDs) |
| Composite | **0.6106** |
| nDCG@20 | 0.4925 |
| Catalog Diversity | 0.0301 |
| LexDiv (Distinct-2) | 0.8381 |
| LLM judge | 4.70 |
| Leaderboard position | **#5** of public Blind-A scoreboard (as of 2026-05-16) |
| Submission zip | `exp/inference/blind_a/r54c_polish_submission.zip` |
| Submission JSON | `exp/inference/blind_a/r54c_polish_submission.json` |
| Metadata | `exp/inference/blind_a/r54c_polish_metadata.json` |
| Predecessor (equivalent score) | `exp/inference/blind_a/r54b_aligned_submission.json` |

R54c and R54b score identically at composite 0.6106; R54c is preferred as
the cleaner artifact (no prefix leaks, no boilerplate, no trailing
questions) but they are operationally interchangeable.

## 2. Submission score table (all Blind-A attempts after R54)

| Submission | Date | nDCG@20 | LexDiv | LLM | Composite | Δ vs R54c | Decision |
|---|---|---|---|---|---|---|---|
| R39 (prior production) | 2026-04-28 | 0.4798 | 0.8198 | 4.70 | 0.6024 | -0.0082 | Replaced by R54b |
| R54 exploratory | 2026-05-15 | 0.4925 | 0.8198 | 4.65 | 0.6050 | -0.0056 | Replaced by R54b |
| R54b aligned | 2026-05-15 | 0.4925 | 0.8387 | 4.70 | **0.6106** | baseline | Production |
| **R54c polish** | 2026-05-15 | 0.4925 | 0.8381 | 4.70 | **0.6106** | 0 | **Production** |
| R55 | 2026-05-16 | 0.4858 | 0.8368 | 4.75 | 0.6108 | +0.0002 | Archived (nDCG drop) |
| R55h hybrid | 2026-05-16 | 0.4894 | 0.8390 | 4.70 | 0.6091 | -0.0015 | Archived (regression) |

The R55 / R55h dips on the actual leaderboard confirmed the dev-side warning
that the cycle was saturating. Subsequent experiments (R56, R57, R57b, R58)
stayed dev-only and all archived as negative without burning a blind slot.

## 3. Failed post-R54c directions (all archived)

| # | Direction | Mechanism | Result |
|---|---|---|---|
| 1 | R55 | Single all-data BGE retriever (vs 5-fold ensemble) | -0.0067 nDCG on blind, flat composite |
| 2 | R55h | Manual 2-row hybrid (R55 picks on 2 sessions) | -0.0015 composite on blind |
| 3 | R56 | Rule-based source-rank protection (5 deployable variants + ORACLE, 3 K values = 18 configs) | All FAIL_REGRESS on dev. Even ORACLE (true diff_artist gating) loses net. |
| 4 | R57 | Structural-metadata forensic refresh on R39+R54 baseline | 9 patterns pass the count/coverage filters, but R49C precedent (negative duration & year implementations) disconfirms cheap implementation |
| 5 | R57b | New categorical LR features (ISRC country, ISRC registrant, artist_id_match_history) — 3 configs | All FAIL_REGRESS. High LightGBM gain importance; aggregate nDCG regresses; same-artist canary fires. |
| 6 | R58 | Learned second-stage LightGBM specialist on LR top-50 — 4 configs × 7 betas = 28 evaluations | All FAIL_REGRESS. Architecture path exhausted. |

Across these, **every beta>0 / every variant on a learned or rule-based
specialist regresses same-artist nDCG** (the canary), and aggregate h7 does
not clear the +0.005 exploratory gate, let alone the +0.010 production
gate. The pattern is consistent: the R39+R54+LambdaRank ranker's top-50
ordering is well-calibrated; cheap interventions cannot recover the
DEMOTED bucket.

For full details and per-experiment memories see:

- `project_r54_blind_transfer.md` (production result)
- `project_r55_outcome.md`
- `project_r56_outcome.md`
- `project_r57_outcome.md`
- `project_r58_outcome.md`
- `project_blind_a_freeze.md` (the freeze rationale)

Feedback memories that encode the lessons:

- `feedback_lr_top50_calibrated.md`
- `feedback_structural_features_exhausted.md`
- `feedback_ensemble_smoothing.md`
- `feedback_no_manual_row_edits.md`
- `feedback_no_hard_negatives.md`
- `feedback_response_polish_saturated.md`
- `feedback_retriever_swap_churn_gates.md`

## 4. Explicit freeze rule

**Until Blind-B arrives or a fundamentally different mechanism appears,
the following are NOT to be done on Blind-A:**

- New LR feature additions (categorical or continuous)
- New post-LR rule-based rerank layers
- New learned reranker / specialist variants (pairwise, MLP, etc.) without
  a new mechanism argument distinct from R58
- Manual row-level edits to R54c track IDs or responses
- Retriever-architecture swaps (single-model, fold-different, hard
  negatives, etc.) without a new mechanism argument
- Response-polish iterations (saturated at LLM 4.70 / LexDiv ~0.84)
- "One more conservative hybrid" — disproven by R55h
- New small experiments framed as "but this time it'll be different"

**Allowed during the freeze:**

- Infrastructure improvements (cache hygiene, replay scripts, doc updates)
- Pre-positioning for Blind-B (data review when available, dust off
  R12/R21/R54 caches as needed)
- Architectural design DOCS (no implementation) if a fundamentally new
  idea appears — must clear (a) different mechanism from R55-R58, (b)
  bounded test design, (c) explicit gates same as R56/R57b/R58.

## 5. What to do when Blind-B arrives

1. **Do NOT auto-port R54c to Blind-B.** The data distribution may have
   shifted. The retrieval and ranking stack should be re-evaluated on
   Blind-B dev/eval before submitting.

2. **First action:** rerun the equivalent of `expR55_post_refresh_decomp.py`
   on Blind-B's dev (if dev is provided) or analogous evaluation set.
   Reproduce R39+R54 baseline on Blind-B data. Verify the bucket
   distribution. If buckets shift materially (e.g. POOL_MISS or
   UNREACHABLE drops), the previously-archived directions may become
   viable again.

3. **Second action:** rebuild the R54 5-fold ensemble blind retrieval for
   Blind-B (analogous to `cache/r54_production/blind_r54_lists.json`).
   Existing R54 models can be re-used; only the blind retrieval needs
   to be regenerated against the Blind-B test queries.

4. **Third action:** rerun R54b/R54c-style submission build on Blind-B
   (drop-in via existing scripts; R54c response-polish hygiene applies).
   This becomes the Blind-B equivalent of R54c — the new baseline to
   defend.

5. **Only then** consider re-opening any of the archived post-R54c
   directions. Specifically, R56 / R57 / R58 might warrant a single
   refresh on Blind-B because:
   - LR's calibration may differ on the new data distribution
   - DEMOTED bucket size may differ
   - Structural patterns may be more discriminative

   These re-opens still need their own design + dev-eval phases. Do NOT
   port over the gates as-is without re-checking the bucket sizes.

6. **What probably stays archived:** R55-style retriever swaps and R55h-
   style manual hybrids. Their failure modes (ensemble smoothing matters;
   intuition without GT-side signal) are architectural and likely transfer
   to Blind-B.

## 6. Artifacts that should NOT be deleted during the freeze

These are the operational dependencies. Don't garbage-collect:

| Path | Why preserved |
|---|---|
| `exp/inference/blind_a/r54c_polish_submission.{json,zip}` | Production submission |
| `exp/inference/blind_a/r54b_aligned_submission.json` | Predecessor (equivalent score) |
| `cache/r54_production/blind_r54_lists.json` | R54 5-fold ensemble blind retrieval (production retriever output) |
| `cache/r54_phase3_lr_model.txt` | Trained LR model with R39+R54 features |
| `cache/r54_phase3_als.npz`, `cache/r54_phase3_payload_maps.pkl`, `cache/r54_phase3_track_pop.json` | LR supporting caches |
| `cache/r21_production/{model, track_embeddings.npy, track_ids.json, dev_r21_oof_lists.json}` | R21 retrieval (input to R39+R54 LR) |
| `exp/eval/_R12_all_turns_payload.pkl` | Dev payload (R12 preprocessing) |
| `cache/r54/phase2_full/oof_r54_lists.json` | R54 Phase 2 OOF (proxy for Phase 3, useful for any future dev forensics) |
| `cache/blind_a/source_cache.pkl` | R54-independent blind source cache (built once; useful for future ranker experiments) |
| `cache/r58/top50_dev.pkl` | Top-50 dev table with features per candidate (useful for any future top-K analysis) |

OK to delete if needed for disk space:

- `cache/r54_production/{model, track_embeddings.npy}` — R55 single-model artifacts (~580 MB total). R55 archived; recoverable from RunPod if needed.
- `cache/r55_compare/`, `cache/r58/` other than `top50_dev.pkl` — derived from training-fold runs and recomputable in minutes.

R54 Phase 3 OOF (`cache/r54/phase3_full/oof_r54_lists.json`, 142 MB) was deleted in the pre-R55 disk cleanup. Phase 2 OOF is the OOF-correct proxy and is preserved. Phase 3 OOF would only need regeneration if a post-Blind-B experiment specifically requires it.

## 7. Branch state

| Branch | Purpose | State |
|---|---|---|
| `main` | Production code through R54c | synced |
| `r55-production-r54` | All post-R54c experiment work (R55-R58, designs, docs, infra) | 15 commits ahead of main; pushed; archived as historical record |
| `r54c-response-polish` | R54c development branch | Merged into main, can be deleted later |
| `r54-second-gen-supervised-retriever` | R54 ensemble development branch | Merged into main |

**Decision on `r55-production-r54`:** keep as-is on origin. It is the
experiment record. Don't merge to main unless a future Blind-B revival
wants to incorporate the infrastructure (source cache, top-50 table) into
the production codebase. The scripts work standalone from the branch.

---

**Effective date: 2026-05-16. Freeze duration: indefinite, ends only on
Blind-B data availability or fundamentally new mechanism proposal.**

# R60: Matched-Pool LambdaRank Training

**Created:** 2026-05-16  
**Branch:** r59-mechanism-reset  
**Status:** Codex cross-check design patch  
**Implementation status:** Design only. No `.py` changes authorized by this document.

---

## 1. Motivation

R59 C3 proved that learned pool admission captured real signal that weighted RRF missed.

| Metric | Value | Source |
|---|---:|---|
| pool_hit@300 gain | +0.0596 | docs/r59_candidates/c3_diagnostic_result.md:24-27 |
| POOL_MISS recovered | 580 / 1163 | docs/r59_candidates/c3_diagnostic_result.md:27 |
| Previously-covered loss | 103 / 4976 (2.1%) | docs/r59_candidates/c3_diagnostic_result.md:28 |
| Net pool recovery | 477 | docs/r59_candidates/c3_diagnostic_result.md:29 |
| diff-artist pool_hit gain | +0.0902 | docs/r59_candidates/c3_diagnostic_result.md:38 |

C3 Phase 2 then showed why pool recovery alone is not enough. The learned pool put more GT tracks inside pool@300, but the frozen production LR mostly buried those newly admitted candidates.

| Metric | Value | Source |
|---|---:|---|
| h7 nDCG@20 delta | -0.00154 | docs/r59_candidates/c3_phase2_frozen_lr_result.md:18 |
| all-dev nDCG@20 delta | -0.00036 | docs/r59_candidates/c3_phase2_frozen_lr_result.md:19 |
| same-artist nDCG@20 delta | -0.00116 | docs/r59_candidates/c3_phase2_frozen_lr_result.md:22 |
| diff-artist nDCG@20 delta | +0.00008 | docs/r59_candidates/c3_phase2_frozen_lr_result.md:23 |
| POOL_MISS converted to frozen-LR top-20 | 48 / 580 (8.28%) | docs/r59_candidates/c3_phase2_frozen_lr_result.md:33-36 |
| POOL_MISS admitted but buried 21-300 | 532 / 580 (91.7%) | docs/r59_candidates/c3_phase2_frozen_lr_result.md:35 |

The working hypothesis is distribution mismatch:

- The frozen LR model was trained on RRF-shaped candidate pools.
- Learned-admission candidates have lower `rrf_rank_inv` / `weighted_rrf_rank_inv`, sparser source agreement, and different R21/R54/ALS evidence patterns.
- The frozen LR therefore treats many recovered GT candidates as weak even after they enter pool@300.

R60 is the smallest falsifiable test of that hypothesis: retrain LambdaRank from scratch on the same learned-admission pool distribution it will see at inference. If this still fails to convert recovered POOL_MISS cases, the C3 admission path is exhausted. If it works, it validates matched pool/ranker training as the missing mechanism rather than another admission-model tweak.

This is not another pool model and not incremental post-LR smoothing. It changes the training distribution of the core ranker. That is why implementation is gated to a fold-0 diagnostic first.

---

## 2. Scope

**Environment**

- Dev-only first; no Blind-A or Blind-B access during the diagnostic.
- Branch: `r59-mechanism-reset`.
- Use `grouped_session_folds(seed=0)` as in `scripts/expS2_lambdarank_grouped.py:44-60`, called from `scripts/expR54_phase3_full5fold_integration.py:255` and `scripts/expR59_c3_pool_admission_diagnostic.py:512`.
- Use OOF R21 and R54 source lists on dev.
- Do not commit from this design task.

**What R60 does**

- Rebuild learned-admission pool@300 OOF using the fixed C3 Phase 1 admission model.
- Train LambdaRank from scratch on learned-admission pools.
- Evaluate against one scoreboard family: C3 Phase 2 frozen-production-LR RRF-pool baseline.
- Report h7, all-dev, same-artist, diff-artist, conversion, top-20 recovered/lost, and churn.
- Test Variant A first. Variants B, C, and full CV5 are not approved until Variant A fold 0 passes and the user separately approves.

**What R60 does not do**

- No blind code, no submission packaging, no blind-list churn generation until dev gates pass and design review approves.
- No hard-negative mining. Memory `feedback_no_hard_negatives` says R23/R23a tested it and hurt scores.
- No new raw structural metadata features. Memory `feedback_structural_features_exhausted` says categorical additions already failed.
- No ensemble architecture change. Memory `feedback_ensemble_smoothing` says the 5-fold ensemble is the safer retriever submission pattern.
- No response-side work. Memory `feedback_response_polish_saturated` says response polishing is saturated.
- No production-model initialization for fold validation. Memory `feedback_clean_fold0` requires clean fold-specific training.
- No admission-model retuning. R60 tests conversion, not more admission optimization.

**Approval boundary**

Only Variant A fold 0 is approved. The rest of this document is design inventory and future gating logic, not implementation authorization.

---

## 3. Feature Count Audit

The current Sonnet draft states that the R54c LR path has 36 features because `scripts/expR55_post_refresh_decomp.py:64-70` ends with:

```python
FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [...]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM + FEAT_R54  # 28 + 5 + 3 = 36
```

That comment is stale. Reading the imported definitions and the production model file gives 37 features:

- `scripts/expS2_lambdarank.py:59-79` defines `FEATURE_NAMES_LR` with 23 features.
- `scripts/expS2_lr_v2.py:39-44` appends 4 V2 features, making `FEATURE_NAMES_V2` length 27.
- `scripts/expR55_post_refresh_decomp.py:64` appends `r21_rank_inv` and `r21_presence`, making `FEAT_BASE` length 29.
- `scripts/expR55_post_refresh_decomp.py:65-69` adds 5 album features and 3 R54 features.
- Therefore the code-resolved R55/R54-style feature count is `29 + 5 + 3 = 37`.
- `cache/r54_phase3_lr_model.txt:8` confirms the production model has `max_feature_idx=36` and 37 feature names.
- `scripts/expR54_phase3_blind_submission.py:10` also documents the production train phase as 37 features.

**R60 must use the code-resolved 37-feature contract unless the user explicitly approves dropping one feature.** A 36-feature implementation would not match the production R54c model or the C3 Phase 2 frozen-LR scoring path.

### Code-resolved 37-feature base set

The following feature order is the implementation contract for Variant A. The names differ slightly between `expR55_post_refresh_decomp.py` imports and the production model text file, but the row positions align.

| Position | expR55/import-chain name | Production-model name | Source |
|---:|---|---|---|
| 1 | `rrf_rank_inv` | `rrf_rank_inv` | `FEATURE_NAMES_LR` |
| 2 | `same_artist` | `last_artist_match` | `FEATURE_NAMES_LR` |
| 3 | `tag_jaccard` | `last_tag_jaccard` | `FEATURE_NAMES_LR` |
| 4 | `artist_tok_overlap` | `query_artist_tok_overlap` | `FEATURE_NAMES_LR` |
| 5 | `title_tok_overlap` | `query_title_tok_overlap` | `FEATURE_NAMES_LR` |
| 6 | `meta_tok_overlap` | `query_meta_tok_overlap` | `FEATURE_NAMES_LR` |
| 7 | `already_played` | `is_played` | `FEATURE_NAMES_LR` |
| 8 | `recency_weighted_meta` | `recency_score` | `FEATURE_NAMES_LR` |
| 9 | `rank_A` | `src_a_rank_inv` | `FEATURE_NAMES_LR` |
| 10 | `rank_B` | `src_b_rank_inv` | `FEATURE_NAMES_LR` |
| 11 | `rank_C` | `src_c_rank_inv` | `FEATURE_NAMES_LR` |
| 12 | `rank_D` | `src_d_rank_inv` | `FEATURE_NAMES_LR` |
| 13 | `rank_F` | `src_f_rank_inv` | `FEATURE_NAMES_LR` |
| 14 | `rank_ALS` | `src_als_rank_inv` | `FEATURE_NAMES_LR` |
| 15 | `in_A` | `src_a_pres` | `FEATURE_NAMES_LR` |
| 16 | `in_B` | `src_b_pres` | `FEATURE_NAMES_LR` |
| 17 | `in_C` | `src_c_pres` | `FEATURE_NAMES_LR` |
| 18 | `in_D` | `src_d_pres` | `FEATURE_NAMES_LR` |
| 19 | `in_F` | `src_f_pres` | `FEATURE_NAMES_LR` |
| 20 | `in_ALS` | `src_als_pres` | `FEATURE_NAMES_LR` |
| 21 | `n_sources` | `n_sources` | `FEATURE_NAMES_LR` |
| 22 | `als_score` | `als_dot` | `FEATURE_NAMES_LR` |
| 23 | `n_hist` | `n_history` | `FEATURE_NAMES_LR` |
| 24 | `track_popularity` | `popularity` | `FEATURE_NAMES_V2` |
| 25 | `artist_popularity` | `pool_artist_frac` | `FEATURE_NAMES_V2` row semantics in expR55:240 |
| 26 | `pool_artist_count` | `pool_artist_count` | `FEATURE_NAMES_V2` |
| 27 | `pool_source_agreement` | `source_count_v2` | `FEATURE_NAMES_V2` |
| 28 | `r21_rank_inv` | `r21_rank_inv` | `FEAT_BASE` addition |
| 29 | `r21_presence` | `r21_presence` | `FEAT_BASE` addition |
| 30 | `same_album_last1` | `same_album_last1` | `FEAT_ALBUM` |
| 31 | `same_album_last3` | `same_album_last3` | `FEAT_ALBUM` |
| 32 | `same_album_any` | `same_album_any` | `FEAT_ALBUM` |
| 33 | `album_history_count` | `album_history_count` | `FEAT_ALBUM` |
| 34 | `pool_same_album_count` | `pool_same_album_count` | `FEAT_ALBUM` |
| 35 | `r54_rank_inv` | `r54_rank_inv` | `FEAT_R54` |
| 36 | `r54_presence` | `r54_presence` | `FEAT_R54` |
| 37 | `r54_cosine` | `r54_cosine` | `FEAT_R54` |

### Variant feature counts

| Variant | Feature count | Rationale |
|---|---:|---|
| A | 37 | Existing R54c/R55-compatible LR feature contract, no admission score features |
| B | 39 | Variant A plus `admission_score` and `admission_rank_inv` |
| C | 37 or 38 | Variant A on hybrid pool, optionally plus `is_rrf_protected` |

Implementation must assert the count before model training. If `len(feature_names) != expected_count`, abort before any result interpretation.

---

## 4. Variants

### Variant A: Learned-Admission Pool + Existing 37-Feature LR

**Approval:** Variant A fold 0 only is approved.

**Pool construction**

- For each dev case, build source-union candidates from A/B/C/D/F/ALS/R21/R54 OOF sources.
- Score candidates with the fixed C3 admission model.
- Take learned-admission top 300 as the LR candidate pool.
- Reuse C3 pool-admission code:
  - `SOURCE_NAMES`: `scripts/expR59_c3_pool_admission_diagnostic.py:48`
  - weighted RRF constants: `scripts/expR59_c3_pool_admission_diagnostic.py:52-56`
  - candidate index/source-union builder: `scripts/expR59_c3_pool_admission_diagnostic.py:244-311`
  - feature builder: `scripts/expR59_c3_pool_admission_diagnostic.py:385-489`
  - CV5 admission trainer: `scripts/expR59_c3_pool_admission_diagnostic.py:504-619`

**Admission model feature count**

The Sonnet draft says 115 admission features. The current code resolves to 99:

- 8 sources in `SOURCE_NAMES` times 9 source-rank fields = 72 features.
- 27 cross-source/session/RRF/score fields in `build_feature_names()` = 27 features.
- Total = 99 features.
- `exp/eval/expR59_c3_pool_admission.json:10-110` confirms 99 feature names in the saved artifact.

This does not change the R60 LR feature count. It only corrects the description of the admission model.

**LR feature set**

- 37 features from the code-resolved R54c/R55 path described in Section 3.
- No `admission_score`.
- No `admission_rank_inv`.
- Keep rank-derived LR features tied to full weighted-RRF source-union rank where C3 already identified that calibration requirement:
  - `scripts/expR59_c3_phase2_frozen_lr_conversion.py:145-151`
  - `scripts/expR59_c3_phase2_frozen_lr_conversion.py:188-193`
  - `scripts/expR59_c3_phase2_frozen_lr_conversion.py:732-746`

**Training procedure**

- Train LightGBM LambdaRank from scratch.
- Use grouped-session folds, not row-random CV.
- Use the R54 integration training pattern:
  - fold split: `scripts/expR54_phase3_full5fold_integration.py:255`
  - train/val row build: `scripts/expR54_phase3_full5fold_integration.py:337-354`
  - LightGBM params: `scripts/expR54_phase3_full5fold_integration.py:359-363`
  - top-20 reconstruction and nDCG: `scripts/expR54_phase3_full5fold_integration.py:374-381`
- Hyperparameters:

```text
objective: lambdarank
metric: ndcg
eval_at: [20]
num_leaves: 31
learning_rate: 0.05
min_data_in_leaf: 10
seed: 0
num_boost_round: 300
```

**Expected outcome**

- If C3 Phase 2 failed because the frozen LR was calibrated to the wrong pool distribution, retraining on learned pools should lift conversion materially above 8.28%.
- Fold-0 target: conversion >= 15%, top-20 net positive, h7 delta nonnegative.
- Full-CV exploratory target, if later approved: conversion >= 15% and h7 delta >= +0.005.
- Full-CV production target, if later approved: conversion >= 25% and h7 delta >= +0.010.

### Variant B: Variant A + Admission Score/Rank

**Approval:** Not approved. Design inventory only.

**Pool construction**

- Identical to Variant A.

**Feature set**

- 37 Variant A features.
- Add two OOF stacking features:
  - `admission_score`: raw C3 LightGBM LambdaRank admission score.
  - `admission_rank_inv`: reciprocal admission rank within source-union candidates, or 0 if missing.
- Total: 39 features.

**Risk**

This is stacking. It is only acceptable if the admission scores used for LR validation are generated OOF from a model that did not train on the held-out case. Do not generate admission scores from an all-dev admission model for dev LR training or validation.

### Variant C: Hybrid RRF-Protected Pool + Learned Fill

**Approval:** Not approved. Design inventory only.

**Pool construction**

For each case:

1. Compute full weighted RRF over A/B/C/D/F/ALS/R21/R54 using the existing weights.
2. Protect top N weighted-RRF candidates, with N in `{50, 100, 150, 200}`.
3. Score remaining source-union candidates with the learned-admission model.
4. Fill to pool@300 with the highest admission scores not already protected.
5. Preserve full weighted-RRF rank features, not hybrid-pool rank features.

**Feature set**

- 37 Variant A features.
- Optional `is_rrf_protected` binary indicator.
- Total: 37 or 38 features.

**When to consider it**

Only consider Variant C if Variant A shows conversion signal but churn is close to the hard stop. Variant C is a churn-control fallback, not the first hypothesis test.

---

## 5. Metrics

### Scoreboard decision

Use one scoreboard family for all R60 gating: the **C3 Phase 2 RRF-pool frozen-production-LR baseline** from `docs/r59_candidates/c3_phase2_frozen_lr_result.md`.

Baseline values:

| Split | n | C3 Phase 2 RRF-pool frozen-LR nDCG@20 |
|---|---:|---:|
| all-dev | 8000 | 0.31588 |
| h7 | 1000 | 0.34838 |
| same-artist | 2857 | 0.62821 |
| diff-artist | 5143 | 0.14237 |

Conversion baseline:

| Metric | Baseline |
|---|---:|
| POOL_MISS admitted by learned pool | 580 |
| POOL_MISS converted to frozen-LR top-20 | 48 |
| POOL_MISS buried 21-300 | 532 |
| Convert rate | 8.28% |

### Why not mix in the R55 decomp metrics?

There are two metric families in the artifacts:

| Family | all-dev nDCG@20 | h7 nDCG@20 | Computation |
|---|---:|---:|---|
| R55 decomp / OOF baseline | 0.22460 | 0.24511 | CV5/fold-derived OOF baseline from `expR55_post_refresh_decomp.json`, reproduced from `cache/r58/top50_dev.pkl` in C3 Phase 2 |
| C3 Phase 2 frozen-production-LR scoreboard | 0.31588 | 0.34838 | Frozen production LR model `cache/r54_phase3_lr_model.txt` scores RRF and learned pools |

The difference is not nDCG horizon, normalization, or label scheme:

- Both use a single positive GT label per query.
- Both compute `1 / log2(rank + 1)` for 1-indexed final rank if rank <= K, else 0.
- Both average per-case values over the selected split.
- C3 Phase 2 also reports @10 and @7, but the chosen gate is @20.

The actual difference is the scoring protocol and ranker source:

- R55 decomp trains fold-specific LambdaRank models and records OOF ranks. See `scripts/expR55_post_refresh_decomp.py:264-306` and metric aggregation at `scripts/expR55_post_refresh_decomp.py:309-326`.
- C3 Phase 2 first reproduces the R55/R58 OOF baseline within epsilon using `verify_oof_baseline_reproduction()` at `scripts/expR59_c3_phase2_frozen_lr_conversion.py:80-128`.
- C3 Phase 2 then loads the frozen production LR model from `cache/r54_phase3_lr_model.txt` and scores RRF vs learned pools without retraining. See `scripts/expR59_c3_phase2_frozen_lr_conversion.py:635-647` and `scripts/expR59_c3_phase2_frozen_lr_conversion.py:682-746`.
- C3 Phase 2 same/diff splits are all-dev same/diff, not h7-only same/diff. See `scripts/expR59_c3_phase2_frozen_lr_conversion.py:321-381`.

The C3 Phase 2 family is the right direct comparator for R60 because it is the family that exposed the 48/580 conversion failure. The caveat is that it uses a frozen production model trained on all dev, so it is not an OOF-clean model-quality estimate. R60 results must therefore begin by reproducing this chosen baseline exactly before interpreting deltas.

### Primary metrics

| Metric | Definition | Gate reference |
|---|---|---|
| h7 nDCG@20 | Mean nDCG@20 on 1000 h7 cases | Delta vs 0.34838 |
| all-dev nDCG@20 | Mean nDCG@20 on all 8000 dev cases | Delta vs 0.31588 |
| same-artist nDCG@20 | Mean nDCG@20 on 2857 same-artist cases | Delta vs 0.62821 |
| diff-artist nDCG@20 | Mean nDCG@20 on 5143 diff-artist cases | Delta vs 0.14237 |
| top-20 recovered | GT absent from baseline top-20 but present in R60 top-20 | Positive net required |
| top-20 lost | GT present in baseline top-20 but absent from R60 top-20 | Positive net required |
| POOL_MISS conversion | Recovered C3 POOL_MISS GT that reaches R60 top-20 | Must beat 8.28% materially |

### Churn metrics

All churn thresholds use blind-equivalent normalization:

```text
top1_blind_eq = (top1_changed_count / split_n) * 80
```

Apply churn gates separately to all-dev, h7, same-artist, and diff-artist splits.

| Split | top-1 soft gate | top-1 hard stop | top-20 overlap gate |
|---|---:|---:|---:|
| all-dev | blind_eq < 25/80 | blind_eq > 35/80 archives or requires explicit override | mean >= 14/20 |
| h7 | blind_eq < 25/80 | blind_eq > 35/80 archives or requires explicit override | mean >= 14/20 |
| same-artist | blind_eq < 25/80 | blind_eq > 35/80 archives or requires explicit override | mean >= 14/20 |
| diff-artist | blind_eq < 25/80 | blind_eq > 35/80 archives or requires explicit override | mean >= 14/20 |

Do not use raw thresholds such as "2500/8000 changed rows." Raw counts can be reported as diagnostics, but all gates are blind-equivalent.

The baseline for dev churn is the C3 Phase 2 RRF-pool frozen-LR top-20, because that is the chosen scoreboard. If a blind path is ever approved, blind churn must be measured against current production blind lists as required by memory `feedback_retriever_swap_churn_gates`.

---

## 6. Gates

### Baseline Reproduction Gate

This is the first implementation step, before Variant A training:

- Reproduce the chosen C3 Phase 2 RRF-pool frozen-LR baseline within `epsilon = 0.0005`.
- Required values:
  - all-dev nDCG@20: 0.31588
  - h7 nDCG@20: 0.34838
  - same-artist nDCG@20: 0.62821
  - diff-artist nDCG@20: 0.14237
  - pool_hit@300 RRF: 0.62200
- If any chosen baseline metric misses by more than epsilon, abort and debug. Do not train Variant A.

### Fold-0 Diagnostic Gate

Fold 0 is a falsifier, not a promotion decision.

Proceed from fold 0 to full CV5 Variant A only if all hold:

| Criterion | Threshold |
|---|---:|
| Baseline reproduction | within epsilon=0.0005 |
| Fold-0 h7 nDCG@20 delta | >= +0.003 vs fold-0 slice of chosen baseline |
| Fold-0 same-artist delta | >= -0.002 vs fold-0 slice of chosen baseline |
| Fold-0 top-20 net | recovered > lost |
| Fold-0 POOL_MISS conversion | >= 15% |
| Fold-0 top1_blind_eq, each split with nontrivial n | < 35/80 |
| Fold-0 top-20 overlap mean, each split with nontrivial n | >= 14/20 |
| User approval | explicit separate approval for full CV5 |

Archive immediately from fold 0 if any hold:

| Criterion | Archive trigger |
|---|---:|
| h7 delta | < 0.000 |
| same-artist delta | < -0.002 |
| top-20 net | < 0 |
| POOL_MISS conversion | <= 10% |
| top1_blind_eq on any tracked split | > 35/80 |
| top-20 overlap mean on any tracked split | < 14/20 |

### Full CV5 Exploratory Gate

Full CV5 is not currently approved. If approved later, Variant A must pass:

| Criterion | Threshold |
|---|---:|
| h7 nDCG@20 delta | >= +0.005 vs 0.34838 |
| all-dev nDCG@20 delta | >= 0.000 vs 0.31588 |
| same-artist delta | >= -0.002 vs 0.62821 |
| diff-artist delta | >= 0.000 vs 0.14237 |
| top-20 net | recovered > lost |
| POOL_MISS conversion | >= 15% |
| top1_blind_eq, all four splits | < 35/80 |
| top-20 overlap mean, all four splits | >= 14/20 |

### Production-Candidate Gate

Passing this gate does not authorize a blind path. It authorizes design review only.

| Criterion | Threshold |
|---|---:|
| h7 nDCG@20 delta | >= +0.010 vs 0.34838 |
| all-dev nDCG@20 delta | >= 0.000 vs 0.31588 |
| same-artist delta | >= -0.002 vs 0.62821 |
| diff-artist delta | >= 0.000 vs 0.14237 |
| top-20 net | recovered > lost |
| POOL_MISS conversion | >= 25% |
| top1_blind_eq, all four splits | < 25/80 preferred, never > 35/80 |
| top-20 overlap mean, all four splits | >= 14/20 |

---

## 7. Risks

### The 37-vs-36 Feature Count Mismatch

**Risk:** A 36-feature implementation would silently test a different ranker from R54c/C3 Phase 2.

**Evidence:** `expR55_post_refresh_decomp.py:70` says `28 + 5 + 3 = 36`, but the import chain and `cache/r54_phase3_lr_model.txt` show 37. The extra feature is the source-agreement duplicate position (`pool_source_agreement` / `source_count_v2`).

**Mitigation:**

- Variant A uses 37.
- Add a pre-training assertion on feature count and feature names.
- If a 36-feature variant is desired, create a separate ablation design and do not call it R54c-compatible.

### Comparing OOF Retraining Against Frozen Production Scoring

**Risk:** C3 Phase 2's 0.31588/0.34838 scoreboard is produced by a frozen production model, not a clean fold-trained LR. A fold-trained R60 model may be penalized against an optimistic baseline.

**Mitigation:**

- Use C3 Phase 2 because it is the direct conversion-failure family.
- Require baseline reproduction before training.
- Report a secondary OOF-clean RRF-pool retrained baseline as diagnostic context only, not as the gate baseline, if later implementation time allows.
- Never mix the 0.22460/0.24511 R55 OOF family into primary R60 gates.

### Overfit to Admission-Model Idiosyncrasies

**Risk:** Matched-pool training may teach LR to trust admission-model artifacts that do not transfer to blind.

**Mitigation:**

- Variant A excludes `admission_score`.
- Compare feature importance to the R54c baseline.
- Enforce churn gates on all-dev, h7, same-artist, and diff-artist.
- Keep B as a later explicit stacking experiment only.

### Same-Artist Canary

**Risk:** same-artist is high-scoring and easy to damage. C3 Phase 2 already regressed same-artist by -0.00116.

**Mitigation:**

- Same-artist delta < -0.002 is an archive trigger.
- Report same-artist recovered/lost and churn separately.
- Do not hide same-artist regression behind diff-artist gains.

### Churn and Blind Transfer

**Risk:** A candidate with local lift can still be too disruptive to production lists.

**Mitigation:**

- Use blind-equivalent normalization for dev churn.
- Apply gates per split, not just all-dev/h7.
- If a blind path is ever authorized, compare against current production blind lists using memory `feedback_retriever_swap_churn_gates`.

### Tenth Post-R54c Negative

**Risk:** R60 would be another post-R54c negative if it fails. Memory `project_r59_c3_outcome` says C3 was already the 9th consecutive negative.

**Mitigation:**

- Run only the smallest falsifiable diagnostic first.
- Archive the C3 admission path if matched training does not improve conversion.
- Do not continue into B/C unless A provides a clean signal and the user separately approves.

---

## 8. Smallest Falsifiable Diagnostic

### Implementation Authorization

User approval scope (2026-05-16): **Variant A fold-0 diagnostic only.**

Variants B, C, and full CV5 are explicitly gated on:

1. Variant A fold-0 pass.
2. Separate user approval.

This section authorizes design and a future fold-0 Variant A implementation plan only. It does not authorize full CV5, B/C, blind packaging, or any `.py` work in this cross-check task.

### Fold-0 Procedure

1. Load dev payload and OOF sources:
   - `exp/eval/_R12_all_turns_payload.pkl`
   - `cache/r21_production/dev_r21_oof_lists.json`
   - `cache/r54/phase2_full/oof_r54_lists.json`
   - `cache/r54_phase3_als.npz`
2. Build fold split with `grouped_session_folds(seed=0)`.
3. Reproduce the chosen C3 Phase 2 RRF-pool frozen-LR baseline within epsilon=0.0005.
4. Train the C3 admission model on folds 1-4 and score fold 0.
5. Build fold-0 learned-admission pool@300.
6. Train the 37-feature LR from scratch on folds 1-4 learned pools.
7. Predict fold 0 learned pools.
8. Compute all fold-0 gate metrics against the fold-0 slice of the chosen C3 Phase 2 baseline.

### Metrics to Report for Fold 0

- Feature-name list and count; expected Variant A count = 37.
- Baseline reproduction table with abs deltas.
- Fold-0 split sizes for all-dev fold slice, h7, same-artist, diff-artist.
- Fold-0 nDCG@20 for all four splits.
- Fold-0 deltas vs C3 Phase 2 fold slice.
- POOL_MISS admitted, converted top-20, buried 21-300, conversion rate.
- top-20 recovered, lost, and net.
- top1_blind_eq per split.
- top-20 overlap mean and median per split.
- Top-15 feature importances by gain.

### Reproducibility Checklist

Before interpreting any R60 number, the implementer must verify:

| Check | How to verify | Pass criterion |
|---|---|---:|
| Instance did not touch blind | Report `dev_only=True`, `blind_access=False` | Required |
| Feature contract | Print exact feature names and count | Variant A = 37 |
| C3 admission feature count | Print `len(c3.FEATURE_NAMES)` | 99 |
| RRF pool baseline | Recompute pool_hit@300 | 0.62200 +/- 0.0005 |
| Chosen nDCG baseline | Reproduce C3 Phase 2 RRF-pool frozen-LR values | all chosen splits within 0.0005 |
| Fold identity | Print fold 0 case count and h7/same/diff counts | Deterministic under seed 0 |
| Clean fold training | Verify LR is trained only on folds 1-4 | No production init |
| Admission OOF discipline | Fold-0 admission scores come from folds 1-4 admission model | No all-dev admission model |
| Churn formula | Print `top1_changed`, `split_n`, and `(count/split_n)*80` | blind-eq used for gates |
| Top-20 overlap | Report mean and median per split | mean >= 14/20 |

If any checklist item fails, abort and debug before proceeding.

### Falsifier

The matched-training hypothesis is false enough to archive if fold 0 still buries nearly all newly admitted POOL_MISS GTs:

- POOL_MISS conversion <= 10%, or
- h7 delta < 0.000, or
- top-20 net < 0, or
- same-artist delta < -0.002, or
- churn exceeds hard stop.

If fold 0 passes, request explicit user approval for full CV5 Variant A. Do not start full CV5 automatically.

---

## 9. Stop Conditions / No-Go Signals

**Abort before training Variant A if:**

- Chosen C3 Phase 2 baseline cannot be reproduced within epsilon=0.0005.
- Feature count does not match the expected Variant A count of 37.
- C3 admission feature count is not 99.
- Fold split differs from `grouped_session_folds(seed=0)`.
- Any source list accidentally uses production full-data retrieval on dev instead of OOF dev lists.

**Abort after fold 0 if:**

- h7 delta < 0.000.
- same-artist delta < -0.002.
- POOL_MISS conversion <= 10%.
- top-20 net < 0.
- Any tracked split has top1_blind_eq > 35/80.
- Any tracked split has top-20 overlap mean < 14/20.

**Archive after full CV5, if later approved, if:**

- Variant A fails the exploratory gate.
- Same-artist canary trips.
- Conversion remains <= 10%.
- Churn hard stop trips on any tracked split.

**Escalate to design review, not blind implementation, if:**

- Best approved variant passes exploratory but not production gate.
- h7 lift is between +0.005 and +0.010.
- Churn is under hard stop but above soft target.

**Permanently archive C3 admission path if:**

- Variant A full CV5 fails after a fold-0 pass, or
- A later approved B/C fails to improve the tradeoff.

---

## 10. What R60 Is Not

### Not a hard-negative mining run

R60 keeps the same single-positive LambdaRank label scheme: GT is 1.0, all other candidates in the query group are 0.0. It does not treat admission-rejected candidates as hard negatives.

### Not a new structural-feature push

Variant A uses the existing 37-feature R54c-compatible contract. Variant B adds only admission-derived stacking features if later approved. No artist_id, ISRC, MusicBrainz, tag, or catalog metadata features are added here.

### Not an ensemble change

R60 uses the same grouped-session fold discipline. It does not replace the ensemble strategy with a single all-data model for dev validation.

### Not response-side work

R60 is pure retrieval/ranking work. It does not alter response generation, LexDiv tuning, or LLM prompting.

### Not a production-model initialization

Do not warm-start from `cache/r54_phase3_lr_model.txt`. The frozen model is allowed only for reproducing the chosen C3 Phase 2 baseline.

### Not blind-path work

No Blind-A/Blind-B access, packaging, or submission work is authorized by this diagnostic. A production-candidate dev result only authorizes design review.

### Not an admission-model tuning run

The C3 admission model is fixed:

- 99 features from `build_feature_names()`.
- LightGBM LambdaRank.
- `eval_at=[300]`.
- `num_boost_round=120`.
- Grouped-session CV5.

R60 tests whether a matched LR can convert the admitted candidates.

---

## Appendix: Key File References

| Component | File | Lines | Notes |
|---|---|---:|---|
| R55 feature composition | `scripts/expR55_post_refresh_decomp.py` | 64-70 | Stale comment says 36; import chain resolves to 37 |
| Base LR feature list | `scripts/expS2_lambdarank.py` | 59-79 | 23 features |
| V2 additions | `scripts/expS2_lr_v2.py` | 39-44 | +4 features |
| R54 production feature names | `scripts/expR54_phase3_blind_submission.py` | 75-94 | 37 production features |
| Frozen production model feature names | `cache/r54_phase3_lr_model.txt` | 8 | `max_feature_idx=36`, 37 names |
| R55 feature builder | `scripts/expR55_post_refresh_decomp.py` | 210-256 | Candidate row fill |
| R55 OOF LR training | `scripts/expR55_post_refresh_decomp.py` | 264-306 | Fold-trained LR and OOF rank capture |
| R55 nDCG aggregation | `scripts/expR55_post_refresh_decomp.py` | 309-326 | 0.24511 h7, 0.22460 all-dev family |
| C3 admission source constants | `scripts/expR59_c3_pool_admission_diagnostic.py` | 48-56 | 8 sources, weighted RRF constants |
| C3 admission feature names | `scripts/expR59_c3_pool_admission_diagnostic.py` | 67-118 | 99 features |
| C3 admission feature matrix | `scripts/expR59_c3_pool_admission_diagnostic.py` | 385-489 | Candidate-level admission rows |
| C3 admission trainer | `scripts/expR59_c3_pool_admission_diagnostic.py` | 504-619 | Grouped-session CV5, `num_boost_round=120` |
| C3 Phase 2 baseline reproduction | `scripts/expR59_c3_phase2_frozen_lr_conversion.py` | 80-128 | Reproduces R55/R58 OOF family |
| C3 Phase 2 nDCG function | `scripts/expR59_c3_phase2_frozen_lr_conversion.py` | 74-77 | `1/log2(rank+1)` if in cutoff |
| C3 Phase 2 frozen LR scoring | `scripts/expR59_c3_phase2_frozen_lr_conversion.py` | 635-647, 682-746 | Loads frozen model and scores RRF/learned pools |
| C3 Phase 2 metric splits | `scripts/expR59_c3_phase2_frozen_lr_conversion.py` | 321-381 | all/h7/same/diff metric blocks |
| C3 Phase 2 churn | `scripts/expR59_c3_phase2_frozen_lr_conversion.py` | 410-431 | Current script only all/h7; R60 must extend to same/diff |
| R54 trainer pattern | `scripts/expR54_phase3_full5fold_integration.py` | 337-381 | Train/val row construction, LR train, top20/nDCG |
| Grouped session folds | `scripts/expS2_lambdarank_grouped.py` | 44-60 | Session-clean CV |
| C3 Phase 1 result | `docs/r59_candidates/c3_diagnostic_result.md` | 20-38 | pool_hit and split gains |
| C3 Phase 2 result | `docs/r59_candidates/c3_phase2_frozen_lr_result.md` | 7-36 | baseline reproduction, scoreboard, conversion |
| Churn memory | `memory/feedback_retriever_swap_churn_gates.md` | 19-29 | top-1 <25/80, hard stop >35/80, top20 overlap >=14 |
| Clean fold memory | `memory/feedback_clean_fold0.md` | 7-11 | no production init for fold validation |

---

## Variant Comparison Table

| Dimension | Variant A | Variant B | Variant C |
|---|---|---|---|
| Approval | Fold 0 only approved | Not approved | Not approved |
| Pool | Learned-admission @300 | Learned-admission @300 | RRF top-N protected + learned fill |
| LR feature count | 37 | 39 | 37 or 38 |
| Admission score as LR feature | No | Yes | No by default |
| Training distribution | 100% learned-pool candidates | 100% learned-pool candidates | Mixed RRF-protected/learned-fill candidates |
| Primary hypothesis | Distribution matching fixes burial | Admission score carries extra LR signal | Churn can be reduced while preserving some conversion |
| Main risk | Pool swap churn | OOF stacking leakage / overfit | N-sweep tuning and diluted conversion |
| First gate | Fold-0 falsifier | Separate approval after A | Separate approval after A |


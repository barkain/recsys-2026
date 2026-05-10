# R49A: Multi-Query R21 Retraining with Augmentation

**Status:** Design (no code)
**Depends on:** R21 baseline, R47 multi-query diagnostic

## Background

R21 is a supervised BGE-base-en-v1.5 dense retriever trained with contrastive loss on dev sessions. It uses a single query per session (concatenation of last 3 user messages). R47 multi-query diagnostic revealed:

| Oracle best-view | Hard cases recovered (of 61) |
|------------------|------------------------------|
| top-300          | 12                           |
| top-500          | 21                           |
| top-1000         | 37                           |

Different query views recover different tracks, indicating the query representation is a bottleneck.

## 1. Objective

Improve R21 query representation for diff-artist and r21_deep misses using multi-query augmentation, with strict 5-fold OOF from day one.

## 2. Current R21 Architecture

| Parameter   | Value                                       |
|-------------|---------------------------------------------|
| Model       | BGE-base-en-v1.5 (768d)                     |
| Loss        | Contrastive, temperature=0.05               |
| Schedule    | 2 epochs, batch_size=32, lr=2e-5            |
| Query       | Concatenation of last 3 user messages        |
| Corpus      | 47071 tracks, encoded as "track_name by artist_name" |
| OOF         | 5-fold, grouped by session, seed=0          |
| Production  | Trained on all 8000 dev cases               |
| RRF weight  | 1.0 into top-300 retrieval pool             |

## 3. Training Pairs

Current construction:
- Each dev case maps a query (user messages) to a ground truth track.
- Positive pair: `(query, gt_track_text)`.
- Negatives: in-batch negatives (primary) + optional hard negatives from pool.
- R21 also experimented with hard negatives from the retrieval pool.

**Design question:** Should R49A change the training pairs, or only the query representation? Recommendation: change only the query representation first (cheaper, isolates the variable). Hard negative mining is orthogonal and can layer on later.

## 4. Query View Generation

| View | Construction | Why it recovers different tracks | Overfitting risk |
|------|-------------|----------------------------------|------------------|
| **A (current)** | Last 3 user messages concatenated | Baseline signal; captures recent conversational context | Low (proven) |
| **B (music history)** | Structured: "User listened to: track1 by artist1, track2 by artist2, ..." from played tracks | Anchors on actual listening; recovers same-vibe tracks the user never named | Medium -- may memorize artist co-occurrence patterns |
| **C (explicit request)** | Last user message only | Captures the most specific intent; less noise from earlier turns | Low, but loses context on multi-turn refinements |
| **D (genre/mood)** | Extract genre/mood keywords from conversation + listening history | Recovers tracks matching abstract descriptors when artist names fail | High -- keyword extraction is noisy, small effective vocab |

### View construction details

- **View B:** Iterate `session["played_tracks"]`, format as semicolon-separated list. Cap at 10 most recent. Available in dev data for all sessions.
- **View C:** `session["messages"][-1]["content"]` where role == user. Trivial extraction.
- **View D:** Regex/heuristic extraction of genre and mood terms from all user messages + track metadata of played tracks. Requires a keyword list.

## 5. Training Approaches

| Option | Description | Training cost | Inference cost | Complexity |
|--------|-------------|---------------|----------------|------------|
| **1: Multi-view augmentation** | Single model; each view generates a separate (query_view, gt) training pair | 3-4x data, same model | 1x (pick best view or fuse) | Low |
| **2: Multi-query inference** | Train single model normally; at inference score each view, take max | 1x | 3-4x | Low |
| **3: Ensemble** | Separate model per view; merge retrieval lists | N x training | N x inference | High |

**Recommendation: Start with Option 2 (multi-query inference).**

Justification:
- Zero-training-risk — uses existing R21 fold models (folds 2-4 exist; 0-1 need retraining regardless). But NOT zero-ranking-risk: globally fusing multi-query scores can hurt h7 (as R26/R36 showed). Must be evaluated through R39/RRF/LambdaRank before any blind use.
- Serves as a diagnostic: if multi-query inference does not improve h7 after LambdaRank, retraining with augmentation will not help either.
- If Option 2 shows gains, follow up with Option 1 to bake the multi-view signal into the model and reduce inference cost.
- **If Option 2 does not improve h7, do not proceed to Option 1 retraining.**

## 6. OOF Protocol (non-negotiable)

- 5-fold OOF using `grouped_session_folds(seed=0)` -- identical folds to R21 production.
- Each fold: train on ~6400 cases, evaluate on ~1600.
- Save per fold:
  - Model checkpoint
  - Fold track embeddings (learned from R46: required for extended retrieval)
  - Fold query embeddings for each view
- Save fold manifest: `{fold_id, model_path, fold_indices, training_config}`.
- **No production model evaluation on dev.** Production model is for blind submission only.
- Baseline comparison: R21 OOF top-300 from existing `dev_r21_oof_lists.json`.

## 7. Evaluation Gates

| Gate | Metric | Threshold |
|------|--------|-----------|
| **Primary** | OOF-clean h7 after R39 LambdaRank CV5 with RRF fusion | **Delta h7 >= +0.010** |
| Secondary | OOF pool hit / candidate admission at top-300 | Improvement over R21 baseline |
| Tertiary | Recovered/lost net | >= +20 |
| Diagnostic | Per-bucket recovery (especially D and E) | Positive net |
| Required check | diff-artist case recovery | Must show improvement |

All evaluation uses OOF embeddings. Never use production model scores on dev.

## 8. Expected Cost

| Phase | Option 1 (augmentation) | Option 2 (multi-query inference) |
|-------|------------------------|----------------------------------|
| Training | ~2h/fold x 5 folds = 10h | 0 (reuse existing models) |
| Encoding (tracks) | ~5 min/fold x 5 = 25 min | ~5 min/fold x 5 = 25 min |
| Encoding (queries) | ~5 min/fold x 5 x 4 views = 100 min | ~5 min/fold x 5 x 4 views = 100 min |
| Retrieval + eval | ~30 min | ~30 min x 4 views = 2h |
| **Total** | **~12-15 hours** | **~3-4 hours** |

Folds can run in parallel if compute allows.

## 9. Risks

1. **Embedding space saturation.** If BGE-base is already at capacity, additional query views will not separate tracks further.
2. **Signal dilution.** Multi-view augmentation (Option 1) may weaken the primary query signal if noisy views dominate gradients.
3. **View D fragility.** Genre/mood keyword extraction is heuristic; noisy input may hurt more than help.
4. **Fold availability.** OOF fold models for folds 0-1 need retraining regardless of approach chosen; folds 2-4 exist.
5. **Inference cost scaling.** Option 2 multiplies retrieval cost by N views. Acceptable for dev evaluation, but production needs a strategy (e.g., pick top-2 views).

## 10. Option 2 Implementation Spec (approved)

R49A-Opt2: narrow multi-query inference diagnostic.

- **Views:** current (last 3 messages), last_query_only, music_history_summary
- **Skip:** genre_mood — heuristic lexical signals overfit (R26/R41a pattern)
- **Models:** OOF fold models only (folds 2-4 exist; 0-1 from r21_supervised)
- **Output:** Per-view OOF top-300 lists
- **Fusion test:**
  1. RRF across views as a new R21_MULTI source replacing R21 in the 7-source fusion
  2. Test weights: 0.3, 0.5, 1.0
  3. R39 album features unchanged, no new features
- **Gate:** OOF-clean Δh7 >= +0.010
- **Stop condition:** If Option 2 does not improve h7, do NOT proceed to Option 1 retraining

## 11. R49A-Opt2 Results: ARCHIVED AS NEGATIVE (2026-05-10)

**OOF-clean baseline reproduced R39 exactly (h7=0.24298).**

| Config | pool_hit | h7 | Δh7 |
|--------|----------|------|------|
| baseline (R39) | 0.6000 | 0.24298 | — |
| R21_MULTI w=0.3 | 0.5735 | 0.23624 | -0.007 |
| R21_MULTI w=0.5 | 0.5845 | 0.24278 | -0.0002 |
| R21_MULTI w=1.0 | 0.5969 | 0.24220 | -0.0008 |

- Multi-query RRF diluted R21 current-view retrieval; pool_hit and h7 both decreased.
- Zero bucket recovery across all configs.
- **Stop condition triggered:** do not proceed to query augmentation retraining (Option 1).
- Additive R21 + R21_MULTI not tested — no positive signal to justify compute.

## 12. Post-Mortem: Why R49A-Opt2 Failed

R49A-Opt2 does NOT prove multi-query is conceptually dead. It proves this specific architecture is bad:

```
encode 3 views → RRF into one source → feed LambdaRank
```

This loses information. RRF treats every view as equally trustworthy, and LambdaRank only sees the final diluted rank list. It cannot learn "this view is useful for this case but harmful for that case."

**The real question is:** Are alternate query views useful as **conditional evidence** (per-candidate, per-view signals the ranker can learn from)? R49A only tested whether they work as a **blind fused retrieval source**, which is much weaker.

**Next direction: R50 view-aware multi-query ranking** — see `docs/r50_view_aware_ranking_design.md`.

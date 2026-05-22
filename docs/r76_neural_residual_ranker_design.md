# R76 — Neural Residual Ranker Design (DESIGN ONLY — NO CODE)

**Status:** design draft. NOT to be implemented until R74 blind result confirms response-side saturation.

## Why this exists

Today's findings closed three retrieval/ranker paths:
- R68 encoder upgrade: retrieval saturated at top-30 (R72)
- R71 stacker on OOF R54c + R68 features: no signal (Δh7 −0.005)
- R75 premise: archived sprints already used OOF gates; no easy revives
- Drift forensics: LR pipeline is nondeterministic; R54c is a sample, not a golden reference

Response-side has lifted us from 0.6106 (R54c) to 0.6234 (R73), but LLM judge has hit a ceiling at 4.85 across two submissions. LexDiv has ~0.03 headroom worth ≤+0.003 composite. To break past 0.625 we need nDCG, and the only meaningful nDCG move left is a **new ranker architecture** that uses signals R54c's LR can't see.

R76 is that ranker.

## Core idea

Train a small neural network as a **residual reranker** on top of frozen R54c's top-K, using:

1. **R54c's own outputs** (its LR score, its rank-position) as features — preserves its calibrated structural information.
2. **Rich semantic features** from a large dense encoder (BGE-large or alternate) applied to query text and candidate text.
3. **The existing structural features** (R39 artist/album overlap, source ranks, ALS dot, recency, etc.) — same 37 columns the LR uses.

The model learns to **reorder R54c's top-K** based on residual signal (what the LR can't see) without ever changing the candidate pool.

Critical design property: **Tracks at positions 1-20 of R54c's top-30 are the only candidates eligible for top-20.** No pool admission, no fusion. Just reordering within R54c's top-K. This guarantees:
- Nothing leaves the LR-validated candidate set
- Same-artist canary is constrained by R54c's existing tendency
- No matched-pool retrain wall (R60), no fusion-buries-singletons issue (R72)

## Architecture options (ranked)

### Option A (recommended): Listwise neural reranker, top-K cascade

- **Input per case**: K=30 candidates from frozen R54c.
- **Per-candidate features (vector)**:
  - 1 dim: frozen R54c score (raw)
  - 1 dim: frozen R54c rank (1..K, normalized)
  - 37 dims: existing R39+R54 features (as fed to R54c LR)
  - 768 dims: BGE-large query+candidate cosine + a few semantic descriptors derived from candidate metadata
  - Total: ~810 dim per candidate
- **Architecture**: small transformer (4-6 layers, d=256, 8 heads) treating the K candidates as a sequence with set-transformer-style attention (no positional encoding — order-invariant). Output: one scalar per candidate.
- **Loss**: LambdaRank loss (NDCG-aware) on (candidate, GT) pairs. Same loss family as LightGBM LambdaRank for consistency.
- **Determinism**: PyTorch with `torch.use_deterministic_algorithms(True)`, fixed seeds, single-threaded BLAS.

Pros: well-studied, ranker quality known to beat LightGBM in literature on rich-feature tasks, deterministic-friendly.
Cons: 810 dim per candidate × 30 candidates × ~5000 train cases × pairs = nontrivial training but well within A100 8-hour run.

### Option B: Pairwise contrastive head

- Same features as A.
- Score head: simple MLP (3 layers, hidden 128).
- Loss: pairwise margin loss on (GT, non-GT) pairs within each case.
- Pros: simpler, more interpretable, fewer hyperparams.
- Cons: doesn't directly optimize NDCG.

### Option C: Cross-encoder rerank (BGE-reranker-v2-m3)

- Skip our own training. Use bge-reranker-v2-m3 on query+candidate text.
- We tested this in R69 and it failed catastrophically (Δh7=-0.25). Reason: cross-encoder ignored structural features (artist overlap, source rank), produced text-only rankings disconnected from R54c's calibrated structural signal.
- **Rejected**. Would need to blend cross-encoder score with R54c score via learned weights — that's just Option A with a slimmer feature set.

## Dataset

### Train/dev split

- Use the existing 5-fold OOF split for fair evaluation:
  - For each fold k ∈ {0..4}:
    - Train R54c-style sibling LR on the OTHER 4 folds (this is `OOF_R54c` we already build in expR71)
    - Score fold-k cases → take OOF top-30 per case
    - These OOF top-30 candidates with OOF R54c scores become the training data for fold-k
  - Train R76 on folds {1,2,3,4}, evaluate on fold-0 (and rotate for full 5-fold OOF measurement)
- **No use of frozen R54c**: avoids the in-sample artifact.

### Per-candidate features (concrete schema)

```
[
  oof_r54c_score: float,
  oof_r54c_rank_normed: float,  # rank/K
  # 37 columns: FEAT_R39_ALL (34) + FEAT_R54 (3) — same as R54c LR sees
  rrf_rank_inv, last_artist_match, last_tag_jaccard, ...
  ...
  r54_rank_inv, r54_presence, r54_cosine,
  # semantic features (BGE-large; 8000 dev cases need pre-computed query+catalog embeddings)
  bge_query_candidate_cosine: float,
  bge_max_sim_to_played: float,   # max cosine to any track in user's played history
  bge_mean_sim_to_played: float,
  bge_topk_artist_centroid_cosine: float,  # cosine to query average of played artist embeddings
]
```

Total ≈ 43 dims per candidate (much smaller than option A's 810 dim full-768-vector approach). Start with this lean schema; expand only if needed.

### Labels

Binary: 1.0 for GT, 0.0 for non-GT. LambdaRank handles position weighting via its NDCG gain function.

## Gates (predeclared)

**Phase 0 (sanity)**: train R76 fold-0 only, evaluate fold-0 dev.
- h7 nDCG Δ ≥ +0.005 vs OOF R54c (same-recipe baseline, fair comparison)
- same-artist Δ ≥ −0.002
- top-1 churn ≤ 25/80
- recovered > lost
- top-20 overlap ≥ 14/20

**Phase 1 (5-fold OOF)**: only if Phase 0 passes.
- h7 nDCG Δ ≥ +0.010 average across 5 folds vs OOF R54c
- same-artist Δ ≥ −0.002 in EVERY fold
- diff-artist Δ ≥ 0 in 4+ of 5 folds

**Phase 2 (production)**: only if Phase 1 passes.
- Train production R76 on all 8000 dev (matching R54c's "Phase TRAIN on all dev" recipe)
- Compare top-20 churn vs frozen R54c on dev: ≤25/80 acceptable
- Visual audit of changed top-1 cases (no obvious quality regressions)
- Run R74-style response polish on the new top-20 if needed
- Package zip, hold for explicit GO

## Cost estimate

- GPU: A100 via Colab, ~4-8 hours total for 5-fold training + Phase 0 + Phase 1 evals
- Cost: ~$25-50 in compute
- Time: 1-2 calendar days incl. Mac-side prep and eval

## Risks (ranked)

1. **Same-artist canary regression**. A neural model has more flexibility than LightGBM and may learn to over-promote semantically-similar-but-different-artist candidates, hurting same-artist nDCG. Mitigation: hard gate on same-artist Δ; explicit monotone constraint or penalty if same-artist regresses.

2. **Overfitting on fold-0 train**. 6400 train cases × 30 candidates ≈ 192K rows is small for a transformer. Mitigation: heavy regularization (dropout 0.3, weight decay), small model (4 layers d=256), early stopping on fold-0 OOF dev.

3. **BGE embeddings still saturate**. R72 showed R68 (BGE-large) has only 10 fold-0 / 2 h7 rescue cases. Even with rich semantic features, the ranker may have ≤ +0.005 h7 ceiling. Mitigation: accept this as upper bound on expected gain.

4. **A100 dependency**. We have it via Colab, but bundle-data-to-Colab was a pain earlier (R69 work). Mitigation: pre-stage cache files to Drive or accept slow upload.

5. **Determinism**. PyTorch CUDA isn't fully deterministic even with all flags. Mitigation: same as today — accept the noise, train each fold once, document variance.

## What this WON'T fix

- The 0.4925 → 0.55 nDCG gap to leaders may not all be closeable by reordering R54c's top-30. If GT is at R54c rank > 30 for some cases, R76 can't see those candidates. Pool admission would be needed (R59 C3 path, archived for valid reasons).
- LLM judge ceiling at 4.85 won't move; that's response-side.
- LexDiv ceiling depends on response style; R76 doesn't touch responses.

Expected nDCG ceiling for R76: probably 0.49 → 0.51 if everything works (matches semintelligence/vkost). +0.018 nDCG ≈ +0.005 to +0.010 composite (back-of-envelope). Lands us at ~0.628-0.633 if combined with R74's LexDiv lift.

## Trigger conditions

R76 implementation begins ONLY IF:

- R74 blind result lands and confirms LexDiv saturation (composite ≤ ~0.627)
- AND user explicitly approves the A100 spend
- AND we have at least 2 calendar days before Blind-B / submission window
- AND we have Codex sign-off on the gate set above

Until then, this is a parked design.

## Files that would be needed

```
scripts/expR76_phase0_neural_residual_fold0.py    # train+eval fold-0
scripts/expR76_build_bge_features.py              # one-time precompute on A100
scripts/expR76_phase1_5fold.py                    # full OOF
scripts/expR76_phase2_blind_train.py              # production train on all 8000 dev
docs/r76_phase0_result.md
docs/r76_phase1_result.md
docs/r76_sprint_summary.md
cache/r76/
  bge_query_embeddings.npy
  bge_track_embeddings.npy
  fold_{k}/
    model.pt
    train_log.json
    val_predictions.json
```

No code written until R74 result lands and trigger conditions are met.

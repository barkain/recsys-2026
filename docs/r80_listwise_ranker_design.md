# R80 — Listwise Neural Ranker over Top-300 (Design)

**Status:** design draft. Phase 0A (Mac data prep) implemented. Phase 0B
(A100 train) awaits explicit user GO.

## What this is and isn't

R80 is a **learned replacement for the R54c LR's top-rank decision
surface**. It is NOT:
- Another retriever swap (R68, R79 — both closed)
- A small residual/specialist on top of R54c (R76 — closed)
- A naked text reranker missing structural features (R67, R69 — closed)

It IS:
- A session-level **listwise** model that takes the existing R54-stacked
  RRF top-300 candidate pool, reads each candidate's full feature vector
  (37 LR features + R54c raw score + R54c rank + 3 R68 features + 1024-dim
  BGE-large embedding for query+candidate), and outputs a learned score
  per candidate
- Optimized with listwise softmax CE (or LambdaLoss/NDCG-weighted) to
  put GT above the other 299 candidates
- A direct A100-justified architecture (transformer over 300 candidates
  per case is meaningful on GPU)

## Why this might succeed where R76 didn't

R76 was a tiny MLP residual (3 layers, hidden=64) that predicted a delta
to add to R54c's score. It collapsed because β converged to 0.82 — the
neural delta drowned R54c's calibration with semantic similarity (chasing
diff-artist neighbors).

R80 differs structurally:
- **Listwise over 300, not residual on 30.** The model sees the entire
  candidate pool, not just R54c's top-30. It can learn relative ordering,
  not just "promote/demote within a small set".
- **Set-attention over candidates.** A SetTransformer or simple
  candidate-to-candidate attention can encode pool-level signals (e.g.,
  "how many candidates share artist", "what's the spread of R54c scores")
  that the LR's per-candidate features can't.
- **Full feature stack, including BGE embeddings as features (not scorer).**
  R76 had only 5 semantic features; R80 has 1024-dim per candidate. The
  transformer can decide how much to weight semantic vs structural.
- **NDCG-aware loss.** Listwise CE with GT one-hot + optional NDCG
  position weighting directly optimizes the ranking metric.

## Architecture

```
Input per case:
  candidates: (300, F) where F = 37 LR + 1 R54c_score + 1 R54c_rank +
                                 3 R68 + 1024 BGE_track + 1024 BGE_query (broadcast)
  = ~2090 dim per candidate
  Project to 256 dim via shared MLP → (300, 256)

Transformer encoder:
  - 4 layers, d_model=256, n_heads=8, ff=512
  - Self-attention over the 300 candidates (no positional encoding —
    order-invariant set transformer style)
  - Dropout 0.2

Scoring head:
  - Linear: (300, 256) → (300, 1)
  - Output is the learned score per candidate

Loss (listwise CE):
  - Take case's softmax over 300 scores
  - Target: one-hot for GT position (if GT in pool, else skip — these
    cases contribute no training signal)
```

Model size: ~5-10M params. Trains in <1 hour on A100 with bf16.

## Per-candidate features (concrete schema)

For each of the 300 candidates in a case:

```
[
  # 37 LR features (R39+R54, same as expR54_phase3_blind_submission)
  rrf_rank_inv, last_artist_match, last_tag_jaccard, ..., 
  r54_rank_inv, r54_presence, r54_cosine,
  
  # R54c LR raw score + rank
  oof_r54c_score: float,                # raw LightGBM output
  oof_r54c_rank_norm: float,            # rank / 300
  
  # R68 features (fold-0 OOF lists)
  r68_rank_inv, r68_presence, r68_cosine,
  
  # Semantic embedding (BGE-large from R68 cache)
  bge_track_emb: float[1024],            # candidate embedding
  bge_query_emb: float[1024],            # case query embedding (broadcast)
  bge_query_candidate_dot: float,        # scalar pre-computed dot product
  bge_max_sim_to_played: float,          # max cosine to any played track
  bge_mean_sim_to_played: float,         # mean cosine to played
  bge_max_artist_centroid_cos: float,    # cosine to artist centroid of played
]
```

Total: ~37 + 2 + 3 + 1024 + 1024 + 5 = ~2095 dim per candidate.

Project to 256 in the first MLP layer to keep memory bounded.

## Phase plan

### Phase 0A — Mac data prep (NO GPU)

Build `cache/r80/listwise_dataset_fold0.pkl`:
- For each fold-0 case (1600), generate top-300 candidates + all features
- Use OOF R54c (trained on folds 1-4) to score
- Pre-compute all BGE-large semantic features from `cache/r68/phase0_fold0/`
- Save as flat tensor per case + metadata

Reuse the OOF R54c LR from R71 Phase 0A. Reuse all R68 artifacts.

Output:
- `cache/r80/listwise_dataset_fold0.pkl` — (1600 cases × 300 candidates × ~2095 dim)
- `cache/r80/eval_baseline.json` — fold-0 OOF R54c top-20 nDCG (≈ R71)

### Phase 0B — A100 training + eval (~$10-15, ~1-2 hours)

Train listwise transformer on fold-0 inner CV (5-way within fold-0):
- 5 inner folds: 1280 train cases / 320 test each
- 20-30 epochs
- AdamW, lr=1e-4, weight_decay=0.01, dropout=0.2
- Gradient clipping 1.0
- bf16 autocast
- Listwise CE loss (mask cases without GT in pool)
- Save best epoch by inner-CV held-out h7 nDCG

Eval on fold-0 dev (all 1600 cases, full 5-fold inner CV aggregated):
- Standalone top-20 from R80 predictions
- Compare to OOF R54c top-20 baseline

**Hard gates (predeclared):**
- h7 nDCG Δ ≥ +0.005 vs baseline 0.2226
- same-artist Δ ≥ -0.002
- recovered > lost on h7
- top-1 churn /80 ≤ 25
- top-20 overlap ≥ 14/20

If ALL pass → Phase 1 (full 5-fold OOF on A100).
If ANY fail → ARCHIVE.

### Phase 1 — 5-fold OOF (CONDITIONAL, ~$50)

- Train 5 R80 models, one per held-out fold
- Aggregate metrics across all 8000 dev cases
- Same gates as Phase 0B but on the larger n

### Phase 2 — Production candidate (CONDITIONAL)

- Train production R80 on all 8000 dev with hard negs from any sibling LR
  (mixed in-sample/OOF, matches R54c's production recipe)
- Encode blind test catalog + queries (catalog only changes per submission)
- Submit standalone R80 top-20 as the candidate
- Compare composite to R78

## Strict no-go conditions

R80 stops if:
- Same-artist canary fails in Phase 0B (semantic collapse like R76)
- Top-1 churn explodes > 50/80 (R76-style overfitting)
- Loss does not converge in Phase 0B
- R78 production state would be threatened

## Cost summary

- Phase 0A: $0 (Mac)
- Phase 0B: ~$10-15 (A100 ~2 hours)
- Phase 1 (conditional): ~$30-50 (5x Phase 0B)
- Phase 2 (conditional): ~$20 (production train + encode)

Total worst-case: ~$80. Stops early if any phase fails.

## Why R80 is genuinely different from past failures

| past sprint | failure mode | how R80 differs |
|---|---|---|
| R67 LLM rerank | naked text, no structural features | full LR features + semantic |
| R69 cross-encoder | same as R67 | same as above |
| R76 neural residual MLP (small) | β=0.82 drowned R54c | listwise replacement, not residual |
| R68 retriever | hit@300 ceiling not top-20 | direct top-20 ranking objective |
| R79 hard-neg retriever | collapse without random negs | uses LR top-300 pool, no retraining of catalog encoder |
| R71 stacker | tiny model on 5 features | full ~2095-dim per candidate, attention |
| R58 LightGBM stage-2 | weak tree ensemble | transformer with cross-candidate attention |

The key novelty: **cross-candidate attention** lets the model see pool-level
context (artist distribution, score spread, presence overlap across sources)
that no prior model has had access to per-case.

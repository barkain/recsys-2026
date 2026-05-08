# R34 Residual Pool Reranker Design

## Status
Proposed next modelling line after R33c clean full-pool MLP failed.

## Motivation
R32 showed the important ceiling: on hist_7 cases, the existing pool@300 contains enough ground-truth tracks to reach much higher nDCG if ranked well. R33c then tested a neural MLP that directly replaced LambdaRank scores. Clean full-pool evaluation failed:

- fold-0 LambdaRank hist_7: 0.2062
- best clean MLP hist_7: 0.1580
- final clean MLP hist_7: 0.1217

The failure mode is not that a neural model cannot learn anything. The best MLP beat raw pool order slightly. The failure is that a free replacement scorer destroys LambdaRank's already-useful ordering. R34 therefore changes the modelling target from replacement to residual correction.

## Hypothesis
A neural scorer may help only as a constrained residual on top of LambdaRank:

```text
final_score(candidate) = normalized_lambdarank_score(candidate) + beta * neural_delta(candidate)
```

The model is allowed to make local corrections but not freely rewrite the ranking. This directly addresses the R33c failure mode.

## Data Protocol
Use the same clean fold-0 diagnostic protocol as R33c-clean.

- Candidate pool: existing R21 production pool@300 tensor cache from `cache/r33c_clean/tensors.npz`.
- Query and track embeddings: clean fold-0 R21 encoder artifacts already baked into the R33c-clean tensor cache.
- Train cases: non-fold0, hist_5+, GT in pool.
- Eval cases: all fold0 hist_7 cases, including GT-not-in-pool as zero-score misses.
- Baseline: CV5/OOF LambdaRank score over the same pool and same feature tensors.

This means the diagnostic answers one specific question: can a residual neural ranker improve fold-0 hist_7 ranking over LambdaRank when the candidate pool is fixed?

## Model
R34 uses a low-capacity residual model, intentionally smaller and more constrained than R33c.

Inputs per candidate:

- `lr_features`: existing 29 LambdaRank features.
- `interaction_features`: cosine similarity, rank inverse, rank normalized.
- `q_emb`: clean R21 query embedding.
- `c_emb`: clean R21 candidate embedding.

Primary fast architecture:

```text
[lr_features, interaction_features]
  -> small MLP -> delta_score

final_score = lr_z + beta * delta_score
```

Optional higher-cost architecture:

```text
q_emb -> Linear(768, 64) -> L2 normalize
c_emb -> Linear(768, 64) -> L2 normalize
embed_score = dot(q_proj, c_proj)

[lr_features, interaction_features, embed_score]
  -> small MLP -> delta_score

final_score = lr_z + beta * delta_score
```

`lr_z` is per-case z-normalized LambdaRank score. This keeps LambdaRank as the anchor and makes beta meaningful across cases.

Start with the fast architecture because it isolates the residual-ranking hypothesis and runs quickly. Only test the projection architecture if the fast residual is positive or near-flat.

## Loss
For each case, score the full pool@300 and train with listwise cross-entropy:

```python
loss_ce = cross_entropy(final_scores / tau, gt_position)
loss_reg = mean(delta_score ** 2)
loss = loss_ce + lambda_delta * loss_reg
```

Only cases with GT in pool are trainable. Eval still includes all hist_7 cases.

Primary hyperparameters:

- `beta`: 0.05, 0.10, 0.20, 0.30
- `tau`: 1.0 initially
- `lambda_delta`: 1e-3 initially
- epochs: 10-20 with checkpointed per-epoch eval

## Decision Gates
Fold-0 clean diagnostic gate:

- `h7_all >= LambdaRank + 0.005`
- recovered top-20 cases > lost top-20 cases
- no severe same-artist collapse; diff-artist improvement is preferred but not required

If gate passes:

1. Expand to 5-fold OOF residual scoring.
2. Train production residual model for Blind-A.
3. Build a conservative blend submission against R27b/R25 tracks.

If gate fails:

Close neural pool reranking with current features/embeddings. The next direction should not be another small MLP on the same representation.

## Why This Is Different From R33c
R33c trained an unconstrained replacement score. It could and did overwrite LambdaRank. R34 preserves LambdaRank and learns only bounded corrections. This is the lowest-risk way to exploit the neural signal without losing the strong handcrafted ranker.

## Expected Outcomes
Interpretation by result:

- If R34 beats LambdaRank: the neural signal is useful but must be residualized.
- If R34 matches LambdaRank: residual constraint prevents damage but adds no information.
- If R34 still underperforms: the current embeddings/features do not contain the missing ranking signal.

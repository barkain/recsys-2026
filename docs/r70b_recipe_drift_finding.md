# R70b — Second major finding: R54c LR is not bitwise reproducible

## What I did

Trained R70b_repro: an LR with the **identical 37-feature schema, identical
hyperparams (lambdarank, num_leaves=31, lr=0.05, min_data=10, seed=0,
num_boost_round=300), identical caches** as R54c. Should match
`cache/r54_phase3_lr_model.txt` bitwise.

## What I got

```
Top-20 ordering match on 100 dev cases: 0/100
Max prediction absolute diff: 1.468
```

Zero out of 100 cases have matching top-20 ordering. Predictions differ by
up to 1.47. **R70b_repro and R54c are materially different models.**

## Where the drift is

Comparing `feature_infos` in the two saved models, three feature columns have
different value ranges in their respective training data:

| feature index | name (from FEAT_R39_ALL) | R54c range | R70b_repro range |
|---:|---|---|---|
| 5  | `query_meta_tok_overlap`  | `[0:13]` | `[0:14]` |
| 21 | `als_dot`                 | `[-0.000151:0.0215]` | `[-0.000190:0.0246]` |
| 36 | `pool_same_album_count`   | `[0:0.9273]` | `[0:0.9204]` |

Other features (32 of them) have identical ranges. So the bulk of the feature
schema is stable; three specific features compute different values today.

## Implications for the sprint history

The 20 sprints of "LR conversion wall" results compared OOF siblings (current
recipe) vs frozen R54c (older recipe). The observed -0.08 h7 / -0.16
same-artist gap is therefore the sum of:

1. **OOF artifact** (memorization gap): frozen R54c saw all 8000 dev in
   training, our siblings train on fold subsets and evaluate on held-out
   fold. R70b 5-fold OOF measures this directly: aggregate Δh7=-0.1042 vs
   frozen R54c.

2. **Recipe drift**: even if our sibling trained on all 8000 dev (no fold
   split), the 3-feature drift means the LR fits different data and produces
   different rankings.

Both effects push siblings down relative to frozen R54c. The relative
contribution of each is unknown.

## Implications for production decisions

- **R63c-repair production (composite 0.6224)** uses frozen R54c. Untouched
  by any of this. Safe.
- **R70_prod (current-recipe LR with 40 feats, +r68)** if shipped to blind
  would carry the recipe drift. If it scores worse on blind than frozen R54c,
  we can't distinguish "r68 features are bad" from "recipe drift hurt".
  **Decision: do NOT ship R70_prod.**
- **The fair test of "do r68 features help?"** is a frozen-ranker-compatible
  stacker: take frozen R54c top-K scores, blend with r68 features, evaluate
  OOF. Bypasses both artifact and drift.

## Likely drift causes (bounded diagnostic, not full fix)

Per Codex's recommendation, not chasing this deeply right now. Likely
candidates:

1. **Python hash randomization** (`PYTHONHASHSEED`). If `_featurize_row` uses
   `set` iteration order anywhere (e.g., for tag overlap, album sets), each
   run produces different feature values.
2. **Set iteration order in tag/title token overlaps.** `query_meta_tok_overlap`
   max changed by exactly +1 — consistent with one extra token being counted
   somewhere.
3. **Floating-point accumulation order in ALS dot products.** Numpy
   reductions can be order-dependent at the bit level (rare but possible
   with `np.dot` vs `np.sum`).
4. **`pool_same_album_count`** depends on album-set ordering — could be
   set-iteration drift.

Caches were re-saved on May 15 16:09 (same minute as R54c LR). Could be that
the maps file was overwritten with slightly different content after R54c was
trained, or that the training pipeline reads inputs in a different order on
re-run.

## Recommendation

Per Codex consult: pivot to **stacker on frozen R54c**.

- Take frozen R54c LR top-30 per case (truly OOF — frozen R54c saw the
  case in training, but we can still rank its candidates).
- For each top-30 candidate, compute: R54c_score, R54c_rank, r68_rank_inv,
  r68_presence, r68_cosine.
- Train small linear/logistic blend on fold-0 OOF train cases.
- Evaluate on fold-0 OOF dev.
- If gain is broad, robust, and not same-artist-only → candidate for blind.
- Otherwise → hold R63c-repair, sprint pivot elsewhere.

This bypasses both the artifact AND the drift.

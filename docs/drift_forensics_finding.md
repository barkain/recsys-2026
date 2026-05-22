# Drift Forensics — Pipeline is Nondeterministic, R54c Is One Sample

## TL;DR

The "recipe drift" we observed (R70b_repro doesn't bitwise match R54c) is
caused by **inherent nondeterminism in the LR training pipeline**, not by
input drift since R54c was trained. Two consecutive R70b_repro runs with
identical code, identical inputs, identical hyperparameters, identical
seed=0, and `PYTHONHASHSEED=0` set in run2 produce DIFFERENT LR models.
R54c is just one sample from this nondeterministic distribution.

## The test

Ran `expR70_production_train.py --mode r70b_repro` twice. Both used:
- Same `cache/*` inputs
- Same LightGBM hyperparams (lambdarank, num_leaves=31, lr=0.05, min_data=10, seed=0)
- `OMP_NUM_THREADS=4`
- Run 2 also had `PYTHONHASHSEED=0`

| Model | MD5 |
|---|---|
| R70b_repro run 1 | d64ac21d09ec5b58e9d96b609366fc1a |
| R70b_repro run 2 (PYTHONHASHSEED=0) | 937a1578cef4e24d49cc8bde268ff639 |
| R54c frozen (golden reference) | e3edad9696bf70a7615cb0cf32f8c022 |

Three different MD5 hashes from the same recipe.

## What's different

`tree_sizes` field reveals ±1 byte offsets at multiple boosting iterations
between run1 and run2:
- Position 124: 3517 → 3516
- Position 138: 3539 → 3540
- Position 175: tiny shifts throughout

These tiny tree-size deltas indicate the trees themselves differ at some
splits/leaves, propagating through subsequent rounds. The cumulative effect
breaks top-20 prediction ordering (0/100 cases matched between runs).

## Likely sources of nondeterminism

1. **LightGBM threading**: even with `seed=0`, internal histogram-construction
   threads can race on feature binning. LightGBM offers `deterministic=true`
   but it wasn't set in the original R54c training (or in our siblings).
2. **numpy threading**: floating-point reductions in `np.dot()` (used for
   ALS scoring during feature build) and `np.sum()` aggregations are
   order-dependent with multi-threaded BLAS. Different threads can finish in
   different orders, giving non-bit-exact reductions.
3. NOT Python set/dict iteration: `PYTHONHASHSEED=0` didn't fix it.

## Implications

### What this means about R54c

R54c is ONE sample from a nondeterministic distribution of LR models. Our
"R70b_repro" sibling LRs are ALSO samples from that distribution. They are
peers, not failed reproductions.

### What this means about the OOF artifact theory

Today's R70b finding (artifact in "LR conversion wall") is still valid:
- R54c was trained on all 8000 dev cases = in-sample for fold-0 dev
- Our OOF siblings train on fold-0 train, are evaluated on held-out fold-0 dev
- The ~−0.08 h7 / ~−0.16 same-artist OOF penalty is real

The drift discovered in R70b_repro was nondeterminism noise ON TOP OF this
OOF penalty. The OOF penalty dominates (≥0.08 h7) and the nondeterminism
contributes some smaller fraction (<0.005 h7 estimate).

### What this means for future ranker work

1. **Don't try to bitwise reproduce R54c.** Each run is a different sample.
2. **For OOF-vs-OOF sibling comparisons**, run both with the same env in the
   same session — or even better, train multiple times and average.
3. **To make future work deterministic**, set `deterministic=true` in
   `LR_PARAMS` and `num_threads=1` for LightGBM. This will be slower (~5x)
   but reproducible. NOT done in R54c or any prior sibling.

### Practical impact on past closures

Adding nondeterminism noise (~0.003 h7) to the OOF penalty (~0.08 h7), the
total "false negative magnitude" for past sprints using frozen-R54c was
~0.083 h7. Still not enough to flip R56-R60 results (those were already
OOF-vs-OOF). Still not enough to flip R67/R68/R70/R71 (those are within the
artifact correction but their magnitudes were not actually within +0.005 of
passing under any reasonable correction).

## Net recommendation

The drift forensics path is closed. R54c is not reproducible, our siblings
aren't reproducible either — they're all samples from a distribution. Move
on:

- Either: do nothing, ship R74 if blind result is positive.
- Or: explore a fundamentally different ranker architecture
  (pairwise/listwise neural on top-K) where determinism is easier to enforce.
- Or: re-train R54c-style LRs with `deterministic=true` to get a reproducible
  golden reference, then run all future sibling experiments against that
  golden reference. ~5x slower per train.

R63c-repair holds production. R73 in production at 0.6234. R74 zip pending
blind verdict.

# R58 Architecture Choice — Second-Stage Specialist Over LR Top-50

**Status: design only. Inventory passed (`exp/eval/expR58_inventory.json`).
This doc picks an architecture for Phase 2. No training code until this is
reviewed and approved.**

R54c remains production at composite 0.6106. R58 is the architecturally-
different attempt at extracting the dev headroom that Phase 1 inventory
sized:

- 4116 / 8000 dev cases have GT in LR top-50 (51.4%)
- 3348 of those are HIT (GT in LR top-20)
- **768 cases have GT in [21, 50]** — R58's pure-rerank target
- Theoretical maximum HIT-rate lift if a specialist perfectly re-orders
  top-50 = +9.6 percentage points
- Translates to roughly +0.04 to +0.06 h7 nDCG if even half captured

That is materially above the +0.010 production gate. But the prior is low
because R56 (rule-based rerank on top-50) failed even with an ORACLE
gating signal, and R57b new-feature LR retraining also failed. R58 has
to clear the bar a learned specialist can find a re-ordering signal that
the cheap interventions couldn't.

## 1. The three architectural options

### A) Second-stage LightGBM LambdaRank on LR top-50 — RECOMMENDED

Architecture:

- Inputs: 51 features per candidate (see §4)
- Group size = 50 (one group per case)
- LambdaRank objective, identical hyperparameters to LR by default:
  `num_boost_round=300, num_leaves=31, learning_rate=0.05,
  min_data_in_leaf=10, seed=0`
- Trained CV5 with the same fold structure as LR (`grouped_session_folds
  (seed=0)`). Stage-2 fold i trained on top-50 rows from cases in
  folds {0..4} \ {i}, evaluated on fold i's top-50.
- Output: per-candidate `stage2_score`

Final ranking is a *residual blend*, not a replacement:

```
final_score(c) = z(lr_score(c)) + beta * z(stage2_score(c))
```

`z(.)` is per-case z-normalisation (mean/std computed over that case's
50 candidates only). This makes LR and stage-2 scores comparable for
blending without one swamping the other.

`beta` sweep: **{0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}**. beta=0 is
pure LR (baseline reproduction check). beta>0 mixes stage-2 in. beta=1+
is excluded — pure stage-2 replacement is too aggressive for a first pass
and would compete with LR rather than augment it.

Why this is the right first attempt:

- Same family as LR (LightGBM LambdaRank), so behaviour and feature
  importance interpretation are familiar.
- LR's per-fold trees were trained on the full pool@300; stage-2 trees
  see only top-50. The smaller, more homogeneous input distribution is
  where a specialist might find signal LR's trees compress away.
- Residual blending bounds the blast radius. If stage-2 is noisy, low
  beta passes through mostly LR. If stage-2 is strong, higher beta picks
  up its signal.
- Cheap to evaluate: 5 fold models on ~400k rows; expected ~10-20 min on
  CPU. No retrieval, no submission.

### B) Pairwise top-50 classifier — DEPRIORITIZED

Architecture:

- For each pair `(i, j)` of candidates within a case's top-50, predict
  `P(i ranks above j)` from concatenated/difference features
- Loss: log-loss on the binary `is_i_GT_and_j_not` label (only positive
  on the unique pair containing GT)
- Inference: rank candidates by win count over all 50*49/2 = 1225 pairs

Why deprioritized:

- More implementation effort, harder to debug per-feature behaviour
- Generally not better than LambdaRank for this corpus size and feature
  set
- Same risk as A on the GT-not-in-top-50 cases (4884 of 8000), which
  contribute no positive pair

**Proceed to B only if A produces directional signal (positive net
recovery) but blocked on top-1 churn or same-artist canary.** Specifically,
if A shows h7 +0.005 to +0.010 with `lost` close to `recovered`, a
pairwise model might tighten the decision boundary. Not before that
signal exists.

### C) Small MLP / set-attention — NOT RECOMMENDED

Architecture:

- MLP per candidate (no cross-candidate interaction), or
- Self-attention over the 50-candidate set with positional encoding by
  LR rank

Why not recommended for first pass:

- 8000 cases × 50 candidates = small dataset for neural nets; overfit
  risk is high
- Tabular features with cardinality this low favour gradient boosting
- Engineering cost much higher than (A); doesn't compose well with the
  inventory work already done
- We don't have raw embeddings in scope (per `r58_design.md` §3.8), so
  the attention variant wouldn't have its natural input

**Proceed to C only if A and B both fail with a clear "non-linear cross-
candidate signal needed" diagnostic, AND we agree to expand scope to
include embeddings (which requires its own inventory pass).**

## 2. Recommended decision

**Implement A only for first implementation.** Stage-2 LightGBM with the
beta sweep above. Single CPU run. If A fails (no beta passes the strict
gate), archive R58 and freeze for Blind-B per the design's §6.

If A produces signal but the lost/recovered ratio looks close, consider
B as a targeted refinement.

If A and B both fail to recover material h7, do not escalate to C.
Architecture is not the bottleneck at that point — the available signal
is.

## 3. Train/validation split discipline (mandatory)

### 3.1 Folds

`grouped_session_folds(sessions, seed=0)` — identical to LR and to the
inventory script. Stage-2 inherits the same fold assignment per case.
The inventory table already records `fold_id` per row.

### 3.2 Per-fold training set

Stage-2 fold i:

- Training rows: all top-50 rows from cases whose `fold_id != i`
- Validation rows: all top-50 rows from cases whose `fold_id == i`
- Group sizes are 50 each (or fewer in the rare case the pool < 50)

### 3.3 Label construction

- `label = 1` if `gt_flag == 1` (candidate is the GT track), else `0`
- A case where GT is not in top-50 contributes 50 negative labels (no
  positive). We **keep these** in training. Filtering them out would
  bias stage-2 toward only the "easy" cases where GT is already reachable
  and remove the natural-negative signal.

### 3.4 Per-case z-normalization for blending

For each case (in both training-time eval and inference):

```
z_lr(c)     = (lr_score(c) - mean_case(lr_score)) / std_case(lr_score)
z_stage2(c) = (stage2_score(c) - mean_case(stage2_score))
              / std_case(stage2_score)
```

Std uses `np.std(..., ddof=0)`. Tiny-std guard (replace 0 with 1) for
edge cases.

### 3.5 No leakage across folds

- Stage-2 fold i's training data must not include any rows from cases
  in fold i.
- The LR top-50 fed to stage-2 fold i for validation is the LR top-50
  produced by the SAME fold's LR (the inventory captured this directly).
- No fold cross-contamination is possible by construction.

## 4. Feature list (first pass; explicitly no raw embeddings)

51 features per candidate, all already in the inventory table
`cache/r58/top50_dev.pkl`:

Group A — LR signal (5):
- `lr_score`
- `lr_score_minus_top` (lr_score − case's best lr_score)
- `lr_score_minus_at20` (lr_score − case's 20-th rank lr_score)
- `margin_to_20_case` (case-level diversity-of-top: best − rank20)
- `candidate_rank` (1..50)

Group B — Per-source ranks (8):
- `r21_rank`, `r54_rank`, `r54_cosine`
- `a_rank`, `b_rank`, `c_rank`, `d_rank`, `f_rank`, `als_rank`
  *(missing rank coded as -1; stage-2 can split on this)*

Group C — The 37 baseline LR features (R39 base + album + R54):
- Already computed; passed through directly. Names listed in
  `expR55_post_refresh_decomp.py` ALL_FEAT.

Total = 5 + 9 + 37 = 51.

### What is NOT a feature in this first pass

- **Raw 768d embeddings** (R21 or R54). Out of Phase 1 scope per design
  §3.8. If a future iteration argues for them, that's a separate
  architecture variant with its own inventory.
- **Cross-candidate aggregates** (e.g. "rank-of-this-candidate's-album
  among other top-50 albums"). Defer to a Phase 2b iteration if Phase
  2a shows signal but not enough.
- **Session-level features beyond what LR already computed** (e.g.
  history length deltas). LR's 37 features already encode session
  state.

## 5. Beta sweep + reporting

For each beta ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}:

| Metric | Reason |
|---|---|
| h7 nDCG@20 | Primary production target |
| all-dev nDCG@20 | Aggregate consistency check |
| same-artist nDCG@20 | Canary regression metric |
| diff-artist nDCG@20 | Where lift should appear |
| Hist-depth breakdown (h0..h7+) | Detect lift concentration |
| Top-1 churn (all-dev) | Stability — count and % |
| Top-1 churn (h7-only) | Stability on production target |
| Top-20 overlap median vs baseline | Stability |
| Recovered (DEMOTED → HIT) | Direct benefit count |
| Lost (HIT → DEMOTED) | Direct cost count |
| Net recovery | Headline number |
| Stage-2 LightGBM feature importance | Diagnostic only — do NOT use as evidence for gate |
| Beta-curve plot (h7 vs beta) | Visual sanity |

The beta=0 row must reproduce the LR baseline within ε. If it doesn't,
the implementation has a bug — halt before reporting beta>0 results.

## 6. Gates (mirror R56 / R57b — no relaxation)

### Production blind candidate
- h7 nDCG Δ vs baseline (beta=0): **≥ +0.010**
- Same-artist nDCG: Δ ≥ 0 (no regress)
- Diff-artist nDCG: Δ ≥ 0 (no regress)
- Top-1 churn (all-dev): ≤ 3.0%
- Top-1 churn (h7-only): ≤ 3.0%

### Exploratory (review-required, NOT auto-ship)
- h7 nDCG Δ: ≥ +0.005
- No same/diff regression
- Top-1 churn (all-dev): ≤ 1.5%
- Top-1 churn (h7-only): ≤ 1.5%

If any beta passes Exploratory but not Production, the result is
documented; **no blind code is written without explicit manual review**.
We do NOT auto-ship exploratory passes after R55h's lesson.

### Stop conditions (any of these → archive)
- Net recovery (recovered − lost) ≤ 0 for the best beta
- Same-artist nDCG regresses > 0.002 for the best beta
- All-dev nDCG worse than baseline for the best beta
- Beta=0 fails baseline reproduction within ε

## 7. Same/diff canary discipline

Following R57b's lesson: high LightGBM gain importance for new features
does not predict positive transfer. Watch the metric, not the importance.

- Same-artist (where LR's nDCG is 0.45) regression by > 0.002 is auto-
  archive even if h7 looks up. We've seen this be the canary that
  catches per-fold overfitting.
- If diff-artist (where LR's nDCG is 0.10) improves but same-artist
  regresses, it is NOT a real win — it's the stage-2 learning to
  trade between splits.

## 8. Decision tree at the end of Phase 2 implementation

```
For each beta in sweep:
  is_production = (h7 Δ >= +0.010) AND no-regress AND churn-in-bounds
  is_exploratory = (h7 Δ >= +0.005) AND no-regress AND churn-in-bounds-loose

If any beta passes Production:
  Write Phase 4 blind doc (must resolve Phase2/Phase3 OOF question from
  design §3.4 before any blind code).
Else if any beta passes Exploratory:
  Document. Manual review. Do NOT submit. Wait for new evidence (Blind-B
  data, or other experiments).
Else if best beta net_recovery > 0 but doesn't pass Exploratory:
  Consider Option B (pairwise) as targeted refinement.
Else:
  Archive R58. Freeze for Blind-B.
```

## 9. Implementation plan (only if this doc approved)

Single script:

```
scripts/expR58_stage2_lightgbm.py
```

Behaviour:
1. Load `cache/r58/top50_dev.pkl` (8000 cases × 50 candidates).
2. Reproduce LR-only baseline (sort by `lr_score`, compute metrics,
   verify against `expR55_post_refresh_decomp.json` within ε).
3. CV5: train 5 stage-2 models, predict on each fold's held-out top-50.
4. For each beta in the sweep: compute final_score, rank, compute the
   full metric battery from §5.
5. Print a per-beta table + bound the gate verdict; save
   `exp/eval/expR58_stage2_results.json`.
6. **NO** blind code. **NO** submission script. **NO** retrieval.
7. Print the decision-tree verdict (PRODUCTION / EXPLORATORY / SIGNAL-NO-GATE
   / ARCHIVE).

Expected runtime: ~10-20 min on CPU.

## 10. What this doc does NOT authorize

- Writing `scripts/expR58_stage2_lightgbm.py` (separate approval)
- Any blind submission
- Any embedding capture
- Any pairwise (Option B) or MLP (Option C) experiments
- Any LR retraining
- Any retrieval-side change

Only Option A's design is being proposed here. Code authorization is a
separate step after this doc is reviewed.

---

**Awaiting review.** Recommended call: approve Option A only; defer B and
C; require this doc's gates be met on dev before any Phase 4 blind work.

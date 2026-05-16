# R58: Residual Top-50 Specialist Reranker — Design Doc

**Status: design only. No implementation. Inventory phase first.**

R54c stays production at composite 0.6106. The post-R54c incremental loop
is saturated (6 consecutive negatives: R55, R55h, R56, R57 forensic,
R57b ISRC, R57b artist_id). The only remaining materially-different path
is an architectural change to the ranker.

## 1. Why R58, not another incremental fix

The post-R54c cycle has disproved every cheap angle:

| Direction | Result |
|---|---|
| Retriever swap (R55) | composite flat, nDCG -0.0067 |
| Manual row hybrid (R55h) | composite -0.0015 |
| Post-LR rule reranker (R56, 18 variants + ORACLE) | all FAIL_REGRESS |
| New LR features (R57b ISRC, artist_id) | all FAIL_REGRESS |
| Forensic-only refresh (R57) | patterns exist, R49C precedent disconfirming |
| Response polish (R54c) | LLM/LexDiv saturated |

The signal is consistent: **the R39+R54+LR pipeline is well-calibrated, but is
also at or near a local ceiling.** Cheap tweaks don't lift it. We need either
to wait (Blind-B) or take one structurally-different swing.

## 2. The R58 hypothesis

**Hypothesis:** LR LambdaRank is doing pointwise relevance well over the full
pool@300, but its top-50 ORDER is missing patterns that a specialist trained
*only on top-50 candidates* could learn — because the specialist isn't
distracted by the rank 50-300 distribution it has to suppress.

Specifically:

- The pool@300 has GT in ~62% of dev cases.
- Of those, LR places GT in top-20 (HIT) for 41.9% of all cases.
- DEMOTED (1628 cases, 20.3%) is GT in pool but LR ranks GT >20.
- R56 showed: LR's *demotion* of GT in DEMOTED is not just "source-signal
  override". It uses other features to learn the demotion. The question is
  whether a top-50 specialist can find a different combination of the same
  features (or new ones) that better resolves the borderline rank 18-25 cases.

Why "residual" framing:

- The specialist is trained to PREDICT THE RESIDUAL between LR's score and
  the true relevance, or equivalently to RE-RANK within top-50 conditional
  on LR's score being one of its features.
- This keeps R58 from re-deciding cases LR confidently solves (top-1 with
  huge LR score margin stays). It only intervenes where LR is uncertain.

Why "top-50 specialist":

- Top-50 is where the action is. GT is at rank 1-50 in 60%+ of cases.
- Restricting input distribution makes learning easier (smaller, more
  homogeneous input space).
- Training-time efficiency: ~50 × 8000 = 400K rows vs 300 × 8000 = 2.4M.

Why "OOF-clean":

- Same CV5 fold structure as LR. Specialist trained on folds A's LR-top-50,
  evaluated on fold A's held-out cases using fold A's CV5 LR top-50.
- No leakage.

Why "dev-only first":

- No blind submission until specialist BEATS LR on dev with h7 +0.010 and
  no same/diff regression — same gates as R57b.
- Given six consecutive negatives, the prior on success is low. Burn-rate
  matters.

## 3. Phase 1 — Artifact inventory (THIS DOC'S FOCUS)

**MUST be completed before any implementation work.** The point of the
inventory is to honestly answer: can R58 be built on what we have, or are
we missing something that would force re-running heavy retrieval?

### 3.1 What R58 needs per-case

For each of the 8000 dev cases, for each of the top-50 candidates:

| Required input | Purpose |
|---|---|
| candidate track_id | identity |
| GT track_id + flag (is this row the GT?) | label |
| Fold assignment (CV5 seed=0) | OOF discipline |
| LR score (from existing CV5 LR) | conditioning signal, possibly a feature |
| LR rank within case (1-50) | rank-position feature |
| The 37 existing LR features per candidate | baseline feature set |
| (optional) pairwise / cross-candidate aggregates | new signal class |

That's the minimum. Architectural options (§5) may need additional inputs.

### 3.2 What we currently have (catalog of artifacts)

| Artifact | Path | Status |
|---|---|---|
| Dev sessions + GT | `exp/eval/_R12_all_turns_payload.pkl` | preserved |
| R21 OOF retrievals | `cache/r21_production/dev_r21_oof_lists.json` | preserved |
| R21 model + embeddings | `cache/r21_production/{model,track_embeddings.npy}` | preserved |
| **R54 Phase 3 OOF** | `cache/r54/phase3_full/oof_r54_lists.json` | **DELETED** (disk cleanup pre-R55) |
| R54 Phase 2 OOF (proxy) | `cache/r54/phase2_full/oof_r54_lists.json` | preserved, 142 MB |
| Phase 3 trained LR | `cache/r54_phase3_lr_model.txt` | preserved |
| Dev-side caches (ALS, pop, maps) | `cache/r54_phase3_*.{npz,json,pkl}` | preserved |
| Source A-F retrieval lists (per dev case) | in R12 payload | preserved |
| LR per-case scores | NOT cached | need to recompute (~5 min CV5 rerun) |
| Per-candidate embeddings (R21, R54) | NOT cached | regenerable from models |
| Cross-candidate / pairwise features | NOT cached | new, would be computed in implementation |

### 3.3 What's missing and the cost to fill

| Missing | Cost to acquire | Blocker? |
|---|---|---|
| Per-case LR top-50 with scores + features | ~5 min CPU (re-run R56/R57b feature build + capture) | NO |
| R54 Phase 3 OOF (the production-baseline version) | ~5 h CPU OR re-run on RunPod (~30 min on A100, ~$1-3) | NO if Phase 2 acceptable as proxy |
| Per-candidate raw embeddings (R21 768d, R54 768d) | Need to encode the candidate text/track text via R21 + R54 models. ~10 min for catalog (R21), ~10 min for catalog (R54). For top-50 only per case: ~5 min if we just look up from pre-computed catalog embeddings. | NO, manageable |
| Cross-candidate features (e.g. "candidate is in same album as another top-10 candidate") | Compute at training/inference time, not cached. | NO |

### 3.4 Phase 2 OOF as a Phase 3 proxy — DEV FEASIBILITY ONLY

R56 and R57b verified that Phase 2 OOF reproduces the R39+R54 baseline
within ε (all-dev nDCG 0.22460 exact, h7 nDCG 0.24511 exact). For an R58
*specialist* that learns residual error, Phase 2 OOF is acceptable as the
"R54 signal" the LR features capture FOR DEV FEASIBILITY.

**Strict scope: Phase 2 OOF is acceptable for dev-only R58 evaluation. Not
acceptable beyond that without an explicit decision.**

If R58 passes the dev gate (§5) and we consider a blind submission, we
have to make ONE of these decisions explicitly (no hand-waving):

- **(a) Regenerate R54 Phase 3 OOF on dev** so the specialist is trained
  on the same R54 features as the production blind R54 ensemble. Cost:
  ~30 min RunPod A100 (~$1-3) or ~5 h local CPU; needs to be planned
  upfront, not at the last minute.
- **(b) Train the specialist on Phase 2 OOF, then apply at blind to
  R54c's actual production blind features.** Mismatch between training
  R54 source (Phase 2 OOF) and inference R54 source (R54 ensemble) is a
  real risk. Quantify it by comparing per-case Phase 2 vs Phase 3 OOF
  cosines on a held-out dev fold *before* committing to this path.
- **(c) Don't submit even if dev passes.** Archive R58 as a dev-only
  validated result; freeze for Blind-B with the methodology documented.

Choice (a) is safest; (b) is faster but adds variance; (c) is the
disciplined fallback. The decision belongs at the end of Phase 3, not
now — but it's flagged here so it does not get skipped.

R58's blind path also relies on the R54 ensemble blind retrieval lists
(`cache/r54_production/blind_r54_lists.json`) being current — those exist
and are already used by R54c. No new blind retrieval is required for R58
itself.

### 3.5 Inventory verdict (preliminary, to be confirmed by inventory script)

**Yes, R58 can probably be built from what we have**, with a single ~5-min
one-time data-prep run to capture per-case LR top-50 + features + scores.
The rest is computable on demand or already cached.

Phase 3 OOF deletion does NOT block R58 dev-only work. Phase 2 OOF is a
faithful proxy for the dev metric battery (already verified ε-exact
reproduction in R56 and R57b).

The verdict here is preliminary. The first concrete script (§3.6) is a
*no-train* inventory pass that confirms artifact existence, sizes, and
schema, captures the per-case top-50 rows, reproduces the R39+R54 dev
baseline within ε, and writes an inventory report. No training. No
specialist. No submission anywhere.

If the inventory confirms feasibility AND no surprises, then we write a
*separate* architecture-choice doc (§4) and only after that approval do
we touch training code.

### 3.6 Inventory script scope (the only thing approved for implementation)

`scripts/expR58_inventory.py` — no-train, no-specialist, no-submission.
What it MUST do:

1. **Artifact existence + size check.** For each of the artifacts listed
   in §3.2, report presence and byte size. Halt with explicit error if
   any required-for-R58 artifact is missing.
2. **Baseline reproduction.** Build features (same as R56/R57b), run CV5
   LambdaRank, compute h7 nDCG / all-dev nDCG / same-artist / diff-artist
   / bucket counts. Compare to `expR55_post_refresh_decomp.json` within
   the same ε used in R56/R57b. Halt if reproduction fails.
3. **Per-case top-50 capture.** For each of the 8000 dev cases, record:
   - session_id
   - fold ID
   - GT track_id, GT position in pool@300 (-1 if not in pool), GT position
     in baseline LR ordering of pool (1..300 or -1)
   - top-50 ordered by LR score (track_id, LR score, LR rank 1..50)
   - For each top-50 candidate, the 37 baseline LightGBM-feature values
     (the existing R39+R54 feature matrix row, already computed)
4. **Cross-source rank lookup per top-50 candidate.** Without recomputing
   any retrieval, for each candidate report ITS RANK (1..300 or -1) in:
   - R21 OOF source
   - R54 Phase 2 OOF source (with cosine, since Phase 2 OOF carries
     cosines)
   - src_A / src_B / src_C / src_D / src_F / ALS lists (from R12 payload
     + ALS cache)
5. **Margin-to-20 helper feature.** For each case, compute
   `margin_to_20 = LR_score - LR_score_at_rank20`. Record per case (and
   optionally per candidate as their score minus the 20-th score). This
   is a derived signal R58 might want without recomputing anything.
6. **Inventory report.** Write a JSON to
   `exp/eval/expR58_inventory.json` with:
   - artifact presence/sizes
   - baseline reproduction PASS/FAIL with ε deltas
   - per-case top-50 table summary (row count, schema, byte size)
   - explicit field availability check (yes/no per field listed in §3.7)
   - HALT verdict: PROCEED-TO-ARCHITECTURE-PHASE or BLOCK-AND-FREEZE
7. **Per-case top-50 table on disk.**
   `cache/r58/top50_dev.parquet` (or pickle if parquet is awkward).
   Columns:
   - case_idx, session_id, fold_id, candidate_rank (1..50),
     candidate_track_id, lr_score, gt_flag, gt_in_pool, gt_lr_rank
   - 37 baseline feature columns
   - per-source rank columns: r21_rank, r54_rank, r54_cosine, a/b/c/d/f
     ranks, als_rank
   - margin_to_20

What it MUST NOT do:

- Train any specialist model.
- Touch blind data.
- Compute pairwise / cross-candidate aggregates (those are a Phase 2
  architecture decision, not Phase 1 inventory scope).
- Encode raw 768d embeddings or any embedding-based feature (see §3.8).
- Tune LR hyperparameters or retrain LR.

### 3.7 Explicit availability check (the inventory must answer YES/NO)

For each candidate-level field below, the inventory report must state
whether it is obtainable WITHOUT recomputing retrieval, AND what the
exact cost is if NO:

| Field | Required-source | Expected verdict |
|---|---|---|
| `lr_score` (CV5 OOF score) | rerun CV5 LR | YES, ~5 min in this script |
| `lr_rank` (within pool, 1..300) | derived from `lr_score` | YES |
| `margin_to_20` (LR_score - LR_score@rank20) | derived | YES |
| `r54_rank` | Phase 2 OOF | YES |
| `r54_cosine` | Phase 2 OOF (carries scores) | YES |
| `r21_rank` | R21 OOF | YES |
| `a_rank` / `b_rank` / `c_rank` / `d_rank` / `f_rank` | R12 payload `src_*` lists | YES |
| `als_rank` | rebuilt from cached ALS factors | YES, fast |
| `same_artist_last`, `tag_jaccard_last`, the 28 R39 base features | already computed in CV5 feature loop | YES |
| 5 R39 album features | already computed | YES |
| 3 R54 features (rank_inv, presence, cosine) | already computed | YES |
| raw R21 / R54 768d embeddings per candidate | encoding pass over top-50 × 8000 | **DEFERRED — NOT in Phase 1 scope** |
| pairwise / cross-candidate aggregates | computed at train/inference time | **DEFERRED — Phase 2** |

If any "expected YES" comes back NO, the inventory should halt with
explicit error and recommend either (a) regenerate the artifact, or
(b) freeze R58.

### 3.8 Raw 768d embeddings are explicitly out of Phase 1 scope

A specialist that consumes the actual 768-dim BGE / R21 query and track
embeddings per candidate is a different architecture commitment (closer to
a learned cross-encoder than to a LightGBM specialist). It is not part of
the default R58 design and not part of the inventory.

If the architecture-choice doc (§4) later argues embeddings are necessary,
that becomes an explicit ARCHITECTURE VARIANT proposal with its own
inventory pass for embedding capture. Phase 1 stops at scalar features.

## 4. Phase 2 — Architecture choice (NOT THIS DOC — design after inventory)

After §3 verifies feasibility, we'd write a second doc choosing among:

**A) Second-stage LightGBM on top-50 with LR_score as a feature.**
Smallest commit, easiest to compare. Trained CV5 on top-50 only, with LR
score as input plus the 37 features. Predicts residual or fresh rank.

**B) Pairwise classifier over top-50 pairs.**
Trained on (candidate_i, candidate_j) pairs from top-50 with label
(is_i_GT, is_j_GT). At inference, sums pairwise wins per candidate. More
expressive, harder to debug.

**C) Small attention/MLP over the top-50 set.**
Genuine architectural commitment. Higher capacity, higher overfit risk.

I expect we'd start with (A) — minimal commitment, comparable to LR, can
fall back to baseline if it doesn't work.

## 5. Phase 3 — Implementation + dev evaluation (NOT THIS DOC)

After §4 architecture decision. Dev-only:
- CV5 train specialist, evaluate on each fold's held-out
- Reproduce R39+R54 baseline first (mandatory, per R56/R57b discipline)
- Report h7 nDCG, all-dev nDCG, same/diff splits, recovered/lost,
  top-1 churn — same metric battery as R56/R57b

## 6. Phase 4 — Blind (ONLY IF dev passes strict gate)

- Gate: h7 +0.010 with no same/diff regression
- Specialist runs on top-50 of R54b's existing LR blind output
- Build R58 submission with bitwise-identical track IDs where specialist
  doesn't move top-20, regenerate responses where it does (same R54c
  hygiene)

If dev gate fails: archive R58. We will have exhausted the cheap and the
hard paths; freeze for Blind-B.

## 7. Risks (preemptive)

| Risk | Mitigation |
|---|---|
| Specialist overfits dev (specifically, the LR's own mistakes on dev) | Strict OOF discipline. Don't peek at fold A's held-out when training fold A's specialist. |
| Phase 2 OOF drift from Phase 3 production | Phase 2 reproduces baseline within ε on dev. Worst case Phase 3 OOF can be regenerated if R58 ships. |
| Top-50 specialist regresses same-artist (the canary metric) | Same hard gate as R57b: same-artist regression > 0.002 is auto-reject. |
| LightGBM gain importance overstates new feature value | We learned this lesson in R57b. Don't trust gain; trust nDCG and recovered/lost. |
| Six consecutive negatives may indicate true saturation, not failure mode | R58 IS the architecturally different attempt. If it fails too, we accept saturation and freeze. |

## 8. Stop conditions

- Inventory shows R58 needs heavy regeneration (>1 day) of artifacts we lack: STOP, freeze for Blind-B.
- Inventory passes; architecture phase shows no clean specialist design: STOP, freeze.
- Phase 3 dev evaluation: specialist regresses or net recovery ≤ 0: STOP, archive.
- Phase 3 dev evaluation: specialist passes only exploratory gate (h7 +0.005), not production: STOP. Do NOT submit. Wait for Blind-B.

## 9. What R58 is NOT

- Not a feature addition to existing LR
- Not a rule-based reranker (R56 already proved that fails)
- Not a manual hybrid (R55h disproved)
- Not a retriever swap (R55 disproved)
- Not a response polish (saturated)
- Not a model with raw 768d embeddings in Phase 1 (see §3.8)
- Not promised to work — it's the last architecturally-different path before freeze

---

**Awaiting review.** First concrete action if approved:
`scripts/expR58_inventory.py` — a no-train, no-specialist, no-submission
script with the scope specified in §3.6. Implementation work begins only
after the inventory script's report shows PROCEED-TO-ARCHITECTURE-PHASE
AND the separate architecture-choice doc (§4) is reviewed.

## 10. Patch history

**v2 (post-review):**

- §3.4 — Phase 2 OOF acceptability now explicitly DEV-FEASIBILITY-ONLY.
  Three explicit choices documented for the blind path (regenerate Phase
  3 OOF, train on Phase 2 then apply to Phase 3 blind, or freeze without
  submitting). The decision belongs at the end of Phase 3, not now, but
  it cannot be hand-waved when we get there.
- §3.6 — added the explicit "no-train inventory script scope" section
  with seven required behaviors and four MUST-NOT bullets. The script
  cannot train any specialist or touch blind data.
- §3.7 — added the explicit availability-check table the inventory must
  answer YES/NO. Names every per-candidate field expected to be available
  and the recovery cost for any that aren't.
- §3.8 — raw 768d embeddings explicitly out of Phase 1 scope. If a future
  architecture variant needs them, that's a separate inventory pass and
  a separate proposal.
- §9 — added "not a model with raw embeddings in Phase 1" to the NOT list.

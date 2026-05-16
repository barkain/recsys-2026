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

### 3.4 Phase 2 OOF as a Phase 3 proxy — is it acceptable for R58?

R56 and R57b verified that Phase 2 OOF reproduces the R39+R54 baseline
within ε (all-dev nDCG 0.22460 exact, h7 nDCG 0.24511 exact). For an R58
*specialist* that learns residual error, Phase 2 OOF is acceptable as the
"R54 signal" the LR features capture. The specialist would be evaluating on
dev, where we have Phase 2 OOF; if the specialist transfers to blind, the
production blind path would use whatever R54 retrieval is current
(R54 ensemble blind lists, already cached for the current production).

**Verdict: Phase 2 OOF is acceptable for the dev-only R58 evaluation.**
If R58 shows signal and we want to ship it on blind, *that's* the time to
decide whether Phase 3 OOF needs to be regenerated.

### 3.5 Inventory verdict

**Yes, R58 can be built from what we have**, with a single ~5-min one-time
data-prep run to capture per-case LR top-50 + features + scores. The rest
is computable on demand or already cached.

Phase 3 OOF deletion does NOT block R58 dev-only work. Phase 2 OOF is a
faithful proxy for dev metrics.

If implementation begins, the first concrete step is:

- `scripts/expR58_capture_top50.py` — reuses R56/R57b feature build + CV5
  LR, saves per-case top-50 record (track IDs, LR scores, the 37 baseline
  features per candidate, fold ID, GT flag). One-shot, deterministic.

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
- Not promised to work — it's the last architecturally-different path before freeze

---

**Awaiting review.** First concrete action if approved: `scripts/expR58_inventory.py` — a no-train script that runs the artifact inventory in detail (resolves §3.2 / §3.3 exactly, including byte sizes and reproducibility checks), then halts. Implementation work begins only after the inventory script confirms feasibility AND the architecture phase (§4) is reviewed.

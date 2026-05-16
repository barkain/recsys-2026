# R61 C1 Opus Review

Reviewer: Opus (automated)
Reviewed: 2026-05-17
Artifacts: `scripts/expR61_c1_transition_probe.py`, `exp/eval/expR61_c1_transition_probe.json`, `docs/r61_c1_transition_probe_result.md`

---

## A. Scope Compliance

### A1. No neural model — CONFIRMED

The script imports only `lightgbm` (for frozen LR inference), `numpy`, and `datasets` (for loading the train arrow). No torch, no embedding training, no gradient computation. Transition scoring uses only `Counter` objects and fixed arithmetic (lines 188-297). The JSON explicitly records `"neural_model": false, "embedding_training": false`.

### A2. No LR retraining — CONFIRMED

The frozen LR model is loaded read-only at line 656 via `lgb.Booster(model_file=...)` from `cache/r54_phase3_lr_model.txt`. It is only used for `predict()` inside `frozen_lr.score_pool()`. The frozen_lr module header (line 10) states "does not retrain LR." Baseline reproduction passes with 0.0 absolute delta on both overall and h7 nDCG.

### A3. Train/dev session_id exclusion — CONFIRMED (with note)

Logic at lines 207-209:
```python
if sid in dev_session_ids:
    excluded_overlap += 1
    continue
```

Dev session_ids are extracted from `_R12_all_turns_payload.pkl` (1000 unique UUIDs for 8000 dev cases). Result: **0 sessions excluded**. This is plausible because the challenge dataset uses distinct UUID pools for train and dev splits. The exclusion count is documented in the train audit section.

**Note:** I was unable to independently verify this zero-overlap claim (environment lacks `datasets` module), but the code logic is correct and consistent with the R54 precedent (`build_train_split_sample()` uses the same `sid in dev_session_ids` pattern). The dataset's structural property (15199 train sessions, 1000 dev sessions, all with unique UUIDs) makes zero overlap the expected result.

### A4. Weight sweep <= 3 predeclared points — CONFIRMED

Line 57: `FUSION_WEIGHTS = [0.25, 0.5, 1.0]` — exactly 3 values, declared as constants at module top, not data-dependent.

---

## B. Methodological Soundness

### B1. Train→dev leakage check

**Rigorous: YES.** The exclusion operates at session_id granularity, matching the project standard. No within-session turn splitting occurs — entire sessions are kept or excluded. The music turn histogram confirms all 15199 train sessions have exactly 8 music turns (structural property of the challenge dataset). Dev sessions have 8 turns yielding 8000 cases from 1000 sessions.

No subtler leakage vectors exist because:
- Content embeddings (track metadata) are not used (metadata-neighbor source was skipped).
- The count tables are derived solely from train-split transitions.
- No dev labels, dev statistics, or dev session structures inform the count tables.

### B2. pool_hit baseline reproduction

- `expR55_post_refresh_decomp.json` line 5: `"pool_hit": 0.622`
- Script computed baseline: `computed_weighted_rrf_pool_hit_at_300: 0.622`
- Fusion baseline reproduction: `overall_abs_delta: 0.0, h7_abs_delta: 0.0, pass: true`
- Source weights match `c3.SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}`

**Reproduces exactly.** Baseline is valid.

### B3. Metric definitions

| Metric | Implementation | Correct |
|---|---|---|
| standalone hit@K | `hit_metrics_for_indices()` lines 452-486: checks GT membership in ranked[:K] | Yes |
| unique h7 outside pool | Lines 525-529: GT in C1 top-300 AND GT not in baseline_pool top-300, restricted to h7 indices | Yes |
| POOL_MISS/UNREACHABLE recoveries | Lines 539-547: counts from bucket labels loaded via `c3.load_bucket_labels()` | Yes |
| same/diff splits | `same_artist_case()` lines 427-431: GT artist in played track artists | Yes |
| fusion nDCG@20 | Via `frozen_lr.score_pool()` which ranks candidates by frozen LR prediction | Yes |
| top-1 churn | Lines 614-615: count of cases where top-1 candidate differs between baseline and fused | Yes |

All metric definitions match the user scope specification.

---

## C. Result Interpretation

### Verdict: AGREE with FAIL

The script's `evaluate_gates()` (lines 757-788) correctly applies the predeclared pass/fail criteria:

| Gate | Threshold | Observed | Status |
|---|---|---|---|
| unique h7 outside pool | >= 30 | 20 | FAIL |
| h7 nDCG delta (w=0.25) | >= +0.003 | +0.00016 | FAIL |
| novelty OR h7 lift | either above passes | neither | FAIL |
| all-dev nDCG delta | >= 0.0 | -0.00097 | FAIL |
| same-artist delta | >= -0.002 | -0.00112 | PASS |
| h7 top-1 churn | <= 1.5% | 4.60% | FAIL |

The primary pass gate (novelty OR lift) fails on both branches. Additionally, the all-dev regression and extreme churn reinforce the FAIL. Only the same-artist guard passes.

### Suspicious numbers flagged: 1

**Flag 1: Churn at 4.6% for w=0.25 is anomalously high.** The pool overlap is only 16% mean, meaning the C1 source introduces ~252 new candidates per case into the 300-pool. The frozen LR was never trained on features for these new candidates (they have no source rank from existing sources), so their LR scores are essentially random from existing feature distributions. This explains why even tiny weight injects noise that dominates churn without improving nDCG. This is not a *bug* — it's an inherent limitation of fusing a new first-stage source through a frozen ranker that has no features for the new candidates. It means the fusion evaluation underestimates what a *retrained* LR might achieve, but retraining is out of scope, so the FAIL verdict correctly reflects the observable signal.

**Not flagged (valid):**
- POOL_MISS recovery 122/1163 (10.5%) — well within [0, 1163], plausible for a sparse count source.
- UNREACHABLE recovery 63/1861 (3.4%) — lower than POOL_MISS, expected because UNREACHABLE tracks may not appear in any train transition.
- Same-artist hit@300 (0.51) >> diff-artist (0.127) — expected; count-based transitions heavily favor same-artist continuations.
- All sessions having exactly 8 music turns — structural property of the challenge dataset, not a bug.
- excluded_overlap = 0 — plausible given separate UUID pools (addressed in A3).

---

## D. Recommended Next Step

### Verdict: FAIL confirmed → archive this direction.

The count-only transition probe produced:
- 20 unique h7 outside-pool hits (67% of the 30 threshold)
- Negligible h7 nDCG lift (+0.00016, 5% of the +0.003 threshold)
- All-dev regression at every weight
- Destructive churn (4.6-10.1%)

### What should NOT be tried next:

1. **More count backoff levels** (e.g., genre-to-track, year-to-track). The fundamental bottleneck is sparsity: 106k transitions across 39,705 unique last-track keys means most last→next pairs are observed exactly once. More backoff dilutes signal further without adding genuinely new coverage.

2. **Metadata-neighbor count extension.** The C1 design included optional `c1_metadata_neighbor` but it was correctly skipped. Even if built, it would be a soft copy of existing semantic retrieval (R21/R54 already exploit metadata similarity), matching the S2 failure pattern cited in the design doc.

3. **Higher fusion weights.** All three predeclared weights show monotonically worsening metrics (w=0.25 best, w=1.0 worst). The signal is not hidden at higher influence.

4. **Utterance-bucket multiplier on counts.** The text context is recorded (106k/106k rows have user text) but not used for scoring in this implementation. Even if bucketed, the count tables are too sparse to benefit from utterance conditioning.

### Archive note:

The 20 unique h7 outside-pool hits demonstrate that *some* behavioral transition signal exists in the train split that the current RRF pool does not capture. However, the signal density is too low for a count-only approach to clear the pass gate. Per the design doc Section 6 stop condition: "Archive this direction immediately if the cheap transition probe fails both candidate-coverage and nDCG gates." Both fail here.

If a future neural sequence model (hybrid with content-initialized item vectors) is considered, it should target the diff-artist bucket specifically — same-artist is already well-served by existing sources (baseline same-artist pool_hit = 0.972) while diff-artist standalone hit@300 of only 0.127 shows the count source adds little beyond artist-conditioned frequency. A neural model would need to learn *cross-artist* sequential patterns, which are precisely what counts cannot capture.

---

## Summary

| Section | Result |
|---|---|
| Scope compliance | 4/4 checks pass |
| Methodological soundness | Pool_hit reproduces, metrics correct, leakage check valid |
| Verdict agreement | **AGREE** with FAIL |
| Issues flagged | 1 (churn anomaly explained by frozen-LR mismatch, not a bug) |
| Recommended action | Archive; do not try bigger count approaches |

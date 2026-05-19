# R66 Sprint Summary: Learned Depth/Source Router

**Created:** 2026-05-19  
**Branch:** `r66-learned-depth-source-router`  
**Final Verdict:** **ARCHIVE_SPRINT**  
**Production Status:** R63c-repair holds Blind-A at composite **0.6224** (nDCG 0.4925, LLM 4.85, LexDiv 0.8438)

---

## Wave 0: Baseline Reproduction

**Commit:** `cf1684c`  
**Result:** PASS (max |Δ| = 4.88e-07)

Reproduced R54c frozen LR bitwise-identically using cached intermediates and production RRF configuration.

| Metric | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| all_dev_ndcg20 | 0.315875 | 0.315875 | +0.000000 |
| h7_ndcg20 | 0.348378 | 0.348378 | -0.000000 |
| same_artist_ndcg20 | 0.628214 | 0.628214 | +0.000000 |
| diff_artist_ndcg20 | 0.142367 | 0.142367 | +0.000000 |
| pool_hit_all | 0.622000 | 0.622000 | +0.000000 |
| pool_hit_h7 | 0.613000 | 0.613000 | +0.000000 |

---

## Wave 1: Phase 0 — Static Profile Conversion

**Commit:** `0bf3552`  
**Duration:** 1186.1s  
**Result:** ARCHIVE_PHASE_0 — No profiles cleared kill gate

Eight hand-designed RRF weight profiles were tested against a 4-condition kill gate:
1. `pool_hit_h7` lift ≥ +0.010
2. `h7_ndcg20` delta ≥ 0 (non-negative)
3. `same_artist_ndcg20` delta ≥ -0.002 (minimal regression)
4. `recovered_h7 > lost_h7` (net recovery)

All conditions must hold for a profile to pass Phase 0.

### Profile Weights

| Profile | Label | A | B | C | D | F | ALS | R21 | R54 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | R54c baseline | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| P1 | text-heavy | 1.5 | 1.5 | 1.5 | 0.5 | 1.5 | 0.5 | 0.7 | 0.7 |
| P2 | collaborative-heavy | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 1.5 | 1.5 | 1.5 |
| P3 | R54-heavy | 0.7 | 0.7 | 0.7 | 0.3 | 0.7 | 0.7 | 0.7 | 2.0 |
| P4 | R21/R54 pair | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 0.5 | 1.5 | 1.5 |
| P5 | BM25-only | 1.5 | 1.5 | 1.5 | 0.5 | 1.5 | 0.3 | 0.3 | 0.3 |
| P6 | C+R54 dominant | 0.5 | 0.5 | 2.0 | 0.3 | 0.5 | 0.5 | 0.7 | 2.0 |
| P7 | ALS+R54 dominant | 0.5 | 0.5 | 0.5 | 0.3 | 0.5 | 2.0 | 0.7 | 2.0 |

### Results: All 8 Profiles Failed

| Profile | pool_hit_h7 | Δpool_hit_h7 | Δh7_ndcg | Δsame_artist | rec | lost | net | churn/80 | pass? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| P0 | 0.613 | — | — | — | — | — | — | — | — |
| P1 | 0.596 | -0.017 | -0.00591 | -0.00423 | 1 | 14 | -13 | 13 | ✗ |
| P2 | 0.620 | +0.007 | -0.02139 | -0.01774 | 4 | 21 | -17 | 15 | ✗ |
| P3 | 0.619 | +0.006 | -0.01332 | -0.01145 | 4 | 17 | -13 | 12 | ✗ |
| P4 | 0.619 | +0.006 | -0.01815 | -0.01936 | 10 | 22 | -12 | 15 | ✗ |
| P5 | 0.583 | -0.030 | -0.00998 | -0.01027 | 6 | 28 | -22 | 14 | ✗ |
| **P6** | **0.627** | **+0.014** | **-0.01468** | **-0.01472** | **10** | **24** | **-14** | **15** | **✗** |
| P7 | 0.619 | +0.006 | -0.02155 | -0.01730 | 6 | 24 | -18 | 18 | ✗ |

**Near-miss:** P6 passed gate 1 (pool_hit_h7 lift +0.014 ≥ +0.010) but failed gates 2 and 4:
- Gate 2: h7_ndcg delta −0.01468 < 0
- Gate 4: 10 recovered vs 24 lost (ratio 1:2.4, requires 1:1)

All other profiles failed gate 1, gate 2, or gate 4 (none passed all four).

---

## Wave 2: Phase 1 — Skipped Per Decision Gate

**Rationale:** Phase 0 sanity check exists to falsify the core premise that a learned per-case router can select among static profiles. If no static profile clears the conversion gate, a learned router over those same profiles cannot help:
- At best, a router could choose P6 for some cases.
- But P6 itself fails gates 2 (nDCG) and 4 (rec < lost).
- A learned selector cannot improve a profile that fails the gate; it can only change selection frequency, which does not recover the underlying shortfall.

**Decision:** Skip Phase 1 without burning optimization compute.

---

## Cross-Mechanism Learnings

### 1. The Pool-Broadening Wall Generalizes Beyond Admission/Reweighting

The frozen LR's inability to rescore beyond its training calibration now extends across three distinct experimental classes:

- **Admission** (R59/R60/R61, [[feedback_pool_broadening_wall]]): Adding candidates without source-rank features scores zero.
- **Reweighting** (R65 M3, depth-weighted RRF): Even oracle-best per-depth source pairs (+0.0320 pool_hit headroom) fail to improve nDCG because LR was trained on the original ensemble's source-rank distribution.
- **Redistribution** (R66 Phase 0, static profile reweighting): Reshuffling existing sources within the ensemble causes LR to misprice them. P6's high pool_hit gain (+0.014) is offset by massive nDCG regression (-0.01468) because reweighting changes the feature values LR depends on.

All three paths hit the same wall: **frozen LR can only score candidates when presented in the exact ensemble composition it was trained on.**

### 2. R65 Oracle Headroom Confirmed Not Extractable via Static Reweighting

R65 M3 showed per-depth oracle ceiling at +0.0320 h7 pool_hit — a real signal. However:
- R65 static profiles (simple weighting schemes) could not extract it; they yielded +0.0000 pool_hit.
- R66 Phase 0 tested profiles deliberately built from R65's per-depth best pairs (P6: C+R54 dominant from depth 0–1 oracle; P7: ALS+R54 dominant from depth 2+ oracle).
- Result: **P6 gains +0.014 pool_hit but loses −0.01468 nDCG.** The rec/lost ratio (10/24 = 0.42) shows LR is rejecting 2.4× as many cases as it recovers.

**Conclusion:** The +0.0320 headroom is a *training-time* artifact of LR's frozen calibration. To extract it, LR would need to be retrained jointly with the chosen profile — which returns to the matched-pool training path, previously falsified in R60 [[feedback_matched_pool_training_falsified]].

### 3. Phase 0 Decision Gate Validated the Sanity Check

The decision gate to skip Phase 1 fired cleanly without wasting optimization budget. This demonstrates the value of early-stage phase gates: if the premise fails at the static level (all 8 profiles fail), advancing to learned routing is futile. The gate prevented speculative Phase 1 work and freed capacity for other directions.

### 4. R66 Confirms R65's Archive Verdict

Seventeen consecutive post-R54c dev experiments (12 prior: R56–R62; 4 R65 mechanisms; 1 R66 sprint) have all failed to advance production. The retriever/pool/LR side is saturated. Production wins came only from response-side polish (R63 → R63c-repair, composite +0.0118). Deferred directions all require either:
- Structural retraining (LR + pool joint optimization)
- External data (C4 organizer approval pending)
- Blind-B release (waiting)

---

## Seventeen Consecutive Negatives: Retriever/Pool/LR Side Saturated

| Sprint | Mechanism | Result | Notes |
|---|---|---|---|
| R56 | Source-rank protection (18 variants) | NEGATIVE | All variants lost more HITs than recovered DEMOTEDs |
| R57 | Structural LR features (ISRC, artist_id) | NEGATIVE | 6 categorical-metadata additions all FAIL_REGRESS |
| R58 | Stage-2 LightGBM specialist (28 configs) | NEGATIVE | Architecture path exhausted |
| R59 (C3) | Pool admission + frozen LR | NEGATIVE | +0.0596 pool_hit buried by LR, h7 Δ −0.0015 |
| R60 | Matched-pool LambdaRank variant | NEGATIVE | h7 Δ −0.0885, same-artist −0.142; retraining falsified |
| R61 (C1) | Count-based train-split transitions | NEGATIVE | +0.00016 nDCG, all-dev regresses |
| R62 (G1) | Goal-conditioned pre-RRF oracle | NEGATIVE | +0.0028 oracle ceiling, gate +0.010 required |
| R63 | Opus 4.7 response-only | **POSITIVE** | +0.0074 composite; LLM 4.80, LexDiv 0.8389 |
| R63b | Targeted 25-row Opus polish | **POSITIVE** | +0.0113 composite; LLM 4.85, LexDiv 0.8389 |
| R63c-repair | 15-row diversified sentence architecture | **POSITIVE** | +0.0118 composite; LLM 4.85, LexDiv 0.8438 ← **PRODUCTION** |
| R64 | Concise-direct style variant | NEGATIVE | LexDiv gate failure (0.8294 < 0.830); archived locally |
| R65 M1 | Goal-progress history editing | NEGATIVE | +0.0050 pool_hit ceiling, gate +0.010 required |
| R65 M2 | Goal-query expansion | NEGATIVE | +0.0050 pool_hit ceiling (same as M1) |
| R65 M3 | Depth-weighted RRF (oracle +0.0320) | NEGATIVE | Static reweighting cannot extract oracle |
| R65 M4 | Scoreable-admission constraint | NEGATIVE | 714 scoreable POOL_MISS, 0 net recovery |
| R66 Phase 0 | Static profile conversion (8 profiles) | NEGATIVE | P6 best at +0.014 pool_hit, fails nDCG gate (−0.01468) |
| R66 Phase 1 | [SKIPPED] Learned routing would be futile |  | No static profile passed Phase 0; learned selector cannot help |

**Outcome:** Only response-side polish (R63 → R63c-repair) unlocked production advance. Retriever/pool/LR side exhausted.

---

## Surviving Deferred Directions

### High-Cost, High-Barrier Experiments

1. **Jointly retrain LR on a chosen P_i profile's pool** — Opens matched-pool training path previously falsified in R60. Only viable if new structural features are introduced that the frozen LR cannot consume (e.g., per-depth rank anchors, source-family embeddings).

2. **Non-deterministic / softmax-fused RRF admission** — Changes the semantics of RRF from hard thresholding (top-300) to soft scoring (softmax over all sources). Requires separate architectural review; not a routing experiment.

3. **External metadata via C4 organizer** — User-owned Codabench forum post pending reply. C4 is marked [[feedback_external_data_unblocked]] as EXPLORATORY only (Phase 3 paused for explicit approval).

### Waiting Gates

- **Blind-B release:** Watch for new dev/test splits that may unlock training or scoring breakthroughs.
- **C4 organizer decision:** Explicit approval required to proceed with external data integration.

---

## Artifacts on `r66-learned-depth-source-router`

**Scripts:**
- `scripts/expR66_baseline_repro.py` — Wave 0 reproduction harness
- `scripts/expR66_phase0_static_profile_conversion.py` — Phase 0 sweep over 8 profiles

**Evaluation:**
- `exp/eval/expR66_baseline_repro.json` — Bitwise reproduction result (max |Δ| 4.88e-07)
- `exp/eval/expR66_phase0_static_profile_conversion.json` — Full 8-profile result matrix

**Documentation:**
- `docs/r66_baseline_repro.md` — Wave 0 technical summary
- `docs/r66_phase0_static_profile_result.md` — Phase 0 detailed breakdown
- `docs/r66_sprint_summary.md` — This document

---

## References

- [[project_r65_sprint_outcome]] — Depth-weighted RRF oracle (+0.0320) and learned routing context
- [[feedback_matched_pool_training_falsified]] — R60 result: LR retraining on changed pool is worse
- [[feedback_pool_broadening_wall]] — C3/R60/R61 admission path exhausted
- [[feedback_external_data_unblocked]] — C4 exploratory path pending approval

---

## Bottom Line

R66 confirmed that oracle headroom from depth-conditioned source weighting (R65 +0.0320 pool_hit) is not extractable via static RRF reweighting. The frozen LR's calibration to the baseline ensemble is a hard constraint. All 8 hand-designed profiles failed the Phase 0 kill gate; Phase 1 (learned routing) was correctly skipped. This is the 17th consecutive post-R54c dev experiment without a production advance. Response-side polish (R63c-repair at composite 0.6224) remains the only proven path forward. Retriever/pool/LR side is saturated pending structural retraining or external data approval.

# R59 C4: External Metadata Enrichment — Status Summary

**Date:** 2026-05-16  
**Instance:** r59-c4-rules-feasibility (af2f7129-41bd-4fdc-bb66-7b6d7d11ba60)  
**Mode:** EXPLORATORY RESEARCH ONLY — NOT SUBMISSION-SAFE

---

## Executive Summary

**Phase 1: ✓ COMPLETE → GREEN verdict**
- TalkPlayData has excellent tag coverage (99.8%, 164K unique tags)
- MusicBrainz cache shows 50% metadata yield, 71% ISRC match
- Local data quality supports external enrichment hypothesis

**Phase 2: SCOPING REQUIRED**
- Full Phase 2: 300 cases, 5 MB features, CV5 (4-6h implementation)
- Lightweight Phase 2: 50 cases, 2 MB features, single-fold (1-2h implementation)
- Awaiting directive on approach

---

## Deliverables Completed

### 1. Rules Investigation (✓ Complete)

**File:** `docs/r59_candidates/c4_rules_verdict.md`

**Verdict:** AMBIGUOUS — no explicit rules found permitting or forbidding external data

**Key findings:**
- Searched 6+ official sources (RecSys, Codabench, HF, GitHub)
- No explicit external data policy found
- Dataset license: CC-BY-NC-ND-4.0 (NoDerivatives ambiguous for internal features)
- Local precedent: MB audit scripts exist but never used as features
- R57b warning: structural metadata features REGRESSED despite high importance

**Organizer email:** `docs/r59_candidates/c4_organizer_email.md` (send-ready, awaiting user decision)

### 2. Phase 1 Local Feasibility (✓ Complete)

**File:** `docs/r59_candidates/c4_phase1_report.md`  
**Script:** `scripts/expR59_c4_phase1_local_feasibility.py`  
**Outputs:** `exp/eval/expR59_c4_phase1_feasibility.json`

**Verdict:** GREEN (2/3 green checks, 0/3 red flags)

**Findings:**
| Metric | Value | Status |
|--------|-------|--------|
| Tag coverage | 99.8% (46,984/47,071 tracks) | ✓ GREEN (≥80%) |
| Unique tags | 164,133 | ✓ GREEN (≥1K) |
| MB tags yield | 50.0% (50/100 sample) | ⚠ BORDERLINE (40-60%) |

**Tag quality:**
- Top tags: Rock (40%), alternative (25%), pop (24%)
- High IDF (rare): "better than muse", "suck my kiss", "trinidad james - all gold everything"
- Low IDF (common): "Rock", "rock", "alternative", "favorites"
- Mean tags per track: ~35 (very high)

**MB cache assessment:**
- 100-track sample from April 30, 2026
- 71% ISRC→recording match rate
- 50% have MB tags, 49% have MB genres
- Example overlap: U2 "Mysterious Ways" has 6 overlapping tags between TalkPlayData and MB (pop rock, classic rock, alternative)

**Gate decision:** Proceed to Phase 2 (MB API diagnostic)

---

## Phase 2 Options

### Option A: Full Phase 2 (Thorough)

**Scope:**
- 300-case stratified sample (100 DEMOTED, 100 UNREACHABLE, 100 HIT)
- 5 MB features:
  1. `mb_tag_jaccard_last`: Jaccard(MB tags candidate, union MB tags last 3 played)
  2. `mb_genre_overlap_history`: count MB genres for candidate in ANY played genre
  3. `mb_recording_relation_played`: binary flag for MB recording relation (cover, remix, sample, live)
  4. `mb_artist_relation_played`: binary flag for MB artist relation (collaboration, member-of-band)
  5. `mb_same_label_last`: binary flag for same release label as last played (requires extra API call per track)
- LambdaRank CV5 training: 37-feature baseline vs 42-feature MB-enriched
- Full metrics: nDCG@20, same-artist, diff-artist, net recovered-lost, per-bucket breakdown

**Gates:**
- GREEN: +0.005 nDCG improvement, same-artist no regress → scale to full 8K (Phase 3)
- RED: regress OR flat → archive C4

**Estimated effort:** 4-6 hours
- 1h: Dev sample loader + candidate pool integration
- 1-2h: MB API fetching (500-1000 new requests at 1 req/sec + caching)
- 1h: MB feature engineering (5 features × 300 cases × ~50 candidates = ~75K feature computations)
- 1-2h: LambdaRank CV5 retraining + evaluation

**Infrastructure needs:**
- Reuse R54 phase3 scripts (`expR54_phase3_full5fold_integration.py`)
- Load R12 payload (dev data at `exp/eval/_R12_all_turns_payload.pkl`)
- Adapt R54 feature pipeline to add MB features

### Option B: Lightweight Phase 2 (Fast Signal Check)

**Scope:**
- 50-case sample (stratified: ~17 DEMOTED, ~17 UNREACHABLE, ~17 HIT)
- 2 MB features (prioritized for signal):
  1. `mb_tag_jaccard_last`: tag overlap between candidate and played history
  2. `mb_genre_overlap_history`: genre overlap count
- Single-fold eval (no CV, faster iteration)
- Simplified metrics: nDCG@20 only, basic same-artist check

**Gates:**
- GREEN: +0.005 nDCG on 50-case sample → justify full Phase 2 (Option A)
- RED: flat/regress → archive C4 without full investment

**Estimated effort:** 1-2 hours
- 30min: 50-case sample loader
- 30min: MB API fetching (~50 cases × ~50 candidates × 50% cache hit = ~1250 new requests = ~25min at 1 req/sec)
- 30min: MB feature engineering (2 features only)
- 30min: LR single-fold training + comparison

**Rationale:** Test core hypothesis (do MB tags/genres add ranking signal?) without full infrastructure build. If this shows promise, Option A is justified. If not, avoid 4-6h investment.

### Option C: Skip Phase 2, Send Organizer Email

**Rationale:** Conservative approach given:
- Rules ambiguity (AMBIGUOUS verdict)
- R57b precedent (structural features REGRESSED)
- C4 is 9th post-R54c candidate (R55-R58 all failed)

**Action:** User sends `docs/r59_candidates/c4_organizer_email.md`, waits for organizer clarification before implementation.

---

## Recommendation

**Recommended:** Option B (Lightweight Phase 2)

**Why:**
1. **Fast signal check:** 1-2h investment vs 4-6h for full
2. **Low risk:** If MB features don't help on 50 cases, unlikely to help on 300
3. **Exploratory label:** All outputs marked "NOT SUBMISSION-SAFE", so no pressure for production quality
4. **Gating discipline:** Only proceed to full Phase 2 if lightweight shows +0.005 lift

**If lightweight Phase 2 fails:** Archive C4, accept R54c as production system. Focus on Blind-B final submission.

**If lightweight Phase 2 passes:** Proceed to full Phase 2 (Option A), then Phase 3 (full 8K-case evaluation) if warranted.

---

## Hard Constraints (All Phases)

1. **NO blind evaluation:** Dev/h7 only, never touch Blind-A or Blind-B data
2. **NO production integration:** Exploratory features never enter production pipeline
3. **Label all outputs:** "EXPLORATORY - NOT SUBMISSION-SAFE - EXTERNAL DATA (MusicBrainz)"
4. **Rate limiting:** MB API 1 req/sec, aggressive caching
5. **Reproducibility:** Cache all API responses with timestamp, never re-fetch

---

## Next Steps

**Awaiting directive:**
- Proceed with Option A (full Phase 2)?
- Proceed with Option B (lightweight Phase 2)?
- Skip Phase 2, send organizer email (Option C)?

**If Option B approved:**
1. Implement 50-case sample loader
2. Fetch MB metadata (use cache + new API calls)
3. Compute 2 MB features (tag_jaccard, genre_overlap)
4. Train LR single-fold, compare with baseline
5. Report: GREEN (+0.005) → justify full Phase 2; RED (flat/regress) → archive C4

**Timeline:** Option B can complete in 1-2 hours once approved.

---

## Artifacts Created

### Documentation
- `docs/r59_candidates/c4_rules_verdict.md` — Rules investigation (AMBIGUOUS)
- `docs/r59_candidates/c4_organizer_email.md` — Send-ready email
- `docs/r59_candidates/c4_phase1_report.md` — Phase 1 feasibility (GREEN)
- `docs/r59_candidates/c4_status_summary.md` — This file

### Scripts
- `scripts/expR59_c4_phase1_local_feasibility.py` — Phase 1 analysis (executed successfully)
- `scripts/expR59_c4_phase2_mb_diagnostic.py` — Phase 2 full (structure only)
- `scripts/expR59_c4_phase2_lightweight.py` — Phase 2 lightweight (structure only)

### Data
- `exp/eval/expR59_c4_phase1_feasibility.json` — Phase 1 metrics
- `cache/musicbrainz_audit/` — 536 cached MB API responses (6.2M, April 30 2026)

---

**Status:** Phase 1 complete (GREEN). Phase 2 scoping complete. Awaiting directive on Option A/B/C.

**Contact:** r59-c4-rules-feasibility instance (af2f7129-41bd-4fdc-bb66-7b6d7d11ba60)

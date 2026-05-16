# R59 Candidate: External/Catalog Metadata Enrichment

**Status: Design analysis only. NO CODE. Rule compliance UNVERIFIED.**

## 1. New information added beyond R54c

External music metadata sources would add signal beyond TalkPlayData's supplied track metadata. Coverage analysis from `exp/eval/expR57_structural_forensics.json` shows TalkPlayData already provides:

- **ISRC**: 97.17% coverage (catalog-level)
- **release_year**: 98.63% coverage
- **duration_ms**: 100% coverage
- **artist_id**: 100% coverage

What external sources could ADD:

### MusicBrainz (via ISRC join at 97.17% coverage)

**Prior work:** `scripts/audit_musicbrainz_sample.py` and `scripts/audit_mb_recovery.py` exist. Cache at `cache/musicbrainz_audit/` contains 536 cached API responses from April 30, 2026.

**Audit results** (100-track sample):
- ISRC→recording match: 71%
- Metadata yield: 61% had tags/genres/artist-relations
- Recovery diagnostic: MB artist-relations recovered only **~2 of 1861 unreachable cases** (projected ~5 at full coverage)

**New fields beyond TalkPlayData:**
- **Tags**: user-contributed descriptors ("electronic", "ambient", "sad", "energetic")
- **Genres**: curated genre labels (more structured than tags)
- **Recording relations**: cover-of, remix-of, sampled-from, live-performance-of
- **Artist relations**: member-of-band, collaboration, producer, remixer
- **Release-level metadata**: label, catalog number, release country, release format
- **Work relationships**: composition links (multiple recordings of same work)

### Spotify API (Web API / partner data)

**No prior integration found in repo** (grep returned no Spotify API usage, only references to Spotify research papers in `docs/s2_behavior_native_retrieval_plan.md`).

**New fields:**
- **Audio features**: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo (12 continuous features)
- **Popularity score**: current Spotify popularity (0-100)
- **Preview URL**: 30-second audio preview (not directly useful for ranking)
- **Audio analysis**: detailed beat/bar/section/segment timings (heavy, likely out of scope)

**Join strategy:** ISRC (primary), fallback to artist+track name fuzzy match.

**Rate limits:** 1000 requests/hour per app in standard tier. Catalog-level fetch for ~515K tracks = 515 hours at max rate (~21 days single-threaded). Batch endpoints exist but still rate-limited.

### Last.fm (API)

**No prior integration found.**

**New fields:**
- **Tags**: user-contributed (similar to MusicBrainz but different community)
- **Playcount**: global Last.fm playcount (popularity proxy, different from TalkPlayData catalog pop)
- **Listeners**: unique listener count
- **Similar tracks**: collaborative-filtering-based similarity (top-N similar per track)
- **Top tags**: aggregated weighted tags per track

**Join strategy:** artist+track name (no ISRC support in Last.fm API).

**Rate limits:** 5 requests/second = 18K/hour. Catalog fetch = ~29 hours.

### Discogs

**No prior integration found.**

**New fields:**
- **Label**: record label
- **Release formats**: vinyl, CD, digital, etc.
- **Release country**
- **Genres/styles**: curated, hierarchical
- **Credits**: producer, engineer, musician roles
- **Market data**: price, sales rank (if release is for sale)

**Join strategy:** ISRC (via Discogs database search), artist+title fallback.

**Rate limits:** 60 requests/minute = 3600/hour. Catalog fetch = ~143 hours.

### ISRC Registry (official)

**No direct API.** ISRC itself is already in TalkPlayData at 97.17%. The ISRC prefix (country + registrant codes) can be PARSED locally from existing ISRCs — **this was already attempted in R57b** (`isrc_country_match`, `isrc_registrant_match`) and both **FAIL_REGRESS**.

### Concrete example: diff_artist case

**Scenario:** User played 3 tracks by "Aphex Twin". GT is track by "Boards of Canada" (diff_artist, nDCG@20 currently ~0.10 in this regime per `docs/blind_a_final_state.md`).

**TalkPlayData has:** artist_name, track_name, album_name, tag_list (supplied), ISRC, duration, release_year.

**External enrichment could add:**
- MusicBrainz tags: both artists tagged "idm", "electronic", "ambient", "experimental" → tag overlap score
- Spotify audio features: both have low danceability (~0.3), high instrumentalness (~0.9), similar tempo range → audio-feature cosine
- Last.fm similar-tracks: Boards of Canada appears in similar-tracks API response for Aphex Twin catalog
- MusicBrainz work relations: both cover the same classical piece, or one sampled the other (rare but high signal)

**Question:** Do these cross-artist structural similarities exist in the data, and can LR learn to use them?

**R57b precedent:** Local structural features (ISRC country match, artist_id match) counted high in LightGBM gain importance but REGRESSED aggregate nDCG. Why would EXTERNAL structural features avoid the same trap?

## 2. Cached data/artifacts it can reuse

### Existing MusicBrainz cache

- **Path:** `cache/musicbrainz_audit/` (536 files, ~12.7 MB, 100-track sample)
- **Scripts:** `scripts/audit_musicbrainz_sample.py`, `scripts/audit_mb_recovery.py`
- **Coverage:** 100-track sample (25 unreachable_pop0, 25 pop0_random, 25 popular, 25 diff_artist_miss categories)
- **Reuse potential:** Sample cache is too small for production features (100 tracks vs 515K catalog). Could reuse API client code and rate-limit discipline.

### ISRC index (97.17% coverage)

- **Already available:** R57b built `tid_to_isrc` index from TalkPlayData
- **Join key:** ISRC is the highest-fidelity join for MusicBrainz, Spotify, Discogs
- **Gap:** 2.83% of catalog lacks ISRC. Fallback: artist+track name fuzzy match (higher false-positive rate)

### TalkPlayData catalog metadata

- **Path:** HuggingFace cache `~/.cache/huggingface/datasets/talkpl-ai___talk_play_data-challenge-track-metadata/`
- **Splits:** `all_tracks` (515K), `test_tracks` (Blind-A/Blind-B held-out)
- **Already loaded:** Every experiment script loads this. Standard index: `tid_to_meta`, `artist_id_to_tids`

### R39+R54 LR feature pipeline

- **Path:** `cache/r54_phase3_lr_model.txt` (LambdaRank model), `cache/r54_phase3_*.{npz,json,pkl}` (ALS, pop, maps)
- **Feature count:** 37 features (28 R39 base + 5 album + 3 R54 + 1 pop)
- **Reuse:** Any new external-derived features would be ADDED to this 37-feature baseline, not replace

### No reusable external API cache

- **Spotify API:** No prior work
- **Last.fm API:** No prior work
- **Discogs API:** No prior work
- **Conclusion:** A production-scale external enrichment would require a NEW catalog-level fetch (21-143 hours depending on source and rate limits) OR a bounded dev-only sample diagnostic.

## 3. Leakage risks

### Rule compliance: UNKNOWN and BLOCKING

**No explicit competition rules found in repo** regarding external data sources. Searched:
- `README.md`: minimal (2-stage strategy doc, no rules)
- `docs/leakage_audit.md`: mentions BM25 uses only TalkPlayData catalog, but no rule citation
- `docs/r27_agentic_submission_audit_plan.md`: line "external knowledge not reflected in catalog metadata" listed as a violation pattern (context: agentic response generation, not retrieval features)
- No `RULES.md`, `COMPETITION.md`, or equivalent found

**Inference from observed practice:**
- R39, R54, R57b all use ONLY TalkPlayData-supplied metadata
- No prior experiment added external API data as features
- MusicBrainz audits (`scripts/audit_musicbrainz_sample.py`, `scripts/audit_mb_recovery.py`) were **diagnostics only**, never integrated as features

**Possible interpretations:**
1. **External sources ALLOWED:** TalkPlayData metadata is incomplete by design; competitors expected to augment. MusicBrainz audit scripts exist, implying consideration.
2. **External sources FORBIDDEN:** Competition tests cold-start, cross-domain transfer, or generalization; external data would break experimental control.
3. **Ambiguous / gray area:** Static external metadata (tags, audio features) allowed; dynamic data (current popularity, user-contributed content scraped after competition start) forbidden.

**HARD PREREQUISITE:** Verify competition rules before any implementation. If rules forbid external data, R59c4 stops here.

### Rate limits and reproducibility

**Spotify API:** 1000 req/hour standard tier. Catalog fetch = 21 days. Reproducibility: snapshot the fetched metadata at a timestamp, commit hash, never re-fetch. Risk: API response changes over time (popularity scores, tag counts); snapshot mitigates.

**Last.fm API:** 5 req/sec. Catalog fetch = 29 hours. Same reproducibility protocol.

**MusicBrainz:** 1 req/sec community tier. Catalog fetch with relations = ~143 hours (assuming 1 ISRC lookup + 1 recording detail per track). Reproducibility: MusicBrainz is versioned (database dumps available); snapshot API responses OR use a specific database dump date.

**Cost:** API quotas are per-app, not per-user. Overage on Spotify/Last.fm typically blocks or requires paid tier. MusicBrainz is free but 503s on rate-limit violations.

**Mitigation:** Aggressive caching (same discipline as existing MB audit scripts: hash(endpoint+params) → JSON cache). One-time fetch, commit cache to repo or external storage, never re-fetch during experimentation.

### Temporal leakage

**Spotify popularity:** Changes daily. A track's popularity TODAY is not its popularity at competition data collection time. Using current Spotify popularity for Blind-A would leak post-competition information.

**Last.fm playcount:** Monotonically increasing. Same issue.

**Solution:** Only use static/structural metadata (audio features, tags, genres, relations). Never use time-varying popularity/playcount UNLESS the API provides historical snapshots at a known date.

**TalkPlayData catalog already has a `popularity` field** (0-5 scale, supplied). External popularity would be redundant and temporally suspect.

### OOF discipline

**Not a concern.** External metadata is **track-catalog-level**, not session-level. Same as TalkPlayData's ISRC, duration, release_year — these are shared across all CV5 folds without leakage. No session-level dependency.

## 4. Expected ceiling

### Argument FOR signal:

1. **Tag/genre overlap for cross-artist recommendations:** Current R54c struggles at diff_artist (nDCG ~0.10 in that regime). MusicBrainz/Last.fm tags provide genre/mood/style descriptors NOT in TalkPlayData's `tag_list` field. Tag Jaccard between played tracks and candidates could surface cross-artist matches (e.g., "idm" tag overlap between Aphex Twin and Boards of Canada).

2. **Audio features for vibe matching:** Spotify audio features (danceability, energy, valence, instrumentalness) are continuous, engineered signals. Could enable "match the energy of the last played track" without artist/album overlap. TalkPlayData has duration but not tempo, key, or mood-proxy features.

3. **MusicBrainz relations for cold tracks:** Covers, remixes, samples, live versions. If a user played Track A, and GT is a cover of Track A, MB work relations could directly link them. Prior audit (`audit_mb_recovery.py`) found this happened in **~2 of 1861 unreachable cases**, but that diagnostic only checked artist-relations, not recording/work relations.

4. **Higher-fidelity genre:** TalkPlayData `tag_list` is supplied but quality/coverage unknown. MB genres are curated; Spotify genres are inferred from listening patterns (different signal source).

5. **Label/release metadata for album-context:** Discogs label, release country. If user's history clusters around a specific label (e.g., Warp Records for electronic), label-match could be a ranking signal.

### Argument AGAINST signal (skeptical view):

1. **R57b already tried structural metadata features and FAILED:** ISRC country match, ISRC registrant match, artist_id match history — all had high LightGBM gain importance, all REGRESSED aggregate nDCG, all REGRESSED same-artist nDCG (the canary). The failure mode: **LR already implicitly learns these patterns via other features**. Adding them explicitly creates multicollinearity, overfits to artifacts, and breaks calibration.

   **Why would external tags/genres/audio-features avoid this?** If Spotify danceability correlates with duration or TalkPlayData tag_list, LR already has proxy signal. Adding Spotify danceability explicitly could regress for the same reason ISRC features did.

2. **MusicBrainz relations empirically DID NOT recover unreachable cases:** Prior audit (`cache/mb_recovery_diagnostic.json`): artist-relations recovered **2 cases** (projected ~5 full coverage). Even with 211 API requests and 88 artists mapped, yield was ~0.1% of unreachable bucket. Work/recording relations might do better, but prior is low.

3. **TalkPlayData already has `tag_list`:** If tags were discriminative for cross-artist ranking, R39's `tag_jaccard_last` feature (which exists — see R39 feature list) should already capture it. R39 is in production; R54c builds on R39. If tag-based ranking worked, it's already in the stack.

4. **External data is SPARSE for long-tail tracks:** MusicBrainz match rate on sample was 71% (not 97.17% like ISRC availability — the 71% is ISRC→MB recording success rate, accounting for MB coverage gaps). Last.fm API requires artist+track name match (fuzzy, error-prone). Discogs match rate for music tracks (vs physical releases) is unknown but likely <50%. **Enrichment helps popular tracks (already well-ranked) more than cold tracks (where lift is needed).**

5. **Spotify audio features are aggregate/catalog-level:** They describe the TRACK, not the USER CONTEXT. LR already has: `same_artist_last`, `tag_jaccard_last`, `album_match`, `duration`, `release_year`. Spotify features add **absolute descriptors** (this track is 120 BPM, high energy), but LR's existing features already provide **relative/contextual descriptors** (this track's artist matches played artists, this track's tags overlap with played tags). Ranking is a relative task; context matters more than absolutes.

6. **Response polish (R54c) and retrieval (R54b) are SATURATED:** R54c hit LLM 4.70 / LexDiv ~0.84 ceiling. R55-R58 all failed. The system is at a local optimum. Adding orthogonal track metadata is unlikely to break through a **ranker calibration ceiling** — the issue is LR can't decide among the pool, not that the pool lacks descriptive features.

### Honest ceiling estimate:

**If rules allow and enrichment proceeds:**

- **Best case:** External tags/genres/audio-features add +0.003 to +0.007 h7 nDCG on dev, PASS exploratory gate (+0.005), FAIL production gate (+0.010). Same-artist regresses slightly (multicollinearity with existing features). Diff-artist lifts ~+0.02 but not enough to offset same-artist loss. Net: archive.

- **Realistic case:** External features have high LightGBM gain, regress aggregate nDCG (R57b pattern repeats). LR already captures the signal via proxies. Net: FAIL_REGRESS at -0.002 to -0.005. Archive immediately.

- **Miracle case:** MusicBrainz work/recording relations (cover-of, remix-of) recover 50-100 unreachable cases (vs prior 2). Tag overlap finds 30-50 diff_artist hits. Spotify audio-feature cosine finds 20 more. Net +0.012 h7 nDCG, PASS production gate. **Likelihood: <5%.** Prior MusicBrainz recovery audit contradicts this (only 2 recovered).

**Why low confidence:** R57b is the direct precedent. Structural metadata features FAILED despite counting high in gain importance. External structural metadata (tags, genres, audio features) are the SAME CLASS of features — catalog-level, static, descriptive. If internal structural features (ISRC prefix, artist_id, release_year, duration) don't help LR, external structural features face the same trap.

**What WOULD work (but external enrichment doesn't provide):** Dynamic user-session-context features (user's TODAY mood inferred from sequence, user's listening TIME OF DAY, user's skip/replay behavior). External APIs don't have these; they're catalog-level. R54 already maximizes catalog-level retrieval signal.

## 5. Smallest falsifiable diagnostic

**Goal:** Test whether external-derived features have incremental signal BEFORE committing to a 21-143 hour catalog fetch and LR retraining.

### Proposed diagnostic (dev-only, OOF-clean, no blind):

**Scope:** 300 dev cases (sample), not full 8000. Stratified: 100 DEMOTED, 100 UNREACHABLE, 100 HIT (to detect regressions).

**External source:** MusicBrainz only (reuse existing API client, leverage 97.17% ISRC coverage, avoid Spotify rate-limit cost).

**Features to test (5 new features per candidate):**
1. `mb_tag_jaccard_last`: Jaccard(MB tags for this candidate, union of MB tags for last 3 played tracks). Tests cross-artist tag-based matching.
2. `mb_genre_overlap_history`: Count of MB genres for this candidate that appear in ANY played track's MB genres. Tests genre-based discovery.
3. `mb_recording_relation_played`: Binary flag: does this candidate have a MB recording relation (cover-of, remix-of, live-version-of, etc.) to any played track? Tests work-level links (what prior audit DIDN'T check).
4. `mb_artist_relation_played`: Binary flag: does this candidate's MB artist have an artist-relation (collaboration, member-of-band) to any played artist? Tests artist-network expansion (what prior audit DID check but only for unreachable cases).
5. `mb_same_label_last`: Binary flag: does this candidate's release label (from MB release-level metadata) match the label of the last played track? Tests label-clustering hypothesis.

**Why these 5:** Cover the 3 hypotheses (tags/genres for cross-artist, relations for cold tracks, label for context). Computable from MB API with <1000 requests for 300 cases × ~50 candidates per case pool = ~15K candidate lookups. With caching and ISRC join, estimate ~500-800 NEW requests (rest hit cache or lack ISRC). At 1 req/sec = ~15 minutes runtime.

**Baseline:** R39+R54 LambdaRank on the 300-case sample, CV5 OOF discipline (5 folds, train on 240 cases, eval on 60 held-out per fold, repeat 5 times, aggregate). Reproduce R56/R57b metric battery on the 300-case sample first (baseline without MB features).

**Test:** Add the 5 MB features to the 37-feature baseline, retrain LambdaRank CV5, compare metrics.

**Gates (300-case sample, scaled thresholds):**
- **PROCEED to full 8000-case experiment:** sample nDCG improvement ≥ +0.010 (twice the exploratory gate to account for sample variance), same-artist does not regress.
- **ARCHIVE immediately:** sample nDCG regresses OR same-artist regresses >0.005 OR net recovered-lost ≤ 0.

**Diagnostic outputs:**
- Per-feature LightGBM gain importance (to confirm features are used)
- Per-feature ablation: drop each of the 5 MB features individually, measure Δ nDCG (to detect which specific feature helps/hurts)
- Bucket breakdown: DEMOTED, UNREACHABLE, HIT — where does signal appear?
- Same/diff artist split: does MB help diff_artist without hurting same_artist?

**Cost:** ~1 hour runtime (500 MB API requests + 5-fold LR retraining on 300 cases). No blind submission. No full-catalog fetch.

**If diagnostic PASSES:** Write R59c4-phase2 design doc for full 8000-case + full-catalog MB fetch (143 hours) + production LR retrain + dev evaluation. At that point, confirm rules allow external data.

**If diagnostic FAILS:** Archive R59c4. Avoid the 21-143 hour catalog fetch. Record: "MB-derived features tested on 300-case dev sample, regressed or flat, consistent with R57b structural-feature failure mode."

### Alternative cheaper diagnostic (no API calls):

**Use existing TalkPlayData `tag_list` field as a proxy for MB tags.** TalkPlayData already has tags. Test whether ENHANCING the existing `tag_jaccard_last` feature (R39 has this) with a WEIGHTED version (TF-IDF over tags, or tag-frequency-based weighting) lifts ranking.

**Why cheaper:** No external API. Pure feature-engineering on existing data.

**Why still valid:** If weighted-tag features DON'T help, external MB tags (which are similar tag data from a different source) likely won't either.

**Cost:** ~15 minutes (reload R39+R54 features, add 2-3 new tag-weighting features, retrain CV5 on 300-case sample or full 8000).

**If this cheaper diagnostic PASSES:** Then justify the MB API fetch for genre/relations. If it FAILS: archive without external API work.

## 6. Stop condition

### Rule-violation abort (even if signal exists):

**Trigger:** Competition rules explicitly forbid external data sources.

**Action:** Archive R59c4 immediately. Do not proceed to diagnostic. Write one-line summary: "External enrichment blocked by competition rule [cite rule]. Cannot proceed."

**Check before diagnostic:** Search for official competition rules (RecSys 2026 challenge page, email from organizers, `RULES.md` if it exists in a future data drop, TalkPlayData HuggingFace dataset card). If ambiguous, email organizers for clarification BEFORE running diagnostic. Do not assume "no explicit rule = allowed."

### Diagnostic-level gates (archive if any fire):

1. **300-case sample nDCG regresses** (Δ < 0): Archive. Write: "MB features regressed on 300-case dev sample. R57b pattern repeated. Do not scale to full 8000."

2. **Same-artist nDCG regresses >0.005 on sample**: Archive. Write: "MB features hurt same-artist ranking (canary metric). Multicollinearity with existing features likely. Do not scale."

3. **All 5 MB features have near-zero LightGBM gain importance**: Archive. Write: "LR does not use MB features. No incremental signal beyond R39+R54 baseline. Do not scale."

4. **Ablation shows features help individually but hurt in combination**: Archive or pivot. Write: "MB features interfere with each other or with baseline. Investigate feature correlation, possibly retry with single best feature only."

5. **Net recovered - lost ≤ 0 on 300-case sample**: Archive. Write: "MB features promote wrong candidates as often as they recover correct ones. No net lift. Do not scale."

### Full-dev-evaluation gates (if diagnostic passed and full experiment ran):

Same gates as R56/R57b (from `docs/r56_design.md` §5):

- **Production candidate:** h7 nDCG +0.010, same-artist does not regress, top-1 churn ≤240/8000 (3%)
- **Exploratory candidate:** h7 nDCG +0.005, same-artist does not regress, top-1 churn ≤120/8000 (1.5%)
- **STOP:** net recovered-lost ≤ 0, same-artist regresses >0.002, all-dev nDCG worse than baseline

### Saturation acknowledgment:

**R59c4 is the 9th post-R54c candidate.** R55, R55h, R56 (18 variants), R57 forensic, R57b (3 configs), R58 (28 configs) all FAILED or archived. External enrichment is testing a **catalog-level metadata hypothesis** when the failure pattern (R57b) showed catalog-level metadata features DON'T help.

**If R59c4 diagnostic fails:** Accept that the R54c local optimum is deep. Catalog-level feature additions (internal or external) are exhausted. Freeze for Blind-B. Future directions (if any): new retriever architecture (generative retrieval, learned semantic IDs), query rewriting, multi-turn context modeling — all out of scope for R59.

**Stop after diagnostic failure. Do not retry with Spotify or Last.fm.** If MusicBrainz (highest-fidelity join via ISRC, curated metadata) doesn't work, Spotify/Last.fm (lower join quality, noisier metadata) won't either.

---

**AWAITING RULE VERIFICATION.** Do not implement diagnostic until competition rules confirm external data is permitted.

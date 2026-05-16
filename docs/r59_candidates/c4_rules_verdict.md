# R59 C4: External Metadata Enrichment — Rules Verdict

**Investigation Date:** 2026-05-16  
**Investigator:** r59-c4-rules-feasibility research instance  
**Target:** MusicBrainz, Spotify, Last.fm, Discogs metadata enrichment for RecSys 2026 Music-CRS Challenge

---

## VERDICT: AMBIGUOUS

**External data usage is neither explicitly permitted nor explicitly forbidden by published competition rules.**

**RECOMMENDATION:** Contact organizers for clarification before proceeding with any external enrichment implementation or API fetches.

---

## Evidence

### 1. Official Competition Sources Searched

| Source | URL | External Data Rules Found? |
|--------|-----|---------------------------|
| RecSys Challenge 2026 (main) | https://www.recsyschallenge.com/2026/ | **NO** — Rules page exists but does not explicitly address external data |
| Codabench Competition Page | https://www.codabench.org/competitions/15786/ | **NO** — Navigation present but no detailed rules displayed |
| ACM RecSys Challenge Page | https://recsys.acm.org/recsys26/challenge/ | **NO** — Overview only, refers to recsyschallenge.com for details |
| nlp4musa music-crs-evaluator | https://github.com/nlp4musa/music-crs-evaluator | **NO** — Submission format documented, no data source restrictions |
| HuggingFace Dataset Card | https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata | **NO** — License visible (CC-BY-NC-ND-4.0), but no explicit enrichment policy |

**Quote from recsyschallenge.com/2026:**
> "Pre-extracted embeddings are supplied to ensure focus on model architecture"

**Implication:** The provision of embeddings and comprehensive metadata *suggests* participants should rely on supplied resources, but this is not stated as a prohibition.

**Quote from music-crs-evaluator README:**
> "Track IDs must match those in TalkPlayData-Challenge-Track-Metadata"

**Implication:** Track recommendations must come from the provided catalog (no external track injection), but this does not address *enriching metadata* for existing catalog tracks.

### 2. Dataset License

**TalkPlayData-1 and TalkPlayData-2:** CC-BY-NC-ND-4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0)

**Source:** https://huggingface.co/datasets/talkpl-ai/TalkPlayData-1, https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2

**Interpretation:**
- **NC (NonCommercial):** Challenge submissions are for research/academic purposes, likely compliant
- **ND (NoDerivatives):** Prohibits creating derivative datasets and redistributing them. **CRITICAL:** This may prohibit *publishing* an enriched catalog (e.g., TalkPlayData + MusicBrainz tags as a new dataset), but does not clearly prohibit *using* external metadata internally for ranking without redistributing the merged data.
- **Ambiguity:** Using external metadata as *features* for a model (without republishing the enriched catalog) may or may not violate ND clause depending on interpretation.

### 3. Local Repository Precedent

**Evidence of prior MusicBrainz investigation:**

| Artifact | Purpose | Status |
|----------|---------|--------|
| `scripts/audit_musicbrainz_sample.py` | MusicBrainz ISRC lookup audit on 100-track sample | **Diagnostic only** |
| `scripts/audit_mb_recovery.py` | Test if MB artist-relations recover unreachable cases | **Diagnostic only** |
| `cache/musicbrainz_audit/` | 536 cached MB API responses (~12.7 MB) from April 30, 2026 | **Never used as features** |
| `cache/mb_audit_results.json` | 100-track sample results: 71% ISRC match, 61% metadata yield | **Diagnostic only** |

**Quote from `scripts/audit_musicbrainz_sample.py:21`:**
```python
USER_AGENT = "RecsysResearch/0.1 (barkai.nadav@gmail.com)"
```

**Interpretation:** MusicBrainz was contacted for research/audit purposes, but the results were never integrated as retrieval features. This precedent suggests:
1. External API exploration has been considered acceptable for *diagnostic* purposes
2. The team has not yet crossed the line to *production feature integration*
3. This may reflect uncertainty about rules OR a conservative interpretation

**Evidence from R27 agentic submission audit plan (`docs/r27_agentic_submission_audit_plan.md:171`):**

> "Reject changes based on:
> - general music taste
> - **external knowledge not reflected in catalog metadata**
> - 'this feels better' without constraint evidence"

**Context:** This rule applies to **response generation and track selection** in agentic review (R27), where agents must not *invent* track IDs or justify recommendations using knowledge not derivable from the provided catalog.

**Interpretation:** This is a *leakage prevention* rule for response text and track ranking *decisions*, not a blanket prohibition on enriching the catalog with external structural metadata (tags, audio features) as *features* for the ranker. However, it does show the team is sensitive to using information "not reflected in catalog metadata."

### 4. R57b Precedent: Structural Features Failed

**Evidence from `docs/r59_candidates/c4_external_enrichment.md §4`:**

> "R57b already tried structural metadata features and FAILED: ISRC country match, ISRC registrant match, artist_id match history — all had high LightGBM gain importance, all REGRESSED aggregate nDCG, all REGRESSED same-artist nDCG (the canary). The failure mode: **LR already implicitly learns these patterns via other features**."

**Relevance to external enrichment:**
- R57b added *internal* structural features (ISRC prefix parsing, artist_id match) derived from TalkPlayData itself
- These features had high model importance but caused ranking regression
- External structural features (MusicBrainz tags, Spotify audio features) are the **same class** of catalog-level descriptive metadata
- **Risk:** External enrichment may repeat R57b's failure mode even if rules allow it

**Note:** This is a *feasibility* concern, not a *rules* concern, but relevant to the go/no-go decision.

---

## Arguments FOR "External Data Allowed" (Inference)

1. **No explicit prohibition found** after extensive search of official sources
2. **MusicBrainz audit scripts exist** in the repo, suggesting exploration was considered acceptable
3. **Synthetic dataset:** TalkPlayData is described as "synthetic, generated through an advanced agentic pipeline with no SiriusXM or Deezer user data" (source: recsyschallenge.com). Unlike real-user-data competitions, there are no user privacy concerns that would motivate restricting enrichment.
4. **ISRC provided at 97.17% coverage:** The dataset includes ISRCs, which are industry-standard identifiers explicitly designed for cross-database linking (MusicBrainz, Spotify, Discogs). Providing ISRCs may *invite* enrichment rather than prohibit it.
5. **Academic research context:** RecSys challenges typically prioritize innovation and do not restrict data sources unless reproducibility or fairness is at risk. Enrichment with public APIs (MusicBrainz, Last.fm) does not create unfair advantage if all participants have access.

## Arguments FOR "External Data Forbidden" (Inference)

1. **"Pre-extracted embeddings are supplied to ensure focus on model architecture"** — This phrasing suggests the organizers want participants to use *provided* resources rather than seeking external data.
2. **CC-BY-NC-ND-4.0 NoDerivatives clause** — Augmenting TalkPlayData with external metadata may create a "derivative dataset," which is prohibited from redistribution. Even if not redistributed, internal use may violate the spirit of ND.
3. **No prior work in this repo integrated external features** — R39, R54, R57b all use only TalkPlayData-supplied metadata. MusicBrainz was explored but never used. This suggests the team interprets rules as restrictive.
4. **Reproducibility concerns:** External APIs change over time (Spotify popularity, Last.fm tag counts). Allowing external enrichment without requiring snapshot timestamps could harm reproducibility of leaderboard results.
5. **Level playing field:** Not all participants may have API expertise, rate-limit resources, or awareness of external sources. Permitting external data could advantage teams with API infrastructure.

---

## Organizer Question (If Ambiguity Persists)

**Draft email to RecSys 2026 Music-CRS organizers:**

> **Subject:** RecSys Challenge 2026 Music-CRS: Clarification on External Metadata Usage
> 
> Dear RecSys 2026 Challenge Organizers,
> 
> I am a participant in the Music-CRS challenge (team: Echo). I would like to clarify the competition rules regarding external data sources.
> 
> **Question:** Are participants permitted to augment the provided TalkPlayData track metadata with external metadata sources (e.g., MusicBrainz tags/genres, Spotify audio features, Last.fm similar-tracks) for use as additional features in the recommendation model?
> 
> **Scenario:** I would use ISRC (provided in TalkPlayData at 97.17% coverage) to join with MusicBrainz recordings, retrieve tags/genres, and compute tag overlap features between candidate tracks and user listening history. The enriched metadata would be used internally for ranking; I would not redistribute the augmented dataset. Track recommendations would still come exclusively from the provided TalkPlayData catalog.
> 
> **My understanding of current rules:**
> - Track IDs must match TalkPlayData-Challenge-Track-Metadata (no external track injection)
> - Pre-extracted embeddings are provided to "ensure focus on model architecture"
> - Dataset license is CC-BY-NC-ND-4.0 (NoDerivatives clause)
> 
> However, I have not found explicit guidance on whether augmenting catalog metadata with external sources is permitted.
> 
> **If external enrichment is allowed, should participants:**
> 1. Document external sources and API versions in the submission paper?
> 2. Provide timestamps/snapshots to ensure reproducibility?
> 3. Limit enrichment to static metadata (tags, genres) and avoid time-varying fields (popularity, playcount)?
> 
> Thank you for your clarification.
> 
> Best regards,  
> [Team Name]

**Recommended action:** Send this email *before* proceeding with diagnostic or implementation.

---

## If ALLOWED: Proposed Bounded Diagnostic

**Scope:** 300-case dev sample (100 DEMOTED, 100 UNREACHABLE, 100 HIT), stratified sampling, CV5 OOF-clean.

**External source:** MusicBrainz only (leverage existing API client, 97.17% ISRC coverage, avoid Spotify/Last.fm rate-limit cost for initial diagnostic).

**Features to test (5 new features):**
1. `mb_tag_jaccard_last`: Jaccard similarity between candidate's MusicBrainz tags and union of MB tags for last 3 played tracks
2. `mb_genre_overlap_history`: Count of MB genres for candidate that appear in ANY played track's MB genres
3. `mb_recording_relation_played`: Binary flag — does candidate have MB recording relation (cover-of, remix-of, live-version-of) to any played track?
4. `mb_artist_relation_played`: Binary flag — does candidate's MB artist have artist-relation (collaboration, member-of-band) to any played artist?
5. `mb_same_label_last`: Binary flag — does candidate's release label match the label of last played track?

**Baseline:** R39+R54 LambdaRank (37 features), CV5 on 300-case sample.

**Test:** Add 5 MB features to 37-feature baseline, retrain LambdaRank CV5, compare metrics.

**Gates:**
- **PROCEED to full 8000-case experiment:** sample nDCG improvement ≥ +0.010, same-artist does not regress
- **ARCHIVE immediately:** sample nDCG regresses OR same-artist regresses >0.005 OR net recovered-lost ≤ 0

**Cost:** ~1 hour (500-800 new MB API requests at 1 req/sec + 5-fold LR retraining on 300 cases).

**Reproducibility measures (if diagnostic proceeds):**
1. Cache all API responses with timestamp
2. Document MusicBrainz API version/endpoint
3. Commit cache to repo (or external storage if too large)
4. Record API request date in experiment artifact
5. Never re-fetch during experimentation (use cache only)

**Reference:** See `docs/r59_candidates/c4_external_enrichment.md` §5 for full diagnostic design.

---

## If FORBIDDEN: Immediate Archive

**Trigger:** Organizers explicitly state external data sources are not permitted.

**Action:**
1. Archive R59 C4 immediately
2. Do not proceed to diagnostic
3. Do not fetch from MusicBrainz, Spotify, Last.fm, Discogs
4. Document ruling in experiment log

**One-line summary for archive:**
> "External enrichment blocked by competition rule [cite organizer email/rules URL]. Cannot proceed."

---

## Reproducibility Implications (If Allowed)

### Catalog Versioning

**MusicBrainz:** Database is versioned; dumps available at https://musicbrainz.org/doc/MusicBrainz_Database. API responses change as community edits accumulate. **Mitigation:** Snapshot API responses at a fixed date (e.g., May 16, 2026), commit cache, cite snapshot date in paper.

**Spotify:** No official database dumps. Web API responses for audio features are stable (tied to Spotify's internal audio analysis pipeline), but popularity/follower counts change daily. **Mitigation:** Use only static fields (audio features, genres); avoid popularity. Document API request date.

**Last.fm:** No versioning. Tags, playcount, listeners change continuously. **Mitigation:** Snapshot API responses; do not re-fetch. Cite snapshot date.

**Discogs:** Database dumps available at https://www.discogs.com/data/. API and dump may diverge. **Mitigation:** Prefer dump over API if possible; cite dump version.

### Rate Limits and Timeline

| Source | Rate Limit | Catalog Fetch Time (515K tracks) | Reproducibility |
|--------|-----------|----------------------------------|-----------------|
| MusicBrainz | 1 req/sec | ~143 hours (6 days) | High (versioned DB dumps) |
| Spotify | 1000 req/hour | ~515 hours (21 days) | Medium (stable audio features) |
| Last.fm | 18K req/hour | ~29 hours | Low (no versioning, tags change) |
| Discogs | 3600 req/hour | ~143 hours | High (database dumps available) |

**Implication:** Full-catalog enrichment is time-intensive (6-21 days). Bounded diagnostic (300-case sample, ~500 API calls) is feasible in ~1 hour.

### Paper Documentation (If External Data Used)

**Mandatory disclosures for submission paper:**
1. External sources used (MusicBrainz, Spotify, etc.)
2. API version/endpoint
3. Snapshot date (when data was fetched)
4. Coverage (% of catalog enriched, % with missing ISRCs)
5. Features derived from external data (list of 5 MB features)
6. Cache location (repo path or external storage URL)
7. Reproducibility: "All external API responses cached and committed at [URL]. No re-fetching performed during experimentation."

---

## External Data Precedent in Repo

### What Has Been Explored (Diagnostics)

1. **MusicBrainz audit (April 30, 2026):**
   - 100-track sample across DEMOTED/UNREACHABLE/popular/diff_artist categories
   - 536 cached API responses in `cache/musicbrainz_audit/`
   - Results: 71% ISRC→recording match, 61% metadata yield (tags/genres/relations)
   - Recovery diagnostic: MB artist-relations recovered ~2 of 1861 unreachable cases (projected ~5 at full coverage)
   - **Conclusion:** Diagnostic only, never used as features

2. **ISRC parsing (R57b):**
   - Extracted ISRC country/registrant codes from TalkPlayData's existing ISRCs
   - Created `isrc_country_match`, `isrc_registrant_match` features
   - **Result:** FAIL_REGRESS — features had high LightGBM gain but caused nDCG regression
   - **Lesson:** Structural metadata features can hurt even when model assigns them high importance

### What Has NOT Been Explored

- Spotify audio features (danceability, energy, valence, etc.)
- Last.fm tags/similar-tracks
- Discogs labels/genres/credits
- Any production integration of MusicBrainz tags/genres/relations as features

---

## Dataset License Terms Relevant to Enrichment

**License:** CC-BY-NC-ND-4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0)

**Full license:** https://creativecommons.org/licenses/by-nc-nd/4.0/

### Key Clauses

1. **BY (Attribution):** Must credit talkpl-ai/TalkPlayData. **Compliant** if paper cites dataset.

2. **NC (NonCommercial):** May not use for commercial purposes. **Compliant** — RecSys challenge is academic research.

3. **ND (NoDerivatives):** May not distribute modified/derivative versions. **AMBIGUITY:**
   - **Clearly prohibited:** Publishing "TalkPlayData + MusicBrainz tags" as a new dataset on HuggingFace
   - **Unclear:** Using external metadata internally to compute ranking features, then submitting predictions to leaderboard (no dataset redistribution)
   - **Potentially compliant:** If enriched metadata is treated as *ephemeral features* (never published), ND may not apply

**Recommendation:** If proceeding with external enrichment, do NOT redistribute augmented catalog. Use external metadata only to compute features for model training/inference; submit predictions only (not enriched catalog).

---

## Summary

| Criterion | Status |
|-----------|--------|
| **Explicit rules found?** | **NO** |
| **Prohibition inferred?** | Weak inference from "pre-extracted embeddings" phrasing and ND license clause |
| **Permission inferred?** | Weak inference from ISRC provision, synthetic data, no explicit ban |
| **Local precedent?** | MusicBrainz explored for diagnostics only; never integrated as features |
| **Dataset license?** | CC-BY-NC-ND-4.0 — NoDerivatives clause creates ambiguity |
| **Reproducibility feasible?** | Yes, with snapshot discipline and cache commits |
| **Feasibility risk?** | High — R57b structural features FAILED; external features may repeat failure mode |

---

## Recommendation

**VERDICT: AMBIGUOUS**

**Next step:** Contact organizers (email draft provided above) for explicit clarification BEFORE:
1. Running the 300-case MusicBrainz diagnostic
2. Fetching any new external API data beyond the existing 536-file MB cache
3. Implementing external-derived features in production pipeline

**If organizers confirm external data is allowed:**
- Proceed with bounded diagnostic (§ "If ALLOWED" above)
- Apply strict reproducibility discipline (cache snapshots, timestamp all fetches, document in paper)
- Acknowledge R57b failure mode risk: external structural features may regress despite adding new information

**If organizers confirm external data is forbidden:**
- Archive R59 C4 immediately
- Do not proceed with diagnostic or implementation
- Accept R54c as production system; focus on Blind-B final submission with existing pipeline

**If organizers do not respond in time:**
- **Conservative interpretation:** Do not use external data (default to restrictive reading of "pre-extracted embeddings" and ND clause)
- **Risk-tolerant interpretation:** Proceed with diagnostic (MusicBrainz only, cache all responses, document fully in paper, disclose to organizers post-submission if rules were unclear)

**Recommended stance:** Conservative. The cost of violating implicit rules (disqualification, reputational risk) exceeds the expected value of external enrichment given R57b's failure precedent.

---

## Sources Consulted

- [RecSys Challenge 2026 (official)](https://www.recsyschallenge.com/2026/)
- [RecSys 26 Challenge (ACM)](https://recsys.acm.org/recsys26/challenge/)
- [Codabench Competition Page](https://www.codabench.org/competitions/15786/)
- [music-crs-evaluator (GitHub)](https://github.com/nlp4musa/music-crs-evaluator)
- [TalkPlayData-1 (HuggingFace)](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-1)
- [TalkPlayData-2 (HuggingFace)](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)
- [TalkPlayData-Challenge-Track-Metadata (HuggingFace)](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata)
- Local repo: `README.md`, `docs/leakage_audit.md`, `docs/r27_agentic_submission_audit_plan.md`, `docs/r59_candidates/c4_external_enrichment.md`, `scripts/audit_musicbrainz_sample.py`, `scripts/audit_mb_recovery.py`, `cache/musicbrainz_audit/`

---

**Document Author:** r59-c4-rules-feasibility research instance  
**Coordination ID:** af2f7129-41bd-4fdc-bb66-7b6d7d11ba60  
**Timestamp:** 2026-05-16T18:00:00Z

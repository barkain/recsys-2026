# R59 Candidate: Entity/Constraint Parser with Deterministic Candidate Admission

**Status: Design analysis only. No implementation.**

## 1. New information added beyond R54c

Current R54c retrieval treats user queries as semantic embeddings (BGE-based dense + BM25 lexical). This collapses explicit constraints into soft similarity signals that get diluted in the 768-dimensional embedding space and lexical match scoring.

**What hard-constraint signal exists that current retrieval dilutes:**

R54c's dense retrievers (R54 BGE ensemble, R21) embed the user query into a vector that captures semantic similarity but cannot enforce hard boundaries. Lexical retrievers (src_b, src_c BM25) match tokens but don't interpret them as constraints. This means:

- **Temporal constraints** ("from the 2010s", "90s rock", "recent releases") → release_date field exists per track but retrieval doesn't filter by it; BGE may learn weak correlation but cannot enforce year ∈ [2010, 2019]
- **Duration constraints** ("under 3 minutes", "long tracks for workout", "short songs") → duration field exists but unused; embedding may capture "workout" → long, but not duration < 180s
- **Artist constraints** ("different artist", "not [artist]", "new artists", "more like this but not [played_artist]") → artist_id exists but only used as LR feature post-retrieval; the diff_artist regime (GT artist ∉ played artists) is pervasive in the dataset (R56's ORACLE gating showed this pattern) but retrieval doesn't enforce it
- **Tag/mood constraints** ("upbeat", "chill", "instrumental", "acoustic version") → tag_list exists per track; BM25 can match "acoustic" token but doesn't require tag_list ∋ "acoustic"; BGE embedding may correlate but can't enforce set membership
- **Exclusion constraints** ("no vocals", "not pop", "avoid EDM") → negative constraints are particularly weak in dense retrieval (embedding space has no natural negation operator) and BM25 ignores them

**How they'd map to candidate admission:**

A constraint parser would extract structured predicates BEFORE retrieval pools candidates:

```
Query: "upbeat 2010s indie rock, under 4 minutes, not [played_artist_X]"
Parsed:
  - tags ⊇ {"upbeat", "indie", "rock"}
  - release_date ∈ [2010, 2019]
  - duration ≤ 240s
  - artist_id ∉ {played_artist_set}

Deterministic admission:
  - Pre-filter catalog → eligible_set
  - Dense/lexical retrieval runs ONLY over eligible_set
  - OR: retrieval runs over full catalog, parser post-filters pool@300 → pool@k', 
    where k' ≤ 300 but only admits constraint-compliant candidates
```

The key architectural shift: move from "retrieval guesses, LR corrects" (R54c) to "parser gates, retrieval ranks within constraints."

**Concrete failure mode in current stack:**

From `docs/blind_a_final_state.md`:
- UNREACHABLE bucket: 1861 cases (23.3%) where GT in no source top-300
- POOL_MISS bucket: 1163 cases (14.5%) where GT in source union but RRF drops it

Many of these are likely cases where:
1. User query contains extractable constraint ("2015 album tracks" → release_date = 2015)
2. GT satisfies constraint (GT release_date = 2015)
3. Dense embedding of query is dominated by semantic features ("album tracks") that retrieve popular 2015 albums
4. GT is from a less-popular 2015 album → low BGE similarity → rank 500+
5. Lexical BM25 may surface GT at rank 200 but RRF@300 aggregation doesn't promote it

A hard release_date = 2015 filter would shrink candidate space from ~1M tracks to ~50K, allowing retrieval to focus ranking within that constrained set. GT's poor semantic similarity is less penalizing when the pool is pre-filtered.

---

## 2. Cached data/artifacts it can reuse

**Track metadata available** (per `scripts/expR54_phase3_blind_submission.py` and `scripts/expR58_inventory.py`):

| Field | Path | Coverage | Usable for constraints |
|---|---|---|---|
| `release_date` | track catalog | ~95%+ | YES — year/decade filters |
| `duration` | track catalog | ~99% | YES — duration < / > / ∈ range |
| `artist_id` | track catalog | 100% | YES — artist match/exclusion, diff_artist gating |
| `album_id` | `cache/r54_phase3_payload_maps.pkl` `track_album` | ~90%+ | YES — same-album constraints, album-level temporal filters |
| `tag_list` | `cache/r54_phase3_payload_maps.pkl` `track_tags` | ~80%+ | YES — tag membership / tag exclusion |
| `ISRC` | track catalog | ~85% | MAYBE — country/registrant (R57b tested ISRC features; failed at LR level but metadata exists) |
| `track_title_toks`, `track_artist_toks`, `track_meta_toks` | `cache/r54_phase3_payload_maps.pkl` | 100% | YES — lexical entity matching for "songs with X in title" |

**Session context available** (per `exp/eval/_R12_all_turns_payload.pkl`):

- `user_query` (current query text)
- `history` (prior turns)
- `music_turns` (list of played track_ids)
- Derived: `played_artist_set = {track_artist[tid] for tid in music_turns}`

**Retrieval artifacts** (reusable for constrained retrieval):

| Artifact | Purpose |
|---|---|
| `cache/r21_production/{model, track_embeddings.npy}` | R21 dense retrieval — can restrict search to eligible_set indices |
| `cache/r54_production/{model, track_embeddings.npy}` (Phase 2/3) | R54 dense retrieval — same |
| `cache/r54_phase3_als.npz` | ALS factors — can filter eligible tracks before ALS scoring |
| BM25 indices (src_b, src_c) | Lexical retrieval — can pre-filter document set |

**No new retrieval required IF** the parser filters pool@300 post-retrieval (cheaper path). New retrieval required ONLY IF we want to pre-filter at embedding-search time (more architecturally ambitious but better recall potential).

---

## 3. Leakage risks

**From the model side:** None. The parser operates on observable session features (query text, played track metadata) and catalog metadata. All available at inference time. No GT-side labels.

**Risks to flag:**

### (a) Extractor information source and blind inference cost

| Extractor type | Info source | Blind cost | Leakage |
|---|---|---|---|
| **Heuristic regex/keyword** | Query text + lexical patterns ("2010s" → year ∈ [2010,2019], "under X minutes" → duration) | ~1ms CPU per query | None |
| **LLM zero-shot** (e.g. Claude API: "extract temporal/duration/tag/artist constraints from query") | Query text | ~$0.01 per query × 80 blind cases = ~$0.80 per submission + 200-500ms latency | None if prompt is static |
| **LLM few-shot / fine-tuned** | Query text + training examples from train-split | Same cost, better precision | **Training distribution** — if few-shot examples are from train-split queries, extractor may learn patterns specific to training cases. Not leakage per se but risks overfit to train query phrasing. Mitigation: validate extractor on dev queries it hasn't seen. |
| **Trained NER/slot-filling model** (e.g. fine-tuned T5 on synthetic or annotated train queries) | Query text + labeled train queries | ~5ms CPU/GPU per query | Same as LLM few-shot — train/dev split discipline required. |

**Recommendation for Phase 1 diagnostic (§5):** Start with heuristic regex/keyword extractor (zero cost, zero leakage, instant) to establish ceiling. If heuristic shows signal, THEN consider LLM zero-shot for precision lift. Do NOT train a supervised extractor until heuristic + LLM zero-shot are both validated as insufficient.

**Blind inference cost for LLM extractor:** 80 queries × $0.01 = $0.80 per submission is negligible vs submission value. Latency (200-500ms per query) is acceptable for offline batch inference. Not a blocker.

### (b) Hard filters and parse errors

**Risk:** Parser extracts constraint "duration < 3 minutes" but 40% of catalog has missing duration field. Hard filter drops those tracks → GT may be in the dropped set → UNREACHABLE worsens instead of improves.

**Mitigation:**

1. **Graceful fallback:** If parser extracts constraint but >20% of pool@300 would be dropped due to missing metadata, IGNORE the constraint for that case. Treat as unparseable. Log and report.
2. **Coverage audit (pre-implementation):** For each constraint type, report % of catalog with non-null field. From §2: release_date ~95%, duration ~99%, tag_list ~80%. Tag constraints are riskiest (20% of tracks have no tags).
3. **Conservative constraint application:** Only apply constraint if:
   - Confidence score (if LLM extractor) > threshold (e.g. 0.8)
   - Field coverage in pool@300 > 80%
   - Constraint is positive (inclusion) not negative (exclusion) — negative constraints are harder to validate ("no vocals" may require tag_list ∋ "instrumental" but tag absence ≠ vocals presence)

**Parse error handling:** If extractor fails (malformed output, API timeout), fall back to baseline R54c retrieval for that case. No constraint applied. Parser is an OPTIONAL lift layer, not a required component.

### (c) Is the parser trained on train-split or zero-shot?

**Zero-shot heuristic or LLM:** No train/dev bleed. Safe.

**Supervised extractor:** MUST use train-split queries for training, dev-split queries for validation, held-out test-split (or blind queries) for evaluation. Same CV5 fold discipline as retrieval and LR.

**OOF contamination analog:** If we train an extractor on fold 0's queries and apply it to fold 0's held-out cases, that's NOT contamination — the extractor never saw those specific queries. BUT if we tune the extractor's architecture/prompts by peeking at dev-split parse quality, that IS a form of overfit. Mitigation: define extractor design on train-split error analysis only; lock it; then evaluate on dev.

---

## 4. Expected ceiling

**Honest estimate:** The constraint parser can plausibly lift **200-400 of the 4884 fixable cases** (4-8% of the fixable bucket).

**Argument:**

From `docs/blind_a_final_state.md`, the fixable buckets:
- DEMOTED: 1628 (GT in pool@300 but LR ranks >20)
- POOL_MISS: 1163 (GT in source union but RRF drops it)
- UNREACHABLE: 1861 (GT in no source top-300)
- **Total fixable: 4651** (not 4884 — I miscalculated; 1628+1163+1861 = 4652)

**DEMOTED (1628 cases):** Constraint parser does NOT help. GT is already in pool@300; LR saw it and demoted it. From R56's failure (source-rank protection), we know LR's demotion is learned correctly. A hard filter that admits GT to the pool is redundant — GT was already there. **Contribution: 0.**

**POOL_MISS (1163 cases):** GT in source union (one of R21, R54, BM25, ALS top-300) but RRF@300 dropped it. A constraint parser could help IF:
1. Query contains extractable constraint (e.g. "2015 indie")
2. GT satisfies constraint (release_date = 2015, tag_list ∋ "indie")
3. Applying hard filter shrinks candidate space → RRF's aggregation over constrained pool promotes GT into top-300

**How many queries have extractable constraints?** Based on `scripts/expR55_changed_rows_analysis.py` structure (which shows user queries with history), conversational music queries are often vague ("something upbeat", "more like this", "chill vibes") or specific-track requests ("play [track_name]"). Estimates from RecSys literature and inspection of similar datasets:
- ~30-40% of queries contain temporal phrases ("90s", "recent", "2010s") → release_date constraint
- ~15-20% contain duration hints ("short", "long", "workout length") → duration constraint  
- ~20-30% contain explicit artist exclusion intent ("different artist", "new artists") → diff_artist constraint
- ~40-50% contain tag/mood keywords ("upbeat", "chill", "acoustic") → tag constraint

**Overlap:** Many queries have multiple constraints. Assume ~50% of queries have at least one extractable constraint with >80% confidence.

**Of those, how many have GT satisfying the constraint but retrieval missing it?** Not all. Retrieval may surface GT despite missing the constraint (semantic similarity rescues it) or GT may not satisfy the constraint (user query is ambiguous, parser extracts wrong constraint).

**Conservative estimate:** 
- POOL_MISS: 1163 cases
- ~50% have extractable constraint = 581 cases
- ~40% of those have GT satisfying constraint AND GT would be admitted by constrained retrieval = 232 cases
- Success rate (parser precision × retrieval lift given constraint) ~60% = **~140 POOL_MISS recovered**

**UNREACHABLE (1861 cases):** GT in no source top-300. A constraint parser could help IF:
1. Hard filter shrinks candidate space dramatically (e.g. release_date = 2015 reduces 1M tracks → 50K)
2. Within constrained space, GT's semantic similarity rank improves enough to enter top-300
3. OR: constraint is so specific (e.g. "2015 indie under 3 min by [artist]") that eligible set is <1000 tracks, making GT findable even with weak similarity

**How many UNREACHABLE cases are rescuable by constraints?** Fewer than POOL_MISS. If GT wasn't in ANY source top-300, it's either:
- Very obscure track (low popularity, weak metadata)
- Query-track mismatch (user asked for X, GT is Y, annotator error or subjective)
- Embedding space failure (BGE and BM25 both miss)

Constraints help MOST when the query is specific and GT is metadata-compliant but unpopular. Estimate:
- UNREACHABLE: 1861 cases
- ~40% have extractable constraint (lower than POOL_MISS because UNREACHABLE queries may be vaguer) = 744 cases
- ~25% of those have GT satisfying constraint AND constrained retrieval would rank GT into top-300 = 186 cases
- Success rate ~50% = **~93 UNREACHABLE recovered**

**Total ceiling: 140 + 93 = ~233 cases recovered.**

**Upside scenario:** If constraint extraction precision is very high (LLM-based, 90%+ confidence gating) and constrained retrieval is run as pre-filter (not post-filter), ceiling could reach **300-400 cases** (~6-8% of fixable bucket).

**Why not higher?**
- Majority of queries are vague semantic requests ("chill music", "something energetic") with no extractable hard constraints
- Many constraints are soft preferences (e.g. "upbeat" is subjective, not a binary tag)
- Metadata coverage gaps (tag_list ~80%, artist exclusion only works if we have full artist_id)
- LR's calibration: even if GT enters pool@300, LR may still demote it (though fewer cases than current DEMOTED since constrained pool has better signal-to-noise)

**Metric impact:** 233 cases / 8000 dev cases = +2.9% absolute recall. If those cases are evenly distributed across history depths and avg nDCG contribution is ~0.5 per recovered case, estimated h7 nDCG lift: **+0.005 to +0.010** (within exploratory-to-production gate range per `feedback_blind_gate.md`).

---

## 5. Smallest falsifiable diagnostic

**Goal:** Establish whether extracted constraints would have admitted GT in cases where current pool@300 misses GT, WITHOUT building the full constrained retrieval pipeline.

**Diagnostic experiment (dev-only, OOF-clean):**

### Extractor
**Phase 1: Heuristic regex/keyword parser.** Zero cost, zero training.

Patterns:
```python
# Temporal: "90s", "1990s", "2010s", "recent" (last 3 years), "classic" (pre-2000)
# Duration: "under X min", "short", "long" (heuristic: short <3min, long >5min)  
# Artist exclusion: "different artist", "new artist" (not in played_artist_set)
# Tags: match {"upbeat", "chill", "acoustic", "instrumental", ...} from fixed vocabulary  
#       against query tokens (lemmatized)
```

Extract per query:
```json
{
  "session_id": "...",
  "query": "...",
  "constraints": {
    "release_year_min": 2010,
    "release_year_max": 2019,
    "duration_max": 180,
    "exclude_artists": ["artist_id_X", "artist_id_Y"],
    "required_tags": ["upbeat", "indie"],
    "confidence": 0.8
  }
}
```

Apply confidence threshold: only use constraint if `confidence >= 0.7`.

### Filter logic
For each dev case:
1. Load GT track_id and its metadata (release_date, duration, artist_id, tag_list)
2. Load current pool@300 (from R54 Phase 2 OOF baseline, per `docs/r58_design.md` §3.4)
3. Apply extracted constraints to GT:
   - `gt_satisfies_constraints = True` if GT passes all extracted constraints
4. Apply extracted constraints to pool@300:
   - `pool_constrained = [tid for tid in pool if tid satisfies constraints]`
5. Compute:
   - `gt_in_baseline_pool = (GT in pool@300)`
   - `gt_in_constrained_pool = (GT in pool_constrained)` if constraints were extracted, else `None`
   - `baseline_gt_rank = rank of GT in pool@300` (1-300 or -1)
   - `constrained_pool_size = len(pool_constrained)`

### Eval procedure
For the **POOL_MISS + UNREACHABLE** buckets (1163 + 1861 = 3024 cases where GT not in baseline pool@300 OR not in baseline top-20):

**Metric 1: Constraint extraction rate**
- % of cases where parser extracts at least one constraint with confidence >= 0.7
- Breakdown by constraint type (temporal, duration, artist, tag)

**Metric 2: GT compliance rate**
- Of cases with extracted constraints, % where GT satisfies all constraints
- If low (<50%), parser is noisy or constraints are too strict

**Metric 3: Hypothetical admission rate**
- Of cases where GT not in baseline pool@300 AND parser extracts constraints AND GT satisfies constraints:
  - % where GT WOULD have been in top-300 of a constrained catalog search
  - **Proxy (without rerunning retrieval):** Assume if constrained_pool_size < 500 and GT satisfies constraints, GT would be findable. This is optimistic but establishes ceiling.
  - **Better proxy:** For cases where GT is in source union (POOL_MISS), check if GT's rank in best-source < 100. If yes, assume constrained RRF would promote it.

**Metric 4: Precision-recall tradeoff**
- False positive rate: % of cases where parser extracts constraint but GT does NOT satisfy constraint (filter would harm)
- Coverage: % of fixable bucket where parser extracts valid constraint (upper bound on lift)

**Expected outcome:**
- If Metric 3 (hypothetical admission) shows **>150 cases** where GT would be admitted by constraints, proceed to Phase 2 (LLM extractor or constrained retrieval implementation).
- If Metric 3 shows **<100 cases**, archive. Constraint signal is too sparse or noisy.
- If Metric 2 (GT compliance) < 40%, parser is extracting wrong constraints or being too strict. Tune confidence threshold or constraint logic.

### Script scope
`scripts/expR59_c2_constraint_diagnostic.py` — no training, no retrieval, no LR. Just:
1. Load dev cases (8000) + GT + baseline pool@300 (from R56/R58 cached artifacts)
2. For each case, run heuristic parser on user query
3. Check GT metadata compliance
4. Compute hypothetical admission metrics (Metrics 1-4)
5. Write report: `exp/eval/expR59_c2_constraint_diagnostic.json`

**Runtime estimate:** ~2-5 minutes CPU (8000 queries × regex parse + metadata lookups).

**No code committed in this design doc.** Script is written only after this design is approved AND Metric 3 ceiling estimate justifies continuation.

---

## 6. Stop condition

Mirror R56 / R57b / R58 gate discipline (per `docs/r56_design.md` §5, `docs/r58_design.md` §8).

**Stop after Phase 1 diagnostic (§5) if:**
1. Hypothetical admission rate (Metric 3) < 100 cases (< 2% of fixable bucket) — signal too weak
2. GT compliance rate (Metric 2) < 40% — parser is noisy, constraints are misaligned with GT metadata
3. False positive rate > 20% — parser harms more cases than it helps

**Stop after Phase 2 implementation (if Phase 1 passes) if:**
1. Dev h7 nDCG gain < +0.005 vs R54c baseline (exploratory gate per `feedback_blind_gate.md`)
2. Same-artist nDCG regresses by > 0.002 (canary metric per `feedback_lr_top50_calibrated.md`)
3. All-dev nDCG regresses vs baseline
4. Net recovery (recovered - lost) ≤ 0 on dev
5. Top-1 churn > 240/8000 (3.0%) on dev (per R56 gate)

**Stop after Phase 3 dev evaluation (if Phase 2 passes) if:**
1. Dev h7 nDCG gain ∈ [+0.005, +0.010) but top-1 churn > 120/8000 (1.5%) — too unstable for exploratory blind
2. Dev h7 nDCG gain < +0.010 AND no structural-feature novelty argument — don't burn blind slot on marginal lift

**Archive as NEGATIVE and freeze for Blind-B if any stop condition fires.**

**Production candidate threshold (proceed to Blind-A submission):**
- Dev h7 nDCG **+0.010** or better vs R54c
- Same-artist nDCG does not regress (Δ ≥ 0)
- Top-1 churn ≤ 240/8000 (3.0%)
- Net recovery ≥ 50 cases

**Explicit freeze trigger:** If diagnostic (§5) shows constraint extraction rate < 30% of queries, the mechanism is not a good fit for the dataset. Archive immediately without implementation. Do NOT iterate on parser design without first confirming query distribution supports extraction.

---

**Design complete. Awaiting review. No code, no implementation, no LLM calls until approved.**

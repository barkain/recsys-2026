# R59 — Mechanism Reset

**Status: design only. Synthesis of 4 candidate analyses from a research team
(2 Codex + 2 Claude instances). No implementation. R54c remains production.**

## 0. Why a reset, not another iteration

R54c at composite 0.6106 (nDCG 0.4925) is a strong local optimum for the
retrievers → RRF pool@300 → R39+R54 LambdaRank → top-20 stack.

Eight post-R54c attempts disconfirmed every cheap angle:

| # | Direction | Outcome |
|---|---|---|
| R55 | single all-data retriever | composite flat, nDCG -0.0067 on blind |
| R55h | 2-row manual hybrid | composite -0.0015 on blind |
| R56 | rule-based source-rank protection (18 variants + ORACLE) | all FAIL_REGRESS dev |
| R57 | structural-metadata forensic refresh | patterns survive, R49C precedent disconfirms |
| R57b | LR with ISRC + artist_id_match features | all FAIL_REGRESS dev |
| R58 | learned stage-2 LightGBM specialist on LR top-50 (28 configs) | all FAIL_REGRESS dev |

The lesson is operational: more LR-feature tweaks, post-LR rerank rules,
manual hybrids, or retriever swaps that reuse the same BGE/R54 supervision
all converge to the same local optimum. To move the number we need a
**different information source** or a **different supervision setup**, not
another feature.

This doc compares four candidate mechanisms with that filter applied. Per-
candidate detail lives in `docs/r59_candidates/c{1,2,3,4}_*.md`. This
synthesis is the routing decision.

## 1. The four candidates at a glance

| ID | Mechanism | What's new vs R54c | Prior | Diag cost | Ceiling | Big risk |
|---|---|---|---|---|---|---|
| **C1** | Behavioral sequence model from train-split session transitions (hybrid ID+metadata) | Ordered item-to-item transitions, utterance-conditioned next-track | Moderate | Low-Med | +0.003 to +0.010 | R31 sparse-ID sequence already failed; must be hybrid |
| **C2** | Entity/constraint parser with deterministic candidate admission | Hard structural constraints from query text (year, duration, tags, artist exclusion) | Moderate | **Very low** (heuristic regex, no training, no API) | +0.005 to +0.010 | Many queries are vague; parse errors; negative constraints |
| **C3** | Learned pool-admission model separate from final ranking | Cross-source interaction + non-monotone admission instead of fixed RRF rank-sum | Moderate | Low | +0.003 to +0.008 | Cannot help UNREACHABLE; some POOL_MISS are noisy single-source RRF correctly drops |
| **C4** | External catalog enrichment (MusicBrainz / Spotify / Last.fm) | Cross-source tags, audio features, work-relations | **Low** (R57b precedent + rule risk) | High (catalog fetch 21-143 h) | Likely 0 in expected case | Rule compliance UNKNOWN; R57b-class failure mode |

Each candidate's analysis fills the 6 sections you specified (new info,
cached reuse, leakage, ceiling, falsifiable diagnostic, stop condition).
Where they overlap is informative, and where they diverge defines the
routing.

## 2. What the candidates have in common

All four analyses converge on three points worth recording up front because
they constrain any next step:

1. **The 4651 fixable cases split unevenly across the four mechanisms.**
   - DEMOTED (1628): only C2/C4 could plausibly contribute, and both at the
     margin. C1 and C3 explicitly do NOT target DEMOTED (R56 already showed
     LR within-top-50 is well-calibrated).
   - POOL_MISS (1163): all four can address some subset.
   - UNREACHABLE (1861): only C1 (new generator) and C4 (relations to
     unreachable tracks) can theoretically expand the candidate universe.
     C3 explicitly cannot help UNREACHABLE.

2. **The same-artist nDCG canary stays in force.** Every candidate analysis
   self-imposes the R56/R57b/R58 stop rule: same-artist regress > 0.002 is
   auto-archive. That's the canonical "feature has high gain but bad
   transfer" detector.

3. **Strict gate inherited.** Production +0.010 h7, exploratory +0.005, top-1
   churn ≤240/8000 (≤120 exploratory). None of the candidate analyses argue
   for relaxing this. After eight negatives the bar stays.

## 3. Where they diverge — the routing decision

Three axes matter:

### 3.1 Cost of the smallest falsifiable diagnostic

| Candidate | Diag scope | Runtime | New artifacts needed |
|---|---|---|---|
| **C2** parser | heuristic regex extractor on 8000 dev queries; check GT metadata compliance vs extracted constraints; no retraining, no API, no model | ~2-5 min CPU | None |
| **C3** admission | LightGBM admission classifier on source-union candidates (CV5 OOF); compare admission@300 vs weighted_rrf@300 | ~10-30 min CPU | None — uses existing source lists |
| **C1** sequence | transition-table aggregation from train-split (no neural training in the cheap probe); score dev candidates | ~30-60 min CPU | Train-split transition table (reusable artifact) |
| **C4** enrichment | 300-case dev sample with 5 MusicBrainz-derived features; 5-fold LR retrain | ~1 h (incl. ~500 new MB API requests) | Bounded MB sample cache |

C2 is cheapest. C4 is most expensive AND blocked on a rule check.

### 3.2 Prior probability the mechanism has any signal

Two distinct priors:

- **C1 (sequence)** — moderate-low. Prior attempts in the same family:
  R31 sparse-ID (h7 hit@200 = 0.025 vs R21 0.435) and S/Qwen (h7 hit@200
  144/1000, weak fusion, Blind-A regression). The new claim is that
  hybrid ID+metadata representation avoids R31's sparse collapse and
  uses behavior-native item heads (not metadata text). Defensible but
  unproven.

- **C2 (parser)** — moderate. The mechanism is genuinely different (hard
  filter vs soft retrieval). Queries do contain extractable temporal,
  duration, tag, and artist constraints. Risk is concentrated in: how
  many cases have extractable constraints, and how well does the heuristic
  extractor handle vague queries.

- **C3 (admission)** — moderate. Source-coverage data from
  `expR55_post_refresh_decomp.json` shows POOL_MISS GTs are distributed
  across all sources (R54 394, R21 368, src_c 362, src_b 286, ALS 193,
  ...) — argues against single-source rules but for combinations. Prior
  is mixed: some POOL_MISS cases are genuinely deep noise that RRF should
  drop.

- **C4 (enrichment)** — low. R57b structural-metadata features had high
  LightGBM gain and still regressed. External structural metadata is the
  same class. The "different mechanism" argument is weak unless we add
  MB recording/work relations (which the prior audit only weakly
  supported — ~2 of 1861 recovered).

### 3.3 Architectural compatibility with what we already know works

R54c works because retrieval feeds a pool@300, LR re-ranks. The candidates
fit that frame as follows:

- **C1** adds a **new retrieval source** (sequence retriever produces top-N
  candidates with scores). Composes naturally via RRF — no LR retraining
  needed for the cheap diagnostic; only an additional RRF source weight.
- **C2** acts at **candidate admission** — filters the pool@300 or runs
  retrieval over a constraint-filtered subset. Pre-LR. Bounded blast radius.
- **C3** replaces or augments **the pool builder** (weighted_rrf →
  learned admission). Pre-LR. Same blast-radius properties as C2.
- **C4** adds **new LR features** (catalog metadata derived from external
  sources). Post-retrieval. This is exactly the failure shape of R57b.

C1, C2, and C3 are pre-LR interventions. They don't compete with LR's
top-50 calibration the way R56/R57b/R58 did. C4 competes with LR's feature
distribution and inherits R57b's failure prior.

## 4. Recommended ordering

Defensive cheap-first ordering with explicit stop gates between phases:

### Phase A — C2 constraint parser (cheapest)

- Heuristic regex extractor on 8000 dev queries. Check `gt_satisfies_constraints` on the dev GT for each constraint type.
- Eval: hypothetical admission rate on POOL_MISS + UNREACHABLE buckets.
- Runtime: minutes. No training, no API, no model.
- **Gate to proceed:** ≥150 cases where GT not in baseline pool@300 AND parser extracts ≥1 high-confidence constraint AND GT satisfies it. Plus GT-compliance rate ≥40% on parsed cases. Per `docs/r59_candidates/c2_entity_constraint_parser.md` §5.
- **If gate fails:** archive C2 (extracted constraints don't align with GT metadata; parser is the wrong tool).

### Phase B — C3 pool-admission model (cheap, OOF-clean)

- LightGBM admission classifier on the union of A/B/C/D/F/ALS/R21/R54
  source-list candidates. CV5 fold-grouped. Compare admission@300 vs
  weighted_rrf@300 on pool_hit and POOL_MISS recovery.
- Runtime: tens of minutes.
- **Gate to proceed:** OOF admission@300 must beat weighted_rrf@300 by
  recovering ≥100 of 1163 POOL_MISS GTs while losing ≤25% of previously-
  covered GTs. Per `docs/r59_candidates/c3_pool_admission.md` §5.
- **If gate fails:** archive C3 (RRF is near-optimal for this source mix).

### Phase C — C1 behavioral sequence (next-cheapest, R31 risk applies)

- Train-split transition table + utterance-bucket aggregation, then dev
  scoring. No neural training in the first probe.
- Runtime: tens of minutes to a few hours depending on backoff design.
- **Gate to proceed:** ≥30 unique h7 GT hits outside the current RRF
  pool@300, OR h7 nDCG +0.003 when added as a low-weight RRF source.
  Same-artist must not regress >0.002. Per
  `docs/r59_candidates/c1_behavioral_sequence.md` §5.
- **If gate fails:** archive C1 (transition signal too sparse; the R31/S
  precedent transfers).

### Phase D — C4 external enrichment (deprioritized; requires rule check)

- **Hard prerequisite:** verify competition rules permit external data
  sources. Per `docs/r59_candidates/c4_external_enrichment.md` §3 and §6,
  no explicit rules found in the repo. Email organizers or find the
  official rule document BEFORE running any external diagnostic.
- **If rules permit:** 300-case dev sample with 5 MusicBrainz-derived
  features (tag Jaccard, genre overlap, recording relations, artist
  relations, label match). One-hour diagnostic with bounded API budget.
- **Gate to proceed:** sample h7 +0.010 with no same-artist regression.
  Doubled threshold to account for 300-case sample variance.
- **If rules forbid or gate fails:** archive C4. Do not retry with
  Spotify or Last.fm (lower-fidelity joins, same metadata class).

## 5. Why C4 is last, not first

C4 (external enrichment) had the most aspirational mechanism description
("could close the gap to el_presidente at 0.57") but it's last in the
queue because:

1. **R57b is a direct precedent.** Internal ISRC country/registrant and
   artist_id_match features were all high-gain in LightGBM and all
   regressed aggregate nDCG. External structural metadata is the same
   feature class.
2. **The prior MusicBrainz audit recovered only ~2 of 1861 unreachable
   cases.** Existing artifact at `cache/musicbrainz_audit/` plus
   `cache/mb_recovery_diagnostic.json` documents this. The "miracle case"
   for C4 contradicts this directly-measured prior.
3. **Rule compliance is UNKNOWN.** No `RULES.md` in the repo; no explicit
   statement on external data sources. Running a diagnostic with
   external data without checking risks disqualification on submission.
4. **A 21-143 hour catalog fetch is expensive** for a mechanism with a
   <5% miracle prior, especially when C1/C2/C3 cost minutes.

The C4 analysis itself argues this explicitly in its skeptical section.

## 6. Stop conditions for the whole R59 effort

If all four phases A→D archive (or D blocks on rules), declare R59
saturated. Freeze stays in place from
`docs/blind_a_final_state.md`. Next move would be:

- Wait for Blind-B data
- Or pivot to a mechanism class genuinely outside the four R59 candidates
  (e.g. cross-encoder reranker with raw embeddings — explicitly out of
  scope for this design)

Per-phase saturation criteria already encoded in the candidate docs.

## 7. What R59 explicitly does NOT cover

- Phase 1 inventory-style code or feature implementation. R59 is
  mechanism-reset design. Implementation is a separate per-phase doc that
  has to clear its phase's gate first.
- Cross-encoder reranker (a Phase-4-style architectural commitment from
  R58 that we deferred). Not in R59's four-candidate set.
- Query rewriting. Adjacent to C2 but distinct (rewriting changes the
  query, not the candidate filter).
- Generative retrieval (semantic IDs / DSI-style). Genuinely different
  but too far from cached artifacts; out of R59 scope.
- Anything that bypasses the OOF discipline. The R55-R58 cycle showed
  every failure was caught by OOF-on-dev diagnostics. We do not relax
  this.

## 8. Open questions for review

1. **Phase ordering** — A→B→C→D is cheap-first. Alternative: parallel
   A+B+C diagnostics (each is ~tens of minutes, no shared dependencies),
   then D conditionally. Argument for parallel: faster total turnaround.
   Argument against: if Phase A surfaces something material, we may
   want to focus there rather than evaluate B+C in noise. Default in
   this doc: sequential, cheapest-first. Open to changing.
2. **C4 rule check** — should this happen NOW (before any of A/B/C), so
   that if rules forbid external data, the team knows to skip D entirely
   without doing a futile rule check at the end? Default: do the rule
   check at the start of Phase D, with the caveat that if rules forbid
   it, the C4 analysis is archived without diagnostic.
3. **Threshold sensitivity in Phase A** — the C2 diagnostic uses heuristic
   regex with a 0.7 confidence threshold. Per C2 §3(a), this is a Phase
   1 choice; LLM extractors are a Phase 2 commitment if regex shows
   signal. Open: should the diagnostic also include LLM zero-shot
   extraction as a side-by-side comparison from the start, or is regex
   the right minimal first test?

## 9. References

- `docs/r59_candidates/c1_behavioral_sequence.md` — Codex GPT-5.5, 161 lines
- `docs/r59_candidates/c2_entity_constraint_parser.md` — Claude Sonnet 4.5, 322 lines
- `docs/r59_candidates/c3_pool_admission.md` — Codex GPT-5.5, 268 lines
- `docs/r59_candidates/c4_external_enrichment.md` — Claude Sonnet 4.5, 315 lines
- `docs/blind_a_final_state.md` — production state and freeze rule
- `docs/r56_design.md` — failed mechanism precedent (rule-based)
- `docs/r58_design.md` — failed mechanism precedent (learned specialist)
- `docs/r58_architecture_choice.md` — architecture comparison framework
- Memory: `[[r58-outcome]]`, `[[r57-outcome]]`, `[[lr-top50-calibrated]]`,
  `[[structural-features-exhausted]]`, `[[no-manual-row-edits]]`

---

**Awaiting review.** Recommended next step: approve Phase A diagnostic
(C2 constraint parser heuristic probe, ~5 min runtime, zero training,
zero API). All other phases gated on Phase A's verdict. R54c stays
production throughout.

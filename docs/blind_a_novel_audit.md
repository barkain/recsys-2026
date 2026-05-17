# Blind-A Novel Audit: Goal Fields

Date: 2026-05-17
Branch: `r59-mechanism-reset`
Scope: audit `conversation_goal` and `goal_progress_assessments` against the
current R54c production pipeline.

Bottom line: both fields are present in schema and unexploited by R54c.
Surviving novelty is limited to pre-RRF/session-representation mechanisms;
LR one-hots and post-LR reranks are closed paths.

## 1. Field-Usage Audit

### Q1. `conversation_goal`

Direct script access: none.

Repo-wide grep found only documentation mentions:

- `docs/r54c_build_chain.md:61`: Blind-A `test` schema includes
  `conversation_goal`.
- `docs/blindb_ingestion_plan.md:7-9`: same Blind-A column list.
- `docs/claude_app_sequence_model_design.md:391-393`: excludes
  goal/specificity conditioning in that design because it assumes an inference
  availability gap.

R54/BGE query construction does not include it:

- `scripts/expR54_phase3_ensemble_blind.py:60-80`: blind BGE query uses only
  `[QUERY]`, `[HISTORY]`, and `[CONTEXT]` from `item["conversations"]`.
- `scripts/expR54_phase3_full5fold_train.py:102-137`: dev/train structured
  BGE query uses current user text, recent user history, and played-track refs.
- `scripts/expR54_phase3_full5fold_train.py:161-196`: train-split pairs are
  built from `conversations`, user messages, and music turns.
- `scripts/expR54_phase3_production_blind.py:120-141`: single-model blind
  builder uses the same no-goal query shape.

Conclusion: structured contents are not used, and raw `conversation_goal` is
not appended to any BGE text blob.

### Q2. `goal_progress_assessments`

Direct script access: none.

Repo-wide grep found only schema documentation:

- `docs/r54c_build_chain.md:61`
- `docs/blindb_ingestion_plan.md:7-9`

Production code consumes ordinary conversation text, not the binary labels:

- `scripts/expR54_phase3_blind_submission.py:47-58`: `parse_last_turn_local`
  reads `item["conversations"]` and returns current user text, prior history,
  and music turns.
- `scripts/expR54_phase3_blind_submission.py:396-400`: R21 blind query is
  recent user text only.
- `offline_retrieval_sweep.py:98-120`: BM25 `query_parts` uses user turns and
  music metadata only.
- `scripts/expR54_phase3_blind_submission.py:418-423`: R54 calls BM25.

Conclusion: the discrete `[MOVES_TOWARD_GOAL | DOES_NOT_MOVE_TOWARD_GOAL]`
signal is unexploited.

### Q3. LR Feature Builders

`scripts/expR55_post_refresh_decomp.py`:

- `scripts/expR55_post_refresh_decomp.py:64-70`: feature set is
  `FEAT_BASE + FEAT_ALBUM + FEAT_R54`.
- `scripts/expR55_post_refresh_decomp.py:101-134`: loads R12 payload, R21 OOF,
  R54 OOF, track maps, ALS, and track albums.
- `scripts/expR55_post_refresh_decomp.py:168-175`: source lists are
  A/B/C/D/F/ALS/R21/R54.
- `scripts/expR55_post_refresh_decomp.py:188-256`: features are source ranks
  and presence, text/token overlaps, played state, recency, ALS, popularity,
  album features, and R54 cosine.
- No `conversation_goal`; no `goal_progress_assessments`.

`scripts/expR54_phase3_blind_submission.py`:

- `scripts/expR54_phase3_blind_submission.py:75-94`: production feature names
  are base source/text/history features, album features, and R54 features.
- `scripts/expR54_phase3_blind_submission.py:120-123`: `_featurize_row` takes
  `case_query`, `history`, and `played`; neither audited field is an argument.
- `scripts/expR54_phase3_blind_submission.py:128-193`: feature construction
  uses query tokens, music history, source ranks/presence, ALS, popularity,
  album history, and R54 scores.
- `scripts/expR54_phase3_blind_submission.py:387-435`: blind phase loads
  Blind-A rows, parses `conversations`, builds sources, and featurizes pool.
- No `conversation_goal`; no `goal_progress_assessments`.

Conclusion: neither field is in the R55 36-feature pipeline or the R54/R54c
37-feature production LR pipeline.

### Q4. Retriever Usage

- R54/BGE uses current-query/history/context text only:
  `scripts/expR54_phase3_ensemble_blind.py:60-80`,
  `scripts/expR54_phase3_full5fold_train.py:102-137`.
- R21 uses recent user text only:
  `scripts/expR54_phase3_blind_submission.py:396-400`,
  `scripts/expR39b_blind_submission.py:386-390`,
  `scripts/expR55_blind_source_cache.py:88-99`.
- BM25 uses user turns and music metadata:
  `offline_retrieval_sweep.py:98-120`,
  `scripts/expR54_phase3_blind_submission.py:418-423`.
- ALS uses played-track histories: `scripts/expS2_lambdarank.py:82-158`.
- RRF uses A/B/C/D/F/ALS/R21/R54 source lists:
  `scripts/expR54_phase3_blind_submission.py:429-431`.

Conclusion: no audited field is used by BGE/R54, BM25, ALS, R21, RRF, or LR.

## 2. Closed-Paths Check

Closed-path prior: R54c is a 37-feature LambdaRank over 8-source RRF@300
(`docs/blind_a_closed_paths.md:17-29`); eleven post-R54c experiments failed
(`docs/blind_a_closed_paths.md:35-49`); the closed classes are documented at
`docs/blind_a_closed_paths.md:53-145`.

Q5a. `conversation_goal` as a discrete LR feature:

- Not exactly the same as prior track-level structural metadata; it is a
  session-level/query-level category.
- Mechanistically it is still a categorical LR feature addition. It is constant
  within a query group and adds no candidate-specific signal by itself.
- The structural-feature memory says five categorical additions regressed and
  warns against more without stronger evidence:
  `/Users/nadavbarkai/.claude/projects/-Users-nadavbarkai-dev-recsys-2026/memory/feedback_structural_features_exhausted.md:10-28`.
- Verdict: different source, same closed mechanism. Not surviving novelty.

Q5b. `goal_progress_assessments` as prior-turn feedback:

- Closed if implemented as generic hard-negative mining against retrieved
  near-misses:
  `docs/blind_a_closed_paths.md:139-145`.
- Different if used as explicit in-session feedback to weight/filter history
  before retrieval. The signal is observed from the dataset, not mined from
  "not GT" candidates.
- The hard-negative memory allows demonstrably bad pairs, while rejecting
  plausible near-miss negatives:
  `/Users/nadavbarkai/.claude/projects/-Users-nadavbarkai-dev-recsys-2026/memory/feedback_no_hard_negatives.md:10-17`.
- Verdict: survives only as history/anchor weighting/filtering, not as
  contrastive hard-negative training.

Q5c. Route retrievers by `conversation_goal` topic:

- Closed if it means post-LR promotion, demotion, rank floors, or score floors:
  `docs/blind_a_closed_paths.md:91-101`.
- Potentially different if routing happens before pool construction by choosing
  query templates or source weights while preserving standard source-rank
  features.
- Still close to retriever-swap/pool-broadening closures:
  `docs/blind_a_closed_paths.md:55-71`,
  `docs/blind_a_closed_paths.md:103-117`.
- Verdict: survives only as pre-RRF input conditioning with a strict
  source-coverage diagnostic.

## 3. Per-Topic Pool-Hit Diagnostic

Diagnostic run:

- Inputs: `cache/r58/top50_dev.pkl`, `exp/eval/_R12_all_turns_payload.pkl`,
  and cached TalkPlayData dev/test Arrow.
- `cache/r58/top50_dev.pkl` includes per-case `gt_in_pool`; it is built by
  `scripts/expR58_inventory.py:420-491` and saved at
  `scripts/expR58_inventory.py:568-573`.
- No retrieval, training, scoring, or modeling experiment was run.
- Overall: 8000 cases, 1000 dev sessions, 0 missing goals, pool_hit 0.6220.

Per `conversation_goal.category` pool_hit:

| category | n | pool_hit | rate |
|---|---:|---:|---:|
| A | 488 | 314 | 0.6434 |
| B | 1136 | 707 | 0.6224 |
| C | 464 | 244 | 0.5259 |
| D | 688 | 411 | 0.5974 |
| E | 760 | 484 | 0.6368 |
| F | 760 | 528 | 0.6947 |
| G | 616 | 368 | 0.5974 |
| H | 1080 | 740 | 0.6852 |
| I | 144 | 87 | 0.6042 |
| J | 616 | 372 | 0.6039 |
| K | 1248 | 721 | 0.5777 |

Interpretation:

- Category dispersion exists: C and K are low, while F and H are high.
- This does not prove independent signal beyond user text. It only shows that
  current pool coverage is goal-conditional.
- It supports a source-coverage/router diagnostic, not an LR one-hot feature.

## 4. Surviving Candidates

G1. Goal-conditioned pre-RRF retrieval/query routing.

- Mechanism: use `conversation_goal.category`, `specificity`, or
  `listener_goal` to choose query variants or source weights before RRF, while
  keeping standard source-rank features for LR scoring.
- Why not closed: query-side input conditioning before pool construction, not
  candidate structural metadata and not post-LR rerank.
- Smallest falsifiable diagnostic: per-category GT coverage in A/B/C/D/F/ALS/
  R21/R54 and in RRF@300. Kill if low-pool categories lack source-specific
  asymmetry.
- Expected ceiling: low to moderate. Moving weak categories halfway to average
  pool_hit is roughly +0.01 to +0.02 pool_hit overall; nDCG lift is smaller
  because LR still must convert recovered candidates.
- Runtime cost: near zero for selecting among existing sources; 2-5x retrieval
  CPU if running extra query variants.

G2. Goal-progress-weighted session history.

- Mechanism: use prior-turn `goal_progress_assessments` to weight/filter user
  utterances and played-track anchors before BGE/BM25/ALS/track-sim retrieval.
- Why not closed: explicit in-session labels change the session representation
  upstream; they are not mined negatives from plausible retrieved candidates.
- Smallest falsifiable diagnostic: count cases with prior DOES_NOT labels,
  compare current pool_hit by label pattern, then rerun cheap BM25/track-sim
  source recall with negative-labeled anchors removed.
- Expected ceiling: bounded by label coverage; probably small unless DOES_NOT
  labels are common and concentrated in low-pool buckets.
- Runtime cost: preprocessing-only for label alignment; low for BM25/track-sim;
  higher for R54/BGE query regeneration.

Non-survivor:

- `conversation_goal` one-hot LR feature. It is another categorical LR feature
  and is closed under feature-broadening.

## 5. Recommended Single Next Experiment

Recommended next move: G1 source-coverage diagnostic, not an LR feature build.

Design: use cached dev sources and `conversation_goal.category`; report GT
coverage in A/B/C/D/F/ALS/R21/R54, source union, RRF@300, and LR buckets; then
compare low-pool C/K against high-pool F/H.

Kill-shot criteria:

- Kill G1 if low-pool categories do not show a source with clear
  category-specific coverage advantage over current RRF admission.
- Kill G1 if a simple category-conditioned RRF/source-choice simulation cannot
  raise all-dev pool_hit by at least +0.010 without increasing unscoreable
  candidates.
- Kill any submission path unless CV5 h7 nDCG clears +0.010 and same-artist
  regression stays within 0.002. Do not start with LR one-hots, post-LR
  goal-topic reranks, or goal-progress hard-negative training.

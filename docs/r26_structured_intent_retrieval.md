# R26 Structured Intent Retrieval

## Purpose

R26 is the next high-variance retrieval direction if R24 teacher distillation does not produce a meaningful lift.

The core hypothesis is that the remaining gap is not solved by more generic embedding sources or harder negatives. The system needs to convert conversational user requests into explicit music-search intent before retrieval.

Current production baseline:

- R21 blind nDCG@20: 0.4734
- R25 response-only composite: 0.5948
- Leader nDCG@20: about 0.57
- Current retrieval stack: A/B/C/D/F + ALS + R21, RRF pool, LambdaRank rerank

The observed gap is mostly track retrieval quality, not response quality.

## Why This Is Different

Prior failed content retrieval attempts used the raw user text or simple metadata expansion:

- R13 query-to-track dense retrieval
- R14 expanded BM25
- Source G generative retrieval
- MusicBrainz enrichment
- R15 weak-source fusion

Those methods treated the conversation as a bag of words or a single embedding query. R26 instead treats the conversation as a structured music intent that must be parsed before retrieval.

Examples of intent fields:

- target artists
- artists to avoid
- genre/style
- mood/energy
- era/scene
- instrumentation/production
- activity context
- similarity anchors
- negative constraints
- desired novelty level

This matters because many Blind-A failures appear to involve conversational constraints like:

- "not that artist again"
- "same feeling but different game soundtrack"
- "golden age jazz-sampled conscious rap"
- "road-trip melodic punk, not too heavy"

Raw embedding retrieval often blurs these constraints.

## Architecture

```text
Conversation history
        |
        v
Structured intent extractor
        |
        +-- canonical query
        +-- artist query
        +-- genre/style query
        +-- mood/energy query
        +-- era/context query
        +-- negative constraints
        |
        v
Multi-query retrieval
        |
        +-- BM25 over track title/artist/album/tags
        +-- BM25 with artist/tag boosts
        +-- R21 BGE over original conversation
        +-- R21/BGE over rewritten intent queries
        +-- optional constrained artist/tag retrieval
        |
        v
RRF fusion with existing A/B/C/D/F/ALS/R21
        |
        v
LambdaRank rerank
        |
        v
Top-20 tracks
```

## Structured Intent Schema

Use a small JSON schema. Keep it stable and easy to validate.

```json
{
  "summary": "short natural-language music request",
  "positive_artists": ["artist names explicitly desired or referenced"],
  "negative_artists": ["artist names explicitly rejected"],
  "similarity_anchors": ["tracks/artists/soundtracks mentioned as references"],
  "genres": ["genre and subgenre terms"],
  "moods": ["mood adjectives"],
  "energy": "low|medium|high|unknown",
  "era": "decade/scene if stated",
  "context": "activity or listening situation",
  "must_have": ["hard constraints"],
  "avoid": ["negative constraints"],
  "query_variants": [
    "artist/style focused query",
    "mood/genre focused query",
    "metadata/BM25 focused query"
  ]
}
```

The extractor may be an LLM API call for the first experiment. If API cost or latency is a concern, cache all dev/blind extractions by session/turn.

## Retrieval Sources

R26 should be additive. It should not replace R21 initially.

Candidate new sources:

### Q1: Intent BM25

BM25 over the existing track metadata, queried with:

```text
summary + genres + moods + era + context + positive_artists
```

### Q2: Artist/Tag Boosted BM25

Same as Q1, but repeat artists, genres, and tags to bias sparse metadata matches.

Example:

```text
artist artist artist genre genre mood era context
```

### Q3: Intent Dense Retrieval

Encode each `query_variants[]` string with the same BGE/R21 retriever and retrieve top-k tracks. This differs from R21 because the input is a clean intent rewrite rather than noisy dialogue.

### Q4: Constraint-Aware Filtered Retrieval

Apply lightweight filtering or penalties:

- remove played tracks
- downweight negative artists
- downweight explicitly rejected soundtracks/games if detectable
- optionally boost positive artists or same-tag candidates

Do not hard-filter aggressively in the first version. Incorrect extraction can be worse than no extraction.

## Fusion Policy

Start conservative:

```text
Base sources:
A=1.0, B=1.0, C=1.0, D=0.5, F=1.0, ALS=1.0, R21=1.0

R26 sources:
Q1_intent_bm25 = 0.25
Q2_artist_tag_bm25 = 0.25
Q3_intent_dense = 0.5
```

Sweep:

```text
R26 total weight: 0.25, 0.5, 1.0
pool_k: 300, 400
```

The first goal is pool coverage, not immediate top-20 ranking. LambdaRank will decide whether the new candidates are rankable.

## Offline Evaluation

Use the established 8,000-case dev payload and grouped-session split discipline.

Stage 1: source diagnostics only

- standalone hit@50/100/200/300
- unique GT hits vs R21
- unique GT hits vs full production pool
- seen/unseen split
- hist_0 and hist_7 split
- same-artist vs different-artist
- overlap with B/C and R21
- count of candidates removed/downweighted by negative constraints

Stage 2: fusion diagnostics

- pool_hit@300
- pool_hit@400
- unique gained/lost vs R21
- slice pool_hit by history depth
- slice pool_hit by seen/unseen

Stage 3: LambdaRank rerank

- CV5 nDCG
- last_turn nDCG
- same_artist nDCG
- diff_artist nDCG
- hist_0 nDCG
- unseen nDCG

## Gates

Do not build blind unless R26 clears offline gates.

Minimum source gate:

```text
unique GT hits vs R21@200 >= 150
pool_hit@300 >= R21 + 0.020
hist_7 pool_hit does not regress
negative-constraint filter removes <5% of true GTs in dev
```

Minimum rerank gate:

```text
last_turn nDCG >= R21 + 0.005
CV5 nDCG >= R21 + 0.005
same_artist does not regress by more than 0.003
unseen or different_artist improves
```

Strong gate:

```text
pool_hit@300 >= 0.650
last_turn nDCG >= R21 + 0.010
```

The calibration proxy from R22 suggested that reaching blind nDCG near 0.57 likely requires pool_hit far above the current 0.60 range. R26 is promising only if it moves pool coverage materially, not by 0.002.

## Implementation Plan

### Phase 0: Extraction Cache

Build:

```text
scripts/expR26_extract_intents.py
```

Inputs:

- R12 dev cases
- optional Blind-A cases later

Outputs:

```text
cache/r26/intents_dev.json
cache/r26/intents_blind_a.json
```

Requirements:

- deterministic JSON validation
- retry on invalid JSON
- no blind extraction until dev gates pass

### Phase 1: Retrieval Diagnostics

Build:

```text
scripts/expR26_intent_retrieval.py
```

Inputs:

- intent cache
- catalog metadata
- R21 lists
- existing payload sources

Outputs:

```text
exp/eval/expR26_stage1_source_diagnostics.json
```

### Phase 2: Fusion + LambdaRank

Build:

```text
scripts/expR26_fusion_lambdarank.py
```

Outputs:

```text
exp/eval/expR26_stage2_fusion.json
```

### Phase 3: Blind Submission

Only after gates pass.

Build or extend:

```text
run_inference_blind_r26.py
```

Use hybrid responses from the current best response artifact unless top-1 is incompatible.

## Risks

### LLM extraction noise

Incorrect negative constraints can remove good candidates. Prefer downweighting before hard filtering.

### Generic intent rewrites

If all extracted queries sound like "energetic emotional track", retrieval will be redundant with R13. Track query diversity and field specificity.

### Overfitting dev prompts

Do not hand-tune prompts on a few known failures only. Evaluate across all 8,000 dev cases.

### Cost and latency

Cache everything. Dev extraction is 8,000 cases; blind is only 80.

## Success Criteria

R26 is successful if it produces a new source that is:

- not redundant with R21 or BM25
- improves pool coverage materially
- survives LambdaRank reranking
- transfers to Blind-A without harming LLM response quality

If R26 only improves lexical matching but does not improve pool_hit or last_turn nDCG, stop and pivot to cross-encoder reranking.

## Next Radical Backup: Cross-Encoder Reranking

If structured retrieval fails, the next most plausible jump is a cross-encoder over top-100 candidates:

```text
input: [conversation] [track metadata]
output: relevance score
```

This directly models semantic fit and constraints. It is more expensive but attacks the exact gap that bi-encoder and LambdaRank features may miss.

Start with fold-0 only, top-100 candidates, and pairwise/listwise training. If it improves local last_turn, distill the cross-encoder scores back into LambdaRank or BGE for production.

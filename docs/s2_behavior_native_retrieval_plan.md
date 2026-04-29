# S2 Behavior-Native Retrieval Plan

## Goal

Build a new high-recall candidate generator that learns from listening/session
behavior directly, rather than predicting into a fixed metadata embedding space.

The target is not a small reranking gain. The target is a material candidate
coverage improvement:

```text
Current best deterministic pool:
  ABCDF top-200 hit ~= 49.2% on 8000-case all-turn benchmark

Needed for leaderboard movement:
  materially higher candidate recall, ideally >60% hit@200
```

The current sequence model `S` passed the unique-hit gate but did not transfer
well to Blind A. Its main limitation is that it predicts into fixed
`metadata-qwen3` space. That space is useful for semantic similarity, but the
remaining gap appears behavioral: session continuation, co-listening, artist
transitions, popularity, and user/session intent.

S2 should learn a behavior-native item space and retrieve candidate tracks from
that space.

## Scientific Grounding

### 1. Spotify / RecSys 2018 Playlist Continuation

Reference:

- Ching-Wei Chen et al., "A Two-stage Model for Automatic Playlist Continuation
  at Scale", RecSys Challenge 2018.
  https://ssanner.github.io/papers/recsys18_challenge.pdf

Useful lessons:

- Winning systems used **two-stage retrieval + reranking**.
- First-stage retrieval optimized high recall with multiple collaborative and
  sequence-based sources.
- Second-stage models reranked large candidate pools.
- Strong first-stage candidate recall was essential. The paper reports very
  high recall at large candidate depths before reranking.

Implication for this project:

```text
Do not expect Powell/LLM reranking to recover missing tracks.
First build a behavior-native high-recall candidate source.
```

### 2. Spotify CoSeRNN

Reference:

- Spotify Research, "Contextual and Sequential User Embeddings for Music
  Recommendation", 2021.
  https://research.atspotify.com/2021/04/contextual-and-sequential-user-embeddings-for-music-recommendation/

Useful lessons:

- Represent the user/session as a vector.
- Combine long-term preference and short-term/contextual offsets.
- Retrieve items by approximate nearest-neighbor search in embedding space.

Implication:

```text
Represent the current conversation/session as a behavior-aware vector.
Score tracks by dot product against learned item embeddings.
```

### 3. Spotify Text2Tracks / Generative Retrieval

Reference:

- Spotify Research, "Text2Tracks: Improving Prompt-based Music Recommendations
  with Generative Retrieval", 2025.
  https://research.atspotify.com/2025/04/text2tracks-improving-prompt-based-music-recommendations-with-generative-retrieval

Useful lessons:

- Track identifiers can be generated/retrieved directly.
- Semantic IDs based on collaborative-filtering representations can outperform
  text/title-based IDs.
- Behavior-derived representations matter for music recommendation.

Implication:

```text
The item space should not be only metadata text/qwen space.
Use collaborative/session behavior to define or train item representations.
```

### 4. Semantic IDs for Joint Generative Search and Recommendation

Reference:

- Spotify Research, "Semantic IDs for Joint Generative Search and
  Recommendation".
  https://research.atspotify.com/publications/semantic-IDs-for-joint-generative-search-and-recommendation

Useful lessons:

- The representation of item IDs strongly affects retrieval/generation quality.
- Joint search+recommendation representations can be better than single-task
  identifiers.

Implication:

```text
S2 should eventually combine user text/query and behavior-native item identity,
not just one modality.
```

### 5. Multimodal Marketplace Recommendation

Reference:

- ZipRecruiter Tech, "Multimodal Learning for Employment Marketplace
  Recommendation".
  https://medium.com/ziprecruiter-tech/multimodal-learning-for-employment-marketplace-recommendation-ee67bdbede53

Useful lessons:

- Use learned entity embeddings for scalable candidate retrieval.
- Pretrain item/entity encoders from behavioral co-occurrence.
- Use dot-product retrieval rather than expensive all-pair scoring.

Implication:

```text
Build trainable track embeddings and a session encoder.
Use dot-product retrieval for top-K candidates.
```

## Current Evidence From This Project

### Deterministic stack saturated

Tried and mostly exhausted:

```text
A' qwen recent similarity
B  BM25 last-music metadata
C  BM25 full history
D  qwen track neighbors
F  CF-BPR item-item
G  session co-occurrence
S  qwen-space sequence model

Rankers:
  8-feature Powell
  source-aware linear
  LambdaRank
  Haiku rerank
  Sonnet rerank experiments
  reserved slots
  per-history-depth policies
```

The best deterministic/weak-learned sources produce small local gains and weak
Blind-A transfer.

### Source S result

Source `S` learned something:

```text
S unique hits vs ABCDF@200: 200+ / 8000
Collapse controlled: ~0-2%
Fusion sanity: ~+0.005 CV5 on dev400
Blind-A result: regression vs F1
```

Interpretation:

```text
The neural sequence idea is valid.
The fixed qwen target space is probably too restrictive.
```

## Core Hypothesis

S2 should optimize the item space and retrieval objective directly:

```text
Old S:
  conversation -> predicted qwen metadata vector -> nearest qwen tracks

New S2:
  conversation/history/query -> session vector
  track ID / behavior / metadata -> trainable track vector
  score = dot(session vector, track vector)
```

This gives the model freedom to learn:

- behavioral co-listening similarity
- next-track transitions
- artist/session continuation
- popularity priors
- when current user text should override history
- when history should dominate text
- non-textual relations absent from metadata embeddings

## Implementation Strategy

Proceed in stages. Do not jump straight to a complex neural system unless the
cheap behavior-native candidate source passes gates.

## Phase 0 — Benchmark Contract

Use the same benchmark definitions for every experiment:

```text
Primary benchmark:
  B1 8000-case all-turn benchmark

Secondary:
  last_turn 1000-case benchmark
  hist-depth slices: hist_0, hist_1, hist_2, hist_3, hist_4plus, hist_5plus

Baseline:
  ABCDF top-200 and top-50 from existing R12/B1 payload where applicable
```

Required metrics:

```text
standalone_hit@20
standalone_hit@50
standalone_hit@200
unique_GT_hits_vs_ABCDF@200
overlap_with_A/B/C/D/F
pool_hit@50 after fusion
CV5 nDCG@20 after fusion+Powell
median GT rank when hit
hist-depth breakdown
```

Hard rule:

```text
No Blind-A submission unless a source passes candidate-generation gates first.
```

## Phase 1 — ALS / WRMF Candidate Source

### Purpose

Build the cheapest behavior-native item space first.

This mirrors the collaborative filtering backbone used in many high-recall
music/playlist recommenders, including the RecSys 2018 winner lineage.

### Data

Build an implicit interaction matrix from train conversations.

Candidate row definitions to test:

```text
Option A: session_id x track_id
  each training session is one user/entity
  interaction = track appeared in session

Option B: prefix_context_id x track_id
  each prefix is one pseudo-user/context
  tracks in prefix are positives
  future/next tracks are held out for eval only

Option C: real user_id x track_id
  use if user_id exists and coverage is adequate
  likely weaker for Blind-A due limited user overlap
```

Start with Option A. It is simple and behavior-native.

### Model

Train implicit ALS / WRMF:

```text
library preference:
  implicit.als.AlternatingLeastSquares if available
  otherwise scipy/sklearn fallback

rank:
  {64, 128, 256}

alpha:
  {10, 40, 100}

regularization:
  {0.01, 0.05, 0.1}

iterations:
  15-30
```

### Inference For Eval Turn

Given a conversation prefix with played tracks:

```text
session_vector = weighted mean of ALS item vectors for played tracks
score(track) = dot(session_vector, ALS item_vector)
exclude already played
return top-200
```

Weights to test:

```text
uniform over played tracks
recency decay 0.8^age
last-5 only
max/mean hybrid
```

For hist_0, ALS has no anchors. Return empty for V0.

### Phase 1 Gates

On B1 all-turn benchmark:

```text
ALS hit@200 >= Source S hit@200
ALS unique hits vs ABCDF@200 >= 300 / 8000
```

Strong pass:

```text
ALS unique hits vs ABCDF@200 >= 500 / 8000
```

Fusion gate:

```text
ABCDF+ALS CV5 lift >= +0.010
```

If ALS unique hits are below 200, stop Phase 1 and do not tune endlessly.

## Phase 2 — S2 Neural Candidate Generator

Run only if Phase 1 shows behavior-native embeddings are useful, or if ALS is
close but insufficient.

### Model Family

Train a next-track model over trainable item embeddings.

Recommended V0:

```text
Inputs:
  previous track IDs
  current user utterance embedding
  turn index / history length

Item embeddings:
  trainable
  initialized from one of:
    ALS item vectors
    CF-BPR projected to model dim
    qwen projected to model dim
    concat/projection of ALS + CF-BPR + qwen

Session encoder:
  GRU4Rec or small SASRec-style causal transformer

Scoring:
  score = session_emb @ item_emb.T

Loss:
  sampled softmax or BPR
  hard negatives from ABCDF candidates
  random negatives from catalog
```

Prefer GRU first for efficiency:

```text
GRU model:
  embedding_dim = 128 or 256
  hidden_dim = 256
  1-2 layers
  dropout = 0.1
```

Only move to SASRec/Transformer if GRU shows a candidate-generation signal.

### Training Examples

For each train session:

```text
prefix tracks: tracks before turn t
current query: user utterance at turn t
positive: track at turn t
```

Include hist_0/query-only examples if possible:

```text
hist_0 input = current query embedding + learned no-history token
```

This matters because Blind-A has many low-history cases.

### Negative Sampling

Use mixed negatives:

```text
in-batch positives
ABCDF hard negatives
same-artist negatives
popular random negatives
uniform random negatives
```

Do not rely only on random negatives. Random negatives make the task too easy
and do not improve ranking against real candidate pools.

### Objectives To Compare

Start with:

```text
sampled softmax / InfoNCE
```

Then compare:

```text
BPR pairwise loss
sampled softmax with hard-negative temperature
multi-positive future loss
```

Multi-positive future loss is important:

```text
For a prefix, predict not only the immediate next track but also future tracks
in the same session.
```

This follows the RecSys 2018 insight that pure next-song prediction can
over-focus on the most recent item.

### Phase 2 Gates

Candidate-generation gate:

```text
S2 unique hits vs ABCDF@200 >= 500 / 8000
```

Fusion gate:

```text
ABCDF+S2 pool_hit@50 improves by >= +0.03
ABCDF+S2 CV5 nDCG improves by >= +0.015
```

If S2 standalone improves but fusion does not:

```text
inspect median rank of S2-unique GTs
if median rank > 80, train for sharper ranking or use S2 as reranker feature
if median rank <= 50, tune fusion/source weights
```

## Phase 3 — Pairwise Reranker Over Large Candidate Pool

Run only after a new candidate source materially improves pool coverage.

Do not repeat previous LambdaRank failure blindly. The prior ranker failed
because the candidate pool and features were weak. A better pool changes the
setup.

### Candidate Pool

Build out-of-fold candidate pools:

```text
ABCDF
ALS
S2
optional S qwen-space
optional G co-occurrence
```

Pool size:

```text
top-100 or top-200 for training
top-50 for final rerank
```

### Features

Include:

```text
source ranks and reciprocal ranks
source presence bits
source scores if available
8 existing Powell features
artist/title/tag/query overlaps
track popularity
artist popularity
album popularity
session homogeneity features
embedding similarities:
  qwen sim to recent tracks
  cf-bpr sim to recent tracks
  ALS sim to session vector
  S2 score
history-depth bucket
turn index
```

### Model

Use LightGBM LambdaRank or XGBoost ranker with strict grouped CV.

Critical:

```text
Candidate pools and features for held-out sessions must be built without using
held-out labels.
```

### Phase 3 Gates

```text
CV5 lift over ABCDF+best_source+Powell >= +0.010
stable across seeds
no large hist-depth regression
```

## Reporting Requirements For Claude

Every report must include:

```text
command run
artifact path
runtime
exact benchmark slice
baseline reproduced
metric table
raw JSON/log evidence path
gate verdict
next recommended action
```

Do not report checkpoint numbers without artifact/log evidence.

## Stop Conditions

Stop a direction if:

```text
unique hits vs ABCDF@200 < 200 / 8000
or fusion CV5 lift < +0.005 after a small sweep
or model only improves standalone metrics but overlaps existing sources heavily
```

Do not spend time polishing response generation until track nDCG has a clear
improvement.

## Recommended Immediate Task For Claude

```text
Implement Phase 1: ALS / WRMF candidate source.

Constraints:
- No API.
- No blind submission.
- Use B1 8000 all-turn benchmark.
- Reuse existing metadata and payloads where possible.
- Output artifact: exp/eval/expS2_als_candidate_source.json

Steps:
1. Build train session-track implicit matrix.
2. Train ALS configs:
   rank in {64, 128, 256}
   alpha in {10, 40, 100}
   regularization in {0.01, 0.05, 0.1}
   start with a small pilot grid; prune aggressively.
3. For each eval case, create a session vector from prior tracks.
4. Retrieve top-200 tracks, excluding already played tracks.
5. Evaluate standalone hit@20/50/200 and unique hits vs ABCDF@200.
6. If unique hits >=300, run fusion sweep:
   w_ALS in {0.25, 0.5, 1.0, 2.0}
   depth in {50, 100, 200}
7. Report gates and do not build blind artifacts.

Primary gate:
  unique hits vs ABCDF@200 >= 500 / 8000

Secondary gate:
  ABCDF+ALS CV5 lift >= +0.010
```

## Expected Outcomes

### Best case

ALS or S2 creates a behavior-native source with:

```text
500+ unique hits vs ABCDF@200
+0.015 or more CV5 lift after fusion
```

This is worth a new blind submission.

### Medium case

ALS produces 250-500 unique hits but weak fusion.

Next action:

```text
train S2 neural candidate generator initialized from ALS
```

### Bad case

ALS produces fewer than 200 unique hits.

Interpretation:

```text
simple behavior-native CF is not enough
move directly to S2 neural sequence model or reconsider data assumptions
```

## Strategic Principle

The leaderboard gap is too large for another small fusion tweak. The next
successful system must improve candidate generation substantially.

S2 should be judged by this question:

```text
Can it retrieve hundreds of correct tracks that ABCDF cannot retrieve?
```

If yes, optimize fusion and ranking. If no, stop.

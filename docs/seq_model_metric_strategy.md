# Sequence Model S — Metric Strategy

## Purpose

Source `S` is a learned sequence-aware retrieval source. Its job is to improve
track accuracy by predicting a target vector in the same space as the
`metadata-qwen3_embedding_0.6b` catalog embeddings, retrieving top candidates,
and adding them to the existing fusion pipeline.

This document explains how `S` affects each competition metric and how to decide
whether a submission using `S` is worth making.

## Competition Metrics

The leaderboard composite is:

```text
Composite = 0.50 * nDCG@20
          + 0.10 * Catalog Diversity
          + 0.10 * Lexical Diversity
          + 0.30 * normalized LLM Judge Score
```

The exact leaderboard normalization is controlled by the evaluator, but the
practical weight ordering is clear:

```text
nDCG@20          highest leverage
LLM Judge Score  second-highest leverage
Lexical Diversity small leverage
Catalog Diversity currently near-pinned
```

## High-Level Effect of Source S

```text
Metric                 Expected effect from Source S
------------------------------------------------------------
nDCG@20                Direct target
Catalog Diversity      Indirect / usually small
Lexical Diversity      No direct effect
LLM Judge Score        No direct effect, but possible indirect risk
Composite Score        Improves only if nDCG lift beats response-side loss
```

Source `S` is not a response generator and not a diversity optimizer. It is a
candidate-generation source. Its success should be judged primarily by whether
it adds ground-truth tracks to the candidate pool and whether those tracks
survive fusion/reranking into the final top-20.

## Pipeline Position

Current deterministic retrieval sources:

```text
A' = qwen3 max similarity over recent played tracks
B  = BM25 over last-music metadata query
C  = BM25 over full-history query
D  = qwen3 track-neighbor retrieval
F  = CF-BPR item-item retrieval
S  = learned sequence model retrieval
```

They combine as:

```text
A' ranked list
B  ranked list
C  ranked list
D  ranked list
F  ranked list
S  ranked list
      |
      v
Weighted RRF fusion
      |
      v
top-50 candidate pool
      |
      v
8-feature Powell postrank
      |
      v
final top-20 predictions
```

Important distinction:

```text
RRF controls candidate admission.
Powell controls final ordering.
S is a new candidate source, not a replacement for Powell.
```

## nDCG@20

This is the primary target.

`S` can improve nDCG in two ways:

1. It retrieves ground-truth tracks that `A'/B/C/D/F` miss.
2. It ranks those tracks high enough that RRF and Powell place them in top-20.

The first condition is more important. If the ground-truth track is not in any
candidate pool, no reranker can recover it.

### Required Local Diagnostics

Before considering a blind submission, evaluate `S` on a held-out validation
slice with session-level splits.

Required metrics:

```text
S_recall@20
S_recall@50
S_recall@200
S_unique_hits@200_vs_ABCDF
ABCDF_baseline_CV5_nDCG@20
ABCDFS_CV5_nDCG@20
delta_CV5 = ABCDFS - ABCDF
collapse_rate = fraction where S top-1 == last played track
```

### nDCG Gates

```text
Hard fail:
  S_unique_hits@200_vs_ABCDF is negligible
  or delta_CV5 < +0.003

Promising:
  S_unique_hits@200_vs_ABCDF >= 3% of evaluated cases
  and delta_CV5 >= +0.005

Submission candidate:
  delta_CV5 >= +0.010
  and the lift is stable across multiple seeds/folds
```

The exact thresholds should be interpreted against current calibration. As of
the deterministic stack, small local gains have transferred weakly to Blind A,
so marginal gains below `+0.005` local are not worth a submission by themselves.

## Catalog Diversity

Catalog Diversity has had low practical leverage in current leaderboard results.
Top submissions have clustered around approximately the same value.

`S` may change Catalog Diversity indirectly by retrieving a different set of
tracks. This is useful only if it does not sacrifice nDCG.

Do not tune `S` for Catalog Diversity unless:

```text
1. nDCG is preserved, and
2. Catalog Diversity changes enough to affect composite, and
3. the change is reproducible across submissions.
```

At present, this is not a primary optimization axis.

## Lexical Diversity

Lexical Diversity is computed from `predicted_response`, not from the retrieval
model itself.

`S` has no direct mechanism to improve Lexical Diversity. If the same response
generation prompt and model are used, Lexical Diversity should remain close to
prior submissions.

Risk: if `S` changes top-1 recommendations, the response generator may produce
different wording, which can slightly move Lexical Diversity. This is secondary.

Recommended handling:

```text
1. First evaluate track predictions without changing response generation.
2. Generate responses only after the track artifact is validated.
3. Compare Lexical Diversity against prior known submissions.
```

## LLM Judge Score

`S` does not directly optimize LLM Judge Score, but it can affect it indirectly.

Prior submissions showed that changing the ranking can change the top-1 track
used by response generation, and that can move LLM Score substantially. A better
nDCG ranking can produce a worse response score if the top-1 is harder to
explain, less coherent with the conversation, or mismatched with the response
prompt.

Observed historical pattern:

```text
v1:
  lower nDCG, higher LLM Score

v2/cfg0209:
  higher nDCG, lower LLM Score

hybrid:
  v2 tracks + v1-style responses partially recovered LLM Score
```

This means response quality and ranking quality are coupled through the selected
recommendation context.

### LLM Risk Rule

Approximate composite tradeoff:

```text
+0.01 nDCG       ~= +0.005 composite
-0.10 LLM score  ~= -0.006 composite
```

So a retrieval improvement can be erased by a response-quality regression.

Example:

```text
If S improves nDCG by +0.02:
  composite gain from nDCG ~= +0.010

If S also drops LLM Score by -0.20:
  composite loss from LLM ~= -0.012

Net effect: likely negative.
```

### Response Strategy for S Submissions

Use a two-stage submission workflow:

```text
Stage 1: Track artifact
  - validate 80 rows
  - 20 unique valid track IDs per row
  - no response generation changes
  - compare top-20 overlap with prior best

Stage 2: Response artifact
  - use the best known response generation strategy
  - ensure non-empty responses
  - remove formatting artifacts such as leading comma bleed
  - compare response lengths and top-1 alignment
```

If `S` changes many top-1 tracks, run a response diagnostic before blind
submission:

```text
top1_changed_count
old_top1_in_new_top20_count
response_mentions_new_top1_artist
response_mentions_new_top1_title
leading_comma_count
empty_response_count
```

## Composite Score

The composite only improves if the nDCG gain survives response-side effects.

The practical decision formula is:

```text
Expected composite delta
  ~= 0.50 * delta_nDCG
   + 0.10 * delta_CatDiv
   + 0.10 * delta_LexDiv
   + 0.30 * delta_normalized_LLM
```

Since Catalog Diversity and Lexical Diversity usually move little, most
decisions reduce to:

```text
Expected composite delta ~= 0.50 * delta_nDCG
                          + 0.30 * delta_normalized_LLM
```

Operational rule:

```text
Submit S only when:
  1. local nDCG lift is real and stable,
  2. S adds unique candidate hits beyond ABCDF,
  3. response generation is expected to preserve LLM Score,
  4. validation artifacts are structurally clean.
```

## Recommended Evaluation Sequence

### 1. Train-Path Sanity

Required before full training:

```text
mixed-length batching unit test passes
self-positive is excluded from in-batch negatives
validation excludes already-played tracks
200-example overfit smoke passes
```

### 2. Source-S Standalone Evaluation

Evaluate the trained checkpoint as a pure retrieval source:

```text
S@20
S@50
S@200
top1_last_played_collapse_rate
unique GT hits vs ABCDF@200
overlap with A'/D/F
```

The most important metric is unique hits, not standalone nDCG.

### 3. Fusion Evaluation

Add `S` to weighted RRF:

```text
ABCDF baseline
ABCDFS with w_S in {0.25, 0.5, 1.0, 2.0}
ABCDFG optional comparison
ABCDFGS optional comparison
```

Evaluate:

```text
pool_hit@50
median GT rank when hit
CV5 nDCG@20
per-history-depth nDCG
```

### 4. Powell Stability

Run session-level CV across multiple seeds. Do not rely on one split.

Minimum report:

```text
mean CV5
std across seeds
per-fold values
learned Powell weights
win/loss cases vs ABCDF
```

### 5. Blind Submission Gate

Only build a blind submission if:

```text
delta_CV5 >= +0.005 stable
or S finds a large number of unique hits and a clear fusion path exists
```

For a serious leaderboard attempt, prefer:

```text
delta_CV5 >= +0.010
```

because prior small local gains have transferred weakly.

## Failure Modes

### 1. Embedding Collapse

Model predicts vectors near the last played track.

Symptoms:

```text
top1 == last_played too often
high overlap with A'
low unique hits vs ABCDF
```

Response:

```text
increase anti-collapse lambda
add explicit played-track exclusion during validation/inference
measure overlap with A'/D
```

### 2. No Unique Candidate Signal

S standalone metrics may look nonzero but mostly duplicate A'/D/F.

Symptoms:

```text
S recall@200 exists
but unique_hits_vs_ABCDF is low
ABCDFS CV5 flat
```

Response:

```text
do not submit
try alternative target space or objective
```

### 3. Fusion Dilution

S adds hits, but RRF pushes them too low or displaces stronger candidates.

Symptoms:

```text
S unique hits high
pool_hit@50 flat or down
median GT rank worsens
CV5 flat
```

Response:

```text
sweep w_S and source depths
test S as top-100 source but low RRF weight
consider candidate-level learned fusion only after S proves unique hits
```

### 4. Response Regression

S improves nDCG but changes top-1 recommendations in a way that lowers LLM
Score.

Symptoms:

```text
nDCG up
LLM Score down
composite flat or worse
```

Response:

```text
reuse best prior response strategy
test hybrid responses when top1 remains in top20
harden prompt to mention selected track explicitly
```

## Current Strategic Read

`S` is the right kind of experiment because it creates a new learned retrieval
signal rather than further tuning deterministic fusion. The deterministic path
has largely saturated. However, `S` should be judged by candidate coverage and
fusion lift, not by training loss or standalone memorization.

The key question is:

```text
Does Source S add ground-truth tracks that ABCDF does not already retrieve,
and can those tracks be promoted into top-20 without damaging response quality?
```

If yes, `S` is a viable path toward a higher nDCG submission. If no, the branch
should be treated as a useful negative result and the next step should be a
different retrieval signal or a stronger supervised candidate-generation model.

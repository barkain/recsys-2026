# R31 Sequential Recommender Design

## Goal

R31 is a true sequential recommendation source for Blind-A last-turn nDCG.

The repeated failure pattern from R26-R30 is that text, intent, generic reranking, and LambdaRank reweighting can improve overall CV or cold-start slices but do not improve the target slice. Blind-A is effectively a deep-history / last-turn task, so the next model should directly learn ordered behavioral dynamics:

```text
track sequence + recent session context -> next track
```

R31 should be evaluated first as a retrieval source. It should not replace R21 or LambdaRank until it proves incremental value.

## Prior Results This Design Responds To

- R21 supervised BGE retriever is the production retrieval breakthrough, but it is still text-centric.
- R26 structured intent retrieval improved pool/CV/hist_0 but hurt blind last-turn.
- R28 deep-history LambdaRank weighting/features failed; current handcrafted features are near ceiling.
- R29 zero-shot cross-encoder failed hist_7; generic relevance is not recommendation relevance.
- R30 deep-history BGE specialist failed clean validation; text bi-encoder specialization gave tiny h7@200 lift and hurt h7@20.

Conclusion: the next substantial nDCG path is not another text feature. It is a sequence model over item IDs.

## High-Level Architecture

Use a SASRec-style Transformer encoder over recent track IDs, optionally augmented with current query text.

```text
recent track IDs            current user query
      |                            |
      v                            v
 track embedding table       frozen/text encoder
      |                            |
 positional embeddings        projection
      |                            |
      +-------------+--------------+
                    |
                    v
         Transformer / GRU sequence encoder
                    |
                    v
          session vector / next-item vector
                    |
                    v
       dot product against candidate track vectors
                    |
                    v
             top-K R31 candidates
```

R31 is a source:

```text
A/B/C/D/F/ALS/R21/R31 -> RRF pool -> LambdaRank -> top-20
```

## Data

### Training Sources

Use only training sessions for the first honest diagnostic:

```text
15,199 train sessions
all music turns available within each session
```

Do not train on dev sessions for fold diagnostics.

### Evaluation Source

Use the existing 8,000 dev cases with grouped-session folds.

For the first diagnostic:

```text
fold 0 only
train: non-fold-0 sessions / train sessions as appropriate
eval: fold-0 dev cases
primary slice: hist_7
secondary slice: hist_5..7
```

### Sequence Construction

For each target event, build:

```text
input sequence = previous music track IDs in chronological order
target = next GT track ID
```

For dev cases:

```text
input sequence = case["music_turns"]
target = case["gt"]
```

Use maximum sequence length:

```text
max_len = 8 initially
```

Later sweep:

```text
max_len in {8, 16, 32}
```

But start with 8 because Blind-A sessions are short and hist_7 is the target.

## Model V0: SASRec-ID

V0 intentionally uses learned item-ID embeddings, not BGE/Qwen embeddings, as the primary representation. This isolates the core question R31 is meant to answer:

```text
Is there ordered behavioral signal in the track sequence that current text/content sources miss?
```

Qwen/BGE embeddings encode metadata/content similarity. R31 V0 should first test sequential collaborative dynamics:

```text
users who go A -> B -> C often next go D
```

This transition can be real even when D is not semantically similar to A/B/C in metadata space.

### Inputs

```text
track_id sequence: [t1, t2, ..., tk]
position ids:      [1,  2,  ..., k]
```

### Embeddings

```text
item_emb:     num_tracks x d_model
pos_emb:      max_len x d_model
padding_idx:  0
```

Recommended starting dimensions:

```text
d_model = 128
n_layers = 2
n_heads = 4
dropout = 0.2
max_len = 8
```

This is intentionally small. We need a fast, stable fold-0 signal before larger models.

Input token:

```text
x_t = item_emb(track_id_t) + pos_emb(position_t)
```

### Encoder

Use causal self-attention:

```text
h = TransformerEncoder(item_emb + pos_emb, causal_mask=True, padding_mask=True)
session_vec = h[last_non_pad_position]
```

### Scoring

Score catalog tracks by dot product:

```text
score(track_j) = session_vec @ item_emb[j]
```

Use tied input/output item embeddings for V0.

This output head is strongest for seen-track behavioral recommendation. It cannot score tracks that are absent from the learned item vocabulary unless those tracks are explicitly assigned embeddings.

## Qwen Embedding Variants

The provided Qwen metadata embeddings can be useful, but they should be introduced as controlled variants after the pure ID baseline. There are two distinct places to use Qwen.

### Variant V1a: Qwen As Input Features

For each prior track, combine a learned ID embedding with a projected Qwen metadata embedding:

```text
id_vec_t     = item_emb(track_id_t)                 # learned, d_model
qwen_vec_t   = qwen_emb(track_id_t)                 # fixed, 1024d
qwen_proj_t  = Linear(1024 -> d_model)(qwen_vec_t)
pos_vec_t    = pos_emb(t)

x_t = LayerNorm(id_vec_t + qwen_proj_t + pos_vec_t)
```

Rationale:

- ID embedding captures behavioral identity and co-listening transitions.
- Qwen embedding adds content continuity and helps rare tracks share statistical strength.
- Position embedding preserves order.

Risk:

- Existing sources A/D already use Qwen heavily. Adding Qwen to R31 can make the source redundant unless the sequence encoder contributes genuinely new ordering signal.

### Variant V1b: Qwen As Output Scoring Space

Instead of scoring candidates with learned item embeddings, predict a vector in Qwen space:

```text
session_vec = Transformer(sequence)
pred_qwen   = normalize(Linear(d_model -> 1024)(session_vec))
score_j     = pred_qwen @ qwen_emb(track_j)
```

Rationale:

- Can score every catalog track with a Qwen embedding, including cold/rare tracks.
- Better for content-continuity recommendations.
- Uses the same catalog vector space as existing Qwen sources, but with a sequence-conditioned query vector.

Risk:

- It may collapse back into content similarity and fail to model behavioral transitions.
- Prior Qwen/content-heavy approaches improved cold-start/CV but often hurt Blind-A last-turn.

### Variant V1c: Two-Head R31

If V0 shows any sequential signal, test two output heads:

```text
ID head:
  score_id(track_j) = session_vec @ item_emb[j]

Qwen head:
  pred_qwen = normalize(Linear(session_vec -> 1024))
  score_qwen(track_j) = pred_qwen @ qwen_emb[j]
```

Produce separate candidate lists:

```text
R31_ID
R31_QWEN
R31_ID_UNION_QWEN
```

Expected behavior:

- `R31_ID` should help seen behavioral hist_7 tracks.
- `R31_QWEN` may help content-continuity and cold/rare tracks.
- The union may be useful if the two lists are complementary.

Decision rule:

```text
Do not add Qwen variants unless V0 has at least weak positive hist_7 signal.
If V0 fails completely, Qwen is unlikely to rescue R31 because Qwen-heavy retrieval has already been tested in other forms.
```

## Model V1: SASRec + Query Adapter

Only try this after V0 establishes a positive sequential signal.

Add current user query as a final context vector:

```text
query_text -> frozen BGE/R21 text encoder -> projection to d_model
session_vec = LayerNorm(seq_vec + query_proj)
```

Do not use LLM-extracted intent in V1. R26 showed intent is anti-correlated with hist_7.

## Training Objective

### Primary Loss

Sampled softmax / in-batch contrastive next-item loss:

```text
positive = target track
negatives = in-batch targets + sampled catalog negatives
loss = cross_entropy(scores / tau, label=positive)
```

Recommended:

```text
tau = 0.07
batch_size = 256 if memory allows, else 128
negatives_per_example = 128 sampled negatives if using sampled softmax
```

If using in-batch only:

```text
batch_size should be >= 128
```

Small negative sets were harmful in prior contrastive experiments.

### Negative Sampling

Avoid “hard negatives” initially. R23 showed plausible music negatives are noisy and can damage the manifold.

Start with:

```text
70% popularity-smoothed random negatives
30% uniform random negatives
```

Popularity-smoothed distribution:

```text
p(track) proportional to count(track)^0.75
```

Do not use same-artist or BM25 hard negatives in V0.

### Exclusion

At inference, exclude already played tracks from candidate lists.

During training, do not mark repeated/plausible tracks as explicit negatives beyond sampled softmax.

## Candidate Coverage For Cold / Unseen Tracks

A pure item-ID sequence model cannot score tracks never seen in training because they have no learned item embedding.

That is acceptable for R31 because its job is not to replace R21/BM25. Its job is to improve **seen-track behavioral continuity** in deep-history cases.

Expected role:

```text
R21/BM25: unseen/content retrieval
ALS/CF/R31: seen behavioral retrieval
LambdaRank: combine
```

If R31 works, later add content-initialized item embeddings:

```text
item_emb_init = projection(BGE/R21 track text embedding)
```

or Qwen-initialized / Qwen-augmented embeddings:

```text
item_emb_init = projection(qwen_track_embedding)
x_t = learned_id_emb + projected_qwen_emb + pos_emb
```

But V0 should use learned ID embeddings to test pure sequential signal.

## Evaluation Protocol

### Stage 0: Sanity

Train on a tiny subset and verify overfit:

```text
train 256 examples
expect train hit@20 > 0.90
```

If it cannot overfit, debug architecture/loss.

### Stage 1: Fold-0 Standalone Retrieval

Train R31 on allowed training data, evaluate on fold-0 dev cases.

Report:

```text
hist_7 hit@20 / 50 / 100 / 200 / 300
hist_5_7 hit@20 / 200
all hit@200
unique GTs vs R21
unique GTs vs current full pool
lost GTs vs R21
seen vs unseen split
top1_repeat_rate
overlap with R21
overlap with ALS/CF-BPR if available
for Qwen variants: overlap with A/D Qwen sources
```

Primary gate:

```text
hist_7 hit@200 >= R21_OOF + 0.05
hist_7 hit@20 >= R21_OOF - 0.005
unique GTs vs R21 on hist_7 >= 20
```

Hard stop:

```text
hist_7 hit@20 drops by more than 0.01
or all@200 collapses by more than 0.03
or top1_repeat_rate is excessive
```

### Stage 2: Fold-0 Fusion

Add R31 as a source to the existing fold-0 fusion.

Test RRF weights:

```text
R31 weight in {0.25, 0.5, 1.0}
pool_k in {300}
optional route: only include R31 for hist>=5
```

Features:

```text
rank_R31
r31_presence
r31_score_norm
r31_agreement_count with R21/ALS/CF
```

For two-head variants, keep features separate:

```text
rank_R31_ID
r31_id_presence
r31_id_score_norm
rank_R31_QWEN
r31_qwen_presence
r31_qwen_score_norm
```

Gate:

```text
hist_7 nDCG@20 >= baseline +0.005
hist_5_7 nDCG@20 >= baseline +0.003
all nDCG@20 not worse by more than 0.002
```

No blind until Stage 2 passes.

### Stage 3: Full OOF

Only if Stage 2 passes fold-0:

```text
train 5 OOF R31 models
generate dev OOF R31 lists
train LambdaRank with R31 features
train production R31 on all allowed data
build blind artifact
```

## Implementation Plan

### Script

Create:

```text
scripts/expR31_sequential_retriever.py
```

Phases:

```text
--phase build_data
--phase overfit_smoke
--phase train_fold0
--phase eval_fold0
--phase fusion_fold0
```

Use phase separation to avoid memory conflicts and make runs resumable.

### Artifacts

```text
cache/r31/vocab.json
cache/r31/train_examples_fold0.npz
cache/r31/model_fold0.pt
cache/r31/fold0_r31_lists.json
exp/eval/expR31_stage1_standalone.json
exp/eval/expR31_stage2_fusion.json
```

### Safety

Default to CPU or conservative MPS:

```text
device = cpu initially
num_workers = 0
pin_memory = False
save checkpoints each epoch
```

If using MPS:

```text
batch_size <= 256
no giant all-catalog score matrix if memory spikes
score candidates in chunks
```

Do not run large MPS jobs without chunking. R29 repeatedly crashed the Mac due MPS OOM.

## Expected Outcomes

### If R31 Works

Expected pattern:

```text
hist_7 hit@200 improves
hist_7 nDCG@20 improves after fusion
unseen recall does not improve much
seen behavioral cases improve
```

This would directly attack the gap to leaders on nDCG.

### If R31 Fails

If pure sequential ID modeling fails fold-0:

```text
the current data may not contain enough sequential signal
or GT generation may be more content/conversation driven than behavior driven
```

Next radical directions would be:

```text
1. supervised generation of candidate track IDs via LLM constrained to catalog
2. larger music-domain cross-encoder fine-tuned on OOF pairs
3. ensemble across multiple independently trained R21-like retrievers
```

But R31 should be tested before those because it is the cleanest untried inductive bias for hist_7.

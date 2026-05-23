# R79 — Top-Rank Discriminator Retriever Design (DESIGN ONLY)

**Status:** design draft. NOT to be implemented without explicit user go.

## Why a different retriever objective

All prior retrieval work optimized **hit@300** (does GT appear anywhere
in the top-300 pool?). R72 confirmed our R54-stacked pool has GT in top-300
for ~62% of cases overall, and R68's encoder upgrade adds only 10 fold-0 /
2 h7 cases that are rescuable into top-30.

The actual bottleneck is different: when GT IS in top-300, the LR ranks it
at position 21-300 (not top-20) in many cases. That's where nDCG is lost.

**R79's idea:** train a retriever whose loss explicitly contrasts GT
against the **wrong candidates R54c currently puts in top-20**, instead
of just "GT vs random non-GT".

## Hypothesis

R68 (BGE-large) was trained with standard contrastive loss: pull GT close
to query, push random catalog tracks away. It learned good general
semantic similarity but didn't learn to discriminate GT from "plausible
neighbors that R54c surfaces".

R79 would use **hard negatives mined from R54c's top-20 false positives**.
For each train case:
- Positive: GT track
- Hard negatives: tracks R54c put in its top-20 that are NOT the GT
- Loss: contrastive (push GT above hard negatives), or LambdaRank-style

This trains the retriever to produce embeddings where GT > hard_negatives
when scored by cosine similarity. If successful, R79's standalone top-20
nDCG should beat R54c's, even if hit@300 stays similar.

## Architecture sketch

- **Encoder**: BGE-base or BGE-large (start small to limit cost)
- **Query encoding**: same structured query format as R54 (`<query> <user_query> <history>`)
- **Track encoding**: track metadata (title + artist + album + tags + year)
- **Training data**: for each fold-0 train case, sample 1 positive (GT) + K hard negatives (R54c top-20 non-GT) + M random negatives (background)
- **Loss**: InfoNCE with hard negatives weighted higher

## Key differences from R68

| property | R68 | R79 |
|---|---|---|
| objective | hit@300 (any-top-300) | top-20 displacement |
| negatives | random | R54c's top-20 false positives |
| loss | softmax InfoNCE | weighted InfoNCE or pairwise margin |
| training data | all 8000 dev (or 4-fold OOF) | per-fold OOF, requires R54c top-20 |
| evaluation | hit@300 + h7 pool_hit | top-20 nDCG, h7 nDCG |
| ceiling | "candidate exists" | "candidate beats LR's mistake" |

## Why this might still fail

1. **R54c's top-20 false positives are HARD by definition.** They're the
   candidates that maximize R54c's calibrated structural features. If R79
   produces embeddings that beat them on cosine, R79 might be learning
   "same-artist-but-wrong-track" or "same-genre-but-wrong-mood" patterns
   that the LR judge will penalize via same-artist canary.

2. **Hard-negative mining historically failed** ([[feedback_no_hard_negatives]]).
   R23/R23a tested it; hurt scores. Music neighbors are plausible and
   pushing GT away from them generalizes poorly. R79 would need to limit
   the strength of hard-negative push to avoid breaking general semantic
   structure.

3. **Composing with R54c LR is unresolved.** Even if R79 produces better
   cosine ranking on fold-0 dev, the LR pipeline needs to know how to use
   R79 features. R71 showed stacker fails. R76 showed neural residual fails.
   The conversion problem persists.

## What's different vs prior R68 / R76 failures

- **R68** (BGE-large): trained with standard contrastive. Found new GT
  candidates at ranks 50-300 (R72). The features were good but the GT
  candidates landed too deep for any downstream reranker to rescue.
- **R79**: targets the conversion failure directly. Doesn't try to find
  NEW candidates; tries to RE-ORDER existing candidates by promoting GT
  over R54c's top-20 mistakes.

## Trigger conditions (BEFORE any implementation)

R79 should only be implemented if ALL of these hold:

1. We have A100 budget (~$50, ~8 hours).
2. We agree the hard-negative-mining concern from [[feedback_no_hard_negatives]]
   has been mitigated by the explicit "vs R54c top-20" targeting (which is
   different from R23's random hard negatives).
3. We have a clear OOF eval framework that compares R79-only top-20 against
   R54c's frozen top-20, NOT against frozen R54c LR with R79 features
   added (R76 closed that path).
4. We are willing to lose if R79 produces semantic-similar-but-wrong
   recommendations and same-artist regresses (the canary risk).
5. Blind-A is still open or we are designing for Blind-B.

## What R79 won't do

- Won't close the LLM gap (4.90 → 4.95 unreachable).
- Won't close the LexDiv gap (already near ceiling).
- Won't help if frozen R54c LR is the final scorer (it doesn't know R79
  features). R79 either replaces R54 as the top-K source OR we accept
  using R79's own top-20 directly as the submission (bypassing LR).

## Why R79 is the only remaining hard-path option

After today's closures (R76 neural residual, R78 response ceiling), every
other angle has been explored and archived. R79 is novel in that it changes
the **retriever's training objective** rather than its features or backbone.

But the empirical history is unfavorable. Expected probability of success:
< 25%.

## Parked

Not implementing without explicit user go.

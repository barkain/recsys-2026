# R83 Sprint — Behavior-Native Sequence Model (SASRec) — ARCHIVE_SPRINT

**Date:** 2026-05-23

## Verdict

ARCHIVE at Phase 0 fold-0. Sessions are too short (~5-7 played tracks) for
a pure sequence model to extract meaningful behavioral signal beyond what
R54c LR already captures from the same data. ~$0.50 A100 spent.

## What we built

SASRec-style causal transformer over played track sequences + utterance:
- Item embeddings: 47K × 128 trainable, init from BGE-large catalog projection
- 2-layer Transformer (d=128, heads=4, ff=256) with causal mask
- Utterance: BGE-large query → 128 projection, prepended as position 0
- Sampled softmax loss with mixed negatives (in_batch=64, random=64, R54c FP=16, same-artist=4)
- Trained on 6400 fold-1..4 cases (20 epochs, bf16, ~30 sec/epoch)

## Phase 0 fold-0 result

| metric | OOF R54c | R83 standalone | Δ |
|---|---:|---:|---:|
| all_fold0 | 0.2123 | 0.0312 | **−0.181** |
| h7 | 0.2226 | 0.0261 | **−0.196** |
| same_artist | 0.4475 | 0.0724 | **−0.375** |
| diff_artist | 0.0955 | 0.0107 | −0.085 |

- h7 recovered = 0, lost = 71 (net −71)
- **h7 top-30 GT hits: 14/200** (7%); unique vs R54c top-20: **1**
- top-1 churn = 78.1/80 (essentially fully changed)
- top-20 overlap = 1.9/20

All gates fail catastrophically.

## Why this failed

**Sessions are too short.** Each case has ~5-7 played tracks
(h7 means exactly 7). Sequence models like SASRec/GRU4Rec work well when
you have hundreds of historical interactions per user — they extract
behavioral patterns from long trajectories. With 7 items there's not
enough sequential context to predict the next listen from history alone.

R54c's "match this query against catalog metadata" signal is
overwhelmingly more informative than "predict next given played-7" at
this dataset scale.

The loss DID converge (144 → 4.6 across 20 epochs), so the model learned
SOMETHING — but it converged to behavior that scores well on its own
sampled-softmax objective and poorly on actual retrieval against the
full 47K catalog. This is the same failure mode as R79 hard-negative
retriever, just from a different angle: the model lacks the regularization
to maintain catalog-wide discrimination.

## What this rules out

**The behavioral signal class is exhausted at this dataset scale.**
Sequence models need long histories; we have short sessions.

Combined with prior closures, the empirical map of nDCG paths is complete:

| signal class | best result |
|---|---|
| Encoder upgrades (BGE-large fine-tune) | R68: hit@300 saturated |
| Pool admission (matched retrain) | R59 C3: rescuable cases ≤2 |
| LR feature substitution/addition | R68.1, R70: OOF artifact |
| Tree stackers (top-30) | R71: no signal |
| Naked LLM/cross-encoder rerank | R67/R69: catastrophic |
| Hard-negative retriever | R79: collapse |
| Neural arch over top-300 | R76/R80/R81: same-artist canary fires |
| LLM-derived intent features | R82: GT < FP on every feature |
| Behavior-native sequence model | **R83: catastrophic** (sessions too short) |

## Production state unchanged

R78 holds at composite 0.6302, position #4 on Blind-A.

## Total spend on Phase 0 R79-R83 + R82 API

- R79: ~$3 A100
- R80: ~$3 A100
- R81 (4 configs): ~$12 A100
- R82: ~$1 API
- R83: ~$0.50 A100
- **Total: ~$20**

Every failure caught cheaply via predeclared gates.

## What remains theoretically possible

1. **External data injection**: Last.fm play counts, Spotify popularity,
   audio features (Spotify API), professional editorial labels. Speculative;
   none are committed dataset assets.
2. **R84 multimodal revisit**: pending, low-cost (Mac). Tests audio/lyrics/
   image embeddings against current R78/R54 with guarded injection.
3. **Multi-task training**: train R83 jointly with R54c labels (next-track
   sequence loss + R54c imitation loss). Speculative; would require new
   data prep + multi-day GPU.

R84 is the only one of these that's cheap. R83's catastrophic failure
suggests R84 is also unlikely to clear gates, but it's a $0 Mac test.

## Files

- `scripts/expR83_sasrec_phase0.py`
- `exp/eval/expR83_phase0.json` (on Colab)
- `docs/r83_phase0_result.md` (on Colab)
- `docs/r83_sprint_summary.md` — this file

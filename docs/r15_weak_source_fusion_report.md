# R15: Weak Source Fusion — Negative Result

## Summary

Attempted to improve pool coverage by fusing three complementary weak retrieval sources into the V3 baseline. Local validation showed gains; blind submission regressed.

**V3 (baseline):** nDCG 0.4420, composite 0.5753
**R15 (this experiment):** nDCG 0.4400, composite 0.5746
**Verdict:** Did not transfer. V3 remains best.

## Experiments

### R13: Query-to-track semantic retrieval (Qwen3-Embedding-0.6B)
- Embed user queries with Qwen3, cosine search against track embeddings
- Q_context variant (query + last user msg): hit@200 = 22.7%, unique unreachable = 70
- **Failed gates:** unique unreachable 70 < 150, diff-artist lift +2.1% < +3%
- Low overlap with A/D (11%), structurally valid but too weak alone

### R14: Expanded BM25 content retrieval
- Tested 20 configs (4 index variants x 5 query variants)
- Best config `base__q_kitchen_sink`: unique unreachable = 75, diff-artist lift +7.1%
- **Failed gates:** unique unreachable 75 < 150

### Source G: LLM generative retrieval (existing, never evaluated)
- Already cached in R12 payload, Haiku-generated track suggestions -> BM25 lookup
- unique unreachable = 19, pop=0 recovery = 0, hist_0 recovery = 0
- **Failed gates:** unique unreachable 19 < 150, diff-artist lift +2.4% < +3%

### Union diagnostic
- Q, R14, G recover almost entirely different unreachable GTs (pairwise overlap ~0-2)
- **Union unique unreachable = 158 >= 150: PASS**
- Union diff-artist recovery: 1988/4968 (40.0%, +4.4% lift)

### R15: Controlled fusion (Stage 2)
- 7 configs tested with 37-feature LambdaRank (27 V2 + 10 weak source features)
- Two configs passed gates:
  - `v3+weak_w0.25_p300`: last-turn 0.2315 (+0.0069), CV5 0.2129
  - `v3+weak_w0.5_p200`: last-turn 0.2301 (+0.0055), CV5 0.2144
- Selected `v3+weak_w0.25_p300` for blind submission

### Blind-A result
| Metric | V3 | R15 | Delta |
|---|---|---|---|
| nDCG@20 | 0.4420 | 0.4400 | -0.0020 |
| CatalogDiv | 0.0305 | 0.0300 | -0.0005 |
| LexDiv | 0.7754 | 0.7783 | +0.0029 |
| LLM | 4.6500 | 4.6500 | 0 |
| Composite | 0.5753 | 0.5746 | -0.0007 |

## Why it failed

- Pool coverage gains (52% -> 56.7%) are real on dev (8000 cases) but only affect ~1-2 of 80 blind rows
- 35/80 top-1 tracks changed — introduced as many wrong changes as right ones
- Weak sources are noisy; local LambdaRank learns to handle them but this doesn't generalize to blind distribution
- The 158 unreachable recoveries are spread across 8000 dev cases — too sparse to matter on 80 blind samples

## Conclusion

Weak-source fusion is not robust for small blind evaluation sets. The approach is sound in principle (complementary sources, low overlap) but the signal is too dilute to transfer. V3 baseline remains the production system. Do not repeat this fusion path without a diagnostic explaining the local-to-blind gap.

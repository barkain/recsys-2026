# R32 Last-Turn Error Taxonomy — Analysis for Codex

## Executive Summary

**The bottleneck is ranking, not retrieval.** Our current pool@300 contains enough GTs to reach 0.598 nDCG@20 on hist_7 with perfect ranking. Leaderboard #1 has 0.53. We already have the candidates — LambdaRank just fails to surface them.

All R26-R31 work on new retrieval sources attacked the wrong problem.

## Oracle Ceilings (hist_7, 1000 dev cases)

```
R21 raw top-20 nDCG@20:         0.086
LambdaRank system (est CV5):    0.230
Perfect rerank pool@300:        0.598   ← beats leaderboard #1
Perfect rerank R21@300 only:    0.455
Perfect rerank ALL sources:     0.727
```

LambdaRank captures only 38% of the available oracle signal (0.230/0.598). 62% of rankable signal is left on the table.

## Source Coverage (hist_7 hit@k, out of 1000)

```
Source          @20    @50   @100   @200   @300
A (qwen3)      187    244    281    309    309
B (BM25)       234    305    341    374    392
C (BM25)       230    350    432    492    523
D (qwen3)      178    218    245    268    268
F (CF-BPR)     128    197    240    285    285
R21 (BGE)      196    309    368    421    455
ALS            126    204    260    305    305
─────────────────────────────────────────────
ALL UNION      473    563    645    703    727
```

C (BM25 full-text) has highest coverage. R21 is #2. Union@300 covers 727/1000.

## Error Categories (hist_7)

```
Category                Count      %    Actionability
─────────────────────────────────────────────────────
in_top20                  322   32.2%   Already correct
in_pool_21_100            182   18.2%   HIGH — close to top-20, ranking fix
in_pool_101_300            94    9.4%   MEDIUM — needs bigger rank jump
in_source_not_pool        145   14.5%   MEDIUM — RRF fusion loses them
seen_not_sourced          108   10.8%   LOW — no source finds them
unseen_unreachable        149   14.9%   ZERO — not in training data
─────────────────────────────────────────────────────
TOTAL                    1000  100.0%
```

**Key insight:** 276 GTs (27.6%) are in pool@300 but ranked below top-20. This is the actionable gap. Another 145 (14.5%) are in sources but lost at RRF fusion.

## Attribute Analysis

### Seen vs Unseen GT
```
seen:   551 (55.1%) — R21@20: 18.1%, pool@300: 66.1%
unseen: 449 (44.9%) — R21@20: 21.4%, pool@300: 52.1%
```

Unseen tracks have slightly higher R21@20 rate (21.4% vs 18.1%) — likely because R21 text embeddings work well for unseen content. Pool coverage is lower for unseen (52% vs 66%).

### Same-Artist vs Different-Artist GT
```
same_artist: 467 (46.7%) — R21@20: 35.8%, pool@300: 94.0%
diff_artist: 533 (53.3%) — R21@20:  5.4%, pool@300: 29.8%
```

**This is the most important split.** Same-artist GTs have 94% pool coverage but only 36% top-20 rate. 272 same-artist GTs are in pool but not top-20 — a same-artist boost alone could recover many of these.

Different-artist is the hard problem: only 30% pool coverage, 5% top-20. These cases require genuine content/conversation understanding.

## Concrete Failure Patterns (from 20 examples)

All 20 failure examples are `in_pool_21_100` — GTs ranked 21-75, close to top-20 boundary.

**Pattern 1: Same-artist continuity (16/20 cases)**
User explicitly enjoys an artist, GT is another track by same artist, but ranker doesn't promote it high enough.
- Beatles: user loves "Oh! Darling", GT is another Beatles track at pool rank 43
- Lacuna Coil: user asks for more, GT at rank 24-33
- Frank Ocean: user wants more deep R&B, GT at rank 23-27
- "Weird Al": user literally says "Play another Weird Al song", GT at rank 31

**Pattern 2: Multiple sources agree but all ranked middling (4/20)**
GT appears in 5-6 sources at ranks 15-50 each. RRF gives a decent pool rank (21-55) but LambdaRank doesn't promote it past other candidates that have one strong source rank.

**Pattern 3: ALS knows but R21/BM25 don't (3/20)**
ALS ranks GT at 1-4, but text sources rank it 30-80. RRF averages them out. If ALS-confident cases were handled differently, these would be recovered.

## What This Means for Modeling

### The ranking gap breakdown
```
0.230 (current) → 0.598 (oracle pool@300)
Gap = 0.368 nDCG

Decomposition:
- 182 in_pool_21_100 cases: ranking fix needed (pool rank 21-100)
- 94 in_pool_101_300 cases: larger ranking fix (pool rank 101-300)
- 145 in_source_not_pool: RRF/pool expansion fix
```

### Why LambdaRank fails
1. **Features are source-rank proxies.** If GT is ranked 80th by R21, the feature says "weak" — but it might be the right track for this specific conversation.
2. **No conversation-conditioned scoring.** LambdaRank doesn't read the conversation to assess candidate relevance.
3. **Same-artist signal underweighted.** Binary feature exists but doesn't capture "user explicitly wants more from this artist."
4. **Equal treatment of all hist depths.** LambdaRank trains on all 8000 cases equally; hist_7 patterns get diluted.

### Highest-value directions

**1. Same-artist reranking heuristic (quick win)**
- For hist_7 cases, boost same-artist candidates by N positions
- Expected recovery: fraction of 272 same-artist pool misses
- Zero ML needed — just a post-hoc adjustment

**2. Fine-tuned conversation→candidate scorer**
- Input: (last 3 user messages, candidate metadata)
- Task: binary classification or pairwise ranking within pool@300
- Training: 598 hist_7 cases where GT is in pool (positive) + hard negatives from same pool
- This directly attacks the ranking bottleneck

**3. Improved RRF fusion (recover in_source_not_pool)**
- 145 cases where GT is in at least one source@300 but not in pool@300
- Larger pool_k, source-specific k values, or cascade fusion could recover these

**4. LLM-based reranking on top-50**
- Use Haiku to score top-50 candidates given conversation
- Expensive but targeted at the exact bottleneck
- Pool@50 oracle would tell us the ceiling for this approach

## Data for Further Analysis

Full results saved to: `exp/eval/expR32_error_taxonomy.json`
Script: `scripts/expR32_error_taxonomy.py`

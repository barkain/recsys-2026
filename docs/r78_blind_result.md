# R78 Blind Result — FLAT WIN, Response-Side Ceiling Confirmed

**Date:** 2026-05-23  Submission ID: 749804  Participant: `dirac`

## Verdict: R78 narrowly becomes production; ceiling reached

R78 lifts composite by +0.0002 over R77 — technically a strict-best
production candidate, but practically flat. Confirms the response-side
ceiling at LLM 4.90 / LexDiv ~0.88.

## Scoring breakdown

| metric | R77 (prev prod) | **R78 (new prod)** | Δ |
|---|---:|---:|---:|
| **composite** | 0.6300 | **0.6302** | **+0.0002** |
| nDCG@20 | 0.4925 | 0.4925 | 0.0000 |
| CatalogDiv | 0.0301 | 0.0301 | 0.0000 |
| LexDiv | 0.8821 | **0.8845** | +0.0024 |
| LLM judge | 4.90 | 4.90 | 0.0000 |

## LLM ceiling confirmed at 4.90

R77 broke 4.85→4.90 with vocabulary-rich style. R78 attempted to push
further to 4.95 via LLM-judge-targeted audit (concrete attributes,
explicit causal links, removed imperative closers like "Crank it loud").
Result: LLM held at 4.90.

**Empirical conclusion:** 4.90 is the practical ceiling for our setup.
semintelligence at 4.95 likely uses a categorically different prompt or
style that our audit+regen pattern cannot reach.

## Response-side fully saturated

Both response-side levers are at their ceilings:
- LLM: 4.90 (2 attempts to lift further failed)
- LexDiv: 0.8845 (within leader range 0.86-0.89)

Targeted regens cannot move composite materially from here.

## Leaderboard

| # | participant | composite | nDCG | LexDiv | LLM |
|---:|---|---:|---:|---:|---:|
| 1 | semintelligence | 0.64 | 0.51 | 0.86 | 4.95 |
| 2 | amaranta_ | 0.64 | 0.55 | 0.89 | 4.65 |
| 3 | vkost | 0.64 | 0.53 | 0.76 | 4.90 |
| **4** | **dirac (R78, us)** | **0.6302** | **0.4925** | **0.8845** | **4.90** |
| 5 | el_presidente | 0.63 | 0.57 | 0.77 | 4.50 |

Gap to top-3 cluster (all at 0.64): **0.0098**. Closing this requires nDCG
movement. Our nDCG at 0.4925 is the lowest in the top-5.

## Where can nDCG come from?

All prior retrieval+ranker paths closed:
- Encoder upgrade: R68/R72 (retrieval saturated at top-30)
- Pool admission: R59 C3, R72
- Stacker: R71, R76
- Naked rerank: R56/R58/R67/R69
- LR feature stacking: R68/R70
- Neural residual: R76

The unexplored angle: **a retriever explicitly trained to beat R54c's
top-20 negatives** (rather than optimize hit@300 as R68 did). R68 found
deep candidates that don't surface to top-20; the right loss is to teach
the retriever to push the RIGHT track INTO top-20, displacing the wrong
candidates R54c currently ranks there.

Design draft: R79 top-rank discriminator retriever. NOT to be implemented
without explicit go.

## State of all paths

| dimension | current | leader | ceiling |
|---|---:|---:|---|
| nDCG | 0.4925 | 0.57 | open in principle, all paths closed in practice |
| CatalogDiv | 0.0301 | 0.03 | at/above leaders |
| LexDiv | 0.8845 | 0.89 | near ceiling |
| LLM | 4.90 | 4.95 | confirmed ceiling for our setup |
| **Composite** | **0.6302** | **0.64** | **gap = 0.0098, needs nDCG** |

## Recommended state

**Freeze response-side.** Further submissions on response polish are not
worth a slot. Either:
1. **Accept R78 at #4 (0.6302).** Defend incumbent.
2. **Design R79 top-rank discriminator retriever**, with A100, only if we
   believe a fundamentally new retrieval objective could close the nDCG gap.
3. **Hold for Blind-B** if/when it opens.

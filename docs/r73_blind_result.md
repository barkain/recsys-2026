# R73 Blind Result — CLEAN WIN, NEW PRODUCTION

**Date:** 2026-05-22  Submission ID: 748216  Participant: `dirac`

## Verdict: R73 becomes production

R73 strictly improves composite over R63c-repair with identical tracks and
the same LLM judge score. The lift came entirely from LexDiv.

## Scoring breakdown

| metric | R63c-repair (prev prod) | **R73 (new prod)** | Δ |
|---|---:|---:|---:|
| **composite** | 0.6224 | **0.6234** | **+0.0010** |
| nDCG@20 | 0.4925 | 0.4925 | 0.0000 |
| CatalogDiv | 0.0301 | 0.0301 | 0.0000 |
| LexDiv | 0.8438 | **0.8536** | **+0.0098** |
| LLM judge | 4.85 | 4.85 | 0.0000 |

## Interpretation

**LLM ceiling confirmed at 4.85.** The 15-row concise-direct polish targeted
LLM uplift via semintelligence-style direct verdict openers, banned-phrase
filtering, and tighter prose. LLM judge did not move. This is the second
consecutive submission (after R63c-repair) where LLM holds at 4.85, so we
treat 4.85 as the practical ceiling for the current track set + response
style family.

**LexDiv carried the lift.** Removing 15 verbose/hedgy outliers freed
corpus-wide bigram space ("comes off", "fits that", "carries that" etc.
were appearing in clusters). LexDiv 0.8438 → 0.8536 = +0.0098. Translates
to ~+0.001 composite, i.e. LexDiv carries approximately 10% of its raw
delta into the composite (consistent with the leaderboard's reported
weighting structure).

## Leaderboard position

| # | participant | composite | nDCG | LexDiv | LLM |
|---:|---|---:|---:|---:|---:|
| 1 | semintelligence | 0.64 | 0.51 | 0.86 | 4.95 |
| 2 | amaranta_ | 0.64 | 0.55 | 0.89 | 4.65 |
| 3 | vkost | 0.64 | 0.53 | 0.76 | 4.90 |
| 4 | el_presidente | 0.63 | 0.57 | 0.77 | 4.50 |
| **5** | **dirac (R73, us)** | **0.6234** | **0.4925** | **0.8536** | **4.85** |
| 6 | jumboleg | 0.61 | 0.54 | 0.77 | 4.50 |

Gap to #4 (el_presidente): **0.0066** — close-able with a LexDiv push.

Gap to #1 (semintelligence / amaranta_ / vkost, all 0.64): **0.0166** —
needs nDCG lift or larger LexDiv lift.

## What this confirms

1. LLM judge has a hard ceiling around 4.85 for our setup — further
   stylistic regens are unlikely to move LLM.
2. LexDiv 0.8536 is below leaders (0.86-0.89) — modest headroom remains
   that's worth pursuing.
3. nDCG 0.4925 is the biggest gap (we're lowest in the top-5), but
   retrieval is saturated per R72 — would need new ranker architecture.

## Next step (open decision)

**Option A: R74 LexDiv-focused response variant.**
Push LexDiv 0.8536 → 0.88+ while preserving LLM 4.85 and tracks. Estimated
composite gain +0.002 to +0.004. Lands us at ~0.625-0.628. Probably #4.

**Option B: nDCG via corrected OOF framework.**
Per Codex consult and R70b artifact finding, the LR conversion "wall" was
largely a measurement artifact. Re-examining R56-R66 sprints under proper
OOF-vs-OOF gates may surface positive variants we incorrectly archived.
Higher uncertainty, higher upside (closing some of the nDCG gap could move
composite by 0.01+). Mac-feasible.

**Option C: Hybrid.** Run R74 LexDiv in foreground (safe, +0.002-0.004),
sweep R56-R66 under corrected OOF in background.

R73 zip ready at `exp/inference/blind_a/r73_concise_direct_submission.zip`
remains the production candidate. Sub ID 748216 confirms it landed and
became #5.

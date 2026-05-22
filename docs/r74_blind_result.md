# R74 Blind Result — CLEAN WIN, NEW PRODUCTION

**Date:** 2026-05-22  Submission ID: 748941  Participant: `dirac`

## Verdict: R74 becomes production

R74 lifts composite vs R73 with identical tracks and identical LLM judge,
entirely via LexDiv. The bigram-repeat-density audit + targeted regen
produced a 2× larger LexDiv delta than R73's style-only audit.

## Scoring breakdown

| metric | R73 (prev prod) | **R74 (new prod)** | Δ |
|---|---:|---:|---:|
| **composite** | 0.6234 | **0.6252** | **+0.0018** |
| nDCG@20 | 0.4925 | 0.4925 | 0.0000 |
| CatalogDiv | 0.0301 | 0.0301 | 0.0000 |
| LexDiv | 0.8536 | **0.8719** | **+0.0183** |
| LLM judge | 4.85 | 4.85 | 0.0000 |

## Key findings (post-R73→R74)

1. **LexDiv → composite scaling is consistent.** R73 saw +0.0098 LexDiv →
   +0.001 composite (ratio ~10). R74 saw +0.0183 LexDiv → +0.0018 composite
   (ratio ~10). This is reliable for further LexDiv pushes.
2. **LLM judge ceiling = 4.85** across THREE submissions (R63c-repair, R73,
   R74). Treat as hard ceiling. Further response style work cannot move LLM.
3. **Bigram-repeat-density audit beats style-only audit.** R73's audit
   targeted "verbose openers / hedge phrases" and delivered +0.0098 LexDiv.
   R74's audit targeted "rows contributing most to corpus bigram redundancy"
   plus 8 audit-derived banned bigrams, and delivered +0.0183 LexDiv.

## Leaderboard

| # | participant | composite | nDCG | LexDiv | LLM |
|---:|---|---:|---:|---:|---:|
| 1 | semintelligence | 0.64 | 0.51 | 0.86 | 4.95 |
| 2 | amaranta_ | 0.64 | 0.55 | 0.89 | 4.65 |
| 3 | vkost | 0.64 | 0.53 | 0.76 | 4.90 |
| 4 | el_presidente | 0.63 | 0.57 | 0.77 | 4.50 |
| **5** | **dirac (R74, us)** | **0.6252** | **0.4925** | **0.8719** | **4.85** |

**Gaps:**
- To #4 (el_presidente at ~0.6295): **0.0043**
- To #3 (vkost at 0.64): **0.0148**
- To #1 (semintelligence at 0.64): **0.0148** (tied with #3)

## Next moves (per Codex strategic read)

**A. Controlled R75 LexDiv ceiling push (low risk, small gain)**
- Same approach as R74 but with even more aggressive audit
- Target rows with highest residual bigram density on the R74 corpus
- Goal: push LexDiv 0.8719 → ~0.88+
- Estimated composite gain: +0.003-0.005 → 0.628-0.630
- Likely passes el_presidente to #4. Still does not reach #1-3.

**B. R76 neural residual ranker (high effort, uncertain large gain)**
- DESIGN DOC EXISTS at `docs/r76_neural_residual_ranker_design.md`
- Feature-aware listwise transformer reordering R54c top-30
- Closes some nDCG gap (0.4925 → ~0.50-0.51 if successful)
- Multi-day on A100, only realistic path to break 0.63

**C. Freeze R74 and defend**
- Position #5 at composite 0.6252
- Strong submission, low risk of regression
- Acceptable if Blind-B / submission window is near

## Recommendation (my read)

**Path A first** (controlled R75 ceiling push): cheap, predictable, doable in
hours. Bank the gain.

**Then Path B in parallel** if A100 budget allows: R76 design is ready, but
implementation is 1-2 days. Decision to start should be informed by R75
result — if R75 pushes us to #4 cleanly, we know the LexDiv path has ~+0.003
more in it but no further; if R75 stalls, response-side is fully done and
the only remaining lever is R76.

R74 zip is at `exp/inference/blind_a/r74_lexdiv_submission.zip`.
R73 superseded.

# R432 Goal-Field Retrieval Audit (Phase 0)

**Date:** 2026-06-05
**Verdict:** **GO to Phase 1.** Adding `conversation_goal` + `user_profile` text to the query
lifts first-turn recall and recovers genuinely union-absent GTs — the first new retrieval signal
since R84. TF-IDF proxy only; the real test is conversion (Phase 2).

## Why this isn't a closed lever

`feedback_retrieval_lever_closed` closed *query→track text encoders over `user_query`* (R96).
R432 is different on two axes: (1) it adds a **new input class** production ignores —
`conversation_goal.listener_goal` + `user_profile.preferred_musical_culture`; (2) it targets the
**first-turn (`n_prior=0`) slice**, which is exactly the Blind-A shape and where production
retrieval is weakest.

## Data structure (important)

The challenge `test.arrow` has **1000 unique sessions**; our 8000 dev cases = those 1000 sessions
× 8 turns (`n_prior` 0–7). **`conversation_goal` and `user_profile` are session-level** (constant
across a session's 8 turns). So the dev **`n_prior=0` slice (1000 cases, one per session) is the
exact Blind-A first-turn proxy.** Join coverage: 8000/8000.

## Results — TF-IDF over track metadata

### First-turn (`n_prior=0`, n=1000 — Blind-A shape)
| variant | r@20 | r@30 | r@100 | r@300 | union-absent@300 |
|---|---:|---:|---:|---:|---:|
| `query` (production input) | 0.2040 | 0.2280 | 0.3000 | 0.3590 | 26 |
| `goal` only | 0.0930 | 0.1290 | 0.2010 | 0.2770 | 93 |
| `query_goal` | 0.2290 | 0.2660 | 0.3500 | 0.4200 | 79 |
| **`query_goal_profile`** | **0.2370** | **0.2720** | **0.3540** | **0.4440** | **96** |

First-turn `recall@20` **+0.033** (0.2040→0.2370), `recall@300` **+0.085**, and union-absent@300
GTs **26 → 96** (~70 GTs the production union@300 never sees, surfaced by goal/profile text).
`query_hist_goal` == `query_goal` on first-turn (no history), as expected.

### All turns (n=8000)
| variant | r@20 | r@300 | union-absent@300 |
|---|---:|---:|---:|
| `query` | 0.1303 | 0.2830 | 107 |
| `query_goal` | 0.1616 | 0.3501 | 207 |
| `query_goal_profile` | 0.1654 | 0.3675 | 243 |
| `query_hist_thought_goal` | 0.2275 | 0.4705 | 282 |

(`*_thought_goal` uses assistant `thought` text — strong but a different mechanism; for Blind-A
first-turn the relevant variant is `query_goal_profile`.)

## Conclusion

Goal/profile augmentation adds real, orthogonal first-turn recall — necessary for an nDCG gain
and mechanistically new. **Necessary, not sufficient:** R103/R431 had real recall that did not
convert through the LR. Phase 1 builds the goal-aware **R54 5-fold OOF** source
(`expR432_phase1_goal_oof.py`, query `[QUERY]…[GOAL]…[PROFILE]…`); Phase 2 integrates it R84-style
(RRF + `goal_rank_inv/presence/score` + sibling LR) and gates on **first-turn ΔnDCG** + all-dev
ΔnDCG + churn/canary safety.

Artifacts: `exp/eval/expR432_goal_field_retrieval_audit.json`,
`scripts/expR432_goal_field_retrieval_audit.py`.

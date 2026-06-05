# R432 Goal-Aware Retrieval — Conversion Result (Phase 1–3)

**Date:** 2026-06-06
**Verdict:** **PASS (dev OOF).** Goal-aware retrieval is the first nDCG conversion path since R84c.
First-turn (n_prior=0 = Blind-A shape) **ΔnDCG@20 = +0.0130 at churn 13.8/80** via a churn-safe,
GT-independent selective rule. Blind transfer is the remaining test.

## Pipeline

- **Phase 1** (`expR432_phase1_goal_oof.py`): goal-aware R54 5-fold OOF source. Query =
  `[QUERY] {user_query} [GOAL] {category specificity listener_goal} [PROFILE] {culture country age}`,
  encoded per held-out fold vs that fold's `track_embs`, top-300 excl. played →
  `cache/r432_goal/goal_oof_lists.json` (8000/8000, all len 300).
- **Phase 2** (`expR103_integrate_eval.py --gte-oof <goal> --dump-percase`): R84-style integration,
  arms A=base / B=goal-pool / C=goal-pool+goal-features, 5-fold OOF sibling LR.
- **Phase 3** (`expR432_phase3_firstturn_selective.py`): first-turn selective deployment search.

## Phase 2 — conversion by slice (dC_vs_A)

| subset | n | A | C | dC-A | churn | overlap |
|---|---:|---:|---:|---:|---:|---:|
| all-dev | 8000 | 0.2256 | 0.2288 | +0.0032 | 32.5 | 15.79 |
| **n0 FIRST-TURN (Blind-A)** | 1000 | 0.2085 | 0.2200 | **+0.0115** | 32.6 | 14.08 |
| h7 (deep history) | 1000 | 0.2493 | 0.2430 | −0.0063 | 30.3 | 15.92 |

The h7 regression is expected and irrelevant: `conversation_goal`/`user_profile` are **session-level**,
so once 7 prior tracks pin the intent they add noise. **Blind-A is first-turn-only**, where the goal
text is the dominant personalization signal — and there it converts (+0.0115). Goal LR features carry
real weight (3.6% gain importance, `gte_cosine` dominant).

## Phase 3 — first-turn selective deployment (the candidate)

Deploy goal-arm-C ordering only where a GT-independent rule holds; else keep production (arm A).

| rule | n_sel | dNDCG | churn/80 | overlap | rec/lost |
|---|---:|---:|---:|---:|---:|
| always-C | 1000 | +0.0115 | 32.6 | 14.08 | 55/32 |
| **`gtecos≥0.65 OR top1-unchanged`** | 766 | **+0.0130** | **13.8** | 15.75 | 44/26 |
| `gtecos≥0.70 OR top1-unchanged` | 705 | +0.0110 | 9.0 | 16.28 | 40/24 |
| oracle (ndC≥ndA) | 881 | +0.0315 | 28.2 | — | — |

The rule keeps production unless the goal candidate promoted to #1 is **high-confidence (cosine ≥ 0.65)**,
reverting the low-confidence injections that caused churn. Net: more gain than always-C at <half the churn.

## Why this is not the R431/R103 wall

R431 (user-cf) failed because **no GT-independent signal separated wins from losses** (discoverability
wall — oracle +0.0236 unreachable). R432 is different: the **goal-source cosine predicts correctness**,
so the selective rule *improves* on always-C (+0.0130 > +0.0115) while cutting churn (13.8 < 32.6). The
signal is genuinely discoverable.

## Caveats / risk

- Dev OOF passes; **blind transfer is unproven** (R90, R107 passed dev and failed blind). This is
  OOF-vs-OOF and churn-safe, which is the right shape, but not a guarantee.
- Threshold 0.65 selected on dev. Overfit risk low: dNDCG is stable (+0.011…+0.013) across cos 0.60–0.70,
  and the rule is a single threshold + a no-op condition.
- h7/multi-turn must NOT get arm-C (regresses); Blind-A being first-turn-only sidesteps this.

## Next (pending go): build Blind-A candidate

1. Build blind goal source: encode Blind-A first-turn goal-queries with the R54 5-fold ensemble vs
   `track_embs`, top-300 (the blind analog of Phase 1).
2. Integrate goal source + goal features into the production blind ranker; apply the selective rule
   (`gtecos≥0.65 OR top1-unchanged`).
3. Patch R106 A-clean responses onto the new ranked tracks.
4. Churn/overlap preflight vs R106 A-clean; submit only if within gates.

Production stays R106 A-clean (0.6377) until a blind candidate clears preflight.

Artifacts: `exp/eval/expR432_integrate_w1.json`(+`_percase.json`),
`exp/eval/expR432_phase2_firstturn.json`, `exp/eval/expR432_phase3_firstturn_selective.json`,
`cache/r432_goal/goal_oof_lists.json`.

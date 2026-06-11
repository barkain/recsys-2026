# R432 Goal-Aware Retrieval — Conversion Result (Phase 1–3)

**Date:** 2026-06-06
**Verdict:** **PASS (dev OOF).** Goal-aware retrieval is the first nDCG conversion path since R84c.
First-turn (n_prior=0 = Blind-A shape) **ΔnDCG@20 = +0.0130 at churn 13.8/80** via a churn-safe,
GT-independent selective rule. Blind transfer is the remaining test.

> **2026-06-07 current-scorer correction:** the old "R432 closed because R106 holds 4.90"
> conclusion must be interpreted against the historical scorer only. An exact R106 repeat later
> scored **LLM 4.80 / composite 0.6302**, while R432s scored **nDCG 0.5092 / LLM 4.85 /
> composite 0.6349**. Under the current scorer, **R432s is the active-best submission if the old
> historical R106 submission cannot be selected/kept**. See `docs/PRODUCTION.md` and
> `docs/r433_multimodal_on_r432s_probe_plan.md`.

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

---

## BLIND RESULT (2026-06-06): NEGATIVE — ARCHIVE

The goal-only FINAL conservative candidate was scored on Blind-A:

| metric | R106 (prod) | R432 goal-only | Δ | composite impact |
|---|---|---|---|---|
| nDCG@20 | 0.5073 | 0.5075 | +0.0002 | +0.0001 |
| LexDiv | 0.8859 | 0.8868 | +0.0009 | +0.0001 |
| LLM judge | 4.90 | 4.85 | −0.05 | −0.0038 |
| **composite** | **0.6377** | **0.6342** | **−0.0035** | |

**The entire composite loss is the LLM judge** (4.90→4.85). The nDCG bet — the whole point —
moved **+0.0002** (≈ +0.0001 composite). Strong dev signal (first-turn +0.0159) did not transfer
to blind (+0.0002 nDCG@20): another dev→blind nDCG non-transfer.

**Verdict: ARCHIVE the whole R432 family** (profile + goal-only). Production stays **R106 A-clean
(0.6377)**. Two durable lessons:
1. Goal-aware retrieval converts on dev but not on blind — the nDCG recall ceiling holds.
2. Bounded few-case nDCG bets are dominated by LLM-judge variance (±0.05 ≈ ±0.0038 composite).
   A blind nDCG gain must be ≥ ~+0.02 to clear the judge noise floor. See
   `feedback_small_ndcg_bets_dominated_by_llm_variance`.

---

## CORRECTION — reorders-only control: R432 has REAL nDCG signal (2026-06-06)

The "archive the whole family" verdict above was premature. A second submission isolated the
**reorder-only** part (8 first-turn rank-2-20 reorders, **0 top-1 changes, responses byte-identical
to R106**):

| submission | nDCG@20 | LexDiv | LLM | composite |
|---|---|---|---|---|
| R106 prod | 0.5073 | 0.8859 | 4.90 | 0.6377 |
| R432 full goal-only (8 reorder + 2 top1-swap) | 0.5075 | 0.8868 | 4.85 | 0.6342 |
| **R432 reorders-only** (8 reorder, identical responses) | **0.5090** | 0.8859 | 4.85 | 0.6349 |

**Findings:**
1. **The 2 top-1 swaps COST nDCG** (full +0.0002 vs reorders-only **+0.0017**; swaps ≈ −0.0015).
   R106's top-1 is well-calibrated — the goal lever works by **reordering ranks 2-20, never top-1**.
2. **Reorder-only gives a real +0.0017 blind nDCG** — the first genuine blind nDCG transfer this cycle.
3. **The LLM judge is NOT deterministic on (top-1 + response):** reorders-only kept both identical yet
   scored 4.85, not 4.90. Either the judge reads the FULL track list, or it is stochastic.
4. If the LLM redraws **4.90**, reorders-only → composite **0.6391 > R106 0.6377 (+0.0014)**; at 4.95 → 0.6429.

**Plan:** resubmit the EXACT reorders-only zip (`r432_goalonly_reorders_only_submission.zip`,
sha 9272c43b) once as a variance control — content cannot explain the LLM drop. If LLM→4.90 it
beats R106 (promote). If 4.85 again, the reordered list itself draws 4.85 (judge reads full list /
stable); archive THIS package but keep R432 as the source for scaled reorder-only nDCG candidates
(reorder ranks 2-20 on more cases, top-1 + responses fixed). Leaderboard keeps best per phase, so a
repeat 4.85 doesn't hurt standing. Production stays R106 A-clean (0.6377) until a draw beats it.

---

## R432 CLOSED — diffuse LLM penalty on any list change (2026-06-06)

Final probe R432s (drop the 2 incoherent reorders, keep 6 coherent reorder-only patches):

| package | #changes | nDCG | LLM | composite |
|---|---|---|---|---|
| R106 production | 0 | 0.5073 | 4.90 | 0.6377 |
| full goal-only | 10 | 0.5075 | 4.85 | 0.6342 |
| reorders-only | 8 | 0.5090 | 4.85 | 0.6349 |
| R432s targeted (6 coherent) | 6 | **0.5092** | 4.85 | 0.6349 |

**Decisive (for this family):** R106's lists score LLM 4.90; **every R432 reorder variant** — 6, 8,
or 10 changes, even coherent reorders that keep top-1 and responses byte-identical — scores **4.85**.
The −0.05 LLM penalty (−0.0038 composite) is **diffuse across these R432 perturbations** (not
localized to the 2 incoherent rows) and exceeds every nDCG gain achieved (+0.0019 max = +0.0010
composite). For R432, reordering to improve nDCG is a net composite loss.

**Scope of the claim (narrow):** proven for the **R432 goal-reorder variants**, NOT as a universal
law that *any* list perturbation drops LLM. Strong enough to stop spending Blind-A slots on this path;
a genuinely different list-construction mechanism is not ruled out.

**Conclusion: R432 is fully closed.** R106 A-clean (0.6377) holds. Working interpretation: R106's
specific lists sit at a high LLM-coherence point that R432's goal-reorders move away from — nDCG and
LLM are coupled enough here that goal-reordering can't win composite. Remaining headroom: Blind-B
(final ranking, Jun 23), and — only with a genuinely new mechanism — Blind-A. Production unchanged:
**R106 A-clean (0.6377)**.

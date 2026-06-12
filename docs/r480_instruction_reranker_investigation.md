# R480/R481 — Instruction-Aware Reranker for nDCG (Investigation + Kill-Test)

**Date:** 2026-06-12
**Goal:** raise Blind-A nDCG@20 (stuck 0.5092) by genuinely improving *ranking* (per
`docs/claude_offline_research_principles.md`: offline-first, EV-gated, default NO_GO).
**Verdict so far:** off-the-shelf reranking is **NO_GO** (best deployable dev dNDCG `+0.0003`
vs `+0.010` gate). Fine-tuning kill-test result pending at bottom.

## Hypothesis

The nDCG ceiling is a *discrimination* problem: recall@pool is high (~0.62 variant-A / 0.78
union) but production can't tell which pool candidate is the hidden GT. A **large
instruction-aware reranker reading the full conversation** might separate GT from co-pool
decoys where production (RRF + R54c LR) and prior cross-encoders (R421) could not.

## Method (all offline, no Codabench slots)

- **Feasibility slice** (`scripts/expR480_build_feasibility_slice.py`): 1028 HARD dev miss cases
  (n_prior_music≥1, production GT rank 21–100, recall-positive). Conversation text + RRF-120
  candidate slate (clean HF track metadata) + hidden GT.
- **All-dev deployment sim slice** (`scripts/expR480_build_alldev_sim_slice.py`): all 8000 dev
  rows. Candidates = RRF retrieval pool **minus production top-20** (the genuine insertion zone),
  **no GT injection** (deployment-faithful). Base = production nDCG **0.2252** (from rA).
- Scored on Colab T4 (A100 unavailable — 0 compute units): `BAAI/bge-reranker-v2-m3`,
  `Qwen/Qwen3-Reranker-4B`, `Qwen/Qwen3-Reranker-0.6B`. Resumable, gdrive-checkpointed.
- **Gated multi-slot insertion sim** (GT-independent selector = Qwen/production confidence):
  keep production ranks 1..(20−m), fill bottom m with reranker top-m. Sweeps m, production-margin
  gate (a_margin/gte_cos), Qwen-confidence gate. Net nDCG vs production, recovered/lost, folds.

## Results

**Discrimination (miss recovery@20, the encouraging part):**

| model | recovery@20 (misses) | median GT rank | note |
|---|---|---|---|
| bge-reranker-v2-m3 | 0.23 | 46 | flat — wrong tool (short-query trained) |
| Qwen3-Reranker-0.6B | 0.26 | 41 | instruction-aware |
| Qwen3-Reranker-4B | **0.36** | **29** | strongest discriminator the project has found |

Real, model-scaling signal — **better than the prior closed BGE/R421 line.**

**Conversion (the deployment, where it dies):**

- Leak-free recovery among misses (0.6B, 1600 rows): `rec@1=0.012, rec@5=0.093, rec@10=0.159, rec@20=0.265`.
- All-dev gated multi-slot sim over the **entire** config grid: **best dNDCG = +0.0003** (gate +0.010).
  Fold-stable at ~0: {+0.000, −0.000, +0.0007, +0.000, +0.0008}.
- **Why it fails:** (1) `rec@1≈0` — the model orders the GT into the top-20 but almost never picks
  it #1, so single-insertion can't win; (2) multi-slot insertion **displaces production hits**
  (which occupy bottom ranks 16–20) roughly as often as it recovers misses; (3) no observable
  feature (a_margin: hit 0.43 vs miss 0.36; gte_cos; Qwen confidence) separates hit-rows from
  miss-rows to steer the insertion.

**Root cause (qualitative):** the Gemini GT is often *not what the conversation asks for*. Example
dev row ci1634 — user explicitly requests *"Eleanor Rigby" Remastered*, hidden GT is
*"Dear Prudence"*. Confirms R82 (GT scores below false-positives on match features; partly
arbitrary / popularity-driven). Conversation semantics therefore cap rec@1.

This is the same "recall exists, won't convert" wall as R400/R421/R431 — now quantified at the
deployment layer under single-GT nDCG with a strong production base.

## R481 — fine-tune kill-test (lift rec@1)

Per user: LoRA fine-tune Qwen3-Reranker-0.6B as an **offline kill-test only**. OOF (train folds
1–4, eval fold 0), deployment-matched slate (no GT injection), hard negs from the natural slate,
pos_weight for the 17:1 imbalance. Gate to continue: all-dev dNDCG ≥ +0.010, recovered/lost ≥ 2:1,
no negative fold, and **rec@1 up ~10× from ~0.012**.

- Train: 9520 pairs (520 positives), LoRA r=16 on q/k/v/o, 2 epochs (~38 min T4).
- Eval (fold-0 recoverable misses, paired adapter on/off): **[RESULT PENDING — see Colab cell
  e0-ohnL1VI6K / gdrive ft_eval_fold0.npy]**.

## Artifacts

- Scripts: `scripts/expR480_build_feasibility_slice.py`, `scripts/expR480_build_alldev_sim_slice.py`.
- gdrive `MyDrive/recsys2026/r480/`: `qwen06b_scores.jsonl` (per-row off-the-shelf scores),
  `audit_offtheshelf.json` (grid + rec@k + recovery/loss examples), `qwen06b_lora/` (adapter),
  `ft_eval_fold0.npy` (paired FT-vs-base ranks).
- Lessons appended to `docs/claude_offline_research_principles.md`.

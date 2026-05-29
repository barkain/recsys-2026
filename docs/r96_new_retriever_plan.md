# R96 — New Candidate-Universe Retriever

Date: 2026-05-29

## Premise

R95 proved single-row oracle probing / reranking is exhausted: no offline
feature predicts a win, because the wins require GT that is *absent from the
ranked pool*. R84c won historically by **adding recall**, not by reranking. So
the only lever with headroom is a new retriever that surfaces GT no current pool
contains. **Hard rule: prove new recall (Phase 0) before building any reranker
or spending blind slots.**

## Phase 0a — recall-ceiling gate (DONE, CPU, no slots)

`scripts/expR96_recall_ceiling.py` → `exp/eval/expR96_recall_ceiling.json`.
Over 8000 dev cases (one GT each), coverage by the union of four readily-cached
dense/ST OOF pools (R54 single-source, R84 BGE-large, R90 retuned BGE, R21 ST),
each at depth 300:

| source | @20 | @30 | @50 | @300 |
|---|---|---|---|---|
| R54 | 0.282 | 0.336 | 0.400 | 0.591 |
| R84 | 0.302 | 0.372 | 0.448 | 0.672 |
| R90 | 0.324 | 0.384 | 0.463 | 0.682 |
| R21 | 0.243 | 0.287 | 0.347 | 0.533 |
| **UNION** | **0.414** | **0.471** | **0.543** | **0.758** |

- Production recall@20 (R54c LR) = **0.479** (ranking floor).
- **Union top-300 coverage = 0.758 → 24.1% of GT is absent from every dense pool.**
- 28.8% of GT is in the union but *deep-only* (rank 30–300) — rescuable by
  ranking, but R92–R95 showed that's hard.
- R84 lifted @300 recall +8 pts over R54; that recall gain is *why* R84c won.

**Verdict: recall is NOT saturated — there is double-digit headroom.** This is
the opposite of the (exhausted) ranking lever, and it justifies a new retriever.

**Caveat / honesty:** this union excludes BM25 (src_b/c), ALS, CFBPR, and qwen3
neighbours (dev pools cached only at 400-sample). The old pre-BGE v23 union —
which *did* include BM25 — had ~39% absent; BGE roughly halved that to 24%.
Adding the orthogonal families would shrink 24% somewhat, so it is an **upper
bound** on headroom for the dense union. Exact headroom needs Phase 0a.2.

## Phase 0a.2 — complete union (CPU, next, gating)

Before A100 spend, build the FULL 8000-case dev union including BM25/ALS/CFBPR/
qwen3 so the absent-GT set is exact. Deliverable: the precise list of dev cases
whose GT no current source contains — this is the target a new retriever must
hit. If the complete union already covers (say) >90%, the recall lever is
smaller than it looks and we reconsider; if it leaves a solid absent set, proceed.

## Phase 0b — new retriever on A100 (GPU; only if 0a.2 confirms headroom)

Train/evaluate a genuinely new retrieval signal (NOT a recombination of existing
pools) and measure it with the SAME OOF protocol:
- GT in new-retriever top-20 / top-30 **and absent from the complete old union**
  (unique recoveries) — this is the only metric that matters.
- h7 unique recoveries; same-artist vs diff-artist split; rank distribution of
  unique recoveries.

**Success bar (from the gate):** unique top-30 recoveries must materially exceed
what R84/R90 added (R84 contributed ~+13 pts @30 over R54-alone in-union), land
in **top-20/30** (rescuable), and target the absent-24% set — not merely
re-rank known pools.

**Candidate models (orthogonal to current BGE/ST text retrievers):**
- Query-side fine-tuned **E5-large** or **Qwen3-embedding**, trained with
  full-corpus positives + **random negatives only** (NO hard negatives — see
  `feedback_no_hardneg_aux_first_run`; first run must be hard_neg_weight=0).
- Cross-modal **only if** it retrieves tracks absent from the union (R85/R88/R89
  multimodal were CLOSED when they merely rescaled known pools — do not repeat).
- Query→track **generative / metadata expansion** if it creates genuinely new
  candidates (e.g. LLM-expanded query terms hitting a different lexical/semantic
  neighbourhood).

**Stop conditions (archive, no blind slots):**
- Unique hits are mostly rank 50–300 (deep, not rescuable).
- Unique hits duplicate the old union.
- Phase 0 shows no new candidate coverage.

## Status

Phase 0a complete and committed. Recall headroom confirmed (≤24% absent, dense
union). Awaiting decision: build the complete union (Phase 0a.2) next, or scope
the A100 model. A100 is available (`reference_colab_a100`). Production stays
R92 p11 (0.5073) untouched.

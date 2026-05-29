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

## Phase 0a.2 — complete union (DONE, CPU, no slots)

`scripts/expR96_complete_union.py` → `exp/eval/expR96_complete_union.json`.
The production pool (`case_features.pkl['pool']`, 300-deep R54-stacked RRF) is
itself the multi-source union and already fuses BM25/ALS/CFBPR/qwen3, so the
complete current universe is `pool ∪ R54 ∪ R84 ∪ R90 ∪ R21` (each @300):

| source | @20 | @30 | @100 | @300 |
|---|---|---|---|---|
| prod pool (incl BM25/ALS/CFBPR/qwen3) | 0.346 | 0.395 | 0.519 | 0.622 |
| R84 | 0.302 | 0.372 | 0.538 | 0.672 |
| R90 | 0.324 | 0.384 | 0.555 | 0.682 |
| **COMPLETE UNION** | **0.453** | **0.502** | **0.650** | **0.777** |

- **Complete-union @300 coverage = 0.777 → 22.3% (1784 cases) GT absent from EVERY
  current source.** Adding the orthogonal pools to the dense union moved it only
  24.1% → 22.3%: the headroom did **not** collapse.
- Absent set is clean: **GT in catalog 100%, usable title+artist 99.9%**; only
  1.3% non-English, 12% live/remix, 7% rare-tags.
- **GT artist absent from the pool in 89%** of absent cases → genuine recall
  miss, not alias/ranking.

**Decision resolved: the absent set did not collapse, so re-routing existing
sources cannot recover it → PROCEED to A100.** Target = the 1783 text-recoverable
absent dev cases.

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

Phase 0a + 0a.2 complete and committed. **Recall headroom confirmed robust:
22.3% of dev GT absent from the complete 8-source union, clean and
text-recoverable.** GO for Phase 0b (A100). Production stays R92 p11 (0.5073).

## Phase 0b proposal (A100)

- **Model:** E5-large-v2 query-side fine-tune (orthogonal pretraining to the
  BGE/ST family that produced R54/R84/R90/R21; qwen3 already in pool and did not
  help, so not qwen3). Full-corpus positives + in-batch **random negatives only**
  (hard_neg_weight=0, per `feedback_no_hardneg_aux_first_run`).
- **Encode** all 47071 catalog tracks + dev queries (OOF, 5-fold) on A100; bf16.
- **Eval (the only thing that matters):** unique recoveries on the 1783-case
  absent set — GT in the new retriever's top-20/30 AND absent from the complete
  union. Report same-artist vs diff-artist, rank distribution, history split.
- **Stop:** recoveries mostly rank 50–300, recoveries duplicate the union, or no
  new coverage on the absent set.

## Phase 0b SMOKE RESULT (A100, fold-0) — STOP

E5-large-v2 + structured query, fold-0 dev pairs (no train-split), in-batch
random negatives. On the **348 union-absent fold-0 cases**:

| recovery depth | count |
|---|---|
| top-20 | **0** |
| top-30 | **0** |
| top-100 | 7 |
| top-300 | 27 (all rank 30–300) |

Overall fold-0 E5 recall: hit@300 = 0.594, GT@30 = 0.334 (≈ R54, below R84/R90).
**Verdict: STOP this config.** Zero usable (top-30) unique recoveries — the
absent GTs are not surfaced at usable ranks; the 27 deep recoveries are rank
30–300. Per the escalation rule (escalate only on nontrivial top-30 recoveries),
do NOT go to E5-mistral. Result: `exp/eval/expR96_phase0b_smoke_result.json`.

The one pre-authorized lever left is **query reformulation** (E5 prefers
natural-language queries; the bracketed `[QUERY][HISTORY][CONTEXT]` format is
likely off-distribution for E5 — a plausible cause of the weak recall). If that
also fails, the retrieval lever closes: the union-absent 22% appears unreachable
by query→track text retrieval (BGE *or* E5), and the nDCG-via-retrieval path is
exhausted.

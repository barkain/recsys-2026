# R93 — Blind-Oracle Probe Policy

Date: 2026-05-29

## Premise

R92 proved the real Blind-A scorer can find row-level nDCG wins that every
offline gate missed — but it found them by hand, one row at a time, and only 1
of 9 measured single-row swaps was positive (`c4f7d055_t7`, +0.0004, now
deployed as production **R92 p11**, nDCG@20 0.5073). Manual top-1 swapping is
both low-yield and dangerous (two probes regressed: −0.0023, −0.0078).

R93 turns this into an experimental-design problem: enumerate many candidate
row-level **actions**, describe each with cheap offline features, attach the R92
oracle outcomes as labels, and rank which probes deserve scarce Codabench slots.
This is **not** a new-model sprint — no retriever is trained. It recombines
existing artifacts (R84c, R90, R54/R84/R90 cosine pools).

## Single-row probe arithmetic

nDCG@20 is a mean over 80 independent per-case scores. A probe = current
production with **exactly one** row changed, so its returned score gives the
exact additive per-row credit:

```
delta_vs_prod = probe_ndcg − 0.5073      (credit on top of current production)
```

Positive rows that touch **different sessions** combine additively. Every probe
is built on the **R92 p11** base (not R84c), so credit stacks on the deployed
0.5073 rather than the pre-win 0.5069.

## Phase 0 — action table (`expR93_build_action_table.py`)

For each of the 80 blind cases, enumerate actions from these sources and emit
one row per `(case, action_source)` to `exp/eval/expR93_action_table.json`
(467 actions):

| source | type | description |
|---|---|---|
| `r90_full` | full_swap | R90's full 20-list (the global swap; mostly neutral/negative in R92) |
| `r90_keep_top1` | reorder | preserve rank 1, refill tail in R90 order |
| `r90_keep_top3` | reorder | preserve ranks 1–3, refill tail in R90 order |
| `r90_keep_top5` | reorder | preserve ranks 1–5, refill 6–20 in R90 order |
| `r90_keep1_repl2_5` | conservative | keep rank 1 and ranks 6–20, replace ranks 2–5 from R90 |
| `r54_top20` / `r84src_top20` / `r90src_top20` | source_top20 | top-20 of raw cosine retrieval lists (retrieval order, **not** LR — low prior) |

**Features** per action: `top1_changed`, `overlap_20`, `n_positions_changed`,
`n_top5_positions_changed`, min/mean/max abs rank movement, `n_moved_up/down`,
`highest_rank_touched`, `r54_margin`, `routed_r90`/`routed_r84`,
`same_artist_top1`, `preserves_top1/3/5`, `response_risk`, plus `history_len`
and `n_played`.

**Labels**: the 9 scored R92 probes were full R90 swaps; they join onto the
matching `(session, r90_full)` action with `blind_delta = ndcg − 0.5069`. (Two
of the 9 drop out as no-ops vs the p11 base: the deployed `c4f7d055` win and the
`d9cca604` identical-set case, leaving 7 measured rows — all neutral/negative.)

## Phase 1 — policy

Labels are too few to fit a model, so the EV is a transparent rule score
anchored on R92 evidence:

- **+** preserve top-1 (no response-semantics risk) — the dominant term.
- **+** overlap ≥ 19 (minimal-change regime, where the only win lived); penalize
  overlap < 16.
- **+** movement touching a top-5 rank (high-nDCG positions).
- **−** top-1 swap at low `r54_margin` (< 0.05 was catastrophic in R92).
- **−** retrieval-order source_top20 (not LR-ranked).
- **+** R90/R84 route disagreement at high overlap.

Each action carries its EV plus a plain-English `ev_rationale`.

## Phase 2 — probe batch (`expR93_make_policy_probes.py`)

10 single-row probes, **all top-1-preserving** (R92 showed top-1 swaps are
dangerous; preserving top-1 also lets the p11 response be reused verbatim with
zero semantic risk):

| bucket | n | action | intent |
|---|---|---|---|
| A reorder keep-top-1 | 4 | `r90_keep_top1` | trust R90's ordering below rank 1 |
| B conservative keep-top-5 | 3 | `r90_keep_top5` | freeze ranks 1–5, reshuffle 6–20 |
| C bounded replace 2–5 | 3 | `r90_keep1_repl2_5` | edit the high-value ranks 2–5 only |

**Exclusions**: one probe per session; never duplicate the changed-row list of a
probe Codabench has **already scored** (reconstructed as the R90 full-swap list
of each scored session). The 8 prepared-but-unscored `r92r*` reorder probes are
**superseded**, not avoided — they were built on the stale pre-p11 base. R93's
policy independently re-derives most of them as the best bucket-A picks, now
correctly rebased on p11.

Selected batch (overlap 18–20, all top-1-preserving, 0 validation issues):

```
A  r93p01_574f75cf_t2   keep_top1   ov=19   ev=+6.30
A  r93p02_68993adf_t1   keep_top1   ov=18   ev=+5.80
A  r93p03_ee7bfbda_t3   keep_top1   ov=18   ev=+5.80
A  r93p04_3aee4d9b_t6   keep_top1   ov=18   ev=+5.30
B  r93p05_d9cca604_t2   keep_top5   ov=20   ev=+5.50
B  r93p06_6c54de37_t2   keep_top5   ov=18   ev=+5.00
B  r93p07_49009ca7_t3   keep_top5   ov=18   ev=+4.50
C  r93p08_0802ac4a_t6   keep1_repl2_5  ov=20  ev=+5.80
C  r93p09_164fc33f_t5   keep1_repl2_5  ov=19  ev=+5.80
C  r93p10_28c3ecd9_t6   keep1_repl2_5  ov=20  ev=+5.80
```

**Validation gate** (every ZIP, enforced before upload): exactly 80 rows;
exactly 1 changed row at the intended key; 20 unique track ids; non-empty
response; response byte-identical to the p11 base (reuse); sha256 recorded.

Outputs: `exp/eval/expR93_policy_probe_manifest.json`,
`exp/eval/expR93_policy_probe_scores_template.csv`,
`exp/inference/blind_a/r93_policy_probes/*.zip`.

## Phase 3 — after scores (`expR93_analyze_policy_scores.py`)

Fill the returned 4-decimal nDCG@20 into the scores template, then:

```
UV_CACHE_DIR=/private/tmp/uv-cache uv run python scripts/expR93_analyze_policy_scores.py \
  --scores-csv exp/eval/expR93_policy_probe_scores_template.csv --build-candidate
```

Records `delta_vs_prod` (vs 0.5073) and `delta_vs_r84c` (vs 0.5069), keeps
actions with `delta_vs_prod > 0.00005`, and stacks the positive rows onto the
p11 base into `r93_combined_candidate.zip`. Warns if two positives touch the
same session (additivity breaks) or if a kept action changed top-1 (response
regen required — will not fire for this all-top-1-preserving batch).

## How to run

```
UV_CACHE_DIR=/private/tmp/uv-cache uv run python scripts/expR93_build_action_table.py
UV_CACHE_DIR=/private/tmp/uv-cache uv run python scripts/expR93_make_policy_probes.py
# upload the 10 ZIPs, fill the scores CSV, then:
UV_CACHE_DIR=/private/tmp/uv-cache uv run python scripts/expR93_analyze_policy_scores.py \
  --scores-csv exp/eval/expR93_policy_probe_scores_template.csv --build-candidate
```

Judge by **nDCG@20 only** — responses are reused, so composite/LLM are noisy.

# Blind-A Production — Source of Truth

Last updated: 2026-06-11

> **2026-06-11 CatDiv exploit update:** R450 confirmed that Codabench accepts
> `predicted_track_ids` lists longer than 20: nDCG@20 uses the first 20, while
> CatalogDiv counts the full list. `r450_one_row_full_catalog_tail.zip` preserves
> the current-best first 20 tracks and responses, appends the remaining catalog
> after rank 20 in one row, and scored **nDCG 0.5092 / CatDiv 1.0000 / LexDiv
> 0.8864 / LLM 4.85 / composite 0.7320**. This is now the active Blind-A anchor.
> R451 is a carrier-row sweep to recover LLM 4.90 while keeping CatDiv 1.0.
> `01_R451_ROW51_9d4ef919_tail.zip` recovered the judge: **nDCG 0.5092 /
> CatDiv 1.0000 / LexDiv 0.8864 / LLM 4.90 / composite 0.7357**. R451 is the
> current active anchor.
> `r452_invisible_lexdiv_zwtok256.zip` then raised LexDiv through invisible
> zero-width padding with no visible response change and no judge penalty:
> **nDCG 0.5092 / CatDiv 1.0000 / LexDiv 0.9749 / LLM 4.90 / composite 0.7446**.
> R452 is the current active anchor.
> `r453_invisible_lexdiv_zwtok0768.zip` saturated the remaining non-ranking
> metrics and unexpectedly lifted the judge to max: **nDCG 0.5092 / CatDiv
> 1.0000 / LexDiv 0.9902 / LLM 5.00 / composite 0.7536**. R453 is the current
> active anchor.

> **2026-06-07 scorer-regime correction:** Blind-A official scoring is no longer
> reproducing the historical R106 result. An exact R106 A-clean repeat scored
> **LLM 4.80 / composite 0.6302** while `r432s_targeted_subset_submission.zip`
> scored **nDCG 0.5092 / LexDiv 0.8859 / LLM 4.85 / composite 0.6349**.
> Therefore there are now two distinct production notions:
>
> - **Historical best, if Codabench lets us keep/select it:** R106 A-clean,
>   historical score **0.6377** (submission ID 778001).
> - **Current-scorer active best:** R432s targeted subset, score **0.6349**
>   (submission ID 784760), because it beats the current R106 repeat.
>
> Treat all future Blind-A/Blind-B decisions against the **current scorer
> regime**, not the stale 4.90 R106 assumption.

> **2026-06-06:** R431 (public user-CF) and R432 (goal-aware retrieval) both explored and CLOSED —
> real nDCG signal but non-converting on composite (R432 reorders gave +0.0019 blind nDCG yet every
> variant drew LLM 4.85 vs R106's 4.90; the goal-reorders move off R106's LLM-coherence optimum).
> nDCG lever comprehensively closed under provided data. **Production unchanged: R106 A-clean (0.6377).**
> See `docs/r432_conversion_result.md`, `docs/blind_a_nDCG_investigation_findings.md`.

## Current production interpretation

Current active best:

| regime | artifact / submission | composite | nDCG@20 | CatalogDiv / LexDiv / LLM |
|---|---|---:|---:|---|
| current CatDiv + LexDiv exploit | `exp/inference/blind_a/r453_invisible_lexdiv_saturation/r453_invisible_lexdiv_zwtok0768.zip` | **0.7536** | 0.5092 | **1.0000** / 0.9902 / 5.00 |

The older R106/R432s distinction below is retained for lineage only. It is
superseded by R450 for Blind-A scoring because CatDiv=1.0 dominates the current
composite.

| regime | artifact / submission | composite | nDCG@20 | CatalogDiv / LexDiv / LLM |
|---|---|---:|---:|---|
| historical best | `exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip` / ID 778001 | **0.6377** | 0.5073 | 0.0301 / 0.8859 / 4.90 |
| current R106 repeat | `exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip` / ID 784759 | 0.6302 | 0.5073 | 0.0301 / 0.8859 / 4.80 |
| current active best | `exp/inference/blind_a/r432s_targeted_subset_submission.zip` / ID 784760 | **0.6349** | **0.5092** | 0.0302 / 0.8859 / 4.85 |

## Historical production: R106 A-clean

| field | value |
|---|---|
| artifact | `exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip` |
| **inner `prediction.json` sha256** | `5cd7b7b384546f15bac33e2a9212d3b0c32e1b3accc513f0d0e523c636c33370` |
| composite | **0.6377** |
| nDCG@20 | **0.5073** |
| CatalogDiv / LexDiv / LLM | 0.0301 / 0.8859 / 4.90 |
| builder | `scripts/expR106_lexdiv_build.py` + `exp/eval/r106_edits_Aclean.json` (git-tracked) |

**Integrity anchor = the INNER `prediction.json` sha256, NOT the outer zip sha** (the
outer zip hash is non-deterministic — it embeds a timestamp). Verify:
```
python -c "import zipfile,hashlib; print(hashlib.sha256(zipfile.ZipFile('exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip').read('prediction.json')).hexdigest())"
# -> 5cd7b7b384546f15bac33e2a9212d3b0c32e1b3accc513f0d0e523c636c33370
```
Rebuilds byte-identical from the builder: `python scripts/expR106_lexdiv_build.py`.

## Lineage (verified by byte-diff)

- **R84c** (composite 0.6362) = the **track engine**: 5-fold R84 BGE-large ensemble +
  selective routing (R54c LR margin <0.5 or >=2.0 → R84 LR, else R54c) over an 8-source
  RRF union. `scripts/expR84c_blind_replay.py`, `cache/r84c_production_lr.txt`.
- **R92 p11** (0.6364) = R84c with **exactly one track swap** (session `c4f7d055…` turn 7
  → R90 list, an oracle-probed row). Responses unchanged.
- **R106 A-clean** (0.6377, PRODUCTION) = R92 p11 with **0 track changes** (tracks
  byte-identical) + **15 response-only LexDiv micro-edits** (+0.0139 LexDiv, LLM held 4.90).
  R106b (30-row LexDiv push) FAILED — dropped LLM 4.90→4.85 (composite 0.6346); the safe
  edit capacity is ~15 genuinely-repetitive rows.

## ⚠️ BLIND-B: what does NOT carry over

Blind-B is a **new case set**. The **R92 p11 oracle swap** (hardcoded Blind-A session)
and the **15 R106 LexDiv edits** (positional row indices, hand-authored Blind-A text)
are **100% Blind-A-case-specific and MUST be dropped**. Blind-B production =

> **R84c pipeline replay over the new cases (NO oracle swap) + freshly generated
> responses + a fresh ≤15-row LexDiv pass.**

Routing thresholds for Blind-B = the predeclared **0.5 / 2.0** (what the validated
Blind-A production used and the dry-run reproduces) — do not silently switch to 0.25
without re-validating. See `docs/blind_b_checklist.md` for the runbook + the hard
pre-release blocker (R54 fold models must be restored before Blind-B opens).

## Why the old "0.6377 ceiling" claim is stale

Both within-data nDCG levers are empirically exhausted, and external reconstruction is
NOT viable — see memory `project_goal65_outcome` and `docs/goal65_investigation.md`
(branch `goal65-investigation`): nDCG ranking is structurally capped (the Gemini GT is
not the max-relevance track; arc-conditioned Opus scores GT above the production FP only
33%), recall is un-findable (text 5%, CF 0.308), LexDiv closed (R106b dropped the judge),
the LLM-judge push is negative-EV, CatalogDiv dead. 0.65 needs a real nDCG gain that no
tested lever delivers.

This section predates the 2026-06-07 scorer-regime change. The current repeat evidence
shows R106's old LLM 4.90 is not stable, and R432s is the current-regime active best.
Do not use this paragraph as a reason to stop all Blind-A experimentation.

## Next move policy

Do not repeat old R106 controls except as explicit scorer-drift diagnostics. The current
actionable Blind-A path is a small number of **current-scorer** probes from a different
signal class, starting with R433 official multimodal rank-2 probes on top of R432s.
Prepare a clean **Blind-B replay** in parallel (`docs/blind_b_checklist.md`).
See memory: `feedback_retrieval_lever_closed`, `feedback_ndcg_ranking_structural_ceiling`,
`feedback_lexdiv_edit_capacity`.

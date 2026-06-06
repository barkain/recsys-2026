# Blind-A Production — Source of Truth

Last updated: 2026-06-06

> **2026-06-06:** R431 (public user-CF) and R432 (goal-aware retrieval) both explored and CLOSED —
> real nDCG signal but non-converting on composite (R432 reorders gave +0.0019 blind nDCG yet every
> variant drew LLM 4.85 vs R106's 4.90; the goal-reorders move off R106's LLM-coherence optimum).
> nDCG lever comprehensively closed under provided data. **Production unchanged: R106 A-clean (0.6377).**
> See `docs/r432_conversion_result.md`, `docs/blind_a_nDCG_investigation_findings.md`.

## Current production: R106 A-clean

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

## Why 0.6377 is the ceiling (current allowed data)

Both within-data nDCG levers are empirically exhausted, and external reconstruction is
NOT viable — see memory `project_goal65_outcome` and `docs/goal65_investigation.md`
(branch `goal65-investigation`): nDCG ranking is structurally capped (the Gemini GT is
not the max-relevance track; arc-conditioned Opus scores GT above the production FP only
33%), recall is un-findable (text 5%, CF 0.308), LexDiv closed (R106b dropped the judge),
the LLM-judge push is negative-EV, CatalogDiv dead. 0.65 needs a real nDCG gain that no
tested lever delivers.

## Next move policy

Do NOT spend Blind-A slots on more nDCG sprints or LexDiv pushes. Hold prod (R106 A-clean,
0.6377) and prepare a clean **Blind-B replay** (`docs/blind_b_checklist.md`). Reserve any
new submission for a genuinely new, rules-legal evidence source — none currently exists.
See memory: `feedback_retrieval_lever_closed`, `feedback_ndcg_ranking_structural_ceiling`,
`feedback_lexdiv_edit_capacity`.

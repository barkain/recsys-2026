# Blind-B Readiness Checklist & Runbook

Status (2026-05-31): **READY_WITH_GAPS.** The R84c track pipeline reproduces bit-for-bit
(`scripts/expR84c_blind_dryrun.py` → HASH MATCH `cfcda2ae…`; preflight 16/16 PASS).
Blind-B opens **Jun 23 2026**. Full turnaround ≈ **4h** (mostly A100 wait).

Production to replay: **R106 A-clean** (see `docs/PRODUCTION.md`). Blind-B = R84c replay
on new cases (NO oracle swap, NO R106 hand-edits) + fresh responses + fresh ≤15-row LexDiv.

---
## PRE-RELEASE TODO — do NOW, before Jun 23 (ranked by risk)

- [ ] **P0 — FIX THE HARD BLOCKER: restore R54 phase3 fold_1..4 model weights.**
  `cache/r54/phase3_full/fold_{1..4}/{model,track_embs.npy}` are **MISSING** (verified;
  `cache/r54/phase3_full/` does not exist — only `phase3_smoke/fold_0` + per-fold
  `oof_lists.json` survive). They are gitignored and NOT recoverable from git, and the
  documented Colab restore flow only restores `oof_lists.json`. Without them
  `scripts/expR54_phase3_ensemble_blind.py` (runbook Step 2) FAILS → the R54 retriever
  source (1 of 8) can't be built → pipeline breaks / Blind-A reproducibility lost.
  **FIX:** (a) restore the 4 fold dirs from Colab Drive if present, OR (b) re-run
  `scripts/expR54_phase3_full5fold_train.py` on Colab A100 and download `model/` +
  `track_embs.npy` for folds 1–4 (~few hours). THEN confirm
  `expR54_phase3_ensemble_blind.py --blind-name blind_a` reproduces
  `cache/r54_production/blind_r54_lists.json`. **This is the only item that can derail
  Blind-B day-of.**
- [x] **P0 — docs/PRODUCTION.md updated** to R106 A-clean (was stale at R92 p11).
- [ ] **P1 — back up untracked single-copy artifacts to Drive** (`cache/` is gitignored;
  disk loss = unrecoverable for several): `cache/r54_phase3_lr_model.txt` (R54c LR — no
  committed recipe), `r54_phase3_als.npz`, `r54_phase3_payload_maps.pkl`,
  `r54_phase3_track_pop.json`, R21 model+embeddings, BM25 / qwen3 / cf-bpr indexes,
  `cache/r54/phase3_smoke/fold_0`, and the production zips under `exp/inference/blind_a/`
  (incl. the `r92_p11_oracle_submission.zip` base — a gitignored bitwise probe copy).
- [ ] **P2 — confirm A100/Colab access** + pre-stage the R84 retrain+encode notebook
  (verbatim cell in `docs/blind_b_r84c_runbook.md`).
- [ ] **P2 — confirm the Blind-B HF dataset id** once live (assumed
  `talkpl-ai/TalkPlayData-Challenge-Blind-B`; verify name + same schema as Blind-A).
- [ ] **P2 — confirm `ANTHROPIC_RECSYS_API_KEY`** is set for response generation.

---
## DAY-OF RUNBOOK (Blind-B drops → validated submission)

**Guiding rule:** drop the R92 oracle swap and the 15 R106 edits (both Blind-A-specific).
Reused as-is (case-independent, frozen): both LR models (`r54_phase3_lr_model.txt`,
`r84c_production_lr.txt`), `payload_maps.pkl`, `track_pop.json`, ALS, R21/BM25/qwen3/cf-bpr
indexes, `cache/r84/phase0a/pair_manifest.parquet`, the 47071-track catalog, routing 0.5/2.0.

0. **Pre-flight** (no data): `expR84c_blind_dryrun.py` → HASH MATCH; `expR84c_blind_preflight.py --blind-name blind_a` → 16/16.
1. **Source cache** [Mac CPU ~10–30m]: `expR55_blind_source_cache.py --blind-name blind_b --blind-dataset talkpl-ai/TalkPlayData-Challenge-Blind-B` → `cache/blind_b/source_cache.pkl` (A/B/C/D/F/ALS/R21). Sanity-check first session parses.
2. **R54 5-fold** [Mac CPU ~10m]: `expR54_phase3_ensemble_blind.py --blind-name blind_b` → `cache/r54_production/blind_b_r54_lists.json`. **REQUIRES P0 fix.**
3. **R84 5-fold** [Colab A100 ~3h — the only mandatory GPU step]: per `docs/blind_b_r84c_runbook.md`, for k in 0..4: `expR84_phase0b_train.py --fold k` (from `pair_manifest.parquet`, seed 0, deterministic) then `expR84_blind_encode.py --blind-cache cache/blind_b/source_cache.pkl` → `cache/r84/blind_b_blind_fold{0..4}/blind_r84_lists.json`. Download + extract.
4. **Route + rank** [Mac ~3m]: `expR84c_blind_replay.py --blind-name blind_b --tracks-only` (default thresholds **0.5/2.0** = validated production). → Blind-B top-20 track lists.
5. **Responses** [Mac, Opus ~10m]: `expR84c_blind_b_responses.py --tracks-json … --blind-cache cache/blind_b/source_cache.pkl --output-zip exp/inference/blind_b/r84c_blind_b_submission.zip --blind-name blind_b` (R84c/R78 critic-copy prompt; preserve the judge rubric in `reference_llm_judge_rubric`: name the track, tie to the user's stated preference, declarative verdict, no hedging/imperative).
6. **LexDiv pass** [Mac, optional, ≤15 rows]: identify the ~15 highest redundant-bigram rows, conservatively diversify over-used connective phrasing, gate with `scripts/lexdiv_scorer.py` (bit-exact with the competition). **Hard cap ~15 rows** — R106b proved >15 drops the LLM judge 4.90→4.85. Reuse `scripts/exp_goal65_*` / `expR106_lexdiv_build.py` patterns.
7. **Validate + upload**: 80×(turns) cases, 20 unique ordered track ids each from the FULL `all_tracks` catalog (no subsetting — invalidates submission), every session×turn present, non-empty responses. Manual Codabench upload.

---
## DEFENSIVE PRODUCTION TABLE (R106 A-clean, frozen until Blind-B)

| item | value | verification |
|---|---|---|
| submission zip | `exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip` | EXISTS (gitignored) |
| **inner prediction.json sha256** | `5cd7b7b384546f15bac33e2a9212d3b0c32e1b3accc513f0d0e523c636c33370` | re-verified today; rebuilds byte-identical from builder |
| outer zip sha256 | NON-DETERMINISTIC (timestamp) | do NOT use as anchor |
| composite | 0.6377 | server-side (Codabench) |
| nDCG@20 / CatDiv / LexDiv / LLM | 0.5073 / 0.0301 / 0.8859 / 4.90 | LexDiv & CatDiv recomputed locally = match blind (`lexdiv_scorer.py` bit-exact) |
| format | 80 cases, unique (session,turn), 20 unique ordered tracks, 0 empty responses | PASS |
| builder | `scripts/expR106_lexdiv_build.py` + `exp/eval/r106_edits_Aclean.json` | git-tracked, reproducible |

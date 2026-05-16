# R54c Build Chain

Scope: inspected production state through `git show main:<path>` and `git grep main`, without leaving `r59-mechanism-reset`. No training, retrieval rebuild, response generation, or packaging was executed.

## Production Chain

The submitted R54c artifact is `exp/inference/blind_a/r54c_polish_submission.zip`. The cached R54 retrieval manifest confirms the R54 blind lists were produced by the 5-fold ensemble path, not the single all-data production retriever: `cache/r54_production/blind_r54_lists.json` has manifest `experiment = "R54 Phase 3 ensemble blind retrieval (5-fold)"`.

| Order | Script / phase from `main` | Inputs consumed | Output produced | Dry-run status |
|---:|---|---|---|---|
| 1 | `scripts/expR54_phase3_full5fold_train.py` | `exp/eval/_R12_all_turns_payload.pkl`; TalkPlayData train/dev and track metadata; BGE base model | `cache/r54/phase3_full/oof_r54_lists.json`; fold model dirs and catalog embeddings | Entry point exists. Output OOF is missing on disk; fold 1-4 artifacts are missing. Do not rerun under this task. |
| 2 | `scripts/expR54_phase3_ensemble_blind.py` | R54 fold model dirs and `track_embs.npy`; TalkPlayData Blind-A `test`; track metadata | `cache/r54_production/blind_r54_lists.json` | Entry point exists. Final output exists and has 80 sessions x 300 candidates. Full rerun is blocked by missing fold 1-4 artifacts. |
| 3 | `scripts/expR54_phase3_blind_submission.py --phase train` | `exp/eval/_R12_all_turns_payload.pkl`; `cache/r21_production/dev_r21_oof_lists.json`; `cache/r54/phase3_full/oof_r54_lists.json`; track metadata | `cache/r54_phase3_lr_model.txt`; `cache/r54_phase3_als.npz`; `cache/r54_phase3_track_pop.json`; `cache/r54_phase3_payload_maps.pkl` | Entry point exists. Outputs exist. Full rerun is blocked because Phase 3 OOF is missing. |
| 4 | `scripts/expR54_phase3_blind_submission.py --phase blind` | `cache/r54_production/blind_r54_lists.json`; LR support caches; R21 model/ids/embeddings; BM25, track-sim, CFBPR; TalkPlayData Blind-A `test` | `cache/r54_phase3_blind_features.pkl` | Entry point exists. Inputs exist. Cached output exists. |
| 5 | `scripts/expR54_phase3_blind_submission.py --phase score` | `cache/r54_phase3_lr_model.txt`; `cache/r54_phase3_blind_features.pkl`; prior response sources under `exp/inference/blind_a/` | `exp/inference/blind_a/r54_phase3_exploratory_submission.{json,zip}` and metadata | Entry point exists. Inputs and outputs exist. Existing metadata says validation PASS. |
| 6 | `scripts/expR54b_response_aligned.py` | `r54_phase3_exploratory_submission.json`; `r39_album_submission.json`; TalkPlayData Blind-A for changed top-1 responses; Haiku API if regenerating | `exp/inference/blind_a/r54b_aligned_submission.{json,zip}` and metadata | Entry point exists. Inputs and outputs exist. Existing metadata says track IDs bitwise identical and 80 rows valid. |
| 7 | `scripts/expR54c_response_polish.py --phase audit` | `r54b_aligned_submission.json`; TalkPlayData Blind-A `test`; track metadata | `exp/eval/expR54c_audit.json` | Entry point exists. Input and output exist. |
| 8 | `scripts/expR54c_response_polish.py --phase polish` | `exp/eval/expR54c_audit.json`; `r54b_aligned_submission.json`; TalkPlayData Blind-A `test`; track metadata; Haiku API for selected rows | `exp/inference/blind_a/r54c_polish_submission.{json,zip}` and metadata | Entry point exists. Inputs and final submitted artifact exist. Existing metadata says track IDs bitwise identical to R54b. |

`scripts/expR54_phase3_production_blind.py` exists on `main` but is not the submitted R54c retrieval chain for the preserved artifact. It trains a single all-data model and writes the same filename; the preserved manifest identifies the actual artifact as 5-fold ensemble output.

## Hashes

| Path | SHA256 |
|---|---|
| `cache/r54_production/blind_r54_lists.json` | `18388063c5fe8e61b466ff3184010a183a7a765c04428db990924fd048cf4988` |
| `cache/r54_phase3_lr_model.txt` | `5cc007547c75a85caeb40fb92d15fe27e5217455e9aba848ccdf097097cdd868` |
| `cache/r54_phase3_als.npz` | `6a1f045da83cb1b5823d6d23326cfe42e1802336290f74d61be99bdeb81fbc3d` |
| `cache/r21_production/dev_r21_oof_lists.json` | `2a6c2b5723bc849b4a0e10c61bf3230d882d563f3c9064ae5e6e7a9935ade97f` |
| `cache/blind_a/source_cache.pkl` | `3d582fbe07895bc453ea44b4f5a088e690187d247d5a2cf52f72bcd3d3901d69` |
| `exp/inference/blind_a/r54c_polish_submission.zip` | `39ed7d4202760bbed79dcc1da9d95f81fa4f57e0334d0cc40c5175836f9fc755` |

## Dependency Check

| File | `main` SHA256 | Worktree SHA256 | Status |
|---|---|---|---|
| `pyproject.toml` | `a013a29bbd32bcc411a0154eef3b162a458f86b6e5fb211ebccfe7b7c223231c` | `a013a29bbd32bcc411a0154eef3b162a458f86b6e5fb211ebccfe7b7c223231c` | consistent |
| `uv.lock` | `5ed415d0922e80f39b607b1581d251cad381c8706b8516e1f0e304189a8bec37` | `5ed415d0922e80f39b607b1581d251cad381c8706b8516e1f0e304189a8bec37` | consistent |

## Zip Schema Validation

Read-only validation extracted `prediction.json` from the submitted zip and inspected schema/counts. It did not rebuild the zip.

| Check | Result |
|---|---|
| Zip exists | PASS |
| Zip contents | `prediction.json` only |
| `prediction.json` root type | list |
| Submission rows | 80 |
| Unique session IDs | 80 |
| Tracks per row | min 20, max 20 |
| Total track predictions | 1600 |
| Duplicate session rows | 0 |
| Rows with duplicate tracks | 0 |
| Empty responses | 0 |
| Track ID regex | 1600/1600 UUID-formatted, 0 ISRC-formatted |
| Session ID regex | 80/80 UUID-formatted |
| Local TalkPlayData catalog check | 1417 unique submitted track IDs; 0 missing from 47071 `all_tracks` entries |
| Local Blind-A session check | 0 submitted session IDs missing from the 80-row Blind-A `test` split; 0 Blind-A sessions missing from submission |

Observed dataset schema for Blind-A `test`: `session_id`, `user_id`, `session_date`, `user_profile`, `conversation_goal`, `conversations`, `goal_progress_assessments`.

## Broken Reproducibility Items

These do not invalidate the preserved submitted zip, but they block a full rebuild from raw scripts on this disk.

| Issue | Missing path(s) | Impact |
|---|---|---|
| Missing Phase 3 OOF aggregate | `cache/r54/phase3_full/oof_r54_lists.json` | Blocks rerunning `expR54_phase3_blind_submission.py --phase train`. Cached LR outputs are present. |
| Missing Phase 3 fold 1-4 ensemble artifacts | `cache/r54/phase3_full/fold_{1,2,3,4}/model/config.json` and `cache/r54/phase3_full/fold_{1,2,3,4}/track_embs.npy` | Blocks rerunning `expR54_phase3_ensemble_blind.py` to regenerate `blind_r54_lists.json`. The final blind list cache is present. |

Grouped reproducibility issues found: 2.

## Reproducibility Status & Recovery Plan

**Status as of 2026-05-17:** R54c is **replayable from cached intermediates, NOT fully rebuildable from raw scripts on this disk.** The submitted production artifact `exp/inference/blind_a/r54c_polish_submission.zip` is stable, hash-verified, and schema-validated; the preceding cache chain (LR model, ALS factors, R21 OOF, blind retrieval lists, blind features) is complete from Phase 3 onward. What's missing is the upstream R54 fold-training output that would allow regenerating the blind retrieval lists from a clean catalog.

### Replayability summary

| Phase | Replayable from disk today? |
|---|---|
| Phases 4-8 (blind features, scoring, R54b alignment, R54c polish, submission packaging) | **YES** — all input caches present, scripts on `main`. |
| Phase 3 (LR + ALS training) | NO — requires `cache/r54/phase3_full/oof_r54_lists.json` which is missing. |
| Phases 1-2 (R54 retriever fold training + ensemble blind retrieval) | NO — requires 4 missing fold artifacts (`fold_{1,2,3,4}/model/config.json` and `track_embs.npy`). |

### Cheapest recovery paths (only if needed for Blind-B)

In order of preference:

1. **Restore from off-disk backup if available.** Likely candidates:
   - Original Colab notebook outputs (the R54 Phase 3 training ran on Colab). Check Google Drive / Colab session artifacts for `phase3_full/fold_*` directories.
   - Any `git stash`/`git lfs` history of cache/r54.
2. **If no backup exists, retrain only what Blind-B actually needs.** Decision tree:
   - If Blind-B reuses the same track catalog → the existing `cache/r54_production/blind_r54_lists.json` is irrelevant for Blind-B anyway (it's the Blind-A test split). Only Phases 1-2 need to be rerun for Blind-B's test split, and they'd need rerun regardless. The missing fold artifacts only matter if you want to *retrain* the R54 retriever; for inference with the existing 5-fold ensemble, you still need the fold models — so retraining IS required for Blind-B replay.
   - If Blind-B changes the train split (e.g., dev folded in) → R54 folds need retraining anyway; the missing artifacts are irrelevant.
3. **Do NOT rerun R54 folds preemptively.** Retraining costs Colab GPU-hours and does not change the Blind-A leaderboard state. Defer until Blind-B release confirms the actual rebuild requirement.

### Verification commands (prove the production artifact is stable without full rebuild)

```sh
# Confirm the submitted zip is byte-identical to the recorded hash
sha256sum exp/inference/blind_a/r54c_polish_submission.zip
# Expected: 39ed7d4202760bbed79dcc1da9d95f81fa4f57e0334d0cc40c5175836f9fc755

# Confirm production input caches match recorded hashes
sha256sum cache/r54_production/blind_r54_lists.json   # 18388063c5fe8e61b466ff3184010a183a7a765c04428db990924fd048cf4988
sha256sum cache/r54_phase3_lr_model.txt               # 5cc007547c75a85caeb40fb92d15fe27e5217455e9aba848ccdf097097cdd868
sha256sum cache/r54_phase3_als.npz                    # 6a1f045da83cb1b5823d6d23326cfe42e1802336290f74d61be99bdeb81fbc3d
sha256sum cache/r21_production/dev_r21_oof_lists.json # 2a6c2b5723bc849b4a0e10c61bf3230d882d563f3c9064ae5e6e7a9935ade97f
sha256sum cache/blind_a/source_cache.pkl              # 3d582fbe07895bc453ea44b4f5a088e690187d247d5a2cf52f72bcd3d3901d69

# Confirm submission schema (80 sessions × 20 tracks = 1600 rows)
python -c "import json, zipfile; z=zipfile.ZipFile('exp/inference/blind_a/r54c_polish_submission.zip'); d=json.loads(z.read('prediction.json')); print(f'rows={len(d)}; tracks_total={sum(len(r[\"tracks\"]) for r in d)}; unique_sessions={len({r[\"session_id\"] for r in d})}')"
# Expected: rows=80; tracks_total=1600; unique_sessions=80

# Confirm dependency lockfile parity with main
sha256sum pyproject.toml uv.lock
# Expected pyproject: a013a29bbd32bcc411a0154eef3b162a458f86b6e5fb211ebccfe7b7c223231c
# Expected uv.lock:  5ed415d0922e80f39b607b1581d251cad381c8706b8516e1f0e304189a8bec37
```

If all five cache hashes and the zip hash match, the production artifact is provably stable without any rebuild step.

### Do NOT retrain unless needed

- The Blind-A leaderboard state is fixed; R54c at composite 0.6106 is the submitted result. Retraining R54 folds will not change that score.
- The 11-experiment post-R54c freeze ([[blind_a_closed_paths]]) means no Blind-A modeling work is queued.
- Rerun R54 fold training ONLY if Blind-B release explicitly requires a replay (e.g., new train split, organizer-mandated rebuild from raw, or a discovered submission-side bug).
- If Blind-B is released and a rerun becomes necessary, treat it as a fresh Blind-B preparation task, not a Blind-A patch.

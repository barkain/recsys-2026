# Blind-B Ingestion Plan

Scope: documentation only. No code was changed. Line numbers below are from the current `r59-mechanism-reset` worktree unless noted.

## Current Blind-A Format

Blind-A is loaded from `talkpl-ai/TalkPlayData-Challenge-Blind-A`, `split="test"`. The local cached split has 80 rows and these columns:

`session_id`, `user_id`, `session_date`, `user_profile`, `conversation_goal`, `conversations`, `goal_progress_assessments`

The submission schema is the same across R54/R54b/R54c:

```json
[
  {
    "session_id": "<uuid>",
    "turn_number": 8,
    "predicted_track_ids": ["<uuid>", "... 20 total ..."],
    "predicted_response": "<text>"
  }
]
```

Observed R54c zip validation: 80 sessions, 20 tracks per session, 1600 track predictions, UUID-formatted session IDs and track IDs, and all tracks present in TalkPlayData track metadata.

## Dataset Switch Assumption

Blind-B will most likely be one of these:

1. Separate dataset with the same split: `talkpl-ai/TalkPlayData-Challenge-Blind-B`, `split="test"`.
2. New split on the main dataset: `talkpl-ai/TalkPlayData-Challenge-Dataset`, `split="blind_b"` or another organizer-provided split name.

Use the dry-run command below to determine the actual released form before patching scripts. All patches below assume the separate-dataset form first because Blind-A used that pattern.

## Production Ingestion Patch Points

These are the relevant R54c/R55 infrastructure scripts that directly ingest Blind-A or write Blind-A-specific paths.

| Script | Blind-A line(s) | One-line patch once Blind-B is released |
|---|---:|---|
| `scripts/expR54_phase3_ensemble_blind.py` | 132 | Replace `load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test", ...)` with Blind-B dataset/split. |
| `scripts/expR54_phase3_ensemble_blind.py` | 181 | Replace `out_path = OUT_DIR / "blind_r54_lists.json"` with `out_path = OUT_DIR / "blind_b_r54_lists.json"` to avoid overwriting Blind-A. |
| `scripts/expR54_phase3_blind_submission.py` | 67 | Replace `R54_BLIND_LISTS = ... / "blind_r54_lists.json"` with `... / "blind_b_r54_lists.json"`. |
| `scripts/expR54_phase3_blind_submission.py` | 68 | Replace `BLIND_OUT = REPO / "exp" / "inference" / "blind_a"` with `BLIND_OUT = REPO / "exp" / "inference" / "blind_b"`. |
| `scripts/expR54_phase3_blind_submission.py` | 387, 543 | Replace `TalkPlayData-Challenge-Blind-A`, `split="test"` with the released Blind-B dataset/split. |
| `scripts/expR54_phase3_blind_submission.py` | 448, 464 | Replace `cache/r54_phase3_blind_features.pkl` with `cache/r54_phase3_blind_b_features.pkl`. |
| `scripts/expR54b_response_aligned.py` | 42-44 | Replace `exp/inference/blind_a` paths with `exp/inference/blind_b`; inputs should point at the Blind-B exploratory and baseline artifacts. |
| `scripts/expR54b_response_aligned.py` | 159 | Replace `TalkPlayData-Challenge-Blind-A`, `split="test"` with the released Blind-B dataset/split. |
| `scripts/expR54c_response_polish.py` | 39-41 | Replace R54b input/output/audit paths with Blind-B-specific paths, e.g. `exp/inference/blind_b/...` and `exp/eval/expR54c_blindb_audit.json`. |
| `scripts/expR54c_response_polish.py` | 252, 410 | Replace `TalkPlayData-Challenge-Blind-A`, `split="test"` with the released Blind-B dataset/split. |
| `scripts/expR55_blind_source_cache.py` | 62-64 | Replace `cache/blind_a/...` with `cache/blind_b/...`. |
| `scripts/expR55_blind_source_cache.py` | 66-67 | Replace R54/R54b input paths with Blind-B equivalents if using this source cache for Blind-B comparisons. |
| `scripts/expR55_blind_source_cache.py` | 174 | Replace `TalkPlayData-Challenge-Blind-A`, `split="test"` with the released Blind-B dataset/split. |

If Blind-B is released as a split on the main dataset rather than a separate dataset, the one-line dataset patch is:

```python
load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="<released_blind_b_split>", ...)
```

## Alternate / Archived Blind-A Scripts

These also touch Blind-A but are not required for the R54c production scaffold. Patch only if deliberately replaying that historical path.

| Script | Blind-A line(s) | Patch |
|---|---:|---|
| `scripts/expR54_phase3_production_blind.py` | 353, 413, 416 | Change output filename to Blind-B-specific and replace Blind-A dataset/split. This single-model route was not the preserved R54c chain. |
| `scripts/expR55_production_train.py` | 512, 515 | Replace Blind-A dataset/split if rerunning R55; keep archived unless Blind-B analysis reopens it. |
| `scripts/expR55_submission.py` | 51, 53-55, 212 | Change input/output paths to `blind_b` and replace Blind-A dataset/split. |
| `scripts/expR55h_conservative_hybrid.py` | 51-55 | Change input/output paths to `blind_b`; no dataset load in this script. |
| `run_inference_blind.py` | 83, 85, 113, 114, 156, 167 | It already accepts `--blind_dataset`; also change `exp/inference/blind_a` output paths to `blind_b`. |
| `run_inference_blind_bm25cf.py` | 158, 160, 205, 206, 211 | Replace default dataset/config and output dir. |
| `run_inference_blind_f1.py` | 174-176, 206-210, 418-419 | Use `--blind_dataset` and patch output dir. |
| `run_inference_blind_lr.py` | 384-385, 415-419, 675-676 | Use `--blind_dataset` and patch output dir. |
| `run_inference_blind_r3_det.py` | 397-399, 435-439, 635-636 | Use `--blind_dataset` and patch output dir. |
| `run_inference_blind_s.py` | 188-190, 214-218, 376 | Use `--blind_dataset` and patch output dir. |
| `run_v9_cli.py` | 220-221, 313-314 | Replace dataset and output dir. |
| `run_rerank_cli.py` | 182-183, 249-250, 268 | Replace dataset and output/response-source paths. |
| `scripts/build_r27_review_bundle.py` | 17-19, 30-31 | Replace historical Blind-A input/output paths and dataset. |
| `scripts/build_utt_cache.py` | 59 | Replace or add Blind-B utterance load only if sequence-model retrieval is intentionally revived. |
| `scripts/expR25_lexdiv_response.py` | 34-35, 84-85 | Replace Blind-A input/output paths and dataset. |
| `scripts/expR26_blind_submission.py` | 47-48, 81, 412, 591 | Replace `intents_blind_a`, output dir, and dataset. |
| `scripts/expR39b_blind_submission.py` | 57, 375, 511 | Replace output dir and dataset. |
| `scripts/expR41a_blind_submission.py` | 57, 460, 617 | Replace output dir and dataset. |
| `scripts/expR53_proxy_score_sample.py` | 73-74, 267, 270, 273 | Replace dataset and sampled submission paths if doing proxy scoring. |
| `scripts/expR53_response_optimization.py` | 26-27, 112-113 | Replace R39 input/output paths and dataset. |
| `scripts/expR54b_blind_change_analysis.py` | 42-43, 100 | Replace R39/R54 input paths and dataset. |

## Dry-Run Commands

First determine how organizers released Blind-B:

```bash
HF_HOME=$PWD/.hf_cache HF_DATASETS_CACHE=$PWD/.hf_cache/datasets .venv/bin/python - <<'PY'
from datasets import DownloadConfig, load_dataset

candidates = [
    ("talkpl-ai/TalkPlayData-Challenge-Blind-B", "test"),
    ("talkpl-ai/TalkPlayData-Challenge-Dataset", "blind_b"),
    ("talkpl-ai/TalkPlayData-Challenge-Dataset", "test"),
]

for name, split in candidates:
    try:
        ds = load_dataset(name, split=split, download_config=DownloadConfig(local_files_only=False))
        print("FOUND", name, split, len(ds), ds.column_names)
        first = ds[0]
        assert "session_id" in first and "conversations" in first
    except Exception as e:
        print("MISS", name, split, type(e).__name__, e)
PY
```

Then verify format only, without model inference:

```bash
HF_HOME=$PWD/.hf_cache HF_DATASETS_CACHE=$PWD/.hf_cache/datasets .venv/bin/python - <<'PY'
from datasets import DownloadConfig, load_dataset

DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-B"
SPLIT = "test"
ds = load_dataset(DATASET, split=SPLIT, download_config=DownloadConfig(local_files_only=False))
required = {"session_id", "conversations"}
assert required.issubset(ds.column_names), ds.column_names
print("rows", len(ds))
print("columns", ds.column_names)
print("first_session_id", ds[0]["session_id"])
print("conversation_turns", len(ds[0]["conversations"]))
PY
```

Once the line patches are made, the ingestion-only check for the R54 ensemble path is:

```bash
# Do not run before paths are patched to Blind-B-specific outputs.
uv run python scripts/expR54_phase3_ensemble_blind.py
```

Expected post-run shape for the cache: `{"lists": {sid: [(track_id, score), ... 300 ...]}, "manifest": ...}` with one key per Blind-B session.

## Expected Blind-B Size / Schema

Best guess from Blind-A precedent:

- Blind-B likely uses the same session record schema as Blind-A: `session_id`, user/session metadata, `conversations`, and goal progress fields.
- The submission zip likely remains a single `prediction.json` with one object per session and exactly 20 `predicted_track_ids` plus `predicted_response`.
- Blind-A had 80 sessions and 1600 track predictions. Blind-B may also be 80 sessions, but do not assume this in validation; use `len(ds)` as the expected row count and `20 * len(ds)` as the expected prediction count.
- Track IDs should remain TalkPlayData catalog UUIDs, with ISRC available only as metadata.

Repo communications checked:

- `docs/blind_a_final_state.md` says to wait for Blind-B, rebuild R54 blind retrieval, rerun R54b/R54c-style submission, then reconsider archived directions only after distribution checks.
- `docs/r59_candidates/c4_organizer_email.md` and related C4 notes contain a draft organizer question about external metadata policy, not an organizer reply. No actual Blind-B release details or organizer response were found in the repo.

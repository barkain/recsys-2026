# Blind-B R84c Runbook

**Goal:** When Blind-B drops, run R84c end-to-end without redesigning under time pressure.

**Validation status:** Dry-run on Blind-A reproduces R84c production hash bytewise (`cfcda2ae309f79e5...`). Harness is deterministic. Verified `2026-05-26`.

---

## Step 0 — Preflight check (anytime, no Blind-B data needed)

```bash
uv run python scripts/expR84c_blind_preflight.py --blind-name blind_a
uv run python scripts/expR84c_blind_dryrun.py    # full Blind-A reproducibility test
```

`blind_dryrun.py` must report `HASH MATCH` before Blind-B drops. If it doesn't, fix the regression before trusting the pipeline.

## Step 1 — Source cache build (Blind-B day, Mac)

When the Blind-B dataset drops, build the per-case source list cache.

`scripts/expR55_blind_source_cache.py` is now parameterized for arbitrary blind splits:

```bash
uv run python scripts/expR55_blind_source_cache.py \
    --blind-name blind_b \
    --blind-dataset talkpl-ai/TalkPlayData-Challenge-Blind-B
```

Defaults derive output paths from `--blind-name`:
- `cache/blind_b/source_cache/<sid>.pkl` (per-session)
- `cache/blind_b/source_cache.pkl` (consolidated)
- `cache/blind_b/source_cache_manifest.json`

R54b sanity validation is auto-skipped for non-blind_a runs (no R54b
submission to compare against). To override paths:

```bash
uv run python scripts/expR55_blind_source_cache.py \
    --blind-name blind_b \
    --blind-dataset <hf_dataset_id> \
    --output-dir cache/blind_b/source_cache \
    --output-cache cache/blind_b/source_cache.pkl \
    --skip-r54b-validation
```

**Verify:** the consolidated pickle should have keys = blind sids and values containing `session_id, turn_number, user_query, history, music_turns, src_a, src_b, src_c, src_d, src_f, als_tracks, als_vec, r21_list, r21_rank_map`. Size ~2-5 MB depending on Blind-B session count.

## Step 2 — R54 5-fold ensemble blind retrieval (Mac OR Colab, ~10 min wall)

R54's 5 phase3 fold models live in `cache/r54/phase3_full/fold_{1..4}/` and `cache/r54/phase3_smoke/fold_0/`. If they're present (they should be — committed), this runs on Mac CPU in ~10 min. Otherwise restore from prior R54 phase3 training.

`scripts/expR54_phase3_ensemble_blind.py` is now parameterized:

```bash
uv run python scripts/expR54_phase3_ensemble_blind.py --blind-name blind_b
# Writes: cache/r54_production/blind_b_r54_lists.json
```

By default it derives:
- `--blind-dataset` → `talkpl-ai/TalkPlayData-Challenge-Blind-B`
- `--output` → `cache/r54_production/blind_b_r54_lists.json`

Override with `--blind-dataset` / `--output` flags if needed.

## Step 3 — R84 5-fold ensemble blind retrieval (Colab A100, ~3 hours wall)

This is the most expensive step. Five R84 fold models must be re-trained from the R84 Phase 0A pair manifest (`cache/r84/phase0a/pair_manifest.parquet`, shipped in git).

**Colab cell (paste verbatim and run):**

```python
import os, sys, subprocess, pathlib, shutil, time
import torch

assert torch.cuda.is_available()
from google.colab import drive
try: drive.mount("/content/drive", force_remount=False)
except ValueError: drive.mount("/content/drive", force_remount=True)
DRIVE = pathlib.Path("/content/drive/MyDrive/r84_blind_b_artifacts")
DRIVE.mkdir(parents=True, exist_ok=True)

REPO = pathlib.Path("/content/recsys-2026")
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "--branch", "r84-full-corpus-retriever",
                    "https://github.com/barkain/recsys-2026.git", str(REPO)], check=True)
os.chdir(str(REPO))
sys.path.insert(0, str(REPO))
# Pull Blind-B source cache that you've uploaded to Drive separately
src = DRIVE / "source_cache.pkl"
if src.exists():
    (REPO / "cache/blind_b").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, REPO / "cache/blind_b/source_cache.pkl")
assert (REPO / "cache/blind_b/source_cache.pkl").exists()
# Restore R12 dev payload (needed for fold-0 model training that's already done)
r12_src = pathlib.Path("/content/drive/MyDrive/r84_phase1_artifacts/_R12_all_turns_payload.pkl")
if r12_src.exists():
    shutil.copy(r12_src, REPO / "exp/eval/_R12_all_turns_payload.pkl")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

for fold in range(5):
    out_dir = REPO / f"cache/r84/blind_b_blind_fold{fold}"
    blind_out = out_dir / "blind_r84_lists.json"
    if blind_out.exists():
        print(f"[FOLD {fold}] SKIP — already in Drive")
        continue
    print(f"\n=== FOLD {fold} TRAIN ===")
    r = subprocess.run([sys.executable, "-u", "scripts/expR84_phase0b_train.py",
                        "--fold", str(fold), "--batch-size", "32",
                        "--output-dir", str(out_dir)], cwd=str(REPO))
    assert r.returncode == 0
    print(f"\n=== FOLD {fold} BLIND ENCODE ===")
    # CRITICAL: pass --blind-cache to point at blind_b not blind_a
    r = subprocess.run([sys.executable, "-u", "scripts/expR84_blind_encode.py",
                        "--model-dir", str(out_dir / "model"),
                        "--output-path", str(blind_out),
                        # Note: existing expR84_blind_encode.py hardcodes BLIND_SRC.
                        # If Blind-B needs different cache, edit BLIND_SRC there OR
                        # set env BLIND_SRC=cache/blind_b/source_cache.pkl
                        ], cwd=str(REPO),
                       env={**os.environ, "PYTHONPATH": str(REPO),
                            "BLIND_SRC": "cache/blind_b/source_cache.pkl"})
    assert r.returncode == 0
    # Save to Drive
    shutil.copy(blind_out, DRIVE / f"blind_r84_fold{fold}.json")
    # Cleanup local model checkpoint to save disk
    shutil.rmtree(out_dir / "model")
    print(f"  fold {fold} -> Drive")

# Bundle artifacts for Mac
import tarfile
tar = DRIVE / "r84_blind_b_artifacts.tar.gz"
with tarfile.open(tar, "w:gz") as tf:
    for fold in range(5):
        p = REPO / f"cache/r84/blind_b_blind_fold{fold}/blind_r84_lists.json"
        if p.exists():
            tf.add(str(p), arcname=str(p.relative_to(REPO)))
print(f"\nTarball: {tar} ({tar.stat().st_size/1e6:.1f} MB)")
```

`scripts/expR84_blind_encode.py` now honors `--blind-cache` flag and the `BLIND_SRC` env var (set in the Colab loop above). Default unchanged for backwards compat.

## Step 4 — Download R84 Blind-B artifacts to Mac

```bash
# After Colab completes, download r84_blind_b_artifacts.tar.gz from Drive web UI
shasum -a 256 ~/Downloads/r84_blind_b_artifacts.tar.gz   # record this for audit
tar -xzf ~/Downloads/r84_blind_b_artifacts.tar.gz -C /Users/nadavbarkai/dev/recsys-2026/
```

## Step 5 — Run R84c replay on Blind-B (Mac, ~3 min)

```bash
uv run python scripts/expR84c_blind_preflight.py --blind-name blind_b
# All required artifacts must be present before continuing

uv run python scripts/expR84c_blind_replay.py --blind-name blind_b --tracks-only
# Produces:
#  - cache/r84_production/blind_b_r84_ensemble_lists.json
#  - exp/inference/blind_b/r84c_replay_blind_b_track_lists.json
#  - exp/eval/expR84c_replay_blind_b_audit.json
```

## Step 6 — Generate Blind-B responses (Mac, ~10 min Opus)

Since Blind-B has no R78 baseline, ALL responses must be generated fresh.

`scripts/expR84c_blind_b_responses.py` uses **R78-style prompting** — empirically the best response generator across this cycle. **NOT R87-style** because R87's evidence-injection variant slightly REGRESSED LexDiv on R84c (0.8720 → 0.8706, blind sub 757667). See [[project_r87_outcome]].

```bash
ANTHROPIC_RECSYS_API_KEY=sk-... \
    uv run python scripts/expR84c_blind_b_responses.py \
        --tracks-json exp/inference/blind_b/r84c_replay_blind_b_track_lists.json \
        --blind-cache cache/blind_b/source_cache.pkl \
        --output-zip exp/inference/blind_b/r84c_blind_b_submission.zip \
        --blind-name blind_b
```

Optional flags:
- `--dry-run` — list cases + print first 2 prompts, no API calls.
- `--limit N` — only generate first N (smoke test before full run).

Validation gates (built in):
- Track IDs unchanged from input (hash check before submission write).
- Word count 60-110 per response.
- No banned phrases (R78 inherited list).
- Empty/invalid responses are counted in metadata for review.

## Step 7 — Audit + manual upload

```bash
# Final sanity
unzip -p exp/inference/blind_b/r84c_blind_b_submission.zip prediction.json | uv run python -m json.tool | head -50
shasum -a 256 exp/inference/blind_b/r84c_blind_b_submission.zip
cat exp/eval/expR84c_replay_blind_b_audit.json
```

**Manual upload to Codabench** at the competition page. Don't auto-upload.

---

## What stays untouched

- `cache/r54_phase3_lr_model.txt` — frozen R54c LR (READ-ONLY)
- `cache/r84c_production_lr.txt` — production R84c LR (READ-ONLY)
- `cache/r84/phase0a/pair_manifest.parquet` + sha256 + build_config — frozen training data spec (READ-ONLY)
- `exp/inference/blind_a/r84c_selective_submission.zip` — current production (READ-ONLY)

## Pre-day hardening (all complete as of 2026-05-26)

1. ✓ `scripts/expR84_blind_encode.py` — accepts `--blind-cache` flag + `BLIND_SRC` env var.
2. ✓ `scripts/expR54_phase3_ensemble_blind.py` — accepts `--blind-name`, `--blind-dataset`, `--output`.
3. ✓ `scripts/expR55_blind_source_cache.py` — accepts `--blind-name`, `--blind-dataset`, `--output-dir`, `--output-cache`, `--manifest-path`.
4. ✓ `scripts/expR84c_blind_b_responses.py` — fresh-response generator for Blind-B (R78-style per R87 lesson).
5. ✓ Blind-A dry-run regression: `scripts/expR84c_blind_dryrun.py` reports `HASH MATCH` (track hash `cfcda2ae309f79e5...`).

## Estimated total wall (Blind-B day)

| step | wall | requires |
|---|---|---|
| Source cache build | ~10-30 min | Mac, depends on dataset size |
| R54 5-fold blind ensemble | ~10 min | Colab A100 OR cached R54 fold models on Mac |
| R84 5-fold train + ensemble | ~3 hours | Colab A100 (sequential 5 × 30 min train + ensemble) |
| Replay + features + score + route | ~3 min | Mac |
| Response generation (~80 cases) | ~10 min | Mac (Opus API) |
| **TOTAL** | **~4 hours** | Mostly Colab waiting |

If Colab disconnects between fold trainings, the per-fold Drive saves let us resume without redoing completed folds.

## Failure modes to watch for

- **R84 fold model checkpoints lost again** → just re-train; deterministic seed=0.
- **HF dataset cache eviction** → Colab will re-download (~5 min for Track-Metadata, ~3 min for Track-Embeddings).
- **Opus rate limit during response gen** → retry with backoff; ~80 calls is well within tier limits.
- **Codabench scorer offline** (as of 2026-05-25 R86 attempt with Gemini deprecation) → hold submission until fixed.

## Key paths reference

| | path |
|---|---|
| R84 manifest | `cache/r84/phase0a/pair_manifest.parquet` |
| Blind-A source | `cache/blind_a/source_cache.pkl` |
| Blind-B source (build day-of) | `cache/blind_b/source_cache.pkl` |
| Blind-A R54 ensemble | `cache/r54_production/blind_r54_lists.json` |
| Blind-B R54 ensemble (build day-of) | `cache/r54_production/blind_b_r54_lists.json` |
| Blind-A R84 5-fold | `cache/r84/blind_fold{0..4}/blind_r84_lists.json` |
| Blind-B R84 5-fold (build day-of) | `cache/r84/blind_b_blind_fold{0..4}/blind_r84_lists.json` |
| Frozen R54c LR | `cache/r54_phase3_lr_model.txt` |
| Production R84c LR | `cache/r84c_production_lr.txt` |
| R84c production submission | `exp/inference/blind_a/r84c_selective_submission.zip` |
| Blind-B output (this run) | `exp/inference/blind_b/r84c_blind_b_submission.zip` |

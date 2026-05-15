# R54 Phase 3 Full — Colab GPU Runbook

Goal: train R54 Phase 3 folds 1-4 on a Colab T4 GPU. Bring back per-fold
`oof_lists.json` artifacts and run integration/evaluation locally.

## Why Colab

- CPU bottleneck: BGE-base training + 47K catalog encoding takes ~5h per fold on local CPU.
- T4 should bring each fold down to ~10-30 min.
- Folds are independent. Cloud execution avoids the local process-reaper problem entirely.
- Final LambdaRank evaluation stays local (uses R39 pipeline + cached features).

## What runs on Colab vs local

| Stage | Where | Why |
|-------|-------|-----|
| Phase 3 training folds 1-4 | Colab T4 | GPU |
| Phase 3 fold 0 | Local (already done as smoke) | Reuse `cache/r54/phase3_smoke/fold_0` |
| Catalog encoding per fold | Colab T4 | GPU |
| OOF list retrieval per fold | Colab T4 | GPU |
| Aggregated OOF assembly | Local | Needs `R12` pickle |
| Standalone evaluation | Local | R21 OOF + R12 |
| R39 + R54 LambdaRank integration | Local | R39 pipeline |

## Prerequisites on local machine

1. `cache/r54/phase3_smoke/fold_0/oof_lists.json` exists (already done).
2. `exp/eval/_R12_all_turns_payload.pkl` exists locally (≈114MB, needed for Colab upload).
3. Git remote is current with all R54 scripts pushed.

## Setup notebook (one-time)

Create a new Colab notebook with T4 runtime. Replace `<REPO_URL>` with the
push target and `<REPO_DIR>` with whatever the clone produces.

### Cell 1: GPU check + clone repo

```python
!nvidia-smi
!git clone <REPO_URL> recsys-2026
%cd recsys-2026
!git checkout r54-second-gen-supervised-retriever
```

### Cell 2: Install deps (via uv)

```python
!pip install -q uv
!uv sync
import torch
print("CUDA:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))
```

If `uv sync` is too slow on Colab, fall back to minimal pip install:

```python
!pip install -q sentence-transformers datasets numpy lightgbm
```

### Cell 3: Upload R12 payload (one-time)

The R12 payload pickle is local-only (preprocessed from dev splits). Upload it.

```python
# Option A: Colab upload widget
from google.colab import files
import os
os.makedirs("exp/eval", exist_ok=True)
print("Upload _R12_all_turns_payload.pkl from your local exp/eval/...")
uploaded = files.upload()  # pick the file
# Move to expected path if needed:
import shutil
src = next(iter(uploaded.keys()))
shutil.move(src, "exp/eval/_R12_all_turns_payload.pkl")
```

```python
# Option B (faster for repeated runs): mount Drive and copy from there
from google.colab import drive
drive.mount("/content/drive")
!cp "/content/drive/MyDrive/r54/_R12_all_turns_payload.pkl" exp/eval/
```

### Cell 4: Pre-download HF datasets (one-time, ~2-3 min)

```python
from datasets import load_dataset
_ = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
_ = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
print("HF datasets cached.")
```

### Cell 5: Sanity — run one fold to validate setup

Start with fold 1 (smallest deviation from a known reference).

```python
!uv run python scripts/expR54_phase3_full5fold_train.py \
    --fold 1 \
    --device cuda \
    --no-aggregate
```

Expected output snippets:
- `Device: cuda`
- `fold 1: ... total=26400`
- `fold 1 val hit@200: XXX/1600 (0.5...)`
- Final timestamp; fold artifact written to
  `cache/r54/phase3_full/fold_1/oof_lists.json`

Sanity check: fold-1 hit@200 should be **at or above ~0.539** (Phase 2 fold-1
standalone was 0.539). The Phase 3 model uses broader data, so equal-or-better.

### Cell 6: Run folds 2, 3, 4

```python
for fold_i in [2, 3, 4]:
    !uv run python scripts/expR54_phase3_full5fold_train.py \
        --fold {fold_i} --device cuda --no-aggregate
```

Each fold should take ~15-30 min on T4. Total ~1-2h.

### Cell 7: Package artifacts for download

```python
import os, shutil
out_dir = "/content/recsys-2026/cache/r54/phase3_full"
os.makedirs("/content/r54_phase3_artifacts", exist_ok=True)
for fold_i in [1, 2, 3, 4]:
    src = f"{out_dir}/fold_{fold_i}/oof_lists.json"
    dst = f"/content/r54_phase3_artifacts/fold_{fold_i}_oof_lists.json"
    shutil.copy(src, dst)
!ls -la /content/r54_phase3_artifacts/
!cd /content && zip -r r54_phase3_artifacts.zip r54_phase3_artifacts/
print("Download r54_phase3_artifacts.zip below.")
```

```python
# Option A: direct download
from google.colab import files
files.download("/content/r54_phase3_artifacts.zip")

# Option B: copy to Drive
!cp /content/r54_phase3_artifacts.zip /content/drive/MyDrive/r54/
```

## Local: install artifacts

```bash
cd /Users/nadavbarkai/dev/recsys-2026
unzip ~/Downloads/r54_phase3_artifacts.zip -d /tmp/r54_artifacts
for f in 1 2 3 4; do
  mkdir -p cache/r54/phase3_full/fold_$f
  cp /tmp/r54_artifacts/r54_phase3_artifacts/fold_${f}_oof_lists.json \
     cache/r54/phase3_full/fold_$f/oof_lists.json
done
# Fold 0 from local smoke
mkdir -p cache/r54/phase3_full/fold_0
cp cache/r54/phase3_smoke/fold_0/oof_lists.json cache/r54/phase3_full/fold_0/oof_lists.json
ls -la cache/r54/phase3_full/fold_*/oof_lists.json
```

Then aggregate to produce `oof_r54_lists.json`:

```bash
uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.expR54_phase3_full5fold_train import aggregate_oof_lists
from pathlib import Path
aggregate_oof_lists(Path('cache/r54/phase3_full'), 8000)
"
ls -la cache/r54/phase3_full/oof_r54_lists.json
```

## Local: run evaluation

```bash
# Standalone — compares P3 vs P2 vs R21 across all 8K cases
uv run python scripts/expR54_phase3_full5fold_standalone.py

# Integration — CV5 LambdaRank, R39+R54 at weights 1.0/1.5/2.0 with R54 features
uv run python scripts/expR54_phase3_full5fold_integration.py
```

Production gate (best non-baseline config):
- Δh7 ≥ +0.010
- Δpool_hit ≥ +0.020
- net > 0
- no same/diff regression (≥ -0.005)

## Failure modes and recovery

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during training | Batch too large for T4 | Add `--max-seq-len 192` (not implemented yet — open PR) or lower batch via env |
| Colab session disconnects mid-fold | Free tier 4h50m limit | Per-fold checkpoint at batch 200; rerun the same fold command, it resumes |
| `hit@200` for a fold << 0.50 | Training divergence / bug | Compare to Phase 2 fold-X reference; redo that fold |
| Missing R12 payload | Forgot to upload | See Setup Cell 3 |
| Artifact zip too big to download | Includes model dirs | Cell 7 only zips `oof_lists.json` (28MB × 4 = 112MB total) |

## Reproducibility constraints

- `--fold N` uses `TRAINING_SEED + N` (= `0 + N`), same NumPy/Torch seeds.
- `train-split sample` is built deterministically with `SAMPLING_SEED=0`,
  same across all folds.
- The `grouped_session_folds(seed=0)` split is the same R21 / R39 / Phase 2
  used — fold-X val cases are identical to other R54 runs.

## What NOT to change

1. Hyperparams (epochs=1, bs=32, lr=2e-5, tau=0.05, max_seq_len=256).
2. Query / track text format.
3. Sampling seed.
4. Fold indices.
5. The model: BGE-base-en-v1.5. No model upgrade in this phase.

Anything else may be tweaked for operational reasons, but flag it in the
fold manifest if you do.

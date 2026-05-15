# R55 RunPod runbook — production-grade R54 training

## Goal

Train a single all-data R55 retriever on an ephemeral GPU (A100 / L40S),
generate blind-A top-300 candidate lists, ship artifacts back to local,
and compare against the R54 ensemble before deciding whether to submit.

Expected total cost: **~$1-3** (A100 ~25-40 min @ ~$1.89/hr, L40S
~40-60 min @ ~$1.19/hr).

## Prerequisites

- RunPod account with $5+ credit
- SSH key uploaded to RunPod settings (Account → SSH Public Keys)
- Local: `cache/r54_production/blind_r54_lists.json` present (the R54 ensemble
  lists are needed for the local churn comparison after training)

## 1. Launch pod

1. Pods → Deploy → GPU Pods → choose **A100 80GB** (preferred) or **L40S 48GB**
2. Template: **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`**
   (any recent PyTorch + CUDA 12.x template is fine — we install our own deps with uv)
3. Volume: **20 GB** is enough (~600 MB final artifacts + cache space)
4. Expose SSH (port 22)
5. Deploy. Note the SSH command from the "Connect" tab — looks like:
   `ssh root@<host> -i ~/.ssh/id_ed25519 -p <port>`

## 2. SSH in and set up

```bash
ssh root@<host> -p <port>          # use the connect command from RunPod

# Inside the pod:
apt update && apt install -y git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /workspace
git clone https://github.com/barkain/recsys-2026.git
cd recsys-2026
git checkout r55-production-r54

uv sync
uv add lightgbm 2>&1 | tail -5    # downstream helper imports need it
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: CUDA: True  NVIDIA A100-SXM4-80GB (or similar)
```

## 3. Fetch R12 payload + HF datasets

```bash
# R12 dev payload (~114 MB) — published as GitHub release asset
mkdir -p exp/eval
curl -L -o exp/eval/_R12_all_turns_payload.pkl \
  https://github.com/barkain/recsys-2026/releases/download/r54-data/_R12_all_turns_payload.pkl
ls -lh exp/eval/_R12_all_turns_payload.pkl

# Prefetch HF datasets (catalog metadata, train split, blind-A)
uv run python -c "
from datasets import load_dataset
load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata')
load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset')
load_dataset('talkpl-ai/TalkPlayData-Challenge-Blind-A')
print('HF datasets cached')
"
```

## 4. Smoke train (50 batches) — verify pipeline before the full run

```bash
# Optional but cheap (~3 min on A100): patch CKPT_EVERY and abort after a few
# batches to verify the loop trains and checkpoints write correctly. Skip if
# you trust the script.
```

## 5. Full training run

```bash
# Run in a tmux session so a dropped SSH won't kill the training
tmux new -s r55
uv run python scripts/expR55_production_train.py --device cuda 2>&1 | tee /tmp/r55_train.log

# Detach: Ctrl-b then d
# Reattach: tmux attach -t r55
```

Expected behavior:
- Loading R12 payload, catalog (~10 s)
- Building train pairs: dev ~8000 + train-split 20000 = ~28000 pairs (~30 s)
- Training: 875 batches, ~1 s/batch on A100 ≈ **~15 min**
  Log lines every 50 batches; checkpoint every 200 batches
- Catalog encoding: 47k tracks at batch size 256 on A100 ≈ **~3 min**
- Blind retrieval: 80 queries × 47k catalog ≈ **~30 s**
- **Total: ~20-25 min on A100, ~40-50 min on L40S**

If the run is interrupted, re-run the same command — the script resumes from
the latest checkpoint in `cache/r55_production/checkpoints/`.

## 6. Verify outputs

```bash
ls -lh cache/r55_production/
# Expected:
#   model/                     (~440 MB)
#   track_embeddings.npy       (~140 MB, 47071 × 768 × 4 bytes)
#   track_ids.json             (~1.5 MB)
#   blind_r55_lists.json       (~1.5 MB)
#   manifest.json              (~1 KB)
# (No checkpoints/ — it's cleaned up on successful completion.)

jq '.manifest | {n_train_pairs, n_dev_pairs, n_train_split_pairs, elapsed_s}' \
  cache/r55_production/blind_r55_lists.json
```

## 7. Bundle artifacts and download to local

```bash
# Drop the heavy model dir from the local-needed bundle (we don't need
# inference weights locally; embeddings + blind lists are enough for
# integration + comparison).
cd cache/r55_production
zip -r /tmp/r55_local.zip \
  track_embeddings.npy track_ids.json blind_r55_lists.json manifest.json
ls -lh /tmp/r55_local.zip   # expect ~140 MB
cd /workspace/recsys-2026

# Optional: bundle the model too, in case we want to re-use it later
zip -r /tmp/r55_model.zip cache/r55_production/model
ls -lh /tmp/r55_model.zip   # expect ~440 MB
```

From your **local** machine:

```bash
# Replace <host> and <port> with the RunPod SSH details
scp -P <port> root@<host>:/tmp/r55_local.zip ~/Downloads/
# Optional: scp the model too
scp -P <port> root@<host>:/tmp/r55_model.zip ~/Downloads/

# Unpack into the same paths the local scripts expect
cd /Users/nadavbarkai/dev/recsys-2026
mkdir -p cache/r55_production
unzip -o ~/Downloads/r55_local.zip -d cache/r55_production
# If you grabbed the model:
unzip -o ~/Downloads/r55_model.zip -d .   # unpacks into cache/r55_production/model
```

## 8. Local: run the churn comparison

```bash
uv run python scripts/expR55_blind_compare.py
```

This computes top-1 / top-20 / top-300 overlap vs the R54 ensemble, R55
cosine distributions, and top-1 churn vs the R54b production top-1s.

**Gates** (from feedback memory `retriever-swap-churn-gates`):

| Gate | Soft | Hard |
|------|------|------|
| top-1 churn vs R54b (retrieval) | < 25 / 80 | > 35 / 80 |
| top-20 overlap median vs R54 ens | >= 14 / 20 | — |

Exit codes:
- `0` — all gates pass, safe to proceed to integration / submission
- `1` — soft fail, review and decide
- `2` — hard fail, do not submit

## 9. Tear down the pod

When you're done, **terminate the pod from the RunPod web UI** (Pods →
your pod → ⋮ → Stop or Terminate). Stopped pods still incur storage cost;
terminate to stop all billing.

## 10. Follow-up (if gates pass)

Not in this runbook. The next steps would be:
- Wire R55 cosines into the LR ranker (drop-in swap for R54 cosine features)
- Build R55 submission with R54c response-polish hygiene
- Submit and compare to R54b/R54c on the leaderboard

Those scripts will be added later (`expR55_submission.py` etc.) if R55 clears
the churn gates.

## Troubleshooting

- **`uv: command not found`** after install — `source $HOME/.local/bin/env`
- **SSH disconnects during training** — use tmux; reattach with `tmux attach -t r55`
- **Out of CUDA memory** at batch start — A100 80GB has plenty for BS=32 at
  MAX_SEQ_LEN=256. If a smaller GPU is used and OOM occurs, drop BATCH_SIZE
  in `expR55_production_train.py` to 16. Note this changes in-batch negative
  count and may alter the model slightly vs Phase 3.
- **HF dataset download fails** — RunPod hosts can be rate-limited; retry, or
  pre-package the parquet/arrow files in `~/.cache/huggingface/datasets` and
  upload to the pod via scp.
- **Cost overrun** — if training is still running after 60 min on A100 or
  90 min on L40S, something is wrong. Check `tmux attach -t r55` for the
  log, look at sec/batch.

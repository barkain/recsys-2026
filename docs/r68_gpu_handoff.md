# R68 Phase 0/1 GPU Handoff

R68 is a large-scale retriever swap: replace R54's BGE-base-en-v1.5 (768-d)
backbone with BGE-large-en-v1.5 (1024-d), keep R54's training recipe
(structured queries, R21 track text, 1 epoch, in-batch InfoNCE, tau=0.05),
keep the frozen R54c LR pool admission, and check whether the larger encoder
recovers GT cases R54 misses.

This document is the runbook for the remote GPU box. All Wave-1 scripts live
in this repo; the Mac orchestrator polls for sync-back artifacts and resumes
the sprint automatically.

## 1. GPU box prerequisites

- CUDA 11.8+ or 12.x driver
- Python 3.10+
- GPU with at least 24 GB VRAM. A100 / H100 are ideal. RTX 4090 works at
  `BATCH_SIZE=16` (Phase 0 script default is 32; reduce if OOM).
- `uv` available on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Internet for the one-time BGE-large model download from HuggingFace
- ~30 GB free disk for HuggingFace cache + per-fold artifacts

## 2. Clone + branch

```
git clone <repo_url> recsys-2026
cd recsys-2026
git checkout r68-large-scale-retrieval
```

## 3. Env setup

```
uv sync
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Confirm `cuda: True` before proceeding. If False, stop and fix the driver /
torch install — there is no point launching CPU-only training (the Mac
already has CPU; this box exists only for GPU acceleration).

## 4. Phase 0 smoke training command

Single fold-0 training to validate the recipe before the full 5-fold run.

```
uv run python scripts/expR68_phase0_fold0_train.py \
    --model BAAI/bge-large-en-v1.5 \
    --fold 0 \
    --output_dir cache/r68/phase0_fold0/
```

Expected runtime: 2–4 hours on A100/H100, 3–5 hours on RTX 4090
(`BATCH_SIZE=16`).

Expected outputs (must all be present for sync verification):

- `cache/r68/phase0_fold0/model/` — SentenceTransformer checkpoint directory
- `cache/r68/phase0_fold0/query_embeddings_dev.npy` — fold-0 dev query
  embeddings (`N_dev_fold0 × 1024` fp32)
- `cache/r68/phase0_fold0/track_embeddings.npy` — track catalog embeddings
  (`N_tracks × 1024` fp32)
- `cache/r68/phase0_fold0/track_ids.json` — ordered track id list aligned to
  `track_embeddings.npy`
- `cache/r68/phase0_fold0/oof_r68_lists_fold0.json` — per fold-0-held-out
  case top-300 candidate list with cosine scores. **THIS is the primary
  sync-trigger artifact the Mac orchestrator polls for.**
- `cache/r68/phase0_fold0/train_log.json` — loss curve, hyperparameters,
  device info, elapsed time

## 5. Phase 1 full 5-fold training command

Only run this **after** Phase 0 has cleared the gate on the Mac side (the
orchestrator will send a PushNotification when ready).

```
uv run python scripts/expR68_phase1_full5fold_train.py \
    --model BAAI/bge-large-en-v1.5 \
    --output_dir cache/r68/phase1_full/
```

Expected runtime: 8–16 hours total (per-fold ~2–3 h × 5 folds).

Expected outputs per fold `i ∈ {0..4}`:

- `cache/r68/phase1_full/fold_{i}/model/`
- `cache/r68/phase1_full/fold_{i}/query_embeddings_oof.npy`
- `cache/r68/phase1_full/fold_{i}/track_embeddings.npy`
- `cache/r68/phase1_full/fold_{i}/track_ids.json`
- `cache/r68/phase1_full/fold_{i}/oof_r68_lists.json`

Sprint sync trigger: `cache/r68/phase1_full/fold_4/oof_r68_lists.json` (last
fold; orchestrator considers the Phase 1 run complete once this exists).

Note: the Phase 1 training script is authored in Wave 2 — do not launch
Phase 1 from this Wave-1 handoff unless the Wave-2 script is present in the
branch.

## 6. Sync back to Mac

From the GPU box:

```
rsync -avz --exclude='*.pt' --exclude='__pycache__' \
    cache/r68/ <mac_user>@<mac_host>:/Users/nadavbarkai/dev/recsys-2026/cache/r68/
```

`--exclude='*.pt'` keeps the rsync small: we mainly need the embeddings, OOF
lists, and the SentenceTransformer config/tokenizer/safetensors. If you
need the raw `pytorch_model.bin` for blind submission, drop the `*.pt`
exclude.

Alternatives:

- Sync to a cloud bucket (S3/GCS) and pull from the Mac.
- If the GPU box is hosted with shared filesystem to the Mac, no rsync needed
  — orchestrator just polls the path directly.

## 7. Sync verification (Mac side does this automatically)

The orchestrator polls these paths to detect sync completion:

- Phase 0: `cache/r68/phase0_fold0/oof_r68_lists_fold0.json`
- Phase 1: `cache/r68/phase1_full/fold_4/oof_r68_lists.json`

Once detected, the Mac runs:

- Phase 0: `scripts/expR68_phase0_eval.py` (Wave 1 deliverable, in branch)
- Phase 1: Wave-2 eval (deferred)

## 8. Troubleshooting

- **OOM at batch 32** — reduce `BATCH_SIZE` to 16 (or 8) in
  `scripts/expR68_phase0_fold0_train.py`. Loss curve will be noisier but
  the 1-epoch budget is unchanged.
- **HuggingFace 401 / 403** — BGE-large is fully open, no token required.
  If you hit a rate limit, set `HF_TOKEN` env var to a personal token.
- **rsync slow** — try `--compress-level=9`, or stage to `/dev/shm` first
  and rsync from there. For multi-GB embeddings, consider `zstd -19` +
  scp.
- **`datasets` cache miss** — the first invocation downloads the TalkPlay
  challenge dataset (~few GB). Pre-warm with
  `uv run python -c "from datasets import load_dataset; load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset')"`
  and similarly for the track metadata.
- **fold-0 reproducibility** — fold split MUST match R54
  (`grouped_session_folds(sessions, seed=0, k=5)`). The training script
  fails fast if `cache/r54/phase2_full/oof_manifest.json` is missing or the
  derived fold-0 `val_indices_sample` does not match.

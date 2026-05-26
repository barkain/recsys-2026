"""R89 Phase 0 — train multimodal fusion track encoder (A100).

Frozen everything except the fusion params:
- Query encoder = raw BGE-large (frozen). Queries pre-encoded.
- 5 modality catalogs = frozen (pre-aligned in modality_cache).
- Trainable = per-modality projections + 5 modality gates + fusion MLP (~3.5M params).

Loss: in-batch InfoNCE + k=64 random catalog negatives per step.
Train fold-0 only.

Outputs:
- cache/r89/phase0_fold0/model.pt
- cache/r89/phase0_fold0/training_log.txt
- cache/r89/phase0_fold0/training_summary.json
- cache/r89/phase0_fold0/query_embs.fp16.npy  (training queries, for re-use)

Run:
  uv run python scripts/expR89_phase0_train_fusion.py --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MOD_CACHE_DIR = REPO / "cache" / "r89" / "modality_cache"
MANIFEST_PATH = REPO / "cache" / "r84" / "phase0a" / "pair_manifest.parquet"
OUT_DIR_DEFAULT = REPO / "cache" / "r89" / "phase0_fold0"

# --- Phase 0 fixed hyperparams ---
QUERY_BGE_MODEL = "BAAI/bge-large-en-v1.5"  # raw, no fine-tune
MAX_SEQ_LEN_QUERY = 384
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 3
TAU = 0.05
K_RANDOM_NEGS = 64
PROJ_DIM = 256
OUT_DIM = 1024
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 0
LOG_EVERY = 50


def ts(): return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def build_track_text(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.txt"

    def log(msg):
        line = f"{ts()} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"R89 Phase 0 fold={args.fold} output={args.output_dir}")
    log(f"hyperparams: batch={args.batch_size} epochs={args.epochs} lr={LR} "
        f"tau={TAU} k_neg={K_RANDOM_NEGS} proj_dim={PROJ_DIM}")

    # --- Imports needed ---
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F_t  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Load modality cache ---
    log("Loading modality cache...")
    modalities = {}
    for key in ["text", "image", "lyrics", "attrs", "audio"]:
        arr = np.load(MOD_CACHE_DIR / f"{key}.fp16.npy")
        modalities[key] = torch.from_numpy(arr.astype(np.float32))
        log(f"  {key}: {arr.shape} dtype={arr.dtype}")
    mask = torch.from_numpy(np.load(MOD_CACHE_DIR / "mask.fp16.npy").astype(np.float32))
    track_ids = json.load(open(MOD_CACHE_DIR / "track_ids.json"))
    dims = json.load(open(MOD_CACHE_DIR / "dims.json"))
    n_tracks = len(track_ids)
    track_id_to_idx = {t: i for i, t in enumerate(track_ids)}
    log(f"  n_tracks={n_tracks}  dims={dims}")

    # --- Load pair manifest, filter to fold's training rows ---
    log("Loading pair manifest...")
    table = pq.read_table(MANIFEST_PATH)
    fold_idx_arr = table.column("fold_idx").to_numpy()
    train_mask = (fold_idx_arr == -1) | (fold_idx_arr == args.fold)
    queries = table.column("query_structured").to_pylist()
    gts = table.column("gt_track_id").to_pylist()
    queries = [queries[i] for i in range(len(queries)) if train_mask[i]]
    gts = [gts[i] for i in range(len(gts)) if train_mask[i]]
    log(f"  fold {args.fold}: {len(queries)} training pairs")

    # Filter pairs to those with GT in our catalog
    valid_indices = [i for i, g in enumerate(gts) if g in track_id_to_idx]
    queries = [queries[i] for i in valid_indices]
    gts = [gts[i] for i in valid_indices]
    gt_idxs = np.array([track_id_to_idx[g] for g in gts], dtype=np.int64)
    log(f"  after catalog filter: {len(queries)} pairs (with GT in catalog)")

    # --- Pre-encode queries with raw BGE-large (one-time) ---
    log(f"Pre-encoding {len(queries)} queries with {QUERY_BGE_MODEL}...")
    from sentence_transformers import SentenceTransformer  # type: ignore
    bge = SentenceTransformer(QUERY_BGE_MODEL, device=args.device)
    bge.max_seq_length = MAX_SEQ_LEN_QUERY
    bge.eval()
    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")

    t0 = time.time()
    chunk = 256
    q_embs_list = []
    with torch.no_grad():
        for i in range(0, len(queries), chunk):
            with torch.amp.autocast(
                device_type="cuda" if use_bf16 else "cpu",
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
                enabled=use_bf16,
            ):
                emb = bge.encode(queries[i:i + chunk], batch_size=chunk,
                                  show_progress_bar=False, convert_to_tensor=True,
                                  normalize_embeddings=True)
            q_embs_list.append(emb.float().cpu())
            if (i + chunk) % 5000 == 0 or i + chunk >= len(queries):
                log(f"  encoded {min(i + chunk, len(queries))}/{len(queries)} "
                    f"({time.time() - t0:.0f}s)")
    q_embs = torch.cat(q_embs_list, dim=0)
    log(f"  query encoding done: shape={q_embs.shape}, "
        f"elapsed={(time.time() - t0)/60:.1f} min")
    # Save for re-use (eval) — quantize to fp16
    np.save(args.output_dir / "training_query_embs.fp16.npy",
            q_embs.numpy().astype(np.float16))
    log(f"  saved training_query_embs.fp16.npy")

    # Free BGE from VRAM
    del bge
    torch.cuda.empty_cache()

    # --- Define fusion model ---
    class TrackFusion(nn.Module):
        def __init__(self, dims, proj_dim=PROJ_DIM, out_dim=OUT_DIM):
            super().__init__()
            self.projs = nn.ModuleDict({
                key: nn.Sequential(
                    nn.Linear(dims[key], proj_dim), nn.GELU(),
                    nn.LayerNorm(proj_dim),
                )
                for key in ["text", "image", "lyrics", "attrs", "audio"]
            })
            self.gates = nn.Parameter(torch.zeros(5))  # logits → sigmoid (~0.5 at init)
            self.fusion = nn.Sequential(
                nn.Linear(proj_dim * 5, 1024), nn.GELU(),
                nn.LayerNorm(1024),
                nn.Linear(1024, out_dim),
            )

        def forward(self, embs_dict, mask_b):
            # embs_dict: {key: (B, dim_key)}, mask_b: (B, 5) modality coverage
            gates = torch.sigmoid(self.gates)  # (5,)
            modality_order = ["text", "image", "lyrics", "attrs", "audio"]
            projs = []
            for i, key in enumerate(modality_order):
                p = self.projs[key](embs_dict[key])  # (B, proj_dim)
                # Apply gate × mask: zero out missing modalities
                p = p * gates[i] * mask_b[:, i:i + 1]
                projs.append(p)
            x = torch.cat(projs, dim=-1)  # (B, proj_dim * 5)
            out = self.fusion(x)
            return F_t.normalize(out, dim=-1)

    model = TrackFusion(dims).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model: {n_params:,} trainable params")

    # Move modality cache to device (fp32, ~600 MB on GPU)
    for key in modalities:
        modalities[key] = modalities[key].to(args.device)
    mask = mask.to(args.device)
    log("  modality cache moved to GPU")

    # --- Train ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    n_pairs = len(queries)
    batches_per_epoch = (n_pairs + args.batch_size - 1) // args.batch_size
    if args.max_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    log(f"batches_per_epoch={batches_per_epoch} total_steps={batches_per_epoch * args.epochs}")

    q_embs_gpu = q_embs.to(args.device)
    log("Starting training...")
    model.train()
    losses_window = []
    t_start = time.time()
    step = 0

    def fetch_modalities(idx_tensor):
        """Given (B,) track index tensor, return embs_dict + mask for that batch."""
        embs_dict = {key: modalities[key][idx_tensor] for key in modalities}
        m = mask[idx_tensor]
        return embs_dict, m

    for epoch in range(args.epochs):
        perm = np.random.RandomState(SEED + epoch).permutation(n_pairs)
        for bi in range(batches_per_epoch):
            i0 = bi * args.batch_size
            i1 = min(i0 + args.batch_size, n_pairs)
            idx = perm[i0:i1]
            B = len(idx)
            if B == 0:
                continue
            q_batch = q_embs_gpu[idx]  # (B, 1024)
            pos_track_idx = torch.from_numpy(gt_idxs[idx]).to(args.device)
            # Sample random catalog negatives (per-step, GT-excluded)
            neg_idx = torch.randint(0, n_tracks, (K_RANDOM_NEGS,), device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if use_bf16 else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                pos_embs, pos_mask = fetch_modalities(pos_track_idx)
                neg_embs, neg_mask = fetch_modalities(neg_idx)
                pos_t = model(pos_embs, pos_mask)  # (B, 1024)
                neg_t = model(neg_embs, neg_mask)  # (K, 1024)
                all_t = torch.cat([pos_t, neg_t], dim=0)  # (B+K, 1024)
                sim = (q_batch @ all_t.T) / TAU  # (B, B+K)
                labels = torch.arange(B, device=args.device)
                loss = F_t.cross_entropy(sim, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses_window.append(float(loss.item()))
            step += 1

            if step % LOG_EVERY == 0:
                recent = losses_window[-LOG_EVERY:]
                avg = sum(recent) / len(recent)
                elapsed = time.time() - t_start
                rate = step / elapsed
                eta = (batches_per_epoch * args.epochs - step) / max(rate, 1e-6) / 60
                gates_now = torch.sigmoid(model.gates).detach().cpu().numpy()
                log(f"  epoch {epoch} step {step}/{batches_per_epoch * args.epochs} "
                    f"loss_avg={avg:.4f} rate={rate:.1f} b/s eta={eta:.1f}m "
                    f"gates={[f'{g:.2f}' for g in gates_now]}")

            if losses_window[-1] != losses_window[-1] or losses_window[-1] > 100:
                log(f"!! CATASTROPHIC LOSS step={step}: {losses_window[-1]}")
                sys.exit(1)

    elapsed = time.time() - t_start
    log(f"Training done in {elapsed/60:.1f} min. Final loss avg = "
        f"{sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)):.4f}")

    # Final gates
    gates_final = torch.sigmoid(model.gates).detach().cpu().numpy().tolist()
    log(f"Final gates: text={gates_final[0]:.3f} image={gates_final[1]:.3f} "
        f"lyrics={gates_final[2]:.3f} attrs={gates_final[3]:.3f} audio={gates_final[4]:.3f}")

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "dims": dims,
        "config": {
            "PROJ_DIM": PROJ_DIM, "OUT_DIM": OUT_DIM,
            "TAU": TAU, "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs, "LR": LR, "K_RANDOM_NEGS": K_RANDOM_NEGS,
        },
    }, args.output_dir / "model.pt")
    log(f"Saved model -> {args.output_dir / 'model.pt'} "
        f"({(args.output_dir / 'model.pt').stat().st_size / 1e6:.1f} MB)")

    summary = {
        "experiment": "R89 Phase 0 fold-0 fusion training",
        "fold": args.fold,
        "n_train_pairs": n_pairs,
        "n_params_trainable": n_params,
        "training_elapsed_min": round(elapsed / 60, 2),
        "final_loss_avg_last50": round(
            sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)), 4
        ),
        "initial_loss_first50": round(
            sum(losses_window[:50]) / max(1, min(len(losses_window), 50)), 4
        ),
        "final_gates": {
            "text": gates_final[0], "image": gates_final[1],
            "lyrics": gates_final[2], "attrs": gates_final[3],
            "audio": gates_final[4],
        },
        "hyperparams": {
            "BATCH_SIZE": args.batch_size, "EPOCHS": args.epochs, "LR": LR,
            "TAU": TAU, "K_RANDOM_NEGS": K_RANDOM_NEGS,
            "PROJ_DIM": PROJ_DIM, "OUT_DIM": OUT_DIM,
            "QUERY_BGE_MODEL": QUERY_BGE_MODEL,
            "MAX_SEQ_LEN_QUERY": MAX_SEQ_LEN_QUERY,
            "bf16": use_bf16,
        },
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary -> training_summary.json")
    log("R89 Phase 0 training complete.")


if __name__ == "__main__":
    main()

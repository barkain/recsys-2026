"""R84 Phase 0B — BGE-large full-corpus retriever training (fold-0).

Consumes the authoritative pair manifest from Phase 0A. NO 20K cap, NO
2-per-session cap, NO hard-negative auxiliary (first run, hard_neg_weight=0.0
per [[feedback_no_hardneg_aux_first_run]]).

Loss: in-batch InfoNCE + k=64 random catalog negatives sampled fresh per step.

Designed to run on Colab A100 with bf16 autocast.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_full5fold_train import (  # noqa: E402
    build_track_text,
    load_catalog,
)
from scripts.expR84_phase0a_census import (  # noqa: E402
    BUILD_CONFIG_PATH,
    MANIFEST_PATH,
    SHA256_PATH,
    assert_manifest_matches_build_config,
    load_pair_manifest,
)

# --- Phase 0B fixed hyperparams (Codex spec) ---
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_SEQ_LEN_QUERY = 384  # justified by Phase 0A census P90=323 tok
MAX_SEQ_LEN_TRACK = 256  # track text rarely exceeds 200 chars
BATCH_SIZE_DEFAULT = 32
LR = 1e-5
EPOCHS = 1
TAU = 0.05
K_RANDOM_NEGS = 64
HARD_NEG_WEIGHT = 0.0  # first run — DO NOT change
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4
SEED = 0
LOG_EVERY = 50

DEFAULT_OUTPUT_DIR = REPO / "cache" / "r84" / "phase0b_fold0"


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def log_truncation_rates(queries: list[str], tokenizer) -> dict:
    """Log truncation rate at 256 vs 384 for queries (Codex requirement)."""
    sample = queries if len(queries) <= 5000 else random.Random(0).sample(queries, 5000)
    lens = [len(tokenizer.encode(q, add_special_tokens=True)) for q in sample]
    n = len(lens)
    return {
        "n_sampled": n,
        "median_tokens": sorted(lens)[n // 2],
        "p90_tokens": sorted(lens)[int(n * 0.9)],
        "p99_tokens": sorted(lens)[int(n * 0.99)],
        "max_tokens": max(lens),
        "trunc_rate_at_256": sum(1 for x in lens if x > 256) / n,
        "trunc_rate_at_384": sum(1 for x in lens if x > 384) / n,
        "trunc_rate_at_512": sum(1 for x in lens if x > 512) / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0, choices=range(5),
                   help="held-out fold index (default 0)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                   help=f"batch size (default {BATCH_SIZE_DEFAULT}; reduce on OOM)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="grad accumulation steps to preserve effective batch on OOM fallback")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true",
                   help="disable bf16 autocast (debugging only — slows training ~2x on A100)")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per epoch for smoke testing (default: all)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    log_path = args.output_dir / "training_log.txt"
    summary_path = args.output_dir / "training_summary.json"

    def log(msg: str, also_print: bool = True):
        line = f"{ts()} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if also_print:
            print(line, flush=True)

    log(f"R84 Phase 0B start. fold={args.fold} output={args.output_dir}")
    log(f"hyperparams: model={MODEL_NAME} max_seq_q={MAX_SEQ_LEN_QUERY} "
        f"max_seq_t={MAX_SEQ_LEN_TRACK} batch={args.batch_size} grad_accum={args.grad_accum} "
        f"lr={LR} epochs={EPOCHS} tau={TAU} k_neg={K_RANDOM_NEGS} "
        f"hard_neg_weight={HARD_NEG_WEIGHT} bf16={not args.no_bf16}")

    # --- Manifest guard ---
    log("Verifying Phase 0A manifest...")
    cfg = assert_manifest_matches_build_config(
        manifest_path=MANIFEST_PATH,
        sha256_path=SHA256_PATH,
        build_config_path=BUILD_CONFIG_PATH,
    )
    log(f"  manifest fingerprint = {cfg['fingerprint'][:16]}")
    log(f"  manifest n_rows = {cfg.get('manifest_n_rows', '?')}")

    # --- Load manifest, filter to fold's training rows ---
    log("Loading manifest...")
    table = load_pair_manifest(MANIFEST_PATH)
    fold_idx_arr = table.column("fold_idx").to_numpy()
    import numpy as np  # type: ignore[reportMissingImports]
    mask = (fold_idx_arr == -1) | (fold_idx_arr == args.fold)
    # Per plan: dev pair fold_idx encodes the held-out fold for which this pair is training.
    # So fold_idx == args.fold selects the right dev pairs (those NOT from held-out fold).
    selected_idx = np.where(mask)[0]
    log(f"  fold {args.fold}: {mask.sum()} training pairs "
        f"(train_split={int((fold_idx_arr == -1).sum())} + "
        f"dev_fold={int((fold_idx_arr == args.fold).sum())})")

    queries = table.column("query_structured").to_pylist()
    gts = table.column("gt_track_id").to_pylist()
    queries = [queries[i] for i in selected_idx]
    gts = [gts[i] for i in selected_idx]
    log(f"  loaded {len(queries)} pairs")

    # --- Load catalog (for negatives + track text builder) ---
    log("Loading catalog...")
    meta, all_track_ids = load_catalog()
    catalog_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
    log(f"  {len(meta)} tracks in catalog")

    # Build track texts for the GT tracks once
    log("Pre-building GT track texts...")
    gt_texts = [build_track_text(g, meta) for g in gts]

    # --- Load model ---
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t  # type: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    log(f"Loading SentenceTransformer({MODEL_NAME}) on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    model.max_seq_length = MAX_SEQ_LEN_QUERY  # set tokenizer cap

    tokenizer = model.tokenizer
    log(f"  hidden_dim = {model.get_sentence_embedding_dimension()}")
    log(f"  max_position_embeddings = {model[0].auto_model.config.max_position_embeddings}")

    # --- Log truncation rates ---
    log("Logging query truncation rates (Codex requirement)...")
    trunc_train = log_truncation_rates(queries, tokenizer)
    log(f"  train queries: median={trunc_train['median_tokens']} P90={trunc_train['p90_tokens']} "
        f"P99={trunc_train['p99_tokens']} max={trunc_train['max_tokens']}")
    log(f"  truncation @256={trunc_train['trunc_rate_at_256']:.3f}  "
        f"@384={trunc_train['trunc_rate_at_384']:.3f}  "
        f"@512={trunc_train['trunc_rate_at_512']:.3f}")
    trunc_track = log_truncation_rates(gt_texts, tokenizer)
    log(f"  gt track texts: median={trunc_track['median_tokens']} P90={trunc_track['p90_tokens']} "
        f"max={trunc_track['max_tokens']}  trunc @256={trunc_track['trunc_rate_at_256']:.3f}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    log(f"autocast dtype = {autocast_dtype}")

    # --- Random catalog neg pool (fresh sampling each batch via indices) ---
    n_catalog = len(all_track_ids)
    rng = np.random.RandomState(SEED + 1)

    # Build a cached set of pre-tokenized track texts for fast negative encoding
    # (Recompute texts on-the-fly per batch is fine — strings are cheap)

    n_total = len(queries)
    batches_per_epoch = (n_total + args.batch_size - 1) // args.batch_size
    if args.max_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    log(f"batches_per_epoch={batches_per_epoch}  total_steps={batches_per_epoch * EPOCHS}")

    # Encode helper — mirrors R54's encode_with_grad pattern (uses sentence-transformers'
    # configured pooling via model.forward, then L2 normalize). Variable max_seq lets
    # us use 384 for queries and 256 for tracks.
    def encode_strings(strings: list[str], max_seq: int) -> "torch.Tensor":
        encoded = tokenizer(
            strings, padding=True, truncation=True,
            max_length=max_seq, return_tensors="pt",
        )
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        out = model.forward(encoded)
        emb = out["sentence_embedding"]
        return F_t.normalize(emb, dim=-1)

    # --- Train ---
    log("Starting training...")
    model.train()
    step = 0
    accum_step = 0
    running_loss = 0.0
    losses_window: list[float] = []
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        perm = np.random.RandomState(SEED + epoch).permutation(n_total)
        for batch_i in range(batches_per_epoch):
            i0, i1 = batch_i * args.batch_size, (batch_i + 1) * args.batch_size
            batch_idx = perm[i0:i1]
            batch_queries = [queries[i] for i in batch_idx]
            batch_gt_texts = [gt_texts[i] for i in batch_idx]
            batch_gt_ids = [gts[i] for i in batch_idx]
            batch_gt_set = set(batch_gt_ids)

            # Sample K_RANDOM_NEGS random catalog tracks, excluding all in-batch GTs
            neg_track_ids = []
            attempts = 0
            while len(neg_track_ids) < K_RANDOM_NEGS and attempts < K_RANDOM_NEGS * 4:
                cand = all_track_ids[rng.randint(0, n_catalog)]
                if cand in batch_gt_set or cand in neg_track_ids:
                    attempts += 1
                    continue
                neg_track_ids.append(cand)
            neg_texts = [build_track_text(t, meta) for t in neg_track_ids]

            # Forward: query embs (seq 384) + pos embs (seq 256) + neg embs (seq 256)
            with torch.amp.autocast(device_type="cuda" if use_bf16 else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                q_emb = encode_strings(batch_queries, MAX_SEQ_LEN_QUERY)  # [B, D]
                p_emb = encode_strings(batch_gt_texts, MAX_SEQ_LEN_TRACK)  # [B, D]
                n_emb = encode_strings(neg_texts, MAX_SEQ_LEN_TRACK)  # [K, D]
                all_cand = torch.cat([p_emb, n_emb], dim=0)  # [B+K, D]
                sim = (q_emb @ all_cand.T) / TAU  # [B, B+K]
                labels = torch.arange(len(batch_queries), device=args.device)
                loss = F_t.cross_entropy(sim, labels)
                loss = loss / max(1, args.grad_accum)
            loss.backward()
            running_loss += loss.item() * max(1, args.grad_accum)
            accum_step += 1

            if accum_step >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_step = 0
                step += 1

            losses_window.append(running_loss / max(1, args.grad_accum))
            running_loss = 0.0

            if (batch_i + 1) % LOG_EVERY == 0:
                recent = losses_window[-LOG_EVERY:]
                avg = sum(recent) / len(recent)
                elapsed = time.time() - t_start
                rate = (batch_i + 1) / elapsed
                eta_min = (batches_per_epoch - batch_i - 1) / max(rate, 1e-6) / 60
                log(f"  epoch {epoch} batch {batch_i + 1}/{batches_per_epoch} "
                    f"loss_avg={avg:.4f} rate={rate:.2f} b/s eta={eta_min:.1f} min")

            # Catastrophic divergence guard
            if losses_window[-1] > 100 or (losses_window[-1] != losses_window[-1]):  # NaN
                log(f"!! CATASTROPHIC LOSS @ batch {batch_i}: {losses_window[-1]} — aborting")
                sys.exit(1)

    elapsed = time.time() - t_start
    log(f"Training done in {elapsed / 60:.1f} min. Final loss avg = "
        f"{sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)):.4f}")

    log(f"Saving model -> {model_dir}")
    model.save(str(model_dir))

    summary = {
        "experiment": "R84 Phase 0B fold-0 training",
        "fold": args.fold,
        "model_name": MODEL_NAME,
        "n_train_pairs": n_total,
        "batches_per_epoch": batches_per_epoch,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "hyperparams": {
            "MAX_SEQ_LEN_QUERY": MAX_SEQ_LEN_QUERY,
            "MAX_SEQ_LEN_TRACK": MAX_SEQ_LEN_TRACK,
            "BATCH_SIZE": args.batch_size,
            "GRAD_ACCUM": args.grad_accum,
            "LR": LR,
            "EPOCHS": EPOCHS,
            "TAU": TAU,
            "K_RANDOM_NEGS": K_RANDOM_NEGS,
            "HARD_NEG_WEIGHT": HARD_NEG_WEIGHT,
            "bf16": use_bf16,
        },
        "training_elapsed_min": round(elapsed / 60, 2),
        "final_loss_avg_last50": round(
            sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)), 4
        ),
        "initial_loss_first50": round(
            sum(losses_window[:50]) / max(1, min(len(losses_window), 50)), 4
        ),
        "truncation_train": trunc_train,
        "truncation_gt_tracks": trunc_track,
        "manifest_sha256": cfg.get("manifest_sha256"),
        "manifest_fingerprint": cfg["fingerprint"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote summary -> {summary_path}")
    log("R84 Phase 0B training complete.")


if __name__ == "__main__":
    main()

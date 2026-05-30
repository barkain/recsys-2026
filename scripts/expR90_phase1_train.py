"""R90 Phase 1 Variant A — clean 2-epoch BGE-large retrain (fold-0 only).

Tests the "more training" hypothesis without checkpoint-restore dependency.
Identical to R84 Phase 0B training except EPOCHS=2 (constant LR=1e-5, no schedule)
so the only experimental variable is total training steps.

Consumes the SAME pair manifest as R84 Phase 0B (cache/r84/phase0a/pair_manifest.parquet)
via load_pair_manifest() so the data side is bit-for-bit identical.

Dry-run mode (--dry-run): loads manifest + first batch, prints diagnostics, exits
before any model load or optimizer step. Runs on CPU/Mac.

Designed for Colab A100 with bf16 autocast.
"""

from __future__ import annotations

import argparse
import json
import os
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

# --- R90 Phase 1 Variant A hyperparams ---
# Differs from R84 Phase 0B only by EPOCHS (1 -> 2). Constant LR; no schedule.
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_SEQ_LEN_QUERY = 384
MAX_SEQ_LEN_TRACK = 256
BATCH_SIZE_DEFAULT = 32
LR = 1e-5
EPOCHS = 2  # vs R84's 1
TAU = 0.05
K_RANDOM_NEGS = 64
HARD_NEG_WEIGHT = 0.0
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4
SEED = 0
LOG_EVERY = 50

DEFAULT_OUTPUT_DIR = REPO / "cache" / "r90" / "phase1_fold0_varA"


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def log_truncation_rates(queries: list[str], tokenizer) -> dict:
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
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per epoch for smoke testing")
    p.add_argument("--dry-run", action="store_true",
                   help="Load manifest + describe first batch, then exit before model load. "
                        "Runs CPU-only.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    log_path = args.output_dir / "training_log.txt"
    summary_path = args.output_dir / "training_summary.json"

    def log(msg: str):
        line = f"{ts()} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"R90 Phase 1 Variant A start. fold={args.fold} output={args.output_dir} "
        f"dry_run={args.dry_run}")
    log(f"hyperparams: model={MODEL_NAME} max_seq_q={MAX_SEQ_LEN_QUERY} "
        f"max_seq_t={MAX_SEQ_LEN_TRACK} batch={args.batch_size} grad_accum={args.grad_accum} "
        f"lr={LR} epochs={EPOCHS} tau={TAU} k_neg={K_RANDOM_NEGS} "
        f"hard_neg_weight={HARD_NEG_WEIGHT} bf16={not args.no_bf16}")

    # --- Manifest guard (same path as R84 phase0b) ---
    log("Verifying Phase 0A manifest (reuses R84's authoritative manifest)...")
    cfg = assert_manifest_matches_build_config(
        manifest_path=MANIFEST_PATH,
        sha256_path=SHA256_PATH,
        build_config_path=BUILD_CONFIG_PATH,
    )
    log(f"  manifest fingerprint = {cfg['fingerprint'][:16]}")
    log(f"  manifest n_rows = {cfg.get('manifest_n_rows', '?')}")

    log("Loading manifest...")
    table = load_pair_manifest(MANIFEST_PATH)
    fold_idx_arr = table.column("fold_idx").to_numpy()
    import numpy as np  # type: ignore[reportMissingImports]
    mask = (fold_idx_arr == -1) | (fold_idx_arr == args.fold)
    selected_idx = np.where(mask)[0]
    log(f"  fold {args.fold}: {mask.sum()} training pairs "
        f"(train_split={int((fold_idx_arr == -1).sum())} + "
        f"dev_fold={int((fold_idx_arr == args.fold).sum())})")

    queries = table.column("query_structured").to_pylist()
    gts = table.column("gt_track_id").to_pylist()
    queries = [queries[i] for i in selected_idx]
    gts = [gts[i] for i in selected_idx]
    log(f"  loaded {len(queries)} pairs")

    log("Loading catalog...")
    meta, all_track_ids = load_catalog()
    log(f"  {len(meta)} tracks in catalog")

    log("Pre-building GT track texts...")
    gt_texts = [build_track_text(g, meta) for g in gts]

    # ---- DRY-RUN PATH (Mac-compatible; exits before model load) ----
    if args.dry_run:
        log("=== DRY-RUN: describing first batch and exiting ===")
        perm = np.random.RandomState(SEED).permutation(len(queries))
        batch_idx = perm[:args.batch_size]
        log(f"  first batch (size={args.batch_size}):")
        log(f"    query[0] (first 200 chars): {queries[batch_idx[0]][:200]!r}")
        log(f"    gt_track[0]: {gts[batch_idx[0]]}")
        log(f"    gt_text[0] (first 200 chars): {gt_texts[batch_idx[0]][:200]!r}")
        log(f"    all 32 gt_track_ids unique: {len(set(gts[i] for i in batch_idx))}")

        # Negative sampling sanity
        rng = np.random.RandomState(SEED + 1)
        batch_gt_set = set(gts[i] for i in batch_idx)
        neg_track_ids = []
        attempts = 0
        while len(neg_track_ids) < K_RANDOM_NEGS and attempts < K_RANDOM_NEGS * 4:
            cand = all_track_ids[rng.randint(0, len(all_track_ids))]
            if cand in batch_gt_set or cand in neg_track_ids:
                attempts += 1
                continue
            neg_track_ids.append(cand)
        log(f"    sampled {len(neg_track_ids)} random catalog negs in {attempts + len(neg_track_ids)} attempts")

        # Token length sanity (no model needed; use base BGE tokenizer)
        log("  sampling tokenization on first 100 queries...")
        from transformers import AutoTokenizer  # type: ignore[reportMissingImports]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        sample_lens = [len(tokenizer.encode(q, add_special_tokens=True))
                       for q in queries[:100]]
        log(f"    median len: {sorted(sample_lens)[50]} P90: {sorted(sample_lens)[90]} "
            f"max: {max(sample_lens)}")

        # Save a small dry-run summary so dryrun is verifiable from disk
        dry_summary = {
            "experiment": "R90 Phase 1 Variant A dry-run",
            "fold": args.fold,
            "model_name": MODEL_NAME,
            "n_train_pairs": len(queries),
            "epochs": EPOCHS,
            "batch_size": args.batch_size,
            "first_batch_first_gt": gts[batch_idx[0]],
            "first_batch_queries_unique": len(set(queries[i] for i in batch_idx)),
            "n_random_negs_sampled": len(neg_track_ids),
            "sample_query_token_lens": {
                "median": sorted(sample_lens)[50],
                "p90": sorted(sample_lens)[90],
                "max": max(sample_lens),
            },
            "manifest_sha256": cfg.get("manifest_sha256"),
            "manifest_fingerprint": cfg["fingerprint"],
            "created_at": datetime.now().isoformat(),
        }
        with open(args.output_dir / "dry_run_summary.json", "w") as f:
            json.dump(dry_summary, f, indent=2)
        log(f"  wrote dry_run_summary.json")
        log("DRY-RUN PASS — no model loaded, no optimizer state created.")
        return

    # --- Real training path (A100) ---
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t  # type: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    log(f"Loading SentenceTransformer({MODEL_NAME}) on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    model.max_seq_length = MAX_SEQ_LEN_QUERY
    tokenizer = model.tokenizer
    log(f"  hidden_dim = {model.get_sentence_embedding_dimension()}")

    log("Logging query truncation rates...")
    trunc_train = log_truncation_rates(queries, tokenizer)
    log(f"  train queries: median={trunc_train['median_tokens']} "
        f"P90={trunc_train['p90_tokens']} P99={trunc_train['p99_tokens']} "
        f"max={trunc_train['max_tokens']}")
    log(f"  truncation @256={trunc_train['trunc_rate_at_256']:.3f}  "
        f"@384={trunc_train['trunc_rate_at_384']:.3f}  "
        f"@512={trunc_train['trunc_rate_at_512']:.3f}")
    trunc_track = log_truncation_rates(gt_texts, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    log(f"autocast dtype = {autocast_dtype}")

    n_catalog = len(all_track_ids)
    rng = np.random.RandomState(SEED + 1)

    n_total = len(queries)
    batches_per_epoch = (n_total + args.batch_size - 1) // args.batch_size
    if args.max_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_batches)
    log(f"batches_per_epoch={batches_per_epoch}  total_steps={batches_per_epoch * EPOCHS}")

    def encode_strings(strings: list[str], max_seq: int) -> "torch.Tensor":
        encoded = tokenizer(strings, padding=True, truncation=True,
                            max_length=max_seq, return_tensors="pt")
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

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

            neg_track_ids = []
            attempts = 0
            while len(neg_track_ids) < K_RANDOM_NEGS and attempts < K_RANDOM_NEGS * 4:
                cand = all_track_ids[rng.randint(0, n_catalog)]
                if cand in batch_gt_set or cand in neg_track_ids:
                    attempts += 1
                    continue
                neg_track_ids.append(cand)
            neg_texts = [build_track_text(t, meta) for t in neg_track_ids]

            with torch.amp.autocast(device_type="cuda" if use_bf16 else "cpu",
                                    dtype=autocast_dtype, enabled=use_bf16):
                q_emb = encode_strings(batch_queries, MAX_SEQ_LEN_QUERY)
                p_emb = encode_strings(batch_gt_texts, MAX_SEQ_LEN_TRACK)
                n_emb = encode_strings(neg_texts, MAX_SEQ_LEN_TRACK)
                all_cand = torch.cat([p_emb, n_emb], dim=0)
                sim = (q_emb @ all_cand.T) / TAU
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
                rate = (batch_i + 1 + epoch * batches_per_epoch) / max(elapsed, 1e-6)
                remaining = (EPOCHS - epoch) * batches_per_epoch - (batch_i + 1)
                eta_min = remaining / max(rate, 1e-6) / 60
                log(f"  epoch {epoch} batch {batch_i + 1}/{batches_per_epoch} "
                    f"loss_avg={avg:.4f} rate={rate:.2f} b/s eta={eta_min:.1f} min")

            if losses_window[-1] > 100 or (losses_window[-1] != losses_window[-1]):
                log(f"!! CATASTROPHIC LOSS @ batch {batch_i}: {losses_window[-1]} — aborting")
                sys.exit(1)

    elapsed = time.time() - t_start
    log(f"Training done in {elapsed / 60:.1f} min. "
        f"Final loss avg = {sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)):.4f}")

    log(f"Saving model -> {model_dir}")
    model.save(str(model_dir))

    summary = {
        "experiment": "R90 Phase 1 Variant A: 2-epoch HF retrain (fold-0)",
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
    log("R90 Phase 1 Variant A training complete.")


if __name__ == "__main__":
    main()

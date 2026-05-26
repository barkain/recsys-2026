"""R90 Phase 1 Variant A — fold-0 eval (Colab A100 or local GPU).

Mirrors expR84_phase0b_eval.py: encodes fold-0 dev queries + full 47K catalog
with the trained R90 Variant A model, writes top-300 OOF lists + LR features
+ cosine cache to cache/r90/phase1_fold0_varA/.

Output schema is bit-for-bit identical to R84's so the existing
expR84_phase0b_compare.py harness can be repointed without code changes.

Dry-run mode (--dry-run): validates model-dir contract OR a missing-but-expected
shape, lists what would be written, exits before any encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_full5fold_train import (  # noqa: E402
    R12_CACHE,
    build_query_structured_from_dev,
    build_track_text,
    load_catalog,
)
from scripts.expR84_phase0a_census import grouped_session_folds  # noqa: E402

MAX_SEQ_LEN_QUERY = 384
MAX_SEQ_LEN_TRACK = 256
TOP_K = 300
DEFAULT_OUTPUT_DIR = REPO / "cache" / "r90" / "phase1_fold0_varA"
DEFAULT_MODEL_DIR = DEFAULT_OUTPUT_DIR / "model"


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="Path to trained R90 model (sentence-transformers .save() dir). "
                        f"Default: {DEFAULT_MODEL_DIR}")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs + model-dir contract, write dry_run_summary, exit.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "eval_log.txt"

    def log(msg: str):
        line = f"{ts()} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"R90 Phase 1 Variant A eval start. fold={args.fold} model_dir={args.model_dir} "
        f"dry_run={args.dry_run}")

    # --- Load dev payload + folds ---
    log("Loading dev payload...")
    if not R12_CACHE.exists():
        log(f"FATAL: R12 cache missing at {R12_CACHE}")
        sys.exit(1)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0)
    fold0_val_idx = folds[args.fold].tolist()
    val_cases = [cases[i] for i in fold0_val_idx]
    log(f"  fold-{args.fold} val cases: {len(val_cases)}")

    log("Loading catalog...")
    meta, all_track_ids = load_catalog()
    log(f"  {len(meta)} tracks")

    # --- DRY-RUN PATH ---
    if args.dry_run:
        log("=== DRY-RUN: validating contracts and exiting ===")
        # Contract 1: val set is exactly 1600 fold-0 cases
        n_val = len(val_cases)
        log(f"  val cases: {n_val} (expected 1600)")
        # Contract 2: catalog is 47071 tracks
        n_cat = len(all_track_ids)
        log(f"  catalog tracks: {n_cat} (expected 47071)")
        # Contract 3: every val case has a GT track in the catalog (sanity)
        catalog_set = set(all_track_ids)
        gt_in_catalog = sum(1 for c in val_cases if c["gt"] in catalog_set)
        log(f"  val GT tracks in catalog: {gt_in_catalog} / {n_val}")

        # Contract 4: model_dir contract — must contain config.json + model weights
        model_dir = args.model_dir
        has_model_dir = model_dir.exists()
        log(f"  model_dir present: {has_model_dir} ({model_dir})")
        required_model_files = ["config.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        config_ok = (model_dir / "config.json").exists() if has_model_dir else False
        any_weights = (any((model_dir / w).exists() for w in weight_files)
                       if has_model_dir else False)
        log(f"  config.json: {config_ok}  any weight file: {any_weights}")
        if has_model_dir and not (config_ok and any_weights):
            log("  WARNING: model_dir incomplete; real eval will fail.")
        if not has_model_dir:
            log("  NOTE: model_dir not present — expected before training runs on Colab. "
                "Real eval will be launched after training completes.")

        # Contract 5: build first query to verify the build path works (no tokenizer needed)
        sample_query = build_query_structured_from_dev(val_cases[0], meta)
        sample_track_text = build_track_text(val_cases[0]["gt"], meta)
        log(f"  sample query (first 200 chars): {sample_query[:200]!r}")
        log(f"  sample track text (first 200 chars): {sample_track_text[:200]!r}")

        dry_summary = {
            "experiment": "R90 Phase 1 Variant A eval dry-run",
            "fold": args.fold,
            "n_val_cases": n_val,
            "n_catalog_tracks": n_cat,
            "gt_tracks_in_catalog": gt_in_catalog,
            "model_dir": str(model_dir),
            "model_dir_present": has_model_dir,
            "config_json_present": config_ok,
            "any_weight_file_present": any_weights,
            "expected_outputs": [
                "oof_r84_lists.json",  # NOTE: keeps R84 filename for compare reuse
                "r84_features.npy",
                "r84_catalog_embs.fp16.npy",
                "r84_dev_query_embs.fp16.npy",
                "eval_summary.json",
            ],
            "created_at": datetime.now().isoformat(),
        }
        with open(args.output_dir / "dry_run_summary.json", "w") as f:
            json.dump(dry_summary, f, indent=2)
        log(f"  wrote dry_run_summary.json")
        log("DRY-RUN PASS — eval will produce R84-schema outputs in this directory.")
        return

    # --- Real eval path ---
    import numpy as np  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t  # type: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    log(f"Loading model from {args.model_dir}")
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.eval()
    tokenizer = model.tokenizer
    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")
    log(f"  bf16={use_bf16}")

    def encode_batched(strings: list[str], max_seq: int, bs: int) -> "np.ndarray":
        out_list = []
        with torch.no_grad():
            for i in range(0, len(strings), bs):
                chunk = strings[i:i + bs]
                with torch.amp.autocast(
                    device_type="cuda" if use_bf16 else "cpu",
                    dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    enabled=use_bf16,
                ):
                    enc = tokenizer(chunk, padding=True, truncation=True,
                                    max_length=max_seq, return_tensors="pt")
                    enc = {k: v.to(args.device) for k, v in enc.items()}
                    o = model.forward(enc)
                    e = F_t.normalize(o["sentence_embedding"], dim=-1)
                out_list.append(e.float().cpu().numpy())
        return np.concatenate(out_list, axis=0)

    log("Logging eval-query truncation...")
    val_queries = [build_query_structured_from_dev(c, meta) for c in val_cases]
    val_lens = [len(tokenizer.encode(q, add_special_tokens=True)) for q in val_queries]
    val_lens_sorted = sorted(val_lens)
    n = len(val_lens_sorted)
    trunc_eval = {
        "n": n,
        "median": val_lens_sorted[n // 2],
        "p90": val_lens_sorted[int(n * 0.9)],
        "p99": val_lens_sorted[int(n * 0.99)],
        "max": max(val_lens),
        "trunc_rate_at_256": sum(1 for x in val_lens if x > 256) / n,
        "trunc_rate_at_384": sum(1 for x in val_lens if x > 384) / n,
        "trunc_rate_at_512": sum(1 for x in val_lens if x > 512) / n,
    }
    log(f"  dev queries: P90={trunc_eval['p90']} max={trunc_eval['max']} "
        f"trunc@384={trunc_eval['trunc_rate_at_384']:.3f}")

    log(f"Encoding {len(all_track_ids)} tracks (max_seq={MAX_SEQ_LEN_TRACK}, bs={args.batch_size})...")
    t0 = time.time()
    track_texts = [build_track_text(t, meta) for t in all_track_ids]
    catalog_embs = encode_batched(track_texts, MAX_SEQ_LEN_TRACK, args.batch_size)
    log(f"  catalog encoded in {(time.time() - t0):.1f} s, shape={catalog_embs.shape}")
    np.save(args.output_dir / "r84_catalog_embs.fp16.npy", catalog_embs.astype(np.float16))
    log(f"  saved -> r84_catalog_embs.fp16.npy")

    log(f"Encoding {len(val_queries)} dev queries (max_seq={MAX_SEQ_LEN_QUERY})...")
    t0 = time.time()
    q_embs = encode_batched(val_queries, MAX_SEQ_LEN_QUERY, args.batch_size)
    log(f"  queries encoded in {(time.time() - t0):.1f} s, shape={q_embs.shape}")
    np.save(args.output_dir / "r84_dev_query_embs.fp16.npy", q_embs.astype(np.float16))

    log(f"Retrieving top-{TOP_K}...")
    t0 = time.time()
    catalog_t = torch.from_numpy(catalog_embs).to(args.device)
    oof_lists = {}
    feature_arr = np.zeros((len(val_cases), 3), dtype=np.float32)
    chunk = 32
    for i0 in range(0, len(val_queries), chunk):
        q_chunk = torch.from_numpy(q_embs[i0:i0 + chunk]).to(args.device)
        sim_chunk = (q_chunk @ catalog_t.T)
        topk_vals, topk_idx = sim_chunk.topk(TOP_K, dim=1)
        topk_vals = topk_vals.float().cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        for j in range(q_chunk.size(0)):
            case_local_idx = i0 + j
            tids = [all_track_ids[k] for k in topk_idx[j]]
            scores = topk_vals[j].tolist()
            oof_lists[fold0_val_idx[case_local_idx]] = list(zip(tids, scores))
            gt = val_cases[case_local_idx]["gt"]
            if gt in tids:
                rank = tids.index(gt) + 1
                feature_arr[case_local_idx, 0] = 1.0 / rank
                feature_arr[case_local_idx, 1] = 1.0
                feature_arr[case_local_idx, 2] = scores[tids.index(gt)]
            else:
                feature_arr[case_local_idx] = 0.0
    log(f"  retrieval done in {(time.time() - t0):.1f} s")

    n_h7 = sum(1 for c in val_cases if c.get("n_prior_music") == 7)
    h7_idx_local = [i for i, c in enumerate(val_cases) if c.get("n_prior_music") == 7]
    hit_at_20_all = sum(
        1 for i, c in enumerate(val_cases)
        if c["gt"] in [t for t, _ in oof_lists[fold0_val_idx[i]][:20]]
    ) / len(val_cases)
    hit_at_300_all = sum(
        1 for i, c in enumerate(val_cases)
        if c["gt"] in [t for t, _ in oof_lists[fold0_val_idx[i]][:300]]
    ) / len(val_cases)
    hit_at_20_h7 = sum(
        1 for i in h7_idx_local
        if val_cases[i]["gt"] in [t for t, _ in oof_lists[fold0_val_idx[i]][:20]]
    ) / max(1, n_h7)
    hit_at_30_h7 = sum(
        1 for i in h7_idx_local
        if val_cases[i]["gt"] in [t for t, _ in oof_lists[fold0_val_idx[i]][:30]]
    ) / max(1, n_h7)
    hit_at_300_h7 = sum(
        1 for i in h7_idx_local
        if val_cases[i]["gt"] in [t for t, _ in oof_lists[fold0_val_idx[i]][:300]]
    ) / max(1, n_h7)

    log(f"R90 source-alone fold-0 metrics (vs R84 phase0b_fold0 hit@20_h7=0.22):")
    log(f"  hit@20 all = {hit_at_20_all:.4f}")
    log(f"  hit@20 h7 = {hit_at_20_h7:.4f}")
    log(f"  hit@30 h7 = {hit_at_30_h7:.4f}")
    log(f"  hit@300 h7 = {hit_at_300_h7:.4f}")

    out_lists = {int(k): [[t, float(s)] for t, s in v] for k, v in oof_lists.items()}
    with open(args.output_dir / "oof_r84_lists.json", "w") as f:
        json.dump(out_lists, f)
    np.save(args.output_dir / "r84_features.npy", feature_arr)

    summary = {
        "experiment": "R90 Phase 1 Variant A fold-0 eval (R84-schema output)",
        "fold": args.fold,
        "n_val_cases": len(val_cases),
        "n_h7": n_h7,
        "top_k": TOP_K,
        "source_alone_metrics": {
            "hit_at_20_all": round(hit_at_20_all, 4),
            "hit_at_300_all": round(hit_at_300_all, 4),
            "hit_at_20_h7": round(hit_at_20_h7, 4),
            "hit_at_30_h7": round(hit_at_30_h7, 4),
            "hit_at_300_h7": round(hit_at_300_h7, 4),
        },
        "truncation_eval_queries": trunc_eval,
        "feature_schema": ["r84_rank_inv", "r84_presence", "r84_cosine"],
        "lists_path": "oof_r84_lists.json",
        "features_path": "r84_features.npy",
        "created_at": datetime.now().isoformat(),
    }
    with open(args.output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("R90 Phase 1 Variant A eval complete.")


if __name__ == "__main__":
    main()

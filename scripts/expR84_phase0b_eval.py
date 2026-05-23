"""R84 Phase 0B — fold-0 eval (Colab A100).

Encodes fold-0 dev queries + full 47K catalog with the trained R84 model,
produces top-300 lists per dev case + LR-style features (r84_rank_inv,
r84_presence, r84_cosine), and a per-case raw cosine score for downstream
RRF/LR-conversion work on the Mac.

Outputs (under output_dir):
- oof_r84_lists.json        : {case_idx: [(tid, cosine_score), ...]} top-300 per case
- r84_features.npy          : (1600, 3) float32 — rank_inv, presence, cosine
- r84_catalog_embs.fp16.npy : (47071, 1024) fp16 — for sibling-LR retrain
- r84_dev_query_embs.fp16.npy : (1600, 1024) fp16
- eval_summary.json         : metric snapshot (no R54c comparison — that's Mac-side)
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
from scripts.expS2_lambdarank_grouped import grouped_session_folds  # noqa: E402

MAX_SEQ_LEN_QUERY = 384
MAX_SEQ_LEN_TRACK = 256
TOP_K = 300
DEFAULT_OUTPUT_DIR = REPO / "cache" / "r84" / "phase0b_fold0"


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to trained R84 model (sentence-transformers .save() dir)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Encoding batch size for catalog/queries")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "eval_log.txt"

    def log(msg: str):
        line = f"{ts()} {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    log(f"R84 Phase 0B eval start. fold={args.fold} model_dir={args.model_dir}")

    # --- Load dev payload + folds ---
    log("Loading dev payload...")
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
    catalog_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
    log(f"  {len(meta)} tracks")

    # --- Load trained model ---
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

    # --- Truncation log on dev queries ---
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
    log(f"  dev queries: P90={trunc_eval['p90']} P99={trunc_eval['p99']} "
        f"max={trunc_eval['max']} trunc@256={trunc_eval['trunc_rate_at_256']:.3f} "
        f"trunc@384={trunc_eval['trunc_rate_at_384']:.3f}")

    # --- Encode catalog ---
    log(f"Encoding {len(all_track_ids)} tracks (max_seq={MAX_SEQ_LEN_TRACK}, bs={args.batch_size})...")
    t0 = time.time()
    track_texts = [build_track_text(t, meta) for t in all_track_ids]
    catalog_embs = encode_batched(track_texts, MAX_SEQ_LEN_TRACK, args.batch_size)
    log(f"  catalog encoded in {(time.time() - t0):.1f} s, shape={catalog_embs.shape}, "
        f"dtype={catalog_embs.dtype}")
    np.save(args.output_dir / "r84_catalog_embs.fp16.npy", catalog_embs.astype(np.float16))
    log(f"  saved -> r84_catalog_embs.fp16.npy "
        f"({(args.output_dir / 'r84_catalog_embs.fp16.npy').stat().st_size / 1e6:.1f} MB)")

    # --- Encode dev queries ---
    log(f"Encoding {len(val_queries)} dev queries (max_seq={MAX_SEQ_LEN_QUERY})...")
    t0 = time.time()
    q_embs = encode_batched(val_queries, MAX_SEQ_LEN_QUERY, args.batch_size)
    log(f"  queries encoded in {(time.time() - t0):.1f} s, shape={q_embs.shape}")
    np.save(args.output_dir / "r84_dev_query_embs.fp16.npy", q_embs.astype(np.float16))

    # --- Top-K retrieval ---
    log(f"Retrieving top-{TOP_K}...")
    t0 = time.time()
    # query_embs @ catalog_embs.T → (n_q, n_catalog). Chunk by query to keep VRAM low.
    catalog_t = torch.from_numpy(catalog_embs).to(args.device)
    oof_lists = {}
    feature_arr = np.zeros((len(val_cases), 3), dtype=np.float32)  # rank_inv, presence, cosine
    chunk = 32
    for i0 in range(0, len(val_queries), chunk):
        q_chunk = torch.from_numpy(q_embs[i0:i0 + chunk]).to(args.device)
        sim_chunk = (q_chunk @ catalog_t.T)  # (chunk, n_catalog)
        topk_vals, topk_idx = sim_chunk.topk(TOP_K, dim=1)
        topk_vals = topk_vals.float().cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        for j in range(q_chunk.size(0)):
            case_local_idx = i0 + j
            tids = [all_track_ids[k] for k in topk_idx[j]]
            scores = topk_vals[j].tolist()
            oof_lists[fold0_val_idx[case_local_idx]] = list(zip(tids, scores))
            # LR features for this case (R54 schema parallel)
            gt = val_cases[case_local_idx]["gt"]
            if gt in tids:
                rank = tids.index(gt) + 1
                feature_arr[case_local_idx, 0] = 1.0 / rank  # rank_inv
                feature_arr[case_local_idx, 1] = 1.0          # presence
                feature_arr[case_local_idx, 2] = scores[tids.index(gt)]
            else:
                feature_arr[case_local_idx] = 0.0
    log(f"  retrieval done in {(time.time() - t0):.1f} s")

    # --- Quick source-alone metrics (for fast smoke; full eval is Mac-side compare) ---
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

    log(f"R84 source-alone fold-0 metrics (quick snapshot):")
    log(f"  hit@20 all = {hit_at_20_all:.4f}")
    log(f"  hit@300 all = {hit_at_300_all:.4f}")
    log(f"  hit@20 h7 = {hit_at_20_h7:.4f}  (n_h7={n_h7})")
    log(f"  hit@30 h7 = {hit_at_30_h7:.4f}")
    log(f"  hit@300 h7 = {hit_at_300_h7:.4f}")

    # --- Persist ---
    lists_path = args.output_dir / "oof_r84_lists.json"
    # Convert numpy int64 keys to int
    out_lists = {int(k): [[t, float(s)] for t, s in v] for k, v in oof_lists.items()}
    with open(lists_path, "w") as f:
        json.dump(out_lists, f)
    log(f"Wrote oof_r84_lists.json ({lists_path.stat().st_size / 1e6:.1f} MB)")
    np.save(args.output_dir / "r84_features.npy", feature_arr)
    log(f"Wrote r84_features.npy shape={feature_arr.shape}")

    summary = {
        "experiment": "R84 Phase 0B fold-0 eval",
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
        "catalog_path": "r84_catalog_embs.fp16.npy",
        "query_path": "r84_dev_query_embs.fp16.npy",
        "lists_path": "oof_r84_lists.json",
        "features_path": "r84_features.npy",
        "created_at": datetime.now().isoformat(),
    }
    with open(args.output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("R84 Phase 0B eval complete.")


if __name__ == "__main__":
    main()

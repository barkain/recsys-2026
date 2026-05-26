"""R84 blind encoding helper — loads a trained R84 model + encodes 80 Blind-A
queries + 47K catalog + saves top-300 per blind case with cosine scores.

Designed to be called once per fold on Colab A100 after that fold's training.
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
    build_query_structured_from_dev,
    build_track_text,
    load_catalog,
)

MAX_SEQ_LEN_QUERY = 384
MAX_SEQ_LEN_TRACK = 256
TOP_K = 300

DEFAULT_BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True,
                   help="JSON file: top-300 per blind sid")
    p.add_argument("--blind-cache", type=Path, default=None,
                   help="Path to blind source_cache.pkl. "
                        "Default: BLIND_SRC env var or cache/blind_a/source_cache.pkl.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    args = p.parse_args()

    # Resolve blind cache path: CLI > env > default
    if args.blind_cache:
        blind_src = args.blind_cache
    elif os.environ.get("BLIND_SRC"):
        env_path = os.environ["BLIND_SRC"]
        blind_src = Path(env_path) if Path(env_path).is_absolute() else REPO / env_path
    else:
        blind_src = DEFAULT_BLIND_SRC

    print(f"{ts()} R84 blind encode  model={args.model_dir.name}  blind={blind_src}")
    assert blind_src.exists(), f"blind source cache missing: {blind_src}"

    # Load blind cases
    with open(blind_src, "rb") as f:
        blind = pickle.load(f)
    blind_sids = sorted(blind.keys())
    print(f"  blind cases: {len(blind_sids)}")

    # Load catalog
    print(f"{ts()} Loading catalog...")
    meta, all_track_ids = load_catalog()
    print(f"  {len(meta)} tracks")

    # Load model
    import numpy as np  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t  # type: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    print(f"{ts()} Loading model...")
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    model.eval()
    tokenizer = model.tokenizer
    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")

    def encode(strings, max_seq, bs):
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

    # Encode catalog
    print(f"{ts()} Encoding {len(all_track_ids)} tracks (max_seq={MAX_SEQ_LEN_TRACK})...")
    t0 = time.time()
    track_texts = [build_track_text(t, meta) for t in all_track_ids]
    catalog_embs = encode(track_texts, MAX_SEQ_LEN_TRACK, args.batch_size)
    print(f"  catalog encoded in {time.time() - t0:.0f}s, shape={catalog_embs.shape}")

    # Build blind queries
    print(f"{ts()} Building blind structured queries...")
    blind_queries = []
    for sid in blind_sids:
        case = blind[sid]
        # blind cases have same shape as dev: user_query, history, music_turns
        q = build_query_structured_from_dev(case, meta)
        blind_queries.append(q)
    print(f"  built {len(blind_queries)} queries")

    # Encode blind queries
    print(f"{ts()} Encoding blind queries (max_seq={MAX_SEQ_LEN_QUERY})...")
    t0 = time.time()
    q_embs = encode(blind_queries, MAX_SEQ_LEN_QUERY, args.batch_size)
    print(f"  encoded in {time.time() - t0:.0f}s, shape={q_embs.shape}")

    # Top-300 retrieval (all queries × catalog dot product)
    print(f"{ts()} Computing top-{TOP_K}...")
    t0 = time.time()
    catalog_t = torch.from_numpy(catalog_embs).to(args.device)
    out_lists = {}
    chunk = 32
    for i0 in range(0, len(blind_queries), chunk):
        q_chunk = torch.from_numpy(q_embs[i0:i0 + chunk]).to(args.device)
        sim = q_chunk @ catalog_t.T
        topk_vals, topk_idx = sim.topk(TOP_K, dim=1)
        topk_vals = topk_vals.float().cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        for j in range(q_chunk.size(0)):
            sid = blind_sids[i0 + j]
            tids = [all_track_ids[k] for k in topk_idx[j]]
            scores = topk_vals[j].tolist()
            out_lists[sid] = [[t, float(s)] for t, s in zip(tids, scores)]
    print(f"  done in {time.time() - t0:.1f}s")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out_lists, f)
    print(f"  saved -> {args.output_path} "
          f"({args.output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{ts()} Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R431 Phase 0: build official user-cf retrieval lists for LR integration.

This uses only public challenge assets:
  - TalkPlayData-Challenge-User-Embeddings: user cf-bpr vectors
  - TalkPlayData-Challenge-Track-Embeddings: track cf-bpr vectors

Output is intentionally shaped like the R103 GTE list file so we can reuse the
existing added-source OOF integration harness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss  # type: ignore[reportMissingImports]
import numpy as np

from scripts.expR180_user_embedding_audit import (
    REPO,
    auto_track_root,
    auto_user_root,
    load_cases,
    load_track_modality,
    load_user_embeddings,
)

OUT_JSON = REPO / "cache/r431_user_cf/user_cf_oof_lists.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-root", type=Path)
    ap.add_argument("--track-root", type=Path)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--search-k", type=int, default=420)
    ap.add_argument("--top-k", type=int, default=300)
    args = ap.parse_args()

    user_root = args.user_root or auto_user_root()
    track_root = args.track_root or auto_track_root()
    out_json = args.out_json if args.out_json.is_absolute() else REPO / args.out_json

    cases = load_cases()
    user_data = load_user_embeddings(user_root)
    if "cf-bpr" not in user_data:
        raise SystemExit(f"user cf-bpr column missing; columns={list(user_data)}")
    user_vectors: dict[str, np.ndarray] = user_data["cf-bpr"]["vectors"]

    track_ids, track_emb, track_valid = load_track_modality(track_root, "cf-bpr")
    valid_emb = np.ascontiguousarray(track_emb[track_valid].astype(np.float32))
    valid_ids = [tid for tid, ok in zip(track_ids, track_valid) if ok]

    faiss.omp_set_num_threads(8)
    index = faiss.IndexFlatIP(valid_emb.shape[1])
    index.add(valid_emb)

    q_cases = []
    q_vecs = []
    for case in cases:
        v = user_vectors.get(case.user_id)
        if v is None or v.shape[0] != valid_emb.shape[1]:
            continue
        q_cases.append(case)
        q_vecs.append(v.astype(np.float32))

    lists: dict[str, list[list[Any]]] = {str(case.idx): [] for case in cases}
    if q_vecs:
        scores, nn = index.search(
            np.ascontiguousarray(np.vstack(q_vecs).astype(np.float32)),
            args.search_k,
        )
        for case, score_row, idx_row in zip(q_cases, scores, nn):
            played = set(case.played)
            row: list[list[Any]] = []
            seen: set[str] = set()
            for score, j in zip(score_row, idx_row):
                if j < 0:
                    continue
                tid = valid_ids[int(j)]
                if tid in played or tid in seen:
                    continue
                seen.add(tid)
                row.append([tid, float(score)])
                if len(row) >= args.top_k:
                    break
            lists[str(case.idx)] = row

    covered = sum(1 for row in lists.values() if row)
    out = {
        "method": "R431 official user cf-bpr -> track cf-bpr retrieval",
        "user_root": str(user_root),
        "track_root": str(track_root),
        "search_k": args.search_k,
        "top_k": args.top_k,
        "n_cases": len(cases),
        "n_query_cases": len(q_cases),
        "coverage": covered / len(cases) if cases else 0.0,
        "lists": lists,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"cases={len(cases)} query_cases={len(q_cases)} coverage={out['coverage']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R180 — official User-Embeddings recall audit.

This is the concrete test for the one official signal that earlier nDCG
campaigns may not have exercised directly: user-level embeddings.

The script is intentionally network-free. It reads cached HF Arrow/Parquet
files for `TalkPlayData-Challenge-User-Embeddings` plus the already-cached
official track embeddings, then measures whether user vectors retrieve the
hidden GT track at usable ranks or add GTs absent from the current source union.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parent.parent
PAYLOAD = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR180_user_embedding_audit.json"

TRACK_MODALITIES = [
    "cf-bpr",
    "audio-laion_clap",
    "image-siglip2",
    "attributes-qwen3_embedding_0.6b",
    "lyrics-qwen3_embedding_0.6b",
    "metadata-qwen3_embedding_0.6b",
]


@dataclass
class Case:
    idx: int
    user_id: str
    gt: str
    played: list[str]
    n_prior: int
    union300: set[str]
    same_artist: bool


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_table(path: Path) -> pa.Table:
    if path.suffix == ".parquet":
        return pq.read_table(path)
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all()


def candidate_roots(*dataset_fragments: str) -> list[Path]:
    roots: list[Path] = []
    for base in [
        REPO / ".hf_cache",
        Path.home() / ".cache" / "huggingface",
        REPO,
    ]:
        if base.exists():
            roots.append(base)
    out: list[Path] = []
    seen: set[Path] = set()
    fragments = [s.lower() for s in dataset_fragments]
    for root in roots:
        for path in root.rglob("*"):
            low = str(path).lower()
            if path.is_dir() and any(fragment in low for fragment in fragments):
                resolved = path.resolve()
                if resolved not in seen:
                    out.append(resolved)
                    seen.add(resolved)
    return out


def data_files(root: Path, include: str | None = None) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in {".arrow", ".parquet"} else []
    files = sorted([*root.rglob("*.arrow"), *root.rglob("*.parquet")])
    if include:
        files = [p for p in files if include in p.name or include in str(p.parent)]
    return files


def auto_track_root() -> Path:
    roots = candidate_roots("track-embeddings", "track_embeddings", "track-embedding", "track_embedding")
    for root in roots:
        if data_files(root, "all_tracks") or data_files(root):
            return root
    raise SystemExit(
        "Missing cached TalkPlayData-Challenge-Track-Embeddings files. "
        "Expected Arrow/Parquet shards for official track embeddings."
    )


def auto_user_root() -> Path:
    roots = candidate_roots("user-embeddings", "user_embeddings", "user-embedding", "user_embedding")
    for root in roots:
        if data_files(root):
            return root
    raise SystemExit(
        "Missing cached TalkPlayData-Challenge-User-Embeddings files.\n"
        "Download/cache the official HF dataset first, then rerun. Example:\n"
        "  HF_HOME=.hf_cache HF_DATASETS_CACHE=.hf_cache/datasets \\\n"
        "  uv run python - <<'PY'\n"
        "from datasets import load_dataset\n"
        "ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-User-Embeddings')\n"
        "print(ds)\n"
        "PY"
    )


def is_vector_type(t: pa.DataType) -> bool:
    return (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
    )


def vector_columns(table: pa.Table) -> list[str]:
    cols: list[str] = []
    for field in table.schema:
        if field.name in {"user_id", "track_id"}:
            continue
        if is_vector_type(field.type):
            cols.append(field.name)
    return cols


def vectors_from_array(col: pa.ChunkedArray | pa.Array, n_rows: int, column: str) -> tuple[np.ndarray, np.ndarray]:
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if pa.types.is_fixed_size_list(col.type):
        dim = int(col.type.list_size)
        flat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
        lengths = np.full(n_rows, dim, dtype=np.int64)
        offsets = np.arange(n_rows + 1, dtype=np.int64) * dim
    else:
        offsets = np.asarray(col.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        lengths = offsets[1:] - offsets[:-1]
        nonempty = lengths[lengths > 0]
        if len(nonempty) == 0:
            raise ValueError(f"{column}: no non-empty vectors")
        dim = int(np.bincount(nonempty).argmax())
        flat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)

    arr = np.zeros((n_rows, dim), dtype=np.float32)
    valid = np.zeros(n_rows, dtype=bool)
    for i, n_items in enumerate(lengths):
        if int(n_items) != dim:
            continue
        v = flat[offsets[i]:offsets[i + 1]]
        norm = float(np.linalg.norm(v))
        if not math.isfinite(norm) or norm < 1e-8:
            continue
        arr[i] = v / norm
        valid[i] = True
    return arr, valid


def load_track_modality(root: Path, column: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids_all: list[str] = []
    arrs: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    files = data_files(root, "all_tracks")
    if not files:
        # Older TalkPlayData-2 caches may not use the all_tracks split name.
        files = data_files(root)
    if not files:
        raise SystemExit(f"No track embedding Arrow/Parquet files under {root}")
    for path in files:
        table = read_table(path)
        if column not in table.column_names:
            continue
        ids = [str(x) for x in table["track_id"].to_pylist()]
        arr, valid = vectors_from_array(table[column], len(ids), column)
        ids_all.extend(ids)
        arrs.append(arr)
        valids.append(valid)
        del table, arr, valid
        gc.collect()
    if not arrs:
        raise ValueError(f"track modality not found: {column}")
    return ids_all, np.vstack(arrs), np.concatenate(valids)


def load_user_embeddings(root: Path) -> dict[str, dict[str, Any]]:
    files = data_files(root)
    if not files:
        raise SystemExit(f"No user embedding Arrow/Parquet files under {root}")

    columns_seen: dict[str, int] = {}
    values: dict[str, dict[str, np.ndarray]] = {}
    valid_by_col: dict[str, dict[str, bool]] = {}
    split_by_user: dict[str, str] = {}

    for path in files:
        table = read_table(path)
        if "user_id" not in table.column_names:
            continue
        vec_cols = vector_columns(table)
        for c in vec_cols:
            columns_seen[c] = columns_seen.get(c, 0) + 1
        users = [str(x) for x in table["user_id"].to_pylist()]
        split = path.stem
        for col in vec_cols:
            arr, valid = vectors_from_array(table[col], len(users), col)
            values.setdefault(col, {})
            valid_by_col.setdefault(col, {})
            for u, v, ok in zip(users, arr, valid):
                # Keep first valid vector for duplicates across splits/cache shards.
                if ok and u not in values[col]:
                    values[col][u] = v.astype(np.float32)
                    valid_by_col[col][u] = True
                    split_by_user.setdefault(u, split)
            del arr, valid
        del table
        gc.collect()

    out: dict[str, dict[str, Any]] = {}
    for col, by_user in values.items():
        if not by_user:
            continue
        sample = next(iter(by_user.values()))
        out[col] = {
            "dim": int(sample.shape[0]),
            "n_users": len(by_user),
            "vectors": by_user,
        }
    if not out:
        raise SystemExit(
            f"Found user files under {root}, but no vector columns. "
            f"Columns seen: {columns_seen}"
        )
    return out


def load_cases() -> list[Case]:
    payload = pickle.load(open(PAYLOAD, "rb"))
    track_artist = payload["track_artist"]
    src_names = [k for k in ("src_a", "src_b", "src_c", "src_d", "src_f", "src_g") if k in payload]
    out: list[Case] = []
    for i, c in enumerate(payload["cases"]):
        gt = str(c["gt"])
        played = [str(t) for t in (c.get("music_turns") or [])]
        union: set[str] = set()
        for src in src_names:
            union.update(str(t) for t in payload[src][i][:300])
        gt_artist = track_artist.get(gt)
        same_artist = bool(gt_artist and any(track_artist.get(t) == gt_artist for t in played))
        out.append(
            Case(
                idx=i,
                user_id=str(c["user_id"]),
                gt=gt,
                played=played,
                n_prior=int(c.get("n_prior_music") or len(played)),
                union300=union,
                same_artist=same_artist,
            )
        )
    return out


def rank_of(gt: str, ranked: list[str], max_rank: int = 300) -> int | None:
    try:
        return ranked[:max_rank].index(gt) + 1
    except ValueError:
        return None


def summarize_cases(cases: list[Case], ranks: dict[int, int | None]) -> dict[str, Any]:
    n = len(cases)
    hit = {20: 0, 30: 0, 100: 0, 300: 0}
    union_absent = {20: 0, 30: 0, 100: 0, 300: 0}
    h7 = {20: 0, 30: 0, 100: 0, 300: 0}
    diff = {20: 0, 30: 0, 100: 0, 300: 0}
    same = {20: 0, 30: 0, 100: 0, 300: 0}
    rank_vals: list[int] = []
    examples: list[dict[str, Any]] = []
    for case in cases:
        r = ranks.get(case.idx)
        if r is None:
            continue
        rank_vals.append(r)
        for k in hit:
            if r <= k:
                hit[k] += 1
                if case.gt not in case.union300:
                    union_absent[k] += 1
                if case.n_prior == 7:
                    h7[k] += 1
                if case.same_artist:
                    same[k] += 1
                else:
                    diff[k] += 1
        if case.gt not in case.union300 and r <= 300 and len(examples) < 20:
            examples.append({
                "case_idx": case.idx,
                "user_id": case.user_id,
                "rank": r,
                "n_prior": case.n_prior,
                "same_artist_history": case.same_artist,
            })
    return {
        "n_cases": n,
        "hit20": hit[20],
        "hit30": hit[30],
        "hit100": hit[100],
        "hit300": hit[300],
        "recall20": hit[20] / n if n else 0.0,
        "recall30": hit[30] / n if n else 0.0,
        "recall100": hit[100] / n if n else 0.0,
        "recall300": hit[300] / n if n else 0.0,
        "median_rank_if_hit300": float(np.median(rank_vals)) if rank_vals else None,
        "union_absent_recoveries": union_absent,
        "h7_hits": h7,
        "same_artist_hits": same,
        "diff_artist_hits": diff,
        "union_absent_examples": examples,
    }


def evaluate_pair(
    cases: list[Case],
    user_vectors: dict[str, np.ndarray],
    track_ids: list[str],
    track_emb: np.ndarray,
    track_valid: np.ndarray,
    *,
    search_k: int,
) -> dict[str, Any]:
    valid_emb = np.ascontiguousarray(track_emb[track_valid].astype(np.float32))
    valid_ids = [tid for tid, ok in zip(track_ids, track_valid) if ok]
    faiss.omp_set_num_threads(8)
    index = faiss.IndexFlatIP(valid_emb.shape[1])
    index.add(valid_emb)

    q_cases: list[Case] = []
    q_vecs: list[np.ndarray] = []
    for case in cases:
        v = user_vectors.get(case.user_id)
        if v is None or v.shape[0] != valid_emb.shape[1]:
            continue
        q_cases.append(case)
        q_vecs.append(v.astype(np.float32))

    if not q_vecs:
        return {"n_query_cases": 0, "reason": "no matching user vectors"}

    _, nn = index.search(np.ascontiguousarray(np.vstack(q_vecs).astype(np.float32)), search_k)
    ranks: dict[int, int | None] = {}
    for case, row in zip(q_cases, nn):
        played = set(case.played)
        ranked: list[str] = []
        for j in row:
            if j < 0:
                continue
            t = valid_ids[int(j)]
            if t in played:
                continue
            ranked.append(t)
            if len(ranked) >= 300:
                break
        ranks[case.idx] = rank_of(case.gt, ranked, 300)

    all_summary = summarize_cases(q_cases, ranks)
    union_miss = [c for c in q_cases if c.gt not in c.union300]
    h7_cases = [c for c in q_cases if c.n_prior == 7]
    diff_cases = [c for c in q_cases if not c.same_artist]
    return {
        "n_query_cases": len(q_cases),
        "coverage": len(q_cases) / len(cases) if cases else 0.0,
        "all": all_summary,
        "union_miss_only": summarize_cases(union_miss, ranks),
        "h7_only": summarize_cases(h7_cases, ranks),
        "diff_artist_only": summarize_cases(diff_cases, ranks),
    }


def verdict(results: dict[str, Any]) -> str:
    rows = results.get("leaderboard", [])
    if not rows:
        return "NO_USER_VECTOR_DATA"
    best = rows[0]
    if best["union_absent_top30"] >= 10 or best["all_recall20"] >= 0.50:
        return "PROCEED_R181_USER_SOURCE_INTEGRATION"
    if best["union_absent_top300"] >= 50:
        return "PROBE_SELECTIVE_USER_SOURCE"
    return "NO_USABLE_USER_EMBEDDING_SIGNAL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-root", type=Path)
    ap.add_argument("--track-root", type=Path)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--track-modalities", nargs="*", default=TRACK_MODALITIES)
    ap.add_argument("--search-k", type=int, default=420)
    args = ap.parse_args()

    out_json = args.out_json if args.out_json.is_absolute() else REPO / args.out_json
    user_root = args.user_root or auto_user_root()
    track_root = args.track_root or auto_track_root()

    log("loading dev payload cases")
    cases = load_cases()
    users = {c.user_id for c in cases}
    log(f"cases={len(cases)} unique_users={len(users)}")

    log(f"loading user embeddings from {user_root}")
    user_cols = load_user_embeddings(user_root)
    log("user vector columns: " + ", ".join(f"{k}({v['n_users']}x{v['dim']})" for k, v in user_cols.items()))

    results: dict[str, Any] = {
        "experiment": "R180 official user embedding recall audit",
        "payload": str(PAYLOAD.relative_to(REPO)),
        "user_root": str(user_root),
        "track_root": str(track_root),
        "n_cases": len(cases),
        "n_unique_case_users": len(users),
        "user_columns": {k: {"dim": v["dim"], "n_users": v["n_users"]} for k, v in user_cols.items()},
        "pairs": {},
        "leaderboard": [],
    }

    track_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for ucol, udata in user_cols.items():
        candidate_track_cols = [m for m in args.track_modalities if m == ucol]
        if not candidate_track_cols:
            # Fall back to any official track modality with matching dimension.
            for m in args.track_modalities:
                try:
                    if m not in track_cache:
                        log(f"loading track modality for dimension check: {m}")
                        track_cache[m] = load_track_modality(track_root, m)
                    if int(track_cache[m][1].shape[1]) == int(udata["dim"]):
                        candidate_track_cols.append(m)
                except Exception as exc:
                    log(f"skip track modality {m}: {exc}")
        for tcol in candidate_track_cols:
            if tcol not in track_cache:
                log(f"loading track modality {tcol}")
                track_cache[tcol] = load_track_modality(track_root, tcol)
            track_ids, track_emb, track_valid = track_cache[tcol]
            if int(track_emb.shape[1]) != int(udata["dim"]):
                continue
            key = f"{ucol}__to__{tcol}"
            log(f"evaluating {key}")
            pair_result = evaluate_pair(
                cases,
                udata["vectors"],
                track_ids,
                track_emb,
                track_valid,
                search_k=args.search_k,
            )
            results["pairs"][key] = pair_result
            all_s = pair_result.get("all", {})
            miss_s = pair_result.get("union_miss_only", {})
            results["leaderboard"].append({
                "pair": key,
                "n_query_cases": pair_result.get("n_query_cases", 0),
                "coverage": pair_result.get("coverage", 0.0),
                "all_recall20": all_s.get("recall20", 0.0),
                "all_recall30": all_s.get("recall30", 0.0),
                "all_recall300": all_s.get("recall300", 0.0),
                "union_absent_top20": miss_s.get("hit20", 0),
                "union_absent_top30": miss_s.get("hit30", 0),
                "union_absent_top100": miss_s.get("hit100", 0),
                "union_absent_top300": miss_s.get("hit300", 0),
            })
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
                f.write("\n")

    results["leaderboard"].sort(
        key=lambda r: (
            r["union_absent_top30"],
            r["union_absent_top20"],
            r["all_recall20"],
            r["union_absent_top300"],
        ),
        reverse=True,
    )
    results["verdict"] = verdict(results)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print("\n=== R180 leaderboard: official user-vector retrieval ===")
    for row in results["leaderboard"][:12]:
        print(
            f"{row['pair']:<65s} "
            f"cov={row['coverage']:.3f} all@20={row['all_recall20']:.4f} "
            f"all@30={row['all_recall30']:.4f} all@300={row['all_recall300']:.4f} "
            f"union_absent top20/30/100/300="
            f"{row['union_absent_top20']}/{row['union_absent_top30']}/"
            f"{row['union_absent_top100']}/{row['union_absent_top300']}"
        )
    print(f"VERDICT: {results['verdict']}")
    print(f"wrote {out_json.relative_to(REPO) if out_json.is_relative_to(REPO) else out_json}")


if __name__ == "__main__":
    main()

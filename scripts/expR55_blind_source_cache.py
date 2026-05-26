#!/usr/bin/env python3
# ruff: noqa: E402,T201,S301
"""One-time build of R54-independent blind source lists for fast comparisons.

Runs all the slow retrievers (A/B/C/D/F/ALS/R21, plus session metadata)
once for the 80 blind-A sessions and saves the result as a reusable cache.
Every future ranking comparison (R55, R56, R57, ...) just plugs its own
retriever output into the R54 slot and runs LR scoring — no need to re-run
BM25/CFBPR/track-sim/A-prime/ALS/R21 again.

Outputs:
  cache/blind_a/source_cache/<sid>.pkl   — per-session record (resumable; if
                                            this script is interrupted, re-running
                                            picks up where it left off)
  cache/blind_a/source_cache.pkl         — consolidated dict {sid: record}
  cache/blind_a/source_cache_manifest.json — summary + validation

Each per-session record:
  session_id, turn_number, user_query, history, music_turns,
  src_a, src_b, src_c, src_d, src_f, als_tracks, als_vec (list or None),
  r21_list, r21_rank_map

Sanity check at the end (optional but enabled by default):
  Reproduce R54b's post-LR top-20 from the cache + cached R54 ensemble lists.
  If track IDs do not match R54b submission, the cache is faulty.

Usage:
  uv run python scripts/expR55_blind_source_cache.py
  uv run python scripts/expR55_blind_source_cache.py --skip-r54b-validation
  uv run python scripts/expR55_blind_source_cache.py --force      # rebuild all
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_ALL,
    POOL_K,
    R21_MODEL,
    R21_TRACK_EMBS,
    R21_TRACK_IDS,
    RRF_K,
    SW,
    _featurize_row,
    als_retrieve_simple,
    load_track_albums,
    parse_last_turn_local,
)

# Defaults for Blind-A (preserved for backwards-compatible reruns)
DEFAULT_BLIND_NAME = "blind_a"
DEFAULT_BLIND_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-A"

# Module-level paths kept for backwards compat with imports; populated in main()
CACHE_DIR = REPO / "cache" / DEFAULT_BLIND_NAME / "source_cache"
CACHE_CONSOLIDATED = REPO / "cache" / DEFAULT_BLIND_NAME / "source_cache.pkl"
MANIFEST = REPO / "cache" / DEFAULT_BLIND_NAME / "source_cache_manifest.json"

R54_BLIND_LISTS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R54B_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r54b_aligned_submission.json"
LR_MODEL_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
POP_CACHE = REPO / "cache" / "r54_phase3_track_pop.json"
MAPS_CACHE = REPO / "cache" / "r54_phase3_payload_maps.pkl"


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def build_one_session(item, retrievers):
    """Run all R54-independent retrievers for a single blind session."""
    sid = str(item["session_id"])
    turn_num, user_query, history, music_turns = parse_last_turn_local(item)
    played_set = set(music_turns)

    (metadata, bm25, track_sim, a_prime, cfbpr,
     r21_model, r21_track_ids_list, r21_track_embs_arr,
     als_factors, als_to_idx, als_ids, query_parts) = retrievers

    # R21
    user_parts = [str(h["content"]) for h in history if h["role"] == "user"]
    user_parts.append(user_query)
    r21_query = " ".join(user_parts[-3:])
    q_emb = r21_model.encode([r21_query], normalize_embeddings=True).astype(np.float32)[0]
    r21_scores = r21_track_embs_arr @ q_emb
    for ti, tid in enumerate(r21_track_ids_list):
        if tid in played_set:
            r21_scores[ti] = -np.inf
    r21_top = np.argpartition(-r21_scores, 300)[:300]
    r21_top = r21_top[np.argsort(-r21_scores[r21_top])]
    r21_list = [r21_track_ids_list[j] for j in r21_top]

    # A/B/C/D/F
    src_a = a_prime.topn(music_turns, topn=100) if music_turns else []
    history_dicts = [{"role": h["role"], "content": h["content"]} for h in history]
    q_b = " ".join(query_parts(history_dicts, user_query, metadata, "last_music_meta"))
    q_c = " ".join(query_parts(history_dicts, user_query, metadata, "full"))
    src_b = bm25.retrieve(q_b or user_query, topk=100)
    src_c = bm25.retrieve(q_c or user_query, topk=100)
    anchor = music_turns[-1] if music_turns else None
    src_d = track_sim.track_id_to_neighbors(anchor, topk=100) if anchor else []
    src_f = cfbpr.topn(music_turns, topn=100) if music_turns else []

    # ALS
    als_tracks, als_vec = als_retrieve_simple(music_turns, als_to_idx, als_factors, als_ids)

    return {
        "session_id": sid,
        "turn_number": turn_num,
        "user_query": user_query,
        "history": history,
        "music_turns": music_turns,
        "src_a": src_a,
        "src_b": src_b,
        "src_c": src_c,
        "src_d": src_d,
        "src_f": src_f,
        "als_tracks": als_tracks,
        "als_vec": als_vec.tolist() if als_vec is not None else None,
        "r21_list": r21_list,
        "r21_rank_map": {tid: r + 1 for r, tid in enumerate(r21_list[:300])},
    }


def load_retrievers():
    """Slow imports + load. Done once."""
    from run_inference_blind_r3_det import APrimeMaxRecent  # type: ignore[reportMissingImports]
    from run_inference_blind_f1 import (  # type: ignore[reportMissingImports]
        CFBPRMaxRecent,
        load_cfbpr_index,
    )
    from offline_retrieval_sweep import (  # type: ignore[reportMissingImports]
        CachedBM25,
        load_track_metadata,
        query_parts,
    )
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever  # type: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    print(f"{ts()} Loading retrievers (BM25, track-sim, A', CFBPR, R21, ALS)...")
    metadata = load_track_metadata()
    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=3)
    cf_ids, cf_vecs, cf_idx = load_cfbpr_index()
    cfbpr = CFBPRMaxRecent(cf_ids, cf_vecs, cf_idx, recent_k=3)

    r21_model = SentenceTransformer(str(R21_MODEL), device="cpu")
    r21_track_ids_list = json.loads(Path(R21_TRACK_IDS).read_text())
    r21_track_embs_arr = np.load(R21_TRACK_EMBS)

    als_data = np.load(ALS_CACHE, allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}

    return (metadata, bm25, track_sim, a_prime, cfbpr,
            r21_model, r21_track_ids_list, r21_track_embs_arr,
            als_factors, als_to_idx, als_ids, query_parts)


def build_cache(force=False, blind_dataset=DEFAULT_BLIND_DATASET,
                cache_dir=None, cache_consolidated=None):
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]

    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    cache_consolidated = cache_consolidated if cache_consolidated is not None else CACHE_CONSOLIDATED
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        db = load_dataset(blind_dataset, split="test",
                          download_config=DownloadConfig(local_files_only=True))
    except Exception:
        print(f"{ts()} HF cache miss; downloading {blind_dataset}...")
        db = load_dataset(blind_dataset, split="test")
    blind_items = list(db)
    print(f"{ts()} {blind_dataset}: {len(blind_items)} sessions")

    # Resumable: skip sessions already cached unless --force
    pending = []
    for item in blind_items:
        sid = str(item["session_id"])
        per_sid_path = cache_dir / f"{sid}.pkl"
        if not force and per_sid_path.exists():
            continue
        pending.append(item)
    print(f"  Pending: {len(pending)}/{len(blind_items)}  "
          f"(force={force})")
    if not pending:
        print(f"  All sessions cached. Skipping retriever load.")
    else:
        retrievers = load_retrievers()
        t0 = time.time()
        for i, item in enumerate(pending):
            sid = str(item["session_id"])
            record = build_one_session(item, retrievers)
            tmp = cache_dir / f"{sid}.pkl.tmp"
            final = cache_dir / f"{sid}.pkl"
            with open(tmp, "wb") as f:
                pickle.dump(record, f)
            tmp.rename(final)
            elapsed = time.time() - t0
            sec_per = elapsed / (i + 1)
            eta = (len(pending) - i - 1) * sec_per
            print(f"  [{i + 1}/{len(pending)}] sid={sid[:8]}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    # Consolidate
    print(f"\n{ts()} Consolidating per-session caches...")
    consolidated = {}
    for item in blind_items:
        sid = str(item["session_id"])
        per_sid_path = cache_dir / f"{sid}.pkl"
        if not per_sid_path.exists():
            raise RuntimeError(f"Missing per-session cache for {sid}")
        with open(per_sid_path, "rb") as f:
            consolidated[sid] = pickle.load(f)
    with open(cache_consolidated, "wb") as f:
        pickle.dump(consolidated, f)
    print(f"  Wrote {cache_consolidated}  ({len(consolidated)} sessions)")
    return consolidated


def validate_against_r54b(cache):
    """Sanity check: reproduce R54b post-LR top-20 from cache + R54 ensemble lists."""
    import lightgbm as lgb  # type: ignore[reportMissingImports]

    print(f"\n{ts()} Sanity check: reproduce R54b top-20 from cache + R54 ensemble lists...")

    if not R54_BLIND_LISTS.exists():
        print(f"  SKIP: {R54_BLIND_LISTS} not found")
        return None

    with open(R54_BLIND_LISTS) as f:
        r54_data = json.load(f)
    r54_by_sid = r54_data["lists"]

    track_album = load_track_albums()
    valid_catalog = set(track_album.keys())
    with open(POP_CACHE) as f:
        track_pop = json.load(f)
    als_data = np.load(ALS_CACHE, allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}
    with open(MAPS_CACHE, "rb") as f:
        maps = pickle.load(f)
    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    lr_model = lgb.Booster(model_file=str(LR_MODEL_PATH))

    # Build top-20 for each session
    repro_top20 = {}
    for sid, rec in cache.items():
        if sid not in r54_by_sid:
            continue
        r54_pairs = r54_by_sid[sid]
        r54_list = [t for t, _ in r54_pairs[:300]]
        r54_score_map = {t: float(s) for t, s in r54_pairs}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_list)}
        als_vec = np.array(rec["als_vec"], dtype=np.float32) if rec["als_vec"] is not None else None

        src_lists = {
            "A": rec["src_a"], "B": rec["src_b"], "C": rec["src_c"],
            "D": rec["src_d"], "F": rec["src_f"],
            "ALS": rec["als_tracks"], "R21": rec["r21_list"], "R54": r54_list,
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        feats = _featurize_row(
            pool, src_lists, rec["r21_rank_map"], r54_rank_map, r54_score_map,
            rec["user_query"], rec["history"], rec["music_turns"], set(rec["music_turns"]),
            ta, tt, ttl, tat, tmt,
            als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
        )
        scores = lr_model.predict(feats)
        ranked = np.argsort(-scores)
        top20 = []
        seen = set()
        for j in ranked:
            tid = pool[int(j)]
            if tid in valid_catalog and tid not in seen:
                top20.append(tid)
                seen.add(tid)
                if len(top20) >= 20:
                    break
        repro_top20[sid] = top20

    # Compare to actual R54b submission
    with open(R54B_SUBMISSION) as f:
        r54b = json.load(f)
    r54b_by_sid = {r["session_id"]: r["predicted_track_ids"] for r in r54b}

    n = 0
    top1_match = 0
    top20_exact_match = 0
    top20_overlap_sum = 0
    diffs = []
    for sid, repro in repro_top20.items():
        if sid not in r54b_by_sid:
            continue
        n += 1
        actual = r54b_by_sid[sid]
        if repro[0] == actual[0]:
            top1_match += 1
        overlap = len(set(repro) & set(actual))
        top20_overlap_sum += overlap
        if repro == actual:
            top20_exact_match += 1
        elif overlap < 18:
            diffs.append({"sid": sid, "overlap": overlap,
                          "repro_top5": repro[:5], "actual_top5": actual[:5]})

    print(f"\n=== R54b reproduction sanity check (n={n}) ===")
    print(f"  top-1 match:           {top1_match}/{n} ({top1_match / n:.1%})")
    print(f"  top-20 exact match:    {top20_exact_match}/{n} ({top20_exact_match / n:.1%})")
    print(f"  top-20 avg overlap:    {top20_overlap_sum / n:.2f}/20")
    if diffs:
        print(f"  {len(diffs)} sessions with <18 overlap (showing first 5):")
        for d in diffs[:5]:
            print(f"    sid={d['sid'][:8]}  overlap={d['overlap']}")
    else:
        print(f"  All sessions reproduce >= 18/20 overlap (cache faithful)")

    cache_quality = "PASS" if top20_exact_match >= n * 0.95 else "WARN"
    return {
        "n": n,
        "top1_match": top1_match,
        "top20_exact_match": top20_exact_match,
        "top20_avg_overlap": top20_overlap_sum / n if n else 0,
        "low_overlap_examples": diffs[:10],
        "quality": cache_quality,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-name", default=DEFAULT_BLIND_NAME,
                    help=f"Blind split label, used to derive output paths "
                         f"(default: {DEFAULT_BLIND_NAME}).")
    ap.add_argument("--blind-dataset", default=DEFAULT_BLIND_DATASET,
                    help=f"HF dataset name for the blind split (default: "
                         f"{DEFAULT_BLIND_DATASET}).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Per-session pickle directory. "
                         "Default: cache/<blind_name>/source_cache/")
    ap.add_argument("--output-cache", type=Path, default=None,
                    help="Consolidated pickle path. "
                         "Default: cache/<blind_name>/source_cache.pkl")
    ap.add_argument("--manifest-path", type=Path, default=None,
                    help="Manifest JSON path. "
                         "Default: cache/<blind_name>/source_cache_manifest.json")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild all per-session caches.")
    ap.add_argument("--skip-r54b-validation", action="store_true",
                    help="Don't reproduce R54b after building cache "
                         "(automatic for non-blind_a runs).")
    args = ap.parse_args()

    # Resolve output paths (default to <blind_name> derived layout)
    cache_dir = args.output_dir if args.output_dir is not None else (
        REPO / "cache" / args.blind_name / "source_cache"
    )
    cache_consolidated = args.output_cache if args.output_cache is not None else (
        REPO / "cache" / args.blind_name / "source_cache.pkl"
    )
    manifest_path = args.manifest_path if args.manifest_path is not None else (
        REPO / "cache" / args.blind_name / "source_cache_manifest.json"
    )

    # R54b validation only meaningful for blind_a (the original training/blind split)
    skip_validation = args.skip_r54b_validation or args.blind_name != DEFAULT_BLIND_NAME

    t0 = time.time()
    cache = build_cache(
        force=args.force, blind_dataset=args.blind_dataset,
        cache_dir=cache_dir, cache_consolidated=cache_consolidated,
    )

    sanity = None
    if not skip_validation:
        sanity = validate_against_r54b(cache)

    manifest = {
        "blind_name": args.blind_name,
        "blind_dataset": args.blind_dataset,
        "n_sessions": len(cache),
        "consolidated_path": str(cache_consolidated),
        "per_session_dir": str(cache_dir),
        "sanity": sanity,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{ts()} Done. Manifest: {manifest_path}  elapsed={time.time() - t0:.0f}s")

    if sanity and sanity["quality"] != "PASS":
        print(f"\n  ⚠️ Sanity check did not fully match R54b. Inspect manifest before using cache.")
        sys.exit(1)


if __name__ == "__main__":
    main()

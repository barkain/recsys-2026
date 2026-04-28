# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Smoke test: precompute features + run E1 seed 0 with weights (1,2,0.5,1)
and time one E3 config to budget runtime."""
from __future__ import annotations

import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from eval_inference import build_ground_truth, cached_test_arrow_path
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, STOPWORDS, reconstruct_context
from mcrs.db_item.music_catalog import MusicCatalogDB

ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
CACHE_FILE = "exp/eval/v23_union_retrievals_cache.json"

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text):
    return {t for t in TOKEN_RE.findall(str(text).lower()) if len(t) > 2 and t not in STOPWORDS}


def stable_hash(s):
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def main():
    t0 = time.time()
    print("Loading...", flush=True)
    with open(CACHE_FILE) as f:
        payload = json.load(f)
    with open(ARTIFACT) as f:
        artifact_rows = json.load(f)
    cases = payload["cases"]
    bm25_meta = payload["bm25_meta"]
    bm25_full = payload["bm25_full"]
    neighbors = payload["neighbors"]
    arrow = cached_test_arrow_path()
    ds = Dataset.from_file(arrow)
    gt_maps = build_ground_truth(ds)
    conv_by_sid = {item["session_id"]: item["conversations"] for item in ds}

    n = len(cases)
    v23_pool = [list(artifact_rows[i]["candidate_pool_track_ids"]) for i in range(n)]
    gts = []
    for c in cases:
        sid = str(c["session_id"])
        uid = c.get("user_id")
        turn = int(c["turn_number"])
        gt_id = None
        if uid is not None:
            gt_id = gt_maps["session_user"].get((sid, str(uid)), {}).get(turn)
        if gt_id is None:
            gt_id = gt_maps["session"].get(sid, {}).get(turn)
        gts.append(gt_id)
    print(f"  GT non-null: {sum(1 for g in gts if g)}/{n}", flush=True)

    print("Init MusicCatalogDB...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )

    print("Collect unique tracks...", flush=True)
    all_tracks = set()
    for i in range(n):
        for tid in v23_pool[i] + bm25_meta[i] + bm25_full[i] + neighbors[i]:
            all_tracks.add(tid)
        for c in conv_by_sid.get(cases[i]["session_id"], []):
            if c["role"] == "music" and int(c["turn_number"]) < int(cases[i]["turn_number"]):
                all_tracks.add(str(c["content"]).strip())
    print(f"  {len(all_tracks)} unique tracks", flush=True)

    print("Precompute per-track artifacts...", flush=True)
    t1 = time.time()
    track_artist = {}
    track_tags = {}
    track_title_toks = {}
    track_artist_toks = {}
    track_meta_toks = {}
    cnt = 0
    for tid in all_tracks:
        try:
            meta = item_db.id_to_full_metadata(tid)
        except KeyError:
            meta = {}
        artist = str(meta.get("artist_name", "")).lower().strip()
        tags_raw = meta.get("tag_list") or []
        if isinstance(tags_raw, list):
            tags = {str(t).lower().strip() for t in tags_raw if str(t).strip()}
        else:
            tags = set()
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = _tokens(title)
        track_artist_toks[tid] = _tokens(meta.get("artist_name", ""))
        meta_parts = [title, str(meta.get("artist_name", "")), album]
        if isinstance(tags_raw, list):
            meta_parts.extend(str(t) for t in tags_raw[:12])
        else:
            meta_parts.append(str(tags_raw))
        track_meta_toks[tid] = _tokens(" ".join(meta_parts))
        cnt += 1
        if cnt % 5000 == 0:
            print(f"  {cnt}/{len(all_tracks)} ({time.time() - t1:.1f}s)", flush=True)
    print(f"  artifacts ready in {time.time() - t1:.1f}s", flush=True)

    print("Build per-session union features...", flush=True)
    t2 = time.time()
    F = len(FEATURE_NAMES)
    union_per_session = []
    union_idx_per_session = []
    feat_per_session = []
    for i, c in enumerate(cases):
        union = dedupe(v23_pool[i] + bm25_meta[i] + bm25_full[i] + neighbors[i])
        union_per_session.append(union)
        union_idx_per_session.append({tid: pos for pos, tid in enumerate(union)})
        ctx = reconstruct_context(conv_by_sid.get(c["session_id"], []), c["turn_number"])
        user_messages = ctx["user_messages"]
        played = ctx["played"]
        now_tokens = _tokens(user_messages[-1]) if user_messages else set()
        all_user_tokens = _tokens(" ".join(user_messages)) if user_messages else set()
        played_set = set(played)
        last_artist = track_artist.get(played[-1], "") if played else ""
        last_tags = track_tags.get(played[-1], set()) if played else set()
        prior = []
        for idx_from_end, tid in enumerate(reversed(played)):
            prior.append((1.0 / (idx_from_end + 1), track_artist.get(tid, ""), track_tags.get(tid, set())))

        K = len(union)
        X = np.zeros((K, F), dtype=np.float64)
        for rank, tid in enumerate(union, start=1):
            cand_artist = track_artist.get(tid, "")
            cand_tags = track_tags.get(tid, set())
            cand_title_tokens = track_title_toks.get(tid, set())
            cand_artist_tokens = track_artist_toks.get(tid, set())
            cand_meta_tokens = track_meta_toks.get(tid, set())
            X[rank - 1, 0] = 1.0 / rank
            X[rank - 1, 1] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
            if last_tags or cand_tags:
                inter = len(cand_tags & last_tags)
                u = len(cand_tags | last_tags)
                X[rank - 1, 2] = inter / u if u else 0.0
            X[rank - 1, 3] = float(len(cand_artist_tokens & now_tokens))
            X[rank - 1, 4] = float(len(cand_title_tokens & now_tokens))
            X[rank - 1, 5] = float(len(cand_meta_tokens & all_user_tokens))
            X[rank - 1, 6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for w, p_art, p_tags in prior:
                am = 1.0 if cand_artist and cand_artist == p_art else 0.0
                if cand_tags or p_tags:
                    inter = len(cand_tags & p_tags)
                    u = len(cand_tags | p_tags)
                    jacc = inter / u if u else 0.0
                else:
                    jacc = 0.0
                rec += w * (am + jacc)
            X[rank - 1, 7] = rec
        feat_per_session.append(X)
        if (i + 1) % 25 == 0:
            print(f"  built {i+1}/{n} ({time.time() - t2:.1f}s)", flush=True)
    print(f"  features ready in {time.time() - t2:.1f}s", flush=True)

    src_ranks = []
    for i in range(n):
        s = {}
        for label, seq in [("A", v23_pool[i]), ("B", bm25_meta[i]), ("C", bm25_full[i]), ("D", neighbors[i])]:
            d = {}
            for rank, tid in enumerate(seq):
                if tid not in d:
                    d[tid] = 1.0 / (60 + rank + 1)
            s[label] = d
        src_ranks.append(s)

    def build_pool(i, weights, topk=50):
        wA, wB, wC, wD = weights
        scores = {}
        sr = src_ranks[i]
        if wA != 0:
            for tid, rr in sr["A"].items():
                scores[tid] = scores.get(tid, 0.0) + wA * rr
        if wB != 0:
            for tid, rr in sr["B"].items():
                scores[tid] = scores.get(tid, 0.0) + wB * rr
        if wC != 0:
            for tid, rr in sr["C"].items():
                scores[tid] = scores.get(tid, 0.0) + wC * rr
        if wD != 0:
            for tid, rr in sr["D"].items():
                scores[tid] = scores.get(tid, 0.0) + wD * rr
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:topk]
        idx_map = union_idx_per_session[i]
        full_X = feat_per_session[i]
        rows_idx = np.array([idx_map[tid] for tid in ranked], dtype=np.int64)
        X_pool = full_X[rows_idx].copy()
        X_pool[:, 0] = 1.0 / np.arange(1, len(ranked) + 1, dtype=np.float64)
        gt_idx = ranked.index(gts[i]) if gts[i] in ranked else None
        return X_pool, gt_idx

    weights = (1.0, 2.0, 0.5, 1.0)
    pool_X = []
    pool_gt = []
    t3 = time.time()
    for i in range(n):
        X, gi = build_pool(i, weights, topk=50)
        pool_X.append(X)
        pool_gt.append(gi)
    print(f"  pool build (1 config): {time.time() - t3:.2f}s, GT-in-pool: {sum(1 for g in pool_gt if g is not None)}/{n}", flush=True)

    def mean_ndcg(idx_list, w):
        if not idx_list:
            return 0.0
        total = 0.0
        for i in idx_list:
            X = pool_X[i]; gi = pool_gt[i]
            if gi is None or X.size == 0:
                continue
            sc = X @ w
            order = np.argsort(-sc, kind="stable")
            for r, j in enumerate(order[:20]):
                if j == gi:
                    total += 1.0 / math.log2(r + 2)
                    break
        return total / len(idx_list)

    def fit(idx_list):
        def neg(w):
            return -mean_ndcg(idx_list, w)
        res = minimize(neg, init_w, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
        return res.x, -res.fun

    init_w = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)

    sessions = [c["session_id"] for c in cases]
    seed = 0
    order = sorted(range(n), key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))
    train_idx = order[:100]
    holdout_idx = order[100:]
    folds = [[] for _ in range(5)]
    for pos, idx in enumerate(order):
        folds[pos % 5].append(idx)

    t4 = time.time()
    w_opt, tr = fit(train_idx)
    ho = mean_ndcg(holdout_idx, w_opt)
    cv = []
    for fold in folds:
        tr_idx = [j for j in range(n) if j not in set(fold)]
        w_f, _ = fit(tr_idx)
        cv.append(mean_ndcg(fold, w_f))
    print("\nseed=0 weights=(1,2,0.5,1):")
    print(f"  train={tr:.4f}, holdout={ho:.4f} (target ~0.1207)")
    print(f"  cv5={np.mean(cv):.4f} ± {np.std(cv, ddof=1):.4f} (target ~0.1494 ± 0.0502)")
    print(f"  time: {time.time() - t4:.2f}s for 1 config × (split + CV5)")
    print(f"  total elapsed: {time.time() - t0:.1f}s")

    # Time one E3 config CV5 only (no split)
    t5 = time.time()
    weights2 = (2.0, 4.0, 1.0, 1.0)
    px = []
    pg = []
    for i in range(n):
        X, gi = build_pool(i, weights2, topk=50)
        px.append(X)
        pg.append(gi)
    cv2 = []
    for fold in folds:
        tr_idx = [j for j in range(n) if j not in set(fold)]
        def neg(w, tr_idx=tr_idx, px=px, pg=pg):
            total = 0.0
            for i in tr_idx:
                X = px[i]; gi = pg[i]
                if gi is None or X.size == 0:
                    continue
                sc = X @ w
                ord_ = np.argsort(-sc, kind="stable")
                for r, j in enumerate(ord_[:20]):
                    if j == gi:
                        total += 1.0 / math.log2(r + 2)
                        break
            return -total / len(tr_idx)
        res = minimize(neg, init_w, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
        w_f = res.x
        total = 0.0
        for i in fold:
            X = px[i]; gi = pg[i]
            if gi is None:
                continue
            sc = X @ w_f
            ord_ = np.argsort(-sc, kind="stable")
            for r, j in enumerate(ord_[:20]):
                if j == gi:
                    total += 1.0 / math.log2(r + 2)
                    break
        cv2.append(total / len(fold))
    print(f"  E3 single-config CV5 (1 seed): {time.time() - t5:.2f}s, mean {np.mean(cv2):.4f}")
    e3_per_seed = (time.time() - t5)
    print(f"  -> E3 budget estimate: 144 cfgs × 5 seeds × {e3_per_seed:.1f}s ≈ {144 * 5 * e3_per_seed / 60:.1f} min")


if __name__ == "__main__":
    main()

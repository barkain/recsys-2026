#!/usr/bin/env python3
"""R432 blind goal source — encode the 80 Blind-A goal-queries with the R54 5-fold ENSEMBLE.

Blind analog of expR432_phase1_goal_oof.py. Blind has no held-out folds, so we ensemble (mean
cosine across the 5 fold models, memory: ensemble beats single on blind), matching the production
R54 blind retrieval. Query construction is IDENTICAL to the dev Phase-1 source:
    [QUERY] {last_user} [GOAL] {category specificity listener_goal} [PROFILE] {culture country age}

Blind-A dataset carries conversation_goal + user_profile directly (no arrow join needed).
Output: cache/r432_goal/blind_goal_lists.json  {"lists": {ci: [[tid, mean_cos], ...]}, "sid": {ci: sid}}
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np  # type: ignore[reportMissingImports]

import argparse
REPO = Path(__file__).resolve().parent.parent
FOLD_DIRS = {0: REPO / "cache/r54/phase3_smoke/fold_0",
             **{k: REPO / f"cache/r54/phase3_full/fold_{k}" for k in range(1, 5)}}
BLIND_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-A"
TOPK = 300
USE_PROFILE = True  # --no-profile -> goal-only


def ts():
    from datetime import datetime
    return f"[{datetime.now():%H:%M:%S}]"


def norm(s):
    return str(s or "").replace("\n", " ").strip()


def load_catalog_ids():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf = Path.home() / ".cache/huggingface/datasets"
    m = sorted(hf.glob("talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
                       "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    return [str(t) for t in Dataset.from_file(str(m[-1])).to_dict()["track_id"]]


def last_user_and_played(item):
    import pandas as pd  # type: ignore[reportMissingImports]
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    users = df[df["role"] == "user"]
    last_user = str(users.iloc[-1]["content"])
    tn = int(users.iloc[-1]["turn_number"])
    played = [str(r["content"]).strip() for _, r in df[df["turn_number"] < tn].iterrows()
              if r["role"] == "music"]
    return last_user, played


def goal_query(item):
    g = item.get("conversation_goal") or {}
    p = item.get("user_profile") or {}
    last_user, played = last_user_and_played(item)
    goal_text = " ".join([norm(g.get("category")), norm(g.get("specificity")),
                          norm(g.get("listener_goal"))]).strip()
    profile_text = " ".join([norm(p.get("preferred_musical_culture")),
                             norm(p.get("country_name")), norm(p.get("age_group"))]).strip()
    parts = [f"[QUERY] {norm(last_user)}"]
    if goal_text:
        parts.append(f"[GOAL] {goal_text}")
    if USE_PROFILE and profile_text:
        parts.append(f"[PROFILE] {profile_text}")
    return " ".join(parts), played


def encode(model_dir, queries):
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
    model = SentenceTransformer(str(model_dir), device="cpu")
    return model.encode(queries, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)


def main():
    global USE_PROFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    USE_PROFILE = not args.no_profile
    out_path = Path(args.out) if args.out else (
        REPO / ("cache/r432_goal/blind_goal_lists.json" if USE_PROFILE
                else "cache/r432_goal_only/blind_goal_lists.json"))
    t0 = time.time()
    print(f"{ts()} R432 blind goal source — R54 5-fold ensemble (USE_PROFILE={USE_PROFILE})")
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    try:
        db = load_dataset(BLIND_DATASET, split="test",
                          download_config=DownloadConfig(local_files_only=True))
    except Exception:
        db = load_dataset(BLIND_DATASET, split="test")
    items = list(db)
    print(f"  {len(items)} blind sessions")
    track_ids = load_catalog_ids()
    n_tracks = len(track_ids)

    queries, sids, played_lists = [], [], []
    for it in items:
        q, played = goal_query(it)
        queries.append(q); sids.append(str(it["session_id"])); played_lists.append(set(played))
    print(f"  sample q0: {queries[0][:150]}", flush=True)

    # ensemble: accumulate mean cosine across folds
    acc = np.zeros((len(items), n_tracks), dtype=np.float32)
    for fk, fd in FOLD_DIRS.items():
        print(f"{ts()} fold {fk}: load embs + encode + score ...", flush=True)
        track_embs = np.load(fd / "track_embs.npy")
        assert track_embs.shape[0] == n_tracks
        q_embs = encode(fd / "model", queries)
        acc += q_embs @ track_embs.T
        del track_embs, q_embs
    acc /= len(FOLD_DIRS)

    lists = {}
    for ci in range(len(items)):
        scores = acc[ci]
        played = played_lists[ci]
        kk = min(TOPK + len(played), n_tracks)
        part = np.argpartition(-scores, kk - 1)[:kk]
        part = part[np.argsort(-scores[part])]
        out = []
        for j in part:
            tid = track_ids[j]
            if tid in played:
                continue
            out.append([tid, float(scores[j])])
            if len(out) >= TOPK:
                break
        lists[ci] = out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"lists": {str(k): v for k, v in lists.items()},
               "sid": {str(k): sids[k] for k in range(len(items))},
               "meta": {"query_variant": "query_goal_profile_structured", "topk": TOPK}},
              open(out_path, "w"))
    print(f"{ts()} Saved {out_path} ({len(lists)} cases, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

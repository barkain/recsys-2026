#!/usr/bin/env python3
"""R432 Phase 1 — build a goal-aware R54 OOF retrieval source.

Phase 0 (TF-IDF audit) showed conversation_goal + user_profile lifts first-turn recall
(+0.033 r@20, 96 vs 26 union-absent@300 on n_prior=0). This builds the real source: the
production R54 5-fold ensemble, but with a goal-augmented query, scored OOF (each held-out
fold encoded by ITS OWN fold model — never the model that trained on it).

Query (additive superset of the production R54 [QUERY] form, isolates the goal contribution):
    [QUERY] {user_query} [GOAL] {category specificity listener_goal} [PROFILE] {culture country age}

Output (r431/r103 source format, consumed by expR103_integrate_eval.py --gte-oof):
    cache/r432_goal/goal_oof_lists.json   {"lists": {case_idx: [[track_id, cosine], ...]}}

Played tracks (case music_turns) are excluded from candidates, matching production retrieval.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import pyarrow as pa  # type: ignore[reportMissingImports]
import pyarrow.ipc as ipc  # type: ignore[reportMissingImports]

import argparse

REPO = Path(__file__).resolve().parent.parent
PAYLOAD = REPO / "exp/eval/_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp/eval/expR68_r54_reference_stats.pkl"
TEST_ARROW = (Path.home() / ".cache/huggingface/datasets/"
              "talkpl-ai___talk_play_data-challenge-dataset/default/0.0.0/"
              "8110a2cfda8f7cfd43805a09eca6c58e0f7b285c/"
              "talk_play_data-challenge-dataset-test.arrow")
FOLD_DIRS = {0: REPO / "cache/r54/phase3_smoke/fold_0",
             **{k: REPO / f"cache/r54/phase3_full/fold_{k}" for k in range(1, 5)}}
TOPK = 300
N_FOLDS = 5
USE_PROFILE = True  # set False via --no-profile (goal-only, removes demographic override)


def ts():
    from datetime import datetime
    return f"[{datetime.now():%H:%M:%S}]"


def norm(s):
    return str(s or "").replace("\n", " ").strip()


def load_catalog_ids():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf = Path.home() / ".cache/huggingface/datasets"
    matches = sorted(hf.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    ds = Dataset.from_file(str(matches[-1]))
    return [str(t) for t in ds.to_dict()["track_id"]]


def load_session_fields():
    with pa.memory_map(str(TEST_ARROW), "r") as src:
        rows = ipc.open_stream(src).read_all().to_pylist()
    out = {}
    for r in rows:
        out[str(r["session_id"])] = {"profile": r.get("user_profile") or {},
                                     "goal": r.get("conversation_goal") or {}}
    return out


def goal_query(case, sf):
    goal = sf.get("goal", {})
    prof = sf.get("profile", {})
    goal_text = " ".join([norm(goal.get("category")), norm(goal.get("specificity")),
                          norm(goal.get("listener_goal"))]).strip()
    profile_text = " ".join([norm(prof.get("preferred_musical_culture")),
                             norm(prof.get("country_name")),
                             norm(prof.get("age_group"))]).strip()
    parts = [f"[QUERY] {norm(case['user_query'])}"]
    if goal_text:
        parts.append(f"[GOAL] {goal_text}")
    if USE_PROFILE and profile_text:
        parts.append(f"[PROFILE] {profile_text}")
    return " ".join(parts)


def encode(model_dir, queries):
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
    model = SentenceTransformer(str(model_dir), device="cpu")
    return model.encode(queries, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)


def main():
    global USE_PROFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-profile", action="store_true", help="goal-only query (drop [PROFILE])")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    USE_PROFILE = not args.no_profile
    out_path = Path(args.out) if args.out else (
        REPO / ("cache/r432_goal/goal_oof_lists.json" if USE_PROFILE
                else "cache/r432_goal_only/goal_oof_lists.json"))
    t0 = time.time()
    print(f"{ts()} R432 Phase 1 — goal-aware R54 OOF source (USE_PROFILE={USE_PROFILE})")
    payload = pickle.load(open(PAYLOAD, "rb"))
    cases = payload["cases"]
    n = len(cases)
    sf = load_session_fields()

    with open(W0_STATS, "rb") as f:
        w0 = pickle.load(f)
    case_fold = [-1] * n
    for row in w0:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    assert all(0 <= f < N_FOLDS for f in case_fold), "unmapped fold"

    track_ids = load_catalog_ids()
    tid_arr = np.array(track_ids)
    n_tracks = len(track_ids)
    print(f"{ts()} {n} cases, {n_tracks} tracks. Building goal queries ...", flush=True)
    queries = [goal_query(cases[i], sf.get(str(cases[i]["session_id"]), {})) for i in range(n)]
    print(f"  sample q0: {queries[0][:160]}", flush=True)

    lists = {}
    for fk in range(N_FOLDS):
        eval_idx = [i for i in range(n) if case_fold[i] == fk]
        fd = FOLD_DIRS[fk]
        print(f"\n{ts()} fold {fk}: {len(eval_idx)} held-out cases; load embs + encode ...", flush=True)
        track_embs = np.load(fd / "track_embs.npy")
        assert track_embs.shape[0] == n_tracks, f"fold {fk} embs {track_embs.shape} != {n_tracks}"
        q_embs = encode(fd / "model", [queries[i] for i in eval_idx])
        print(f"{ts()} fold {fk}: scoring ({len(eval_idx)}x{n_tracks}) ...", flush=True)
        for row_i, i in enumerate(eval_idx):
            scores = q_embs[row_i] @ track_embs.T  # (n_tracks,)
            played = set(str(t) for t in cases[i]["music_turns"])
            kk = min(TOPK + len(played), n_tracks)
            part = np.argpartition(-scores, kk - 1)[:kk]
            part = part[np.argsort(-scores[part])]
            out_pairs = []
            for j in part:
                tid = track_ids[j]
                if tid in played:
                    continue
                out_pairs.append([tid, float(scores[j])])
                if len(out_pairs) >= TOPK:
                    break
            lists[i] = out_pairs
        del track_embs, q_embs
        print(f"{ts()} fold {fk} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"lists": {str(k): v for k, v in lists.items()},
               "meta": {"query_variant": "query_goal_profile_structured", "topk": TOPK,
                        "n": n, "built_s": time.time() - t0}}, open(out_path, "w"))
    print(f"\n{ts()} Saved {out_path} ({len(lists)} cases, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

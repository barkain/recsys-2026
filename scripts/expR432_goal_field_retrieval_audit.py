#!/usr/bin/env python3
"""R432: conversation_goal / user_profile retrieval audit.

Blind-A exposes `conversation_goal` and `user_profile`; production R54/R84 query
builders ignore them. This script tests whether goal-aware text retrieval has
usable nDCG/recall signal, especially on first-turn cases (Blind-A shape).

This is a cheap lexical TF-IDF audit, not a production candidate. If it shows
strong signal, the next step is to build the same query variant with the actual
R54/BGE ensemble and integrate it as a source.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent.parent
PAYLOAD = REPO / "exp/eval/_R12_all_turns_payload.pkl"
TEST_ARROW = Path(
    "/Users/nadavbarkai/.cache/huggingface/datasets/"
    "talkpl-ai___talk_play_data-challenge-dataset/default/0.0.0/"
    "8110a2cfda8f7cfd43805a09eca6c58e0f7b285c/"
    "talk_play_data-challenge-dataset-test.arrow"
)
META = REPO / "cache/metadata/track_metadata_all_tracks.json"
OUT_JSON = REPO / "exp/eval/expR432_goal_field_retrieval_audit.json"
OUT_MD = REPO / "docs/r432_goal_field_retrieval_audit.md"


def g1(m: dict[str, Any], k: str) -> str:
    v = m.get(k, "")
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def read_arrow(path: Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all().to_pylist()


def norm(s: Any) -> str:
    return str(s or "").replace("\n", " ").strip()


def track_doc(tid: str, m: dict[str, Any]) -> str:
    tags = m.get("tag_list", [])
    if isinstance(tags, list):
        tags_s = " ".join(str(x) for x in tags[:20])
    else:
        tags_s = str(tags or "")
    parts = [
        g1(m, "track_name"),
        g1(m, "artist_name"),
        g1(m, "album_name"),
        g1(m, "release_date")[:4],
        tags_s,
        g1(m, "isrc"),
    ]
    return " | ".join(p for p in parts if p)


def load_session_fields() -> dict[str, dict[str, Any]]:
    rows = read_arrow(TEST_ARROW)
    out = {}
    for r in rows:
        out[str(r["session_id"])] = {
            "profile": r.get("user_profile") or {},
            "goal": r.get("conversation_goal") or {},
            "progress": r.get("goal_progress_assessments") or [],
        }
    return out


def load_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = pickle.load(open(PAYLOAD, "rb"))
    session_fields = load_session_fields()
    src_names = [k for k in ("src_a", "src_b", "src_c", "src_d", "src_f", "src_g") if k in payload]
    cases = []
    for i, c in enumerate(payload["cases"]):
        sid = str(c["session_id"])
        sf = session_fields.get(sid, {})
        union = set()
        for src in src_names:
            union.update(str(t) for t in payload[src][i][:300])
        cases.append({
            "idx": i,
            "session_id": sid,
            "turn_number": int(c.get("turn_number") or 0),
            "n_prior": int(c.get("n_prior_music") or len(c.get("music_turns") or [])),
            "user_query": norm(c.get("user_query")),
            "history": c.get("history") or [],
            "gt": str(c["gt"]),
            "union300": union,
            "goal": sf.get("goal") or {},
            "profile": sf.get("profile") or {},
        })
    return cases, payload


def query_for(case: dict[str, Any], variant: str) -> str:
    goal = case["goal"]
    prof = case["profile"]
    goal_text = " ".join([
        norm(goal.get("category")),
        norm(goal.get("specificity")),
        norm(goal.get("listener_goal")),
    ]).strip()
    profile_text = " ".join([
        norm(prof.get("preferred_musical_culture")),
        norm(prof.get("country_name")),
        norm(prof.get("age_group")),
    ]).strip()
    hist_user = " ".join(
        norm(h.get("content"))
        for h in case["history"]
        if h.get("role") == "user"
    )
    hist_thought = " ".join(
        norm(h.get("thought"))
        for h in case["history"]
        if h.get("thought")
    )

    if variant == "query":
        return case["user_query"]
    if variant == "goal":
        return goal_text
    if variant == "query_goal":
        return f"{case['user_query']} {goal_text}"
    if variant == "query_goal_profile":
        return f"{case['user_query']} {goal_text} {profile_text}"
    if variant == "query_hist_goal":
        return f"{case['user_query']} {hist_user} {goal_text}"
    if variant == "query_hist_thought_goal":
        return f"{case['user_query']} {hist_user} {hist_thought} {goal_text} {profile_text}"
    raise ValueError(variant)


def rank_metrics(cases: list[dict[str, Any]], rankings: list[list[str]]) -> dict[str, Any]:
    subsets = {
        "all": [True] * len(cases),
        "n0_first_turn": [c["n_prior"] == 0 for c in cases],
        "n_prior_gt0": [c["n_prior"] > 0 for c in cases],
    }
    out: dict[str, Any] = {}
    for name, mask in subsets.items():
        idxs = [i for i, ok in enumerate(mask) if ok]
        hit = {20: 0, 30: 0, 100: 0, 300: 0}
        absent = {20: 0, 30: 0, 100: 0, 300: 0}
        ranks = []
        for i in idxs:
            gt = cases[i]["gt"]
            row = rankings[i]
            try:
                r = row.index(gt) + 1
            except ValueError:
                continue
            ranks.append(r)
            for k in hit:
                if r <= k:
                    hit[k] += 1
                    if gt not in cases[i]["union300"]:
                        absent[k] += 1
        n = len(idxs)
        out[name] = {
            "n": n,
            "recall20": hit[20] / n if n else 0.0,
            "recall30": hit[30] / n if n else 0.0,
            "recall100": hit[100] / n if n else 0.0,
            "recall300": hit[300] / n if n else 0.0,
            "hits": hit,
            "union_absent_hits": absent,
            "median_rank_if_hit300": float(np.median([r for r in ranks if r <= 300])) if any(r <= 300 for r in ranks) else None,
        }
    return out


def retrieve_topk(q_mat: sparse.csr_matrix, doc_mat: sparse.csr_matrix, track_ids: list[str], topk: int, chunk: int) -> list[list[str]]:
    rankings: list[list[str]] = []
    doc_t = doc_mat.T.tocsr()
    for start in range(0, q_mat.shape[0], chunk):
        scores = (q_mat[start:start + chunk] @ doc_t).toarray()
        for row in scores:
            if topk >= row.shape[0]:
                order = np.argsort(-row)
            else:
                part = np.argpartition(-row, topk)[:topk]
                order = part[np.argsort(-row[part])]
            rankings.append([track_ids[int(j)] for j in order[:topk]])
    return rankings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    cases, _ = load_cases()
    meta = json.load(open(META))
    track_ids = sorted(meta)
    docs = [track_doc(t, meta[t]) for t in track_ids]

    variants = [
        "query",
        "goal",
        "query_goal",
        "query_goal_profile",
        "query_hist_goal",
        "query_hist_thought_goal",
    ]
    all_queries = []
    offsets = {}
    for v in variants:
        offsets[v] = (len(all_queries), len(all_queries) + len(cases))
        all_queries.extend(query_for(c, v) for c in cases)

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    doc_mat = vec.fit_transform(docs)
    q_all = vec.transform(all_queries)

    results = {}
    best_rankings = {}
    for v in variants:
        a, b = offsets[v]
        rankings = retrieve_topk(q_all[a:b], doc_mat, track_ids, args.topk, args.chunk)
        results[v] = rank_metrics(cases, rankings)
        best_rankings[v] = rankings

    out = {
        "experiment": "R432 conversation_goal/user_profile TF-IDF retrieval audit",
        "n_cases": len(cases),
        "n_tracks": len(track_ids),
        "variants": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    lines = [
        "# R432 Goal-Field Retrieval Audit",
        "",
        "TF-IDF over track metadata. This tests public `conversation_goal`/`user_profile` signal before building a production BGE/R54 source.",
        "",
        "## First-turn subset (`n_prior=0`, Blind-A shape)",
        "",
        "| variant | recall@20 | recall@30 | recall@100 | recall@300 | union-absent@300 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        s = results[v]["n0_first_turn"]
        lines.append(f"| {v} | {s['recall20']:.4f} | {s['recall30']:.4f} | {s['recall100']:.4f} | {s['recall300']:.4f} | {s['union_absent_hits']['300']} |")
    lines += ["", "## All turns", "", "| variant | recall@20 | recall@30 | recall@100 | recall@300 | union-absent@300 |", "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        s = results[v]["all"]
        lines.append(f"| {v} | {s['recall20']:.4f} | {s['recall30']:.4f} | {s['recall100']:.4f} | {s['recall300']:.4f} | {s['union_absent_hits']['300']} |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_MD.relative_to(REPO)}")
    for v in variants:
        s = results[v]["n0_first_turn"]
        print(f"{v:24s} n0 r20={s['recall20']:.4f} r300={s['recall300']:.4f} absent300={s['union_absent_hits']['300']}")


if __name__ == "__main__":
    main()

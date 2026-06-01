#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R400 Phase-0 DATA AGENT — canonical supervised training corpus + leakage audit + per-user train-history index.

R400 goal: a large supervised session/choice recommender to beat R106 A-clean (nDCG@20 0.5073) on nDCG.
The GT next-track is a Gemini pick from each user's hidden 16-32 track LFM-2b session pool.

PHASE-0 BREAKTHROUGH this script materializes the data for:
  74.2% of dev users (371/500) and 43.1% of blind users (25/58) ALSO appear in TRAIN sessions
  (same users, DIFFERENT sessions — 0 session leakage). Each train user has ~100+ history tracks.
  -> a user->item TRAIN-history taste lever, untested (R104 used only within-conversation played tracks).

Produces (materialized under cache/r400/):
  1. user_train_history.json  — user_id -> ordered de-duplicated list of all track_ids played across
     that user's TRAIN sessions (the taste signal). Plus history-size distributions for
     all-train / dev-overlap / blind-overlap user populations.
  2. train_corpus.jsonl       — every train prefix->next transition as
     {user_id, session_id, turn_index, history_prefix (track_ids played so far),
      conv_text (conversation-text-so-far user/assistant turns), target_track_id}.
     train_corpus_meta.json holds count + schema.
  3. leakage_audit.json        — dev∩train SESSIONS, dev/blind∩train USERS, and the
     dev-GT-in-user-TRAIN-history rate (REAL signal, not leakage: train session != dev session).

Run:  /Users/nadavbarkai/dev/recsys-2026/.venv/bin/python scripts/expR400_build_full_training_corpus.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

HF_BASE = Path.home() / ".cache" / "huggingface" / "datasets"
TRAIN_ARROW = (HF_BASE / "talkpl-ai___talk_play_data-challenge-dataset" / "default" / "0.0.0"
               / "8110a2cfda8f7cfd43805a09eca6c58e0f7b285c"
               / "talk_play_data-challenge-dataset-train.arrow")
BLIND_ARROW = (HF_BASE / "talkpl-ai___talk_play_data-challenge-blind-a" / "default" / "0.0.0"
               / "9be92e3b93ebdaa5d5d60831b3b060d7c3a3faa8"
               / "talk_play_data-challenge-blind-a-test.arrow")
DEV_PAYLOAD = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"

OUT_DIR = REPO / "cache" / "r400"


def _dist(sizes: list[int]) -> dict:
    """Summary stats for a list of history sizes."""
    if not sizes:
        return {"n_users": 0}
    a = np.asarray(sizes, dtype=np.float64)
    return {
        "n_users": int(a.size),
        "min": int(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": round(float(a.mean()), 2),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": int(a.max()),
        "n_zero": int((a == 0).sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[R400] loading train sessions ...")
    tr = datasets.Dataset.from_file(str(TRAIN_ARROW))
    n_sessions = len(tr)
    print(f"[R400] train sessions: {n_sessions}")

    # --- pass over train: build per-user history + supervised transition corpus ---
    user_hist_ordered: dict[str, list[str]] = defaultdict(list)   # ordered, dedup-on-append
    user_hist_seen: dict[str, set[str]] = defaultdict(set)
    train_user_ids: set[str] = set()
    train_session_ids: set[str] = set()
    # per-user set of ALL tracks they ever played in train (for GT-in-history audit)
    user_track_set: dict[str, set[str]] = defaultdict(set)

    corpus_path = OUT_DIR / "train_corpus.jsonl"
    n_transitions = 0          # all music picks (incl. cold-start first pick of a session)
    n_transitions_prefix = 0   # picks with a non-empty played history prefix (~106393, task's count)
    turn_index_hist: dict[int, int] = defaultdict(int)

    with open(corpus_path, "w") as fout:
        for row in tr:
            uid = str(row["user_id"])
            sid = str(row["session_id"])
            train_user_ids.add(uid)
            train_session_ids.add(sid)
            conv = row["conversations"] or []

            played_so_far: list[str] = []   # track_ids played earlier IN THIS SESSION (the prefix)
            text_turns: list[dict] = []      # user/assistant text turns seen so far (conv-text-so-far)
            music_turn_idx = 0               # 0-based index of the music turn within this session

            for turn in conv:
                role = turn.get("role")
                content = turn.get("content")
                if role == "music":
                    target = str(content)
                    # supervised transition: predict THIS music pick from prefix + conv-text-so-far
                    rec = {
                        "user_id": uid,
                        "session_id": sid,
                        "turn_index": music_turn_idx,          # which music pick in the session (0-based)
                        "has_prefix": len(played_so_far) > 0,   # False only for the session's first pick
                        "history_prefix": list(played_so_far),  # tracks played before this pick (this session)
                        "conv_text": text_turns,                # [{role, content}] user/assistant text so far
                        "target_track_id": target,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_transitions += 1
                    if played_so_far:
                        n_transitions_prefix += 1
                    turn_index_hist[music_turn_idx] += 1

                    # update per-user GLOBAL history (across all this user's train sessions)
                    if target not in user_hist_seen[uid]:
                        user_hist_seen[uid].add(target)
                        user_hist_ordered[uid].append(target)
                    user_track_set[uid].add(target)

                    played_so_far.append(target)
                    music_turn_idx += 1
                else:
                    # user / assistant / (any non-music) text turn -> part of conv-text-so-far
                    if content is not None and role in ("user", "assistant", "system"):
                        text_turns.append({"role": role, "content": str(content)})

    print(f"[R400] supervised transitions written: {n_transitions} "
          f"(of which {n_transitions_prefix} have a non-empty prefix) -> {corpus_path}")

    # --- materialize per-user TRAIN-history index ---
    user_train_history = {uid: user_hist_ordered[uid] for uid in train_user_ids}
    hist_path = OUT_DIR / "user_train_history.json"
    with open(hist_path, "w") as f:
        json.dump(user_train_history, f)
    print(f"[R400] per-user train history written: {len(user_train_history)} users -> {hist_path}")

    # --- dev + blind user/session sets ---
    print("[R400] loading dev payload + blind-a ...")
    payload = pickle.load(open(DEV_PAYLOAD, "rb"))
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    dev_user_ids = {str(c["user_id"]) for c in cases}
    dev_session_ids = {str(c["session_id"]) for c in cases}

    bl = datasets.Dataset.from_file(str(BLIND_ARROW))
    blind_user_ids = {str(u) for u in bl["user_id"]}
    blind_session_ids = {str(s) for s in bl["session_id"]}

    # --- history-size distributions for the 3 populations ---
    all_sizes = [len(user_hist_ordered[u]) for u in train_user_ids]
    dev_overlap_users = sorted(dev_user_ids & train_user_ids)
    blind_overlap_users = sorted(blind_user_ids & train_user_ids)
    dev_sizes = [len(user_hist_ordered[u]) for u in dev_overlap_users]
    blind_sizes = [len(user_hist_ordered[u]) for u in blind_overlap_users]

    hist_dists = {
        "all_train_users": _dist(all_sizes),
        "dev_overlap_users": _dist(dev_sizes),
        "blind_overlap_users": _dist(blind_sizes),
    }

    # --- LEAKAGE AUDIT ---
    dev_inter_sessions = sorted(dev_session_ids & train_session_ids)
    blind_inter_sessions = sorted(blind_session_ids & train_session_ids)
    dev_inter_users = len(dev_user_ids & train_user_ids)
    blind_inter_users = len(blind_user_ids & train_user_ids)

    # dev-GT-in-user-TRAIN-history rate (REAL signal, not leakage: train session != dev session).
    # Evaluate per UNIQUE dev (user, gt). A dev user has 16 cases (8 per session x 2 sessions);
    # we report at the case level AND at the unique (user,gt) level. Restrict the "hittable"
    # denominator to dev users that actually have a train history (the overlap population),
    # since a non-overlap user CANNOT have the GT in a (non-existent) train history.
    n_cases = len(cases)
    case_overlap = 0           # dev cases whose user is in train
    case_gt_in_hist = 0        # of those, GT track is in that user's TRAIN history
    case_gt_in_hist_h7 = 0     # same, restricted to n_prior_music==7 (the h7 subset)
    case_overlap_h7 = 0
    uniq_pairs: set[tuple] = set()
    uniq_pairs_hit: set[tuple] = set()
    # also: same-session baseline — is GT already in the dev case's OWN played prefix? (sanity, should be rare)
    case_gt_in_dev_prefix = 0

    for c in cases:
        uid = str(c["user_id"])
        gt = str(c["gt"])
        npri = int(c["n_prior_music"])
        dev_prefix = {str(t) for t in c["music_turns"]}
        if gt in dev_prefix:
            case_gt_in_dev_prefix += 1
        tset = user_track_set.get(uid)
        if tset is None:
            continue  # user not in train -> no train history
        case_overlap += 1
        hit = gt in tset
        if hit:
            case_gt_in_hist += 1
        uniq_pairs.add((uid, gt))
        if hit:
            uniq_pairs_hit.add((uid, gt))
        if npri == 7:
            case_overlap_h7 += 1
            if hit:
                case_gt_in_hist_h7 += 1

    leakage_audit = {
        "train_n_sessions": n_sessions,
        "train_n_users": len(train_user_ids),
        "dev_n_cases": n_cases,
        "dev_n_users": len(dev_user_ids),
        "dev_n_sessions": len(dev_session_ids),
        "blind_n_users": len(blind_user_ids),
        "blind_n_sessions": len(blind_session_ids),
        # 1. SESSION leakage (must be 0)
        "dev_inter_train_SESSIONS": len(dev_inter_sessions),
        "blind_inter_train_SESSIONS": len(blind_inter_sessions),
        "session_leakage": (len(dev_inter_sessions) + len(blind_inter_sessions)) > 0,
        # 2. USER overlap (the lever)
        "dev_inter_train_USERS": dev_inter_users,
        "dev_inter_train_USERS_frac": round(dev_inter_users / len(dev_user_ids), 4),
        "blind_inter_train_USERS": blind_inter_users,
        "blind_inter_train_USERS_frac": round(blind_inter_users / len(blind_user_ids), 4),
        # 3. dev GT in user's TRAIN history (REAL signal quantification)
        "dev_cases_total": n_cases,
        "dev_cases_user_in_train": case_overlap,
        "dev_cases_gt_in_train_history": case_gt_in_hist,
        "dev_gt_in_train_history_rate_over_overlap_cases": round(case_gt_in_hist / max(case_overlap, 1), 4),
        "dev_gt_in_train_history_rate_over_all_cases": round(case_gt_in_hist / max(n_cases, 1), 4),
        "dev_h7_cases_user_in_train": case_overlap_h7,
        "dev_h7_cases_gt_in_train_history": case_gt_in_hist_h7,
        "dev_h7_gt_in_train_history_rate": round(case_gt_in_hist_h7 / max(case_overlap_h7, 1), 4),
        "dev_uniq_user_gt_pairs_in_train": len(uniq_pairs),
        "dev_uniq_user_gt_pairs_gt_in_history": len(uniq_pairs_hit),
        "dev_uniq_pair_gt_in_history_rate": round(len(uniq_pairs_hit) / max(len(uniq_pairs), 1), 4),
        # sanity: GT in the dev case's OWN played prefix (same-session; expected very low)
        "dev_cases_gt_in_own_dev_prefix": case_gt_in_dev_prefix,
        "notes": (
            "0 session leakage: dev/blind sessions are disjoint from train sessions. "
            "dev-GT-in-user-TRAIN-history is REAL signal, NOT leakage: the GT track was played by the "
            "SAME user in a DIFFERENT (train) session, so a user->item taste model can legitimately use it. "
            "Corpus uses ONLY train transitions; dev is held out for OOF."
        ),
    }

    # --- corpus meta / schema ---
    corpus_meta = {
        "corpus_path": str(corpus_path),
        "n_transitions": n_transitions,
        "n_transitions_with_prefix": n_transitions_prefix,
        "n_transitions_cold_start": n_transitions - n_transitions_prefix,
        "n_train_sessions": n_sessions,
        "n_train_users": len(train_user_ids),
        "turn_index_distribution": {str(k): turn_index_hist[k] for k in sorted(turn_index_hist)},
        "schema": {
            "user_id": "str — LFM-2b user id (joins to user_train_history)",
            "session_id": "str — train session id (disjoint from dev/blind sessions)",
            "turn_index": "int — 0-based index of this music pick within the session",
            "has_prefix": "bool — False only for the session's first music pick (cold-start)",
            "history_prefix": "list[str] — track_ids played earlier IN THIS SESSION (in order)",
            "conv_text": "list[{role,content}] — user/assistant text turns BEFORE this pick",
            "target_track_id": "str — the GT next-track for this transition (supervision label)",
        },
        "history_size_distributions": hist_dists,
        "files": {
            "user_train_history": str(hist_path),
            "train_corpus": str(corpus_path),
            "leakage_audit": str(OUT_DIR / "leakage_audit.json"),
        },
    }
    with open(OUT_DIR / "train_corpus_meta.json", "w") as f:
        json.dump(corpus_meta, f, indent=2)
    with open(OUT_DIR / "leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=2)
    with open(OUT_DIR / "history_size_distributions.json", "w") as f:
        json.dump(hist_dists, f, indent=2)

    # --- console report ---
    print("\n" + "=" * 70)
    print("R400 PHASE-0 CORPUS REPORT")
    print("=" * 70)
    print(f"train sessions={n_sessions}  train users={len(train_user_ids)}")
    print(f"supervised transitions={n_transitions}  "
          f"(with-prefix={n_transitions_prefix} ~task's 106393, cold-start={n_transitions - n_transitions_prefix})")
    print("\nHISTORY-SIZE DISTRIBUTIONS (unique tracks per user across train sessions):")
    for pop, d in hist_dists.items():
        print(f"  {pop:22} {d}")
    print("\nLEAKAGE AUDIT:")
    print(f"  dev∩train SESSIONS   = {leakage_audit['dev_inter_train_SESSIONS']}  (must be 0)")
    print(f"  blind∩train SESSIONS = {leakage_audit['blind_inter_train_SESSIONS']}  (must be 0)")
    print(f"  dev∩train USERS      = {dev_inter_users}/{len(dev_user_ids)} "
          f"({leakage_audit['dev_inter_train_USERS_frac']})")
    print(f"  blind∩train USERS    = {blind_inter_users}/{len(blind_user_ids)} "
          f"({leakage_audit['blind_inter_train_USERS_frac']})")
    print(f"  dev-GT-in-user-TRAIN-history (over overlap cases) = "
          f"{case_gt_in_hist}/{case_overlap} "
          f"({leakage_audit['dev_gt_in_train_history_rate_over_overlap_cases']})")
    print(f"  dev-GT-in-user-TRAIN-history (over ALL 8000 cases) = "
          f"{leakage_audit['dev_gt_in_train_history_rate_over_all_cases']}")
    print(f"  dev-GT-in-history h7 subset = {case_gt_in_hist_h7}/{case_overlap_h7} "
          f"({leakage_audit['dev_h7_gt_in_train_history_rate']})")
    print(f"  dev-GT-in-OWN-dev-prefix (same-session sanity) = {case_gt_in_dev_prefix}")
    print("\nMATERIALIZED:")
    for k, v in corpus_meta["files"].items():
        print(f"  {k:22} {v}")
    print("  train_corpus_meta       " + str(OUT_DIR / "train_corpus_meta.json"))
    print("=" * 70)


if __name__ == "__main__":
    main()

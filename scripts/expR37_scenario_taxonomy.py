#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R37: Scenario taxonomy diagnostic.

For each dev case, compute observable scenario features and oracle labels.
Output tables showing when audio/Q3/R22b should be allowed vs when
same-artist continuation should be preserved.

No model training. Pure analysis.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R22B_OOF = REPO / "cache" / "r22b" / "dev_r22b_lists.json"
Q3_LISTS = REPO / "cache" / "r26" / "q3_dense_results.json"

NEG_PATTERNS = re.compile(
    r"\b(no more|not |don'?t|other than|beyond|different|without|avoid|tired of|enough)\b", re.I)
CONT_PATTERNS = re.compile(
    r"\b(more like|similar|another|same vibe|same style|keep|again|continue)\b", re.I)
DISC_PATTERNS = re.compile(
    r"\b(branch out|something else|surprise|new|different genre|change|switch|explore)\b", re.I)


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()
    print(f"{ts()} R37: Scenario Taxonomy Diagnostic")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R22B_OOF) as f:
        r22b_source = json.load(f)
    with open(Q3_LISTS) as f:
        q3_source = json.load(f)

    # Build current pools
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    als_factors, als_track_ids, als_track_to_idx = build_als()
    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

    # Build audio lists
    print(f"{ts()} Building audio retrieval lists...")
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    audio_track_ids = []
    audio_vecs = []
    for item in ds:
        audio_track_ids.append(str(item["track_id"]))
        v = item["audio-laion_clap"]
        if v is not None and len(v) == 512:
            audio_vecs.append(v)
        else:
            audio_vecs.append([0.0] * 512)
    audio_embs = np.array(audio_vecs, dtype=np.float32)
    norms = np.linalg.norm(audio_embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    audio_embs = audio_embs / norms
    audio_tid_to_idx = {tid: i for i, tid in enumerate(audio_track_ids)}
    del ds, audio_vecs

    # Load cached LR scores from R34
    R34_LR = REPO / "cache" / "r34_residual" / "lr_scores.npy"
    lr_scores = np.load(R34_LR)
    print(f"  LR scores: {lr_scores.shape}")

    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]

    print(f"{ts()} Computing scenario features for {len(h7)} hist_7 cases...")

    rows = []
    for i in h7:
        c = cases[i]
        gt = c["gt"]
        played = c["music_turns"]
        query = c["user_query"]

        # Query features
        has_neg = bool(NEG_PATTERNS.search(query))
        has_cont = bool(CONT_PATTERNS.search(query))
        has_disc = bool(DISC_PATTERNS.search(query))

        # History features
        recent3_artists = [ta.get(t, "") for t in played[-3:]]
        all_artists = [ta.get(t, "") for t in played]
        artist_counts = Counter(a for a in all_artists if a)
        n_unique_artists = len(artist_counts)
        dominant_artist = artist_counts.most_common(1)[0] if artist_counts else ("", 0)
        dom_frac = dominant_artist[1] / len(played) if played else 0
        recent3_same = len(set(a for a in recent3_artists if a)) == 1 and recent3_artists[0] != ""

        # GT oracle labels
        gt_artist = ta.get(gt, "")
        gt_same = gt_artist and gt_artist in {ta.get(t, "") for t in played}
        gt_same_recent3 = gt_artist and gt_artist in set(recent3_artists)

        # Pool/source coverage
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_list = []
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_list = [als_track_ids[j] for j in top]

        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_list, "R21": r21_source[i]}
        pool = set(weighted_rrf(sl, sw, topk=300, k=20))
        gt_in_pool = gt in pool

        gt_in_r21 = gt in set(r21_source[i][:300])
        gt_in_r22b = gt in set(r22b_source[i][:300])
        gt_in_q3 = gt in set(q3_source[i][:300])
        gt_in_bm25c = gt in set(payload["src_c"][i][:300])
        gt_in_als = gt in set(als_list[:300])

        # Audio coverage
        recent_audio = played[-3:]
        recent_aidx = [audio_tid_to_idx[t] for t in recent_audio if t in audio_tid_to_idx]
        gt_in_audio = False
        if recent_aidx and gt in audio_tid_to_idx:
            avg = audio_embs[recent_aidx].mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            sims = audio_embs @ avg
            played_aidx = {audio_tid_to_idx[t] for t in played if t in audio_tid_to_idx}
            for pi in played_aidx:
                sims[pi] = -np.inf
            top300 = np.argpartition(-sims, 300)[:300]
            gt_audio_idx = audio_tid_to_idx.get(gt)
            if gt_audio_idx is not None and gt_audio_idx in set(top300):
                gt_in_audio = True

        # LR rank bucket (from cached R34 scores)
        pool_list = list(pool)  # need ordered pool
        sl_ordered = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                      "C": payload["src_c"][i], "D": payload["src_d"][i],
                      "F": payload["src_f"][i], "ALS": als_list, "R21": r21_source[i]}
        pool_list = weighted_rrf(sl_ordered, sw, topk=300, k=20)
        ps = len(pool_list)
        lr_sc = lr_scores[i, :ps]
        lr_rank_bucket = "not_pool"
        lr_rank = -1
        if gt_in_pool and ps > 0:
            gt_pool_idx = pool_list.index(gt) if gt in pool_list else -1
            if gt_pool_idx >= 0:
                lr_ranked = np.argsort(-lr_sc)
                gt_lr_pos = np.where(lr_ranked == gt_pool_idx)[0]
                if len(gt_lr_pos) > 0:
                    lr_rank = int(gt_lr_pos[0]) + 1
                    if lr_rank <= 20:
                        lr_rank_bucket = "top20"
                    elif lr_rank <= 100:
                        lr_rank_bucket = "21_100"
                    else:
                        lr_rank_bucket = "101_300"

        # Observable pool composition features
        pool_top20 = pool_list[:20]
        top20_artists = [ta.get(tid, "") for tid in pool_top20]
        played_artist_set = {ta.get(t, "") for t in played} - {""}
        top20_same_count = sum(1 for a in top20_artists if a and a in played_artist_set)
        top20_same_share = top20_same_count / max(len(pool_top20), 1)
        pool_same_count = sum(1 for tid in pool_list if ta.get(tid, "") in played_artist_set)
        pool_same_share = pool_same_count / max(ps, 1)

        # Scenario classification
        if has_cont and not has_neg and not has_disc:
            scenario = "continuation"
        elif has_neg or has_disc:
            scenario = "pivot"
        else:
            scenario = "neutral"

        rows.append({
            "case_idx": i, "scenario": scenario,
            "has_neg": has_neg, "has_cont": has_cont, "has_disc": has_disc,
            "n_unique_artists": n_unique_artists,
            "dom_frac": round(dom_frac, 2),
            "recent3_same_artist": recent3_same,
            "top20_same_share": round(top20_same_share, 2),
            "pool_same_share": round(pool_same_share, 2),
            "lr_rank": lr_rank, "lr_rank_bucket": lr_rank_bucket,
            "gt_same_artist": bool(gt_same),
            "gt_same_recent3": bool(gt_same_recent3),
            "gt_in_pool": gt_in_pool,
            "gt_in_r21": gt_in_r21, "gt_in_r22b": gt_in_r22b,
            "gt_in_q3": gt_in_q3, "gt_in_bm25c": gt_in_bm25c,
            "gt_in_als": gt_in_als, "gt_in_audio": gt_in_audio,
        })

    # ---------------------------------------------------------------
    # Table 1: Scenario × gt_same_artist rate
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("TABLE 1: Scenario × GT Same-Artist Rate")
    print(sep)
    scenarios = ["continuation", "pivot", "neutral"]
    print(f"  {'Scenario':<15} {'count':>6} {'gt_same%':>10} {'gt_diff%':>10}")
    print(f"  {'-'*41}")
    for sc in scenarios:
        subset = [r for r in rows if r["scenario"] == sc]
        if not subset:
            continue
        same_rate = sum(1 for r in subset if r["gt_same_artist"]) / len(subset)
        print(f"  {sc:<15} {len(subset):>6} {same_rate*100:>9.1f}% {(1-same_rate)*100:>9.1f}%")
    all_same = sum(1 for r in rows if r["gt_same_artist"]) / len(rows)
    print(f"  {'ALL':<15} {len(rows):>6} {all_same*100:>9.1f}% {(1-all_same)*100:>9.1f}%")

    # ---------------------------------------------------------------
    # Table 2: Scenario × source coverage
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 2: Scenario × Source GT Coverage (h7)")
    print(sep)
    sources_to_check = ["gt_in_pool", "gt_in_r21", "gt_in_audio", "gt_in_r22b",
                        "gt_in_q3", "gt_in_bm25c", "gt_in_als"]
    print(f"  {'Scenario':<15} " + " ".join(f"{s.replace('gt_in_',''):>8}" for s in sources_to_check))
    print(f"  {'-'*79}")
    for sc in scenarios:
        subset = [r for r in rows if r["scenario"] == sc]
        if not subset:
            continue
        counts = " ".join(
            f"{sum(1 for r in subset if r[s])/len(subset)*100:>7.1f}%"
            for s in sources_to_check)
        print(f"  {sc:<15} {counts}")
    counts_all = " ".join(
        f"{sum(1 for r in rows if r[s])/len(rows)*100:>7.1f}%"
        for s in sources_to_check)
    print(f"  {'ALL':<15} {counts_all}")

    # ---------------------------------------------------------------
    # Table 3: Audio helps/hurts by scenario
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 3: Audio-Only GTs by Scenario (in audio but not pool)")
    print(sep)
    print(f"  {'Scenario':<15} {'audio_only':>12} {'audio_unique_same':>18} {'audio_unique_diff':>18}")
    print(f"  {'-'*63}")
    for sc in scenarios:
        subset = [r for r in rows if r["scenario"] == sc]
        if not subset:
            continue
        audio_only = sum(1 for r in subset if r["gt_in_audio"] and not r["gt_in_pool"])
        audio_only_same = sum(1 for r in subset if r["gt_in_audio"] and not r["gt_in_pool"]
                             and r["gt_same_artist"])
        audio_only_diff = sum(1 for r in subset if r["gt_in_audio"] and not r["gt_in_pool"]
                             and not r["gt_same_artist"])
        print(f"  {sc:<15} {audio_only:>12} {audio_only_same:>18} {audio_only_diff:>18}")

    # ---------------------------------------------------------------
    # Table 4: Baseline miss analysis by scenario
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 4: Baseline Miss Buckets by Scenario")
    print(sep)
    print(f"  {'Scenario':<15} {'in_pool':>8} {'not_pool':>9} {'not_any_src':>12}")
    print(f"  {'-'*44}")
    for sc in scenarios:
        subset = [r for r in rows if r["scenario"] == sc]
        if not subset:
            continue
        in_pool = sum(1 for r in subset if r["gt_in_pool"])
        not_pool_in_src = sum(1 for r in subset if not r["gt_in_pool"] and
                             any(r[s] for s in ["gt_in_r21", "gt_in_r22b", "gt_in_q3",
                                                "gt_in_bm25c", "gt_in_als", "gt_in_audio"]))
        not_any = sum(1 for r in subset if not any(r[s] for s in
                      ["gt_in_pool", "gt_in_r21", "gt_in_r22b", "gt_in_q3",
                       "gt_in_bm25c", "gt_in_als", "gt_in_audio"]))
        print(f"  {sc:<15} {in_pool:>8} {not_pool_in_src:>9} {not_any:>12}")

    # ---------------------------------------------------------------
    # Table 5: High-precision routing candidates
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 5: Oracle Diagnostics (uses GT labels, not observable at inference)")
    print(sep)

    rules = [
        ("pivot + audio_has_gt", lambda r: r["scenario"] == "pivot" and r["gt_in_audio"]),
        ("pivot + diff_artist", lambda r: r["scenario"] == "pivot" and not r["gt_same_artist"]),
        ("neutral + diff_artist", lambda r: r["scenario"] == "neutral" and not r["gt_same_artist"]),
        ("continuation + same_artist", lambda r: r["scenario"] == "continuation" and r["gt_same_artist"]),
        ("recent3_same + gt_same", lambda r: r["recent3_same_artist"] and r["gt_same_artist"]),
        ("recent3_same + gt_diff", lambda r: r["recent3_same_artist"] and not r["gt_same_artist"]),
        ("dom_frac>=0.5 + gt_same", lambda r: r["dom_frac"] >= 0.5 and r["gt_same_artist"]),
        ("dom_frac>=0.5 + gt_diff", lambda r: r["dom_frac"] >= 0.5 and not r["gt_same_artist"]),
        ("audio_only (not in pool)", lambda r: r["gt_in_audio"] and not r["gt_in_pool"]),
    ]

    print(f"  {'Rule':<35} {'match':>6} {'gt_same%':>10} {'audio_cov%':>11}")
    print(f"  {'-'*62}")
    for rule_name, rule_fn in rules:
        matched = [r for r in rows if rule_fn(r)]
        if not matched:
            print(f"  {rule_name:<35} {0:>6}")
            continue
        same_rate = sum(1 for r in matched if r["gt_same_artist"]) / len(matched)
        audio_cov = sum(1 for r in matched if r["gt_in_audio"]) / len(matched)
        print(f"  {rule_name:<35} {len(matched):>6} {same_rate*100:>9.1f}% {audio_cov*100:>10.1f}%")

    # ---------------------------------------------------------------
    # Table 5b: LR rank bucket distribution
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 5b: LR Rank Bucket Distribution (h7)")
    print(sep)
    for bucket in ["top20", "21_100", "101_300", "not_pool"]:
        subset = [r for r in rows if r["lr_rank_bucket"] == bucket]
        if not subset:
            continue
        same_rate = sum(1 for r in subset if r["gt_same_artist"]) / len(subset)
        audio_rate = sum(1 for r in subset if r["gt_in_audio"]) / len(subset)
        print(f"  {bucket:<12} {len(subset):>6} same={same_rate*100:.1f}% audio={audio_rate*100:.1f}%")

    # ---------------------------------------------------------------
    # Table 6: Observable router evaluation
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("TABLE 6: Observable Router Rules → Oracle Outcomes")
    print(sep)

    obs_rules = [
        ("has_neg", lambda r: r["has_neg"]),
        ("has_disc", lambda r: r["has_disc"]),
        ("has_cont", lambda r: r["has_cont"]),
        ("scenario=pivot", lambda r: r["scenario"] == "pivot"),
        ("scenario=continuation", lambda r: r["scenario"] == "continuation"),
        ("recent3_same", lambda r: r["recent3_same_artist"]),
        ("dom_frac>=0.5", lambda r: r["dom_frac"] >= 0.5),
        ("n_unique<=2", lambda r: r["n_unique_artists"] <= 2),
        ("n_unique>=5", lambda r: r["n_unique_artists"] >= 5),
        ("top20_same>=0.5", lambda r: r["top20_same_share"] >= 0.5),
        ("top20_same<0.1", lambda r: r["top20_same_share"] < 0.1),
        ("pool_same>=0.5", lambda r: r["pool_same_share"] >= 0.5),
    ]

    print(f"  {'Rule':<22} {'n':>5} {'same%':>7} {'diff%':>7} {'pool%':>7} "
          f"{'lr20%':>7} {'miss%':>7} {'audio%':>7} {'q3%':>7} {'r22b%':>7}")
    print(f"  {'-'*96}")

    global_n = len(rows)
    global_same = sum(1 for r in rows if r["gt_same_artist"]) / global_n
    global_pool = sum(1 for r in rows if r["gt_in_pool"]) / global_n
    global_lr20 = sum(1 for r in rows if r["lr_rank_bucket"] == "top20") / global_n
    global_miss = sum(1 for r in rows if r["gt_in_pool"] and r["lr_rank_bucket"] != "top20") / global_n
    global_audio = sum(1 for r in rows if r["gt_in_audio"]) / global_n
    global_q3 = sum(1 for r in rows if r["gt_in_q3"]) / global_n
    global_r22b = sum(1 for r in rows if r["gt_in_r22b"]) / global_n

    print(f"  {'GLOBAL':<22} {global_n:>5} {global_same*100:>6.1f}% {(1-global_same)*100:>6.1f}% "
          f"{global_pool*100:>6.1f}% {global_lr20*100:>6.1f}% {global_miss*100:>6.1f}% "
          f"{global_audio*100:>6.1f}% {global_q3*100:>6.1f}% {global_r22b*100:>6.1f}%")

    for rule_name, rule_fn in obs_rules:
        matched = [r for r in rows if rule_fn(r)]
        if len(matched) < 20:
            continue
        nm = len(matched)
        same = sum(1 for r in matched if r["gt_same_artist"]) / nm
        pool = sum(1 for r in matched if r["gt_in_pool"]) / nm
        lr20 = sum(1 for r in matched if r["lr_rank_bucket"] == "top20") / nm
        miss = sum(1 for r in matched if r["gt_in_pool"] and r["lr_rank_bucket"] != "top20") / nm
        audio = sum(1 for r in matched if r["gt_in_audio"]) / nm
        q3 = sum(1 for r in matched if r["gt_in_q3"]) / nm
        r22b = sum(1 for r in matched if r["gt_in_r22b"]) / nm
        print(f"  {rule_name:<22} {nm:>5} {same*100:>6.1f}% {(1-same)*100:>6.1f}% "
              f"{pool*100:>6.1f}% {lr20*100:>6.1f}% {miss*100:>6.1f}% "
              f"{audio*100:>6.1f}% {q3*100:>6.1f}% {r22b*100:>6.1f}%")

    out_path = REPO / "exp" / "eval" / "expR37_scenario_taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{ts()} Saved {len(rows)} rows to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

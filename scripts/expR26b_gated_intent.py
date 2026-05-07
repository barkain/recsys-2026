#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R26b: Gated intent retrieval — test 5 routing variants.

A. cold_only:             n_prior_music <= 1
B. explicit_negative:     negative_artists/avoid/pivot language
C. cold_or_negative:      A OR B
D. strict_pivot:          B AND (R21/Q3 overlap < 12 OR R21 top-1 violates negative)
E. feature_only:          no Q3 in pool, only intent features in LambdaRank

Primary gate: last_turn >= R21 + 0.005 (blind is last-turn only).
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
Q3_LISTS = REPO / "cache" / "r26" / "q3_dense_results.json"
INTENTS_DEV = REPO / "cache" / "r26" / "intents_dev.json"
RRF_K = 20
POOL_K = 300

SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
SW_GATED = {**SW_BASE, "Q3": 0.5}

FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
INTENT_FEATURE_NAMES = [
    "q3_rank_inv", "q3_presence",
    "intent_pos_artist_match", "intent_genre_match",
    "intent_neg_artist_violation", "intent_avoid_violation",
    "q3_r21_top20_agree", "is_gated",
]
FEATURE_NAMES_GATED = FEATURE_NAMES_R21 + INTENT_FEATURE_NAMES

PIVOT_PHRASES = ["no more", "not ", "different", "branch out", "tired of",
                 "other than", "instead of", "change", "switch", "something else",
                 "move away", "avoid"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_intents():
    with open(INTENTS_DEV) as f:
        return {(r["session_id"], r["turn_number"]): r.get("intent") for r in json.load(f)}


def has_explicit_negative(intent, user_query=""):
    """Check if intent has explicit negatives or pivot language."""
    if not intent:
        return False
    if intent.get("negative_artists"):
        return True
    for field in ["avoid", "must_have"]:
        for item in intent.get(field, []):
            item_lower = str(item).lower()
            if any(p in item_lower for p in PIVOT_PHRASES):
                return True
    query_lower = user_query.lower()
    if any(p in query_lower for p in PIVOT_PHRASES[:6]):
        return True
    return False


def compute_gate_masks(cases, intent_map, r21_source, q3_source):
    """Compute per-case gate masks for all variants."""
    n = len(cases)
    masks = {v: np.zeros(n, dtype=bool) for v in ["A", "B", "C", "D", "E"]}

    for i, c in enumerate(cases):
        intent = intent_map.get((c["session_id"], c["turn_number"]))
        cold = c["n_prior_music"] <= 1
        neg = has_explicit_negative(intent, c["user_query"])

        masks["A"][i] = cold
        masks["B"][i] = neg
        masks["C"][i] = cold or neg

        # D: strict pivot — neg AND (low R21/Q3 overlap OR R21 top-1 violates negative)
        if neg and intent:
            r21_top20 = set(r21_source[i][:20])
            q3_top20 = set(q3_source[i][:20])
            overlap = len(r21_top20 & q3_top20)
            masks["D"][i] = overlap < 12
        # E: feature_only — never admit Q3 to pool
        masks["E"][i] = False

    for v in masks:
        print(f"  Gate {v}: {int(masks[v].sum())}/{n} ({masks[v].mean()*100:.1f}%)")
    return masks


def build_features_variant(cases, payload, als_source, als_vecs, als_factors,
                           als_track_to_idx, track_pop, r21_source, q3_source,
                           intent_map, gate_mask, feature_names):
    n = len(cases)
    n_feat = len(feature_names)
    has_intent = "q3_rank_inv" in feature_names

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        intent = intent_map.get((c["session_id"], c["turn_number"]))
        gated = bool(gate_mask[i]) if gate_mask is not None else False

        src_lists: dict[str, list[str]] = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        sw = SW_BASE
        if gated:
            src_lists["Q3"] = q3_source[i]
            sw = SW_GATED

        pool = weighted_rrf(src_lists, sw, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank: dict[str, dict[str, int]] = {}
        for sname, slist in src_lists.items():
            src_rank[sname] = {tid: rank + 1 for rank, tid in enumerate(slist)}

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior_list = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

        pos_artists_lower = set()
        neg_artists_lower = set()
        genres_lower = set()
        avoid_lower = set()
        if intent:
            pos_artists_lower = {a.lower() for a in intent.get("positive_artists", [])}
            neg_artists_lower = {a.lower() for a in intent.get("negative_artists", [])}
            genres_lower = {g.lower() for g in intent.get("genres", [])}
            avoid_lower = {a.lower() for a in intent.get("avoid", [])}

        q3_rank_map = {tid: r + 1 for r, tid in enumerate(q3_source[i][:300])}
        r21_top20 = set(r21_source[i][:20])

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior_list:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sname in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sname, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

            if has_intent:
                ca_lower = ca.lower() if ca else ""
                ct_lower = {t.lower() for t in ct} if ct else set()
                row[29] = 1.0 / q3_rank_map[tid] if tid in q3_rank_map else 0.0
                row[30] = 1.0 if tid in q3_rank_map else 0.0
                row[31] = 1.0 if ca_lower and ca_lower in pos_artists_lower else 0.0
                row[32] = float(len(ct_lower & genres_lower)) / max(len(genres_lower), 1) if genres_lower else 0.0
                row[33] = 1.0 if ca_lower and ca_lower in neg_artists_lower else 0.0
                row[34] = 1.0 if ca_lower and any(ca_lower in a or a in ca_lower for a in avoid_lower) else 0.0
                row[35] = 1.0 if tid in q3_rank_map and tid in r21_top20 else 0.0
                row[36] = 1.0 if gated else 0.0

    return X, gt_idx, sizes


def run_cv(X, gt_idx, sizes, cases, sessions, feature_names, intent_map, seed=0):
    n = X.shape[0]
    folds = grouped_session_folds(sessions, seed=seed)
    case_ndcg = np.zeros(n)

    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        train_list = [j for j in range(n) if j not in val_idx]
        val_list = sorted(val_idx)

        X_flat_train, y_train, g_train = [], [], []
        X_flat_val, y_val, g_val = [], [], []

        for idx in train_list:
            s = int(sizes[idx])
            for k in range(s):
                X_flat_train.append(X[idx, k])
                y_train.append(1.0 if k == gt_idx[idx] else 0.0)
            g_train.append(s)

        for idx in val_list:
            s = int(sizes[idx])
            for k in range(s):
                X_flat_val.append(X[idx, k])
                y_val.append(1.0 if k == gt_idx[idx] else 0.0)
            g_val.append(s)

        ds_tr = lgb.Dataset(np.array(X_flat_train), label=np.array(y_train),
                            group=g_train, feature_name=list(feature_names))
        ds_va = lgb.Dataset(np.array(X_flat_val), label=np.array(y_val),
                            group=g_val, reference=ds_tr)

        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "eval_at": [20], "num_leaves": 31, "learning_rate": 0.05,
            "min_data_in_leaf": 10, "verbose": -1, "seed": seed,
        }
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        preds_va = model.predict(np.array(X_flat_val))
        offset = 0
        for idx in val_list:
            s = int(sizes[idx])
            sc = preds_va[offset:offset + s]
            offset += s
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    slices = {}
    for depth in range(8):
        idx_list = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
        if idx_list:
            slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx_list]))

    return {
        "cv5": float(np.mean(case_ndcg)),
        "last_turn": slices.get("hist_7", 0),
        "hist_0": slices.get("hist_0", 0),
        "hist_7": slices.get("hist_7", 0),
        "slices": slices,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R26b: Gated Intent Retrieval (5 variants)")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(Q3_LISTS) as f:
        q3_source = json.load(f)
    intent_map = load_intents()

    print(f"\n{ts()} Gate coverage:")
    masks = compute_gate_masks(cases, intent_map, r21_source, q3_source)

    print(f"\n{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()
    configs: dict[str, dict] = {}

    # Baseline: R21 only
    print(f"\n{ts()} Baseline: R21 (29 features)")
    no_gate = np.zeros(n, dtype=bool)
    X0, gt0, sz0 = build_features_variant(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, q3_source,
        intent_map, no_gate, FEATURE_NAMES_R21)
    pool_hit_base = float(np.mean(gt0 >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit_base:.4f}")
    r0 = run_cv(X0, gt0, sz0, cases, sessions, FEATURE_NAMES_R21, intent_map)
    r0["pool_hit"] = pool_hit_base
    configs["R21_base"] = r0
    print(f"  CV5={r0['cv5']:.5f}  last_turn={r0['last_turn']:.5f}  hist_0={r0['hist_0']:.5f}")

    # Variants A-E
    variant_configs = [
        ("A_cold", masks["A"], FEATURE_NAMES_GATED, "cold only (n<=1)"),
        ("B_neg", masks["B"], FEATURE_NAMES_GATED, "explicit negative/pivot"),
        ("C_cold_neg", masks["C"], FEATURE_NAMES_GATED, "cold OR negative"),
        ("D_strict", masks["D"], FEATURE_NAMES_GATED, "strict pivot (neg + low overlap)"),
        ("E_feat_only", no_gate, FEATURE_NAMES_GATED, "intent features only, no Q3 pool"),
    ]

    for name, mask, feat_names, desc in variant_configs:
        n_gated = int(mask.sum())
        print(f"\n{ts()} {name}: {desc} ({n_gated}/{n} = {n_gated/n*100:.1f}%)")
        X, gt, sz = build_features_variant(
            cases, payload, als_source, als_vecs, als_factors,
            als_track_to_idx, track_pop, r21_source, q3_source,
            intent_map, mask, feat_names)
        pool_hit = float(np.mean(gt >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")
        r = run_cv(X, gt, sz, cases, sessions, feat_names, intent_map)
        r["pool_hit"] = pool_hit
        r["n_gated"] = n_gated
        configs[name] = r
        print(f"  CV5={r['cv5']:.5f}  last_turn={r['last_turn']:.5f}  hist_0={r['hist_0']:.5f}")

    # Summary table
    sep = "=" * 70
    print(f"\n{sep}")
    print("R26b GATED INTENT RETRIEVAL — ALL VARIANTS")
    print(sep)
    header = f"  {'Variant':<18} {'gated%':>7} {'pool':>8} {'CV5':>10} {'last_t':>10} {'h0':>10} {'h7':>10} {'Dp':>8} {'Dl':>8}"
    print(header)
    print(f"  {'-'*89}")
    for name, r in configs.items():
        pct = r.get("n_gated", 0) / n * 100 if "n_gated" in r else 0
        dp = r["pool_hit"] - pool_hit_base
        dl = r["last_turn"] - r0["last_turn"]
        print(f"  {name:<18} {pct:>6.1f}% {r['pool_hit']:>8.4f} {r['cv5']:>10.5f} "
              f"{r['last_turn']:>10.5f} {r['hist_0']:>10.5f} {r['hist_7']:>10.5f} "
              f"{dp:>+8.4f} {dl:>+8.5f}")

    # Gate check (strict)
    print(f"\n{sep}")
    print("STRICT GATE CHECK (primary: last_turn)")
    best_name = None
    best_last = -1.0
    for name, r in configs.items():
        if name == "R21_base":
            continue
        last_d = r["last_turn"] - r0["last_turn"]
        pool_d = r["pool_hit"] - pool_hit_base
        cv5_d = r["cv5"] - r0["cv5"]
        hist7_d = r["hist_7"] - r0["hist_7"]
        g_last = last_d >= 0.005
        g_pool = pool_d >= 0.010
        g_cv5 = cv5_d >= 0.0
        g_hist7 = hist7_d >= -0.001
        status = "PASS" if (g_last and g_pool and g_cv5 and g_hist7) else "FAIL"
        print(f"  {name:<18} last={last_d:+.5f}{'*' if g_last else ' '} "
              f"pool={pool_d:+.4f}{'*' if g_pool else ' '} "
              f"cv5={cv5_d:+.5f}{'*' if g_cv5 else ' '} "
              f"h7={hist7_d:+.5f}{'*' if g_hist7 else ' '} → {status}")
        if g_last and r["last_turn"] > best_last:
            best_last = r["last_turn"]
            best_name = name

    if best_name:
        print(f"\n  BEST: {best_name} (last_turn={best_last:.5f})")
    else:
        print("\n  NO VARIANT PASSES last_turn gate. Do not submit.")

    out_path = REPO / "exp" / "eval" / "expR26b_gated_intent.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": configs, "best": best_name}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

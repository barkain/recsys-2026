#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R16 Stage 2: LambdaRank fusion with graph retrieval candidates.

Tests whether graph-based pool construction lifts nDCG through LambdaRank.
Includes ablation: graph candidates WITHOUT graph features (pool-only gain).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20

FEATURE_NAMES_R16 = FEATURE_NAMES_V2 + [
    "graph_rank_inv",
    "graph_presence",
    "graph_score_norm",
    "graph_degree_log",
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_graph_from_train(variant="full"):
    """Build graph and return adjacency + indices."""
    from datasets import DownloadConfig, load_dataset, Dataset, concatenate_datasets

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]

    all_tracks = set()
    all_sessions = set()
    all_users = set()
    session_tracks = {}
    session_users = {}

    for item in train:
        sid = str(item["session_id"])
        uid = str(item.get("user_id", ""))
        all_sessions.add(sid)
        if uid:
            all_users.add(uid)
            session_users[sid] = uid
        tracks = [str(c["content"]).strip() for c in item["conversations"] if c["role"] == "music"]
        session_tracks[sid] = tracks
        all_tracks.update(tracks)

    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    meta_ds = concatenate_datasets(splits)
    mcols = meta_ds.to_dict()

    all_artists = set()
    all_tags = set()
    track_artists = {}
    track_tags_map = {}
    for i in range(len(mcols["track_id"])):
        tid = str(mcols["track_id"][i])
        artists = [str(a) for a in (mcols["artist_id"][i] or [])]
        tags = [str(t) for t in (mcols["tag_list"][i] or [])[:10]]
        track_artists[tid] = artists
        track_tags_map[tid] = tags
        all_artists.update(artists)
        all_tags.update(tags)

    track_list = sorted(all_tracks)
    track_idx = {t: i for i, t in enumerate(track_list)}
    n_tracks = len(track_list)

    session_list = sorted(all_sessions)
    session_idx = {s: n_tracks + i for i, s in enumerate(session_list)}
    user_list = sorted(all_users)
    user_idx = {u: n_tracks + len(session_list) + i for i, u in enumerate(user_list)}
    artist_list = sorted(all_artists)
    artist_idx = {a: n_tracks + len(session_list) + len(user_list) + i for i, a in enumerate(artist_list)}
    tag_list_sorted = sorted(all_tags)
    tag_idx = {t: n_tracks + len(session_list) + len(user_list) + len(artist_list) + i
               for i, t in enumerate(tag_list_sorted)}
    N = n_tracks + len(session_list) + len(user_list) + len(artist_list) + len(tag_list_sorted)

    rows, cols_list, weights = [], [], []

    include_cooccur = variant in ("full", "cooccur_only", "behavioral")
    include_transition = variant in ("full", "behavioral")
    include_user = variant in ("full", "behavioral")
    include_session = variant in ("full", "behavioral")
    include_artist = variant in ("full",)
    include_tag = variant in ("full",)

    if include_session:
        for sid, tracks in session_tracks.items():
            if sid not in session_idx: continue
            si = session_idx[sid]
            for pos, tid in enumerate(tracks):
                if tid not in track_idx: continue
                ti = track_idx[tid]
                w = 1.0 / (pos + 1)
                rows.extend([si, ti]); cols_list.extend([ti, si]); weights.extend([w, w])

    if include_cooccur:
        cooccur = Counter()
        for sid, tracks in session_tracks.items():
            unique = list(dict.fromkeys(t for t in tracks if t in track_idx))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    cooccur[(unique[i], unique[j])] += 1
        for (t1, t2), cnt in cooccur.items():
            i1, i2 = track_idx[t1], track_idx[t2]
            rows.extend([i1, i2]); cols_list.extend([i2, i1]); weights.extend([float(cnt)] * 2)

    if include_transition:
        trans = Counter()
        for sid, tracks in session_tracks.items():
            for i in range(len(tracks) - 1):
                if tracks[i] in track_idx and tracks[i+1] in track_idx:
                    trans[(tracks[i], tracks[i+1])] += 1
        for (t1, t2), cnt in trans.items():
            rows.append(track_idx[t1]); cols_list.append(track_idx[t2])
            weights.append(float(cnt) * 2.0)

    if include_user:
        ut = Counter()
        for sid, tracks in session_tracks.items():
            uid = session_users.get(sid)
            if not uid or uid not in user_idx: continue
            for tid in tracks:
                if tid in track_idx: ut[(uid, tid)] += 1
        for (uid, tid), cnt in ut.items():
            ui, ti = user_idx[uid], track_idx[tid]
            rows.extend([ui, ti]); cols_list.extend([ti, ui])
            weights.extend([float(cnt), float(cnt) * 0.5])

    if include_artist:
        for tid in track_list:
            if tid not in track_artists: continue
            ti = track_idx[tid]
            for aid in track_artists[tid]:
                if aid in artist_idx:
                    ai = artist_idx[aid]
                    rows.extend([ti, ai]); cols_list.extend([ai, ti]); weights.extend([1.0, 1.0])

    if include_tag:
        for tid in track_list:
            if tid not in track_tags_map: continue
            ti = track_idx[tid]
            for tag in track_tags_map[tid]:
                if tag in tag_idx:
                    tgi = tag_idx[tag]
                    rows.extend([ti, tgi]); cols_list.extend([tgi, ti]); weights.extend([0.3, 0.3])

    adj = sparse.csr_matrix((weights, (rows, cols_list)), shape=(N, N), dtype=np.float32)
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    adj_norm = sparse.diags(1.0 / row_sums, format="csr") @ adj

    track_degrees = np.array(adj[:n_tracks].sum(axis=1)).flatten()

    return adj_norm, track_list, track_idx, n_tracks, track_degrees


def ppr(adj_norm, seed_indices, seed_weights, n_tracks, alpha=0.15, max_iter=20, tol=1e-6):
    N = adj_norm.shape[0]
    p = np.zeros(N, dtype=np.float32)
    for idx, w in zip(seed_indices, seed_weights):
        p[idx] = w
    if p.sum() > 0:
        p /= p.sum()
    r = p.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * (adj_norm.T @ r) + alpha * p
        if np.abs(r_new - r).sum() < tol:
            break
        r = r_new
    return r[:n_tracks]


def run_ppr_for_cases(adj_norm, cases, track_idx, track_list, n_tracks, alpha=0.15):
    """Run PPR for all cases, return ranked lists and raw scores."""
    ppr_lists = []
    ppr_scores_per_case = []
    for c in cases:
        played = c["music_turns"]
        seeds, weights = [], []
        for j, tid in enumerate(reversed(played)):
            if tid in track_idx:
                seeds.append(track_idx[tid])
                weights.append(1.0 / (j + 1))
        if not seeds:
            ppr_lists.append([])
            ppr_scores_per_case.append({})
            continue

        scores = ppr(adj_norm, seeds, weights, n_tracks, alpha=alpha)
        for tid in played:
            if tid in track_idx:
                scores[track_idx[tid]] = 0.0

        top_k = min(500, max(1, (scores > 0).sum()))
        top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        result = [track_list[j] for j in top_idx if scores[j] > 0]
        score_map = {track_list[j]: float(scores[j]) for j in top_idx if scores[j] > 0}
        ppr_lists.append(result)
        ppr_scores_per_case.append(score_map)
    return ppr_lists, ppr_scores_per_case


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx: scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

    # Build graph variants
    graph_configs = {
        "cooccur_only": "cooccur_only",
        "behavioral": "behavioral",
        "full": "full",
    }
    graph_data = {}
    for gname, gvariant in graph_configs.items():
        print(f"\n{ts()} Building graph: {gname}...", flush=True)
        adj, tl, ti, nt, deg = build_graph_from_train(gvariant)
        print(f"{ts()} Running PPR for {n} cases...", flush=True)
        lists, scores = run_ppr_for_cases(adj, cases, ti, tl, nt, alpha=0.15)
        graph_data[gname] = {"lists": lists, "scores": scores, "track_idx": ti, "degrees": deg}
        print(f"  Done.", flush=True)
        del adj

    n_feat_v3 = len(FEATURE_NAMES_V2)
    n_feat_r16 = len(FEATURE_NAMES_R16)

    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    # Configs to test
    configs = [
        {"name": "v3_baseline", "graph": None, "graph_w": 0, "use_graph_feats": False, "pool_k": 200},
        {"name": "v3+cooccur_w0.5", "graph": "cooccur_only", "graph_w": 0.5, "use_graph_feats": True, "pool_k": 200},
        {"name": "v3+cooccur_w0.25", "graph": "cooccur_only", "graph_w": 0.25, "use_graph_feats": True, "pool_k": 200},
        {"name": "v3+behavioral_w0.5", "graph": "behavioral", "graph_w": 0.5, "use_graph_feats": True, "pool_k": 200},
        {"name": "v3+full_w0.5", "graph": "full", "graph_w": 0.5, "use_graph_feats": True, "pool_k": 200},
        {"name": "v3+full_w0.25", "graph": "full", "graph_w": 0.25, "use_graph_feats": True, "pool_k": 200},
        # Ablation: graph pool WITHOUT graph features
        {"name": "v3+full_w0.5_NO_FEATS", "graph": "full", "graph_w": 0.5, "use_graph_feats": False, "pool_k": 200},
    ]

    results = {}
    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }

    for cfg in configs:
        cname = cfg["name"]
        graph_name = cfg["graph"]
        graph_w = cfg["graph_w"]
        use_feats = cfg["use_graph_feats"]
        pool_k = cfg["pool_k"]
        feat_names = FEATURE_NAMES_R16 if use_feats else FEATURE_NAMES_V2
        nf = len(feat_names)

        print(f"\n{ts()} === {cname} ===", flush=True)

        gd = graph_data.get(graph_name)
        g_lists = gd["lists"] if gd else [[] for _ in range(n)]
        g_scores = gd["scores"] if gd else [{} for _ in range(n)]
        g_ti = gd["track_idx"] if gd else {}
        g_deg = gd["degrees"] if gd else np.zeros(0)

        X = np.zeros((n, pool_k, nf), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            src_lists = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
            }
            if graph_name:
                src_lists["G_PPR"] = g_lists[i]
                sw = dict(base_weights)
                sw["G_PPR"] = graph_w
            else:
                sw = base_weights

            pool = weighted_rrf(src_lists, sw, topk=pool_k, k=RRF_K)
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])

            src_rank = {sn: {tid: r+1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}
            user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                         + [c["user_query"]])
            played = c["music_turns"]
            n_hist = len(played)
            now_tok = tokens(user_msgs[-1]) if user_msgs else set()
            all_tok = tokens(" ".join(user_msgs))
            played_set = set(played)
            l_artist = ta.get(played[-1], "") if played else ""
            l_tags = tt.get(played[-1], set()) if played else set()
            prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j, t in enumerate(reversed(played))]
            sv = als_vecs[i]
            pool_artists = [ta.get(tid, "") for tid in pool[:pool_k]]
            artist_counts = Counter(a for a in pool_artists if a)

            g_rank = {tid: r+1 for r, tid in enumerate(g_lists[i][:500])} if graph_name else {}

            for rank, tid in enumerate(pool[:pool_k], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank-1]
                row[0] = 1.0/rank
                row[1] = 1.0 if ca and ca == l_artist else 0.0
                if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
                row[3] = float(len(tat.get(tid, set()) & now_tok))
                row[4] = float(len(ttl.get(tid, set()) & now_tok))
                row[5] = float(len(tmt.get(tid, set()) & all_tok))
                row[6] = 1.0 if tid in played_set else 0.0
                rec = 0.0
                for wd, pa, pt in prior:
                    am = 1.0 if ca and ca == pa else 0.0
                    tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                    rec += wd * (am + tm)
                row[7] = rec
                for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                    sr = src_rank[sn].get(tid)
                    row[8+fi] = 1.0/sr if sr else 0.0
                for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                    row[14+fi] = 1.0 if tid in src_rank[sn] else 0.0
                row[20] = sum(1 for sn in ["A","B","C","D","F","ALS"] if tid in src_rank[sn])
                if sv is not None:
                    aidx = als_track_to_idx.get(tid)
                    if aidx is not None: row[21] = float(np.dot(sv, als_factors[aidx]))
                row[22] = float(n_hist)
                pop = track_pop.get(tid, 0)
                row[23] = pop / max_pop
                row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
                row[25] = float(artist_counts.get(ca, 0)) if ca else 0
                row[26] = row[20]

                if use_feats:
                    row[27] = 1.0/g_rank[tid] if tid in g_rank else 0.0
                    row[28] = 1.0 if tid in g_rank else 0.0
                    row[29] = g_scores[i].get(tid, 0.0) * 1000  # normalize
                    tidx = g_ti.get(tid)
                    row[30] = np.log2(g_deg[tidx] + 2) if (tidx is not None and tidx < len(g_deg)) else 0.0

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{pool_k}: {pool_hit:.4f}", flush=True)

        # Grouped-session CV5
        X_flat = X.reshape(-1, nf)
        labels = np.zeros(n * pool_k, dtype=np.float32)
        for i in range(n):
            if gt_idx[i] >= 0:
                labels[i * pool_k + gt_idx[i]] = 1.0

        seeds = [0, 1, 2]
        cv5_seeds, lt_seeds = [], []
        slice_data = defaultdict(list)

        for seed in seeds:
            folds = grouped_session_folds(sessions, seed)
            fold_ndcgs = []
            for fold in folds:
                held = set(fold.tolist())
                train_c = [j for j in range(n) if j not in held]
                val_c = fold.tolist()
                train_flat = [j*pool_k+k for j in train_c for k in range(int(sizes[j]))]
                val_flat = [j*pool_k+k for j in val_c for k in range(int(sizes[j]))]
                g_train = np.array([int(sizes[j]) for j in train_c], dtype=np.int32)
                g_val = np.array([int(sizes[j]) for j in val_c], dtype=np.int32)
                dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat],
                                     group=g_train, feature_name=feat_names, free_raw_data=False)
                dval = lgb.Dataset(X_flat[val_flat], labels[val_flat],
                                   group=g_val, reference=dtrain, free_raw_data=False)
                model = lgb.train(lgb_params, dtrain, num_boost_round=300,
                                  valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
                val_scores = model.predict(X_flat[val_flat])
                offset = 0
                case_ndcgs = []
                for j in val_c:
                    sz = int(sizes[j])
                    if sz == 0:
                        case_ndcgs.append(0.0); continue
                    sc = val_scores[offset:offset+sz]
                    gt = gt_idx[j]
                    if gt >= 0:
                        gt_score = sc[gt]
                        rank0 = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                        ndcg = 1.0/np.log2(rank0+2) if rank0 < 20 else 0.0
                    else:
                        ndcg = 0.0
                    case_ndcgs.append(ndcg)

                    played = cases[j]["music_turns"]
                    gt_tid = cases[j]["gt"]
                    if len(played) == 7:
                        slice_data[f"{cname}_lt"].append(ndcg)
                    if played:
                        la = ta.get(played[-1], "")
                        ga = ta.get(gt_tid, "")
                        if isinstance(la, list): la = la[0] if la else ""
                        if isinstance(ga, list): ga = ga[0] if ga else ""
                        if ga and la:
                            if ga == la: slice_data[f"{cname}_sa"].append(ndcg)
                            else: slice_data[f"{cname}_da"].append(ndcg)
                    if len(played) == 0:
                        slice_data[f"{cname}_h0"].append(ndcg)
                    if track_pop.get(gt_tid, 0) == 0:
                        slice_data[f"{cname}_p0"].append(ndcg)
                    offset += sz
                fold_ndcgs.append(float(np.mean(case_ndcgs)))
            cv5_seeds.append(float(np.mean(fold_ndcgs)))
            lt_vals = slice_data.get(f"{cname}_lt", [])
            if lt_vals:
                lt_seeds.append(float(np.mean(lt_vals[-len(lt_vals)//len(seeds):])))

        cv5 = float(np.mean(cv5_seeds))
        lt = float(np.mean(slice_data.get(f"{cname}_lt", [0])))
        sa = float(np.mean(slice_data.get(f"{cname}_sa", [0])))
        da = float(np.mean(slice_data.get(f"{cname}_da", [0])))
        h0 = float(np.mean(slice_data.get(f"{cname}_h0", [0])))
        p0 = float(np.mean(slice_data.get(f"{cname}_p0", [0])))

        print(f"  CV5={cv5:.4f}  last_turn={lt:.4f}")
        print(f"  same_artist={sa:.4f}  diff_artist={da:.4f}  hist_0={h0:.4f}  pop_0={p0:.4f}")

        results[cname] = {
            "pool_hit": pool_hit, "cv5": cv5, "last_turn": lt,
            "same_artist": sa, "diff_artist": da, "hist_0": h0, "pop_0": p0,
            "pool_k": pool_k, "graph": graph_name, "graph_w": graph_w,
            "use_graph_feats": use_feats,
        }

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{ts()} R16 Stage 2 complete. Elapsed: {elapsed:.1f}s")

    bl = results["v3_baseline"]
    print(f"\nBaseline: CV5={bl['cv5']:.4f}  last_turn={bl['last_turn']:.4f}  "
          f"sa={bl['same_artist']:.4f}  da={bl['diff_artist']:.4f}")

    print(f"\nGate evaluation:")
    for name, r in results.items():
        if name == "v3_baseline": continue
        lt_d = r["last_turn"] - bl["last_turn"]
        cv5_d = r["cv5"] - bl["cv5"]
        sa_d = r["same_artist"] - bl["same_artist"]
        da_d = r["diff_artist"] - bl["diff_artist"]
        gate_lt = lt_d >= 0.005
        gate_cv = cv5_d >= 0.005
        gate_sa = sa_d > -0.003
        gate_da = da_d > 0
        all_pass = gate_lt and gate_cv and gate_sa and gate_da
        print(f"  {name}:")
        print(f"    CV5={r['cv5']:.4f} (Δ={cv5_d:+.4f} {'PASS' if gate_cv else 'FAIL'})  "
              f"lt={r['last_turn']:.4f} (Δ={lt_d:+.4f} {'PASS' if gate_lt else 'FAIL'})")
        print(f"    sa={r['same_artist']:.4f} (Δ={sa_d:+.4f} {'PASS' if gate_sa else 'FAIL'})  "
              f"da={r['diff_artist']:.4f} (Δ={da_d:+.4f} {'PASS' if gate_da else 'FAIL'})")
        print(f"    pool_hit={r['pool_hit']:.4f}  → {'PASS' if all_pass else 'FAIL'}")

    out_path = REPO_ROOT / "exp" / "eval" / "expR16_stage2_fusion.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()

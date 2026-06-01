#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R400 Phase-0 RECALL CEILING.

Measures dev OOF candidate recall@{20,100,300,1000} for:
  (a) production union  = r54pool ∪ r84pool
  (b) USER-HISTORY retrieval (the new lever): per dev case, if user_id in train,
      take that user's TRAIN-history tracks (across ALL their train sessions),
      centroid their embeddings (cf-bpr; also metadata-qwen3), cosine vs the full
      catalog, top-1000 (exclude conversation already-played tracks, NOT the GT).
  (c) DIRECT signal: fraction of dev cases where GT is literally in the user's
      train-history track set (overall / h7 / by user-in-train).
  (d) combined union (a)+(b)+official-NN(played) -> recall@{300,1000}.

Writes exp/eval/expR400_recall_ceiling.json.
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
from scripts.exp_goal65_eval import load_dev  # noqa: E402

TRAIN_ARROW = (
    "/Users/nadavbarkai/.cache/huggingface/datasets/"
    "talkpl-ai___talk_play_data-challenge-dataset/default/0.0.0/"
    "8110a2cfda8f7cfd43805a09eca6c58e0f7b285c/"
    "talk_play_data-challenge-dataset-train.arrow"
)
R12 = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
OUT = REPO / "exp" / "eval" / "expR400_recall_ceiling.json"
KS = [20, 100, 300, 1000]
DECAY = 0.85  # for played-track NN (same as exp_goal65_cfbpr)


# ---------------------------------------------------------------- embeddings
def load_emb(col: str, dim: int):
    """Robust row-by-row catalog matrix load for one embedding column."""
    ds = datasets.load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
    )["all_tracks"].select_columns(["track_id", col])
    ids = [str(t) for t in ds["track_id"]]
    raw = ds[col]
    M = np.zeros((len(ids), dim), dtype=np.float32)
    nmiss = 0
    for i, v in enumerate(raw):
        if v is not None and len(v) == dim:
            M[i] = v
        else:
            nmiss += 1
    norms = np.linalg.norm(M, axis=1)
    Mn = M / np.clip(norms[:, None], 1e-8, None)
    tid2idx = {t: i for i, t in enumerate(ids)}
    print(f"{col}: {M.shape} missing/cold={nmiss} ({nmiss/len(ids):.3f})")
    return ids, Mn, norms, tid2idx


# ---------------------------------------------------------------- train history
def build_user_history(dev_uids: set[str]):
    """Map user_id -> ordered list of train-history track_ids (across all sessions)."""
    ds = datasets.Dataset.from_file(TRAIN_ARROW)
    uid_col = ds["user_id"]
    conv_col = ds["conversations"]
    hist = defaultdict(list)
    train_uids = set()
    for u, conv in zip(uid_col, conv_col):
        u = str(u)
        train_uids.add(u)
        if u not in dev_uids:
            continue
        for turn in conv:
            if turn["role"] == "music":
                hist[u].append(str(turn["content"]))
    hist = {u: v for u, v in hist.items()}
    print(f"train users={len(train_uids)}  dev-user histories built={len(hist)}")
    lens = [len(v) for v in hist.values()]
    if lens:
        print(f"  history len: mean={np.mean(lens):.1f} median={np.median(lens):.0f} "
              f"min={min(lens)} max={max(lens)}")
    return hist, train_uids


# ---------------------------------------------------------------- retrieval
def centroid_topk(track_ids, Mn, norms, tid2idx, ids, exclude: set, k=1000, decay=None):
    """Mean (optionally decayed) of embeddings of track_ids -> cosine vs catalog -> top-k
    excluding `exclude`."""
    idxs = [tid2idx[t] for t in track_ids if t in tid2idx and norms[tid2idx[t]] > 1e-8]
    if not idxs:
        return []
    if decay is not None:
        w = np.array([decay ** (len(idxs) - 1 - j) for j in range(len(idxs))], dtype=np.float32)
        v = (w[:, None] * Mn[idxs]).sum(0)
    else:
        v = Mn[idxs].sum(0)
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return []
    s = Mn @ (v / nv)
    out = []
    for j in np.argsort(-s):
        t = ids[j]
        if t in exclude:
            continue
        out.append(t)
        if len(out) >= k:
            break
    return out


def recall_at(rank_list, gt, ks):
    """rank_list: ordered candidate track_ids. Returns {k: 0/1 hit}."""
    pos = {}
    for i, t in enumerate(rank_list):
        if t == gt:
            pos = i + 1
            break
    else:
        pos = None
    return {k: int(pos is not None and pos <= k) for k in ks}


def main():
    dev = load_dev()
    n = dev["n"]
    gt = dev["gt"]
    npri = dev["n_prior"]
    payload = pickle.load(open(R12, "rb"))
    cases = payload["cases"]
    case_uid = [str(c["user_id"]) for c in cases]
    played = [c["music_turns"] for c in cases]
    track_artist = dev["track_artist"]
    dev_uids = set(case_uid)

    hist, train_uids = build_user_history(dev_uids)

    # which cases have user in train (and have >=1 history track w/ embedding later)
    user_in_train = [case_uid[i] in train_uids for i in range(n)]
    h7 = [npri[i] == 7 for i in range(n)]
    same_artist = dev["same_artist"]

    print(f"\ncases user-in-train: {sum(user_in_train)}/{n}")
    print(f"cases h7: {sum(h7)}/{n}")

    # embeddings
    cf_ids, cf_Mn, cf_norms, cf_idx = load_emb("cf-bpr", 128)
    md_ids, md_Mn, md_norms, md_idx = load_emb("metadata-qwen3_embedding_0.6b", 1024)

    # ------- (c) DIRECT: GT literally in user's train history track set
    direct_all = direct_h7 = 0
    direct_in_n = direct_in_hit = 0       # among user-in-train cases
    direct_notin_n = 0                    # user NOT in train (always 0 hit)
    direct_h7_in_n = direct_h7_in_hit = 0
    hist_sets = {u: set(v) for u, v in hist.items()}
    for i in range(n):
        u = case_uid[i]
        in_train = user_in_train[i]
        hs = hist_sets.get(u, set())
        hit = gt[i] in hs
        direct_all += hit
        if h7[i]:
            direct_h7 += hit
        if in_train:
            direct_in_n += 1
            direct_in_hit += hit
            if h7[i]:
                direct_h7_in_n += 1
                direct_h7_in_hit += hit
        else:
            direct_notin_n += 1

    # ------- precompute production union pools per case
    prod_pool = []
    for i in range(n):
        # ordered: keep r54 first then r84 (order only matters for @k truncation by source;
        # we measure union membership at k by taking top-k of EACH source then union — matching
        # the harness convention)
        prod_pool.append((dev["r54pool"][i], dev["r84pool"][i]))

    def prod_union_at(i, k):
        r54, r84 = prod_pool[i]
        return set(r54[:k]) | set(r84[:k])

    # ------- (a) production union recall@k
    rec_prod = {k: 0 for k in KS}
    for i in range(n):
        for k in KS:
            if gt[i] in prod_union_at(i, k):
                rec_prod[k] += 1

    # ------- (b) USER-HISTORY retrieval recall@k (cf-bpr and metadata-qwen3 centroids)
    # measured over user-in-train cases that yield a non-empty retrieval
    rec_cf = {k: 0 for k in KS}
    rec_md = {k: 0 for k in KS}
    rec_cf_h7 = {k: 0 for k in KS}
    n_uh = 0        # user-in-train cases with usable cf history
    n_uh_h7 = 0
    # unique-vs-production: GT surfaced by user-history@1000/@300 but ABSENT from prod union@300
    uniq_cf_300 = uniq_cf_1000 = 0
    uniq_md_300 = uniq_md_1000 = 0
    # cache per-user retrieval (same user repeats across turns; played-track exclusion differs
    # per turn, but history centroid is per-user — only the exclude set varies).
    # We exclude per-case played; cheap enough to recompute since played small.
    cf_lists = {}   # cache by (user, frozenset exclude) is overkill; recompute per case
    for i in range(n):
        u = case_uid[i]
        if not user_in_train[i]:
            continue
        h = hist.get(u, [])
        if not h:
            continue
        excl = set(played[i])  # exclude conversation already-played, NOT the GT
        cf_list = centroid_topk(h, cf_Mn, cf_norms, cf_idx, cf_ids, excl, k=1000)
        if not cf_list:
            continue
        n_uh += 1
        is_h7 = h7[i]
        if is_h7:
            n_uh_h7 += 1
        rcf = recall_at(cf_list, gt[i], KS)
        for k in KS:
            rec_cf[k] += rcf[k]
            if is_h7:
                rec_cf_h7[k] += rcf[k]
        # metadata centroid
        md_list = centroid_topk(h, md_Mn, md_norms, md_idx, md_ids, excl, k=1000)
        rmd = recall_at(md_list, gt[i], KS) if md_list else {k: 0 for k in KS}
        for k in KS:
            rec_md[k] += rmd[k]
        # unique vs production union@300
        prod300 = prod_union_at(i, 300)
        if gt[i] not in prod300:
            if rcf[300]:
                uniq_cf_300 += 1
            if rcf[1000]:
                uniq_cf_1000 += 1
            if rmd[300]:
                uniq_md_300 += 1
            if rmd[1000]:
                uniq_md_1000 += 1

    # ------- (d) combined union: prod ∪ user-history-cf ∪ played-NN(cf) -> recall@{300,1000}
    # played-NN: cf-bpr decayed centroid of the conversation's played tracks (the R104 lever)
    rec_comb = {k: 0 for k in [300, 1000]}
    rec_comb_h7 = {k: 0 for k in [300, 1000]}
    rec_prod_played = {k: 0 for k in [300, 1000]}  # prod ∪ played-NN only (no user-history)
    n_h7 = sum(h7)
    for i in range(n):
        u = case_uid[i]
        excl = set(played[i])
        # played-NN cf (decayed) top-1000
        played_nn = centroid_topk(played[i], cf_Mn, cf_norms, cf_idx, cf_ids, excl,
                                  k=1000, decay=DECAY) if played[i] else []
        # user-history cf top-1000
        uh = []
        if user_in_train[i]:
            h = hist.get(u, [])
            if h:
                uh = centroid_topk(h, cf_Mn, cf_norms, cf_idx, cf_ids, excl, k=1000)
        for k in [300, 1000]:
            base = prod_union_at(i, min(k, 300))  # prod pools max 300 deep
            comb = set(base) | set(played_nn[:k]) | set(uh[:k])
            pp = set(base) | set(played_nn[:k])
            hit = gt[i] in comb
            rec_comb[k] += hit
            rec_prod_played[k] += gt[i] in pp
            if h7[i]:
                rec_comb_h7[k] += hit

    # ------- artist buckets on production recall@300 (context)
    def bucket_recall(mask, k=300):
        idx = [i for i in range(n) if mask[i]]
        if not idx:
            return 0.0, 0
        hits = sum(1 for i in idx if gt[i] in prod_union_at(i, k))
        return hits / len(idx), len(idx)

    sa_rec, sa_n = bucket_recall(same_artist)
    da_rec, da_n = bucket_recall([not s for s in same_artist])
    h7_prod_rec, h7_n = bucket_recall(h7)

    # ---- assemble report
    rep = {
        "n_cases": n,
        "n_user_in_train_cases": int(sum(user_in_train)),
        "n_h7_cases": int(n_h7),
        "distinct_dev_users": len(dev_uids),
        "distinct_train_users": len(train_uids),
        "(a)_production_union_recall": {f"@{k}": round(rec_prod[k] / n, 4) for k in KS},
        "(b)_user_history_cf_recall_over_uitrain": {
            "n_eval": n_uh,
            **{f"@{k}": round(rec_cf[k] / max(n_uh, 1), 4) for k in KS},
        },
        "(b)_user_history_metadata_recall_over_uitrain": {
            "n_eval": n_uh,
            **{f"@{k}": round(rec_md[k] / max(n_uh, 1), 4) for k in KS},
        },
        "(b)_user_history_cf_recall_h7": {
            "n_eval": n_uh_h7,
            **{f"@{k}": round(rec_cf_h7[k] / max(n_uh_h7, 1), 4) for k in KS},
        },
        "(b)_user_history_UNIQUE_vs_prod_union300": {
            "cf_unique@300_absent_from_prod300": uniq_cf_300,
            "cf_unique@1000_absent_from_prod300": uniq_cf_1000,
            "md_unique@300_absent_from_prod300": uniq_md_300,
            "md_unique@1000_absent_from_prod300": uniq_md_1000,
            "cf_unique@300_as_recall_lift_over_8000": round(uniq_cf_300 / n, 4),
            "cf_unique@1000_as_recall_lift_over_8000": round(uniq_cf_1000 / n, 4),
        },
        "(c)_GT_in_train_history_DIRECT": {
            "overall_rate_over_8000": round(direct_all / n, 4),
            "overall_count": direct_all,
            "h7_rate_over_h7": round(direct_h7 / max(n_h7, 1), 4),
            "h7_count": direct_h7,
            "user_in_train_rate": round(direct_in_hit / max(direct_in_n, 1), 4),
            "user_in_train_count": f"{direct_in_hit}/{direct_in_n}",
            "user_NOT_in_train_count": f"0/{direct_notin_n}",
            "h7_user_in_train_rate": round(direct_h7_in_hit / max(direct_h7_in_n, 1), 4),
            "h7_user_in_train_count": f"{direct_h7_in_hit}/{direct_h7_in_n}",
        },
        "(d)_combined_union_recall": {
            "prod_union_only": {f"@{k}": round(rec_prod[k] / n, 4) for k in [300, 1000]},
            "prod+played_NN(cf)": {f"@{k}": round(rec_prod_played[k] / n, 4) for k in [300, 1000]},
            "prod+played_NN+user_history(cf)": {f"@{k}": round(rec_comb[k] / n, 4) for k in [300, 1000]},
            "combined_h7": {f"@{k}": round(rec_comb_h7[k] / max(n_h7, 1), 4) for k in [300, 1000]},
        },
        "artist_buckets_prod_union@300": {
            "same_artist": {"recall": round(sa_rec, 4), "n": sa_n},
            "diff_artist": {"recall": round(da_rec, 4), "n": da_n},
            "h7": {"recall": round(h7_prod_rec, 4), "n": h7_n},
        },
    }
    OUT.write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep, indent=2))
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

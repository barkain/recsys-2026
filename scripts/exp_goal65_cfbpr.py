#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Track B1: does the official cf-bpr (128-d CF) embedding add GT recall via item->item
from the conversation's played tracks? The GT is co-listened with the played tracks
(competition mechanic), so cf-bpr nearest-neighbours of played should surface it."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pickle
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from datasets import load_dataset
from scripts.exp_goal65_eval import load_dev

DECAY = 0.85


def load_cf():
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"].select_columns(
        ["track_id", "cf-bpr"])
    ids = [str(t) for t in ds["track_id"]]
    raw = ds["cf-bpr"]
    cf = np.zeros((len(ids), 128), dtype=np.float32)
    nmissing = 0
    for i, v in enumerate(raw):
        if v is not None and len(v) == 128:
            cf[i] = v
        else:
            nmissing += 1
    norms = np.linalg.norm(cf, axis=1)
    print(f"cf-bpr: {cf.shape}  missing/degenerate (no CF vec, cold): {nmissing} ({nmissing/len(ids):.3f})")
    cfn = cf / np.clip(norms[:, None], 1e-8, None)
    return ids, cfn, norms, {t: i for i, t in enumerate(ids)}


def main():
    ids, cfn, norms, tid2idx = load_cf()
    dev = load_dev()
    payload = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))
    played = [c["music_turns"] for c in payload["cases"]]
    gt, n, npri = dev["gt"], dev["n"], dev["n_prior"]

    def topk(pl, k=300):
        idxs = [tid2idx[t] for t in pl if t in tid2idx and norms[tid2idx[t]] > 1e-8]
        if not idxs:
            return []
        w = np.array([DECAY ** (len(idxs) - 1 - j) for j in range(len(idxs))], dtype=np.float32)
        v = (w[:, None] * cfn[idxs]).sum(0)
        nv = np.linalg.norm(v)
        if nv < 1e-8:
            return []
        s = cfn @ (v / nv)
        pls = set(pl)
        out = []
        for j in np.argsort(-s):
            t = ids[j]
            if t in pls:
                continue
            out.append(t)
            if len(out) >= k:
                break
        return out

    rec20 = rec100 = rec300 = uniq300 = warm = 0
    h7 = [i for i in range(n) if npri[i] == 7]
    rec_h7 = uniq_h7 = warm_h7 = 0
    union_miss = 0
    cf_rescue_miss = 0
    for i in range(n):
        if not played[i]:
            continue
        warm += 1
        top = topk(played[i])
        g = gt[i]
        in300 = g in top[:300]
        rec300 += in300
        rec100 += g in top[:100]
        rec20 += g in top[:20]
        union = set(dev["r54pool"][i][:300]) | set(dev["r84pool"][i][:300])
        gt_in_union = g in union
        if in300 and not gt_in_union:
            uniq300 += 1
        if not gt_in_union:
            union_miss += 1
            if in300:
                cf_rescue_miss += 1
        if npri[i] == 7:
            warm_h7 += 1
            rec_h7 += in300
            if in300 and not gt_in_union:
                uniq_h7 += 1

    print(f"\nwarm cases (have played history): {warm}/{n}")
    print(f"cf-bpr item->item recall (over warm): @20={rec20/warm:.4f} @100={rec100/warm:.4f} @300={rec300/warm:.4f}")
    print(f"cf-bpr UNIQUE GTs (cf@300, ABSENT from r54∪r84 union@300): {uniq300} ({uniq300/n:.4f} of 8000)")
    print(f"h7 (warm n={warm_h7}): cf recall@300={rec_h7/max(warm_h7,1):.4f}  unique-vs-union={uniq_h7}")
    print(f"\nunion-MISS warm cases (GT not in r54∪r84@300): {union_miss}")
    print(f"  of those, cf-bpr@300 RECOVERS GT: {cf_rescue_miss} ({cf_rescue_miss/max(union_miss,1):.4f})")
    print(f"  -> potential recall lift if added: +{cf_rescue_miss} cases (+{cf_rescue_miss/n:.4f} recall@300 over r54∪r84={dev_recall(dev):.4f})")


def dev_recall(dev):
    n = dev["n"]
    return sum(1 for i in range(n)
               if dev["gt"][i] in (set(dev["r54pool"][i][:300]) | set(dev["r84pool"][i][:300]))) / n


if __name__ == "__main__":
    main()

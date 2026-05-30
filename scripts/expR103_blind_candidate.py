#!/usr/bin/env python3
"""R103 blind candidate — selective GTE patch on R92 p11 production (NOT an LR rerank).

Per the verified dev selective-deployment frontier (scripts/expR103_selective_eval.py),
the deployed object is: base = R92 p11 production blind lists, patched ONLY on rows a
GT-INDEPENDENT selector picks, by switching to arm-C (GTE-aug LR) ranking. Production is
kept EXACTLY elsewhere -> churn = patched rows.

Two candidates (build both; submit R103a first, R103b only if R103a transfers >=0):
  R103a row-budget (conservative):
    select if c_top1_gte_cos*(1 + c_top1_diff_artist + 2*c_top1_base_absent) >= 0.6195
  R103b hybrid (aggressive):
    switch if n_prior_music!=7 and gte_present and (diff_artist==0 or (diff_artist==1 and
            base_absent==0 and a_margin<2.0)); promote@2 remaining gte_present rows.

Feature stack (matches integration EXACTLY): 37 = FEAT_R39_ALL(34) + R84(rank_inv,presence,
cosine); +3 GTE(rank_inv,presence,cosine) = 40. Pool = R54-stacked RRF + GTE source (w=1.0,
cap 300), so GTE blind candidates can enter. arm A = 37-feat LR, arm C = 40-feat LR, both
all-data (8000 dev). Selector signals computed vs the R92 p11 BASE list (the deployed base).

Inputs:
  dev:   _R12 payload, R21/R54/ALS, R84 OOF (5 folds), GTE OOF (cache/r103_gte/oof_gte_lists.json)
  blind: cache/blind_a/source_cache.pkl, cache/r54_production/blind_r54_lists.json,
         cache/r84_production/blind_r84_ensemble_lists.json,
         cache/r103_gte/blind_r103_ensemble_lists.json  (<- from Colab, keyed by case_idx; sid map)
  base:  exp/inference/blind_a/r92_p11_oracle_submission.zip (predicted_track_ids + responses)
Output:
  exp/inference/blind_a/r103a_blind_track_lists.json, r103b_blind_track_lists.json
  exp/eval/expR103_blind_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import FEAT_R39_ALL, _featurize_row  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import load_supporting_maps  # noqa: E402

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}
GTE_W = 1.0
RRF_K, POOL_K, TOP_K = 20, 300, 20
N_R39 = len(FEAT_R39_ALL)
FEAT_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
FEAT_R103 = FEAT_R84 + ["gte_rank_inv", "gte_presence", "gte_cosine"]
LR_PARAMS = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20], "num_leaves": 31,
             "learning_rate": 0.05, "min_data_in_leaf": 10, "verbose": -1, "seed": 0}
LR_ROUNDS = 300

R84_OOF = [REPO / "cache/r84/phase0b_fold0/oof_r84_lists.json"] + \
    [REPO / f"cache/r84/phase1_fold{k}/oof_r84_lists.json" for k in range(1, 5)]
GTE_OOF = REPO / "cache/r103_gte/oof_gte_lists.json"
W0_STATS = REPO / "exp/eval/expR68_r54_reference_stats.pkl"
BLIND_SRC = REPO / "cache/blind_a/source_cache.pkl"
R54_BLIND = REPO / "cache/r54_production/blind_r54_lists.json"
R84_BLIND_ENS = REPO / "cache/r84_production/blind_r84_ensemble_lists.json"
GTE_BLIND_ENS = REPO / "cache/r103_gte/blind_r103_ensemble_lists.json"
R92_ZIP = REPO / "exp/inference/blind_a/r92_p11_oracle_submission.zip"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_oof(files):
    m = {}
    for f in files:
        for ci, pairs in json.load(open(f)).items():
            ids = [p[0] for p in pairs]
            m[int(ci)] = {"ranks": {t: r + 1 for r, t in enumerate(ids)},
                          "scores": {p[0]: float(p[1]) for p in pairs}}
    return m


def load_gte_oof(path):
    d = json.load(open(path)); lists = d.get("lists", d)
    m = {}
    for ci, pairs in lists.items():
        ids = [p[0] for p in pairs]
        m[int(ci)] = {"ids": ids, "ranks": {t: r + 1 for r, t in enumerate(ids)},
                      "scores": {p[0]: float(p[1]) for p in pairs}}
    return m


def sub_r84(feats, pool, r84):
    for i, tid in enumerate(pool):
        feats[i, N_R39] = (1.0 / r84["ranks"][tid]) if tid in r84["ranks"] else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in r84["ranks"] else 0.0
        feats[i, N_R39 + 2] = r84["scores"].get(tid, 0.0)
    return feats


def gte_cols(pool, ranks, scores):
    g = np.zeros((len(pool), 3), dtype=np.float32)
    for i, tid in enumerate(pool):
        if tid in ranks:
            g[i, 0] = 1.0 / ranks[tid]; g[i, 1] = 1.0; g[i, 2] = scores.get(tid, 0.0)
    return g


def artist_of(maps, tid):
    a = maps["track_artist"].get(tid, "")
    return (a or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.6195, help="R103a row-budget cutoff")
    args = ap.parse_args()
    t0 = datetime.now()
    print(f"{ts()} R103 blind candidate — selective GTE patch on R92 p11")
    if not GTE_BLIND_ENS.exists():
        sys.exit(f"MISSING GTE blind ensemble: {GTE_BLIND_ENS}\nRun scripts/expR103_blind_encode.py on Colab first.")

    # ---- dev: train all-data arm A (37) + arm C (40) ----
    print(f"{ts()} loading dev payload + sources ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]; n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    case_index = c3.build_case_index(payload, r21_source, r54_source, r54_scores,
                                     als_factors, als_track_ids, als_to_idx)
    w0 = pickle.load(open(W0_STATS, "rb"))
    case_fold = {row["case_idx"]: int(row["fold_idx"]) for row in w0}
    r84_per_fold = {k: load_oof([R84_OOF[k]]) for k in range(5)}
    gte = load_gte_oof(GTE_OOF)

    def feat37(i, src_lists, pool):
        c = cases[i]
        r21m = {t: r + 1 for r, t in enumerate(src_lists["R21"][:POOL_K])}
        r54m = {t: r + 1 for r, t in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(pool, src_lists, r21m, r54m, r54_scores[i],
                              c["user_query"], c["history"], c["music_turns"], set(c["music_turns"]),
                              maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
                              maps["track_artist_toks"], maps["track_meta_toks"],
                              als_factors, als_to_idx, case_index["als_session_vecs"][i],
                              track_pop, max_pop, track_album).astype(np.float32)

    print(f"{ts()} building dev features (8000) ...", flush=True)
    XA, yA, gA, XC, yC, gC = [], [], [], [], [], []
    for i in range(n):
        src = c3.make_source_lists(payload, r21_source, r54_source, case_index["als_source"], i)
        gmaps = gte.get(i, {"ids": [], "ranks": {}, "scores": {}})
        src_aug = {**src, "GTE": gmaps["ids"][:POOL_K]}
        pool = weighted_rrf(src_aug, {**SW_BASELINE, "GTE": GTE_W}, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]; gp = pool.index(gt) if gt in pool else -1
        r84 = r84_per_fold[case_fold[i]][i]
        fA = sub_r84(feat37(i, src, pool), pool, r84)
        fC = np.concatenate([fA, gte_cols(pool, gmaps["ranks"], gmaps["scores"])], axis=1)
        for r in range(len(pool)):
            XA.append(fA[r]); yA.append(1.0 if r == gp else 0.0)
            XC.append(fC[r]); yC.append(1.0 if r == gp else 0.0)
        gA.append(len(pool)); gC.append(len(pool))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n}", flush=True)
    print(f"{ts()} training all-data arm A (37) + arm C (40) ...", flush=True)
    lrA = lgb.train(LR_PARAMS, lgb.Dataset(np.asarray(XA, np.float32), label=np.asarray(yA, np.float32),
                    group=gA, feature_name=FEAT_R84), num_boost_round=LR_ROUNDS)
    lrC = lgb.train(LR_PARAMS, lgb.Dataset(np.asarray(XC, np.float32), label=np.asarray(yC, np.float32),
                    group=gC, feature_name=FEAT_R103), num_boost_round=LR_ROUNDS)
    del XA, yA, XC, yC

    # ---- blind: featurize + score arm A/C ----
    print(f"{ts()} loading blind inputs ...", flush=True)
    blind = pickle.load(open(BLIND_SRC, "rb"))
    r54_blind = json.load(open(R54_BLIND))
    r84_ens = json.load(open(R84_BLIND_ENS))
    gte_blind_raw = json.load(open(GTE_BLIND_ENS))
    sid_by_ci = gte_blind_raw.get("sid", {})
    gte_blind_by_sid = {}  # session_id -> [(tid, cos)]
    for ci, lst in gte_blind_raw["lists"].items():
        sid = sid_by_ci.get(str(ci))
        if sid:
            gte_blind_by_sid[sid] = [(p[0], float(p[1])) for p in lst]

    def r84_blind_maps(sid):
        pairs = r84_ens.get(sid, [])
        return {"ranks": {t: r + 1 for r, t in enumerate([p[0] for p in pairs][:POOL_K])},
                "scores": {p[0]: float(p[1]) for p in pairs}}

    # R92 p11 production base
    z = zipfile.ZipFile(R92_ZIP)
    prod = json.loads(z.read([x for x in z.namelist() if x.endswith(".json")][0]))
    prod_by_sid = {p["session_id"]: p for p in prod}

    print(f"{ts()} scoring {len(blind)} blind cases ...", flush=True)
    rows = []
    for sid, case in blind.items():
        src = {"A": case.get("src_a", []), "B": case.get("src_b", []), "C": case.get("src_c", []),
               "D": case.get("src_d", []), "F": case.get("src_f", []), "ALS": case.get("als_tracks", []),
               "R21": case.get("r21_list", []),
               "R54": [t for t, _ in r54_blind.get(sid, [])][:POOL_K]}
        gpairs = gte_blind_by_sid.get(sid, [])
        gids = [t for t, _ in gpairs]
        gmaps = {"ranks": {t: r + 1 for r, t in enumerate(gids)}, "scores": {t: float(s) for t, s in gpairs}}
        src_aug = {**src, "GTE": gids[:POOL_K]}
        pool = weighted_rrf(src_aug, {**SW_BASELINE, "GTE": GTE_W}, topk=POOL_K, k=RRF_K)
        r21m = {t: r + 1 for r, t in enumerate(src["R21"][:POOL_K])}
        r54m = {t: r + 1 for r, t in enumerate(src["R54"][:POOL_K])}
        r54sc = {t: float(s) for t, s in r54_blind.get(sid, [])}
        f37 = _featurize_row(pool, src, r21m, r54m, r54sc, case["user_query"], case["history"],
                             case["music_turns"], set(case["music_turns"]),
                             maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
                             maps["track_artist_toks"], maps["track_meta_toks"],
                             als_factors, als_to_idx, case.get("als_vec"),
                             track_pop, max_pop, track_album).astype(np.float32)
        fA = sub_r84(f37, pool, r84_blind_maps(sid))
        fC = np.concatenate([fA, gte_cols(pool, gmaps["ranks"], gmaps["scores"])], axis=1)
        sA = lrA.predict(fA); sC = lrC.predict(fC)
        oA = np.argsort(-sA, kind="mergesort"); oC = np.argsort(-sC, kind="mergesort")
        played = set(str(t) for t in case.get("music_turns", []))

        def top20(order):
            out = []
            for j in order:
                t = pool[int(j)]
                if t in played:
                    continue
                out.append(t)
                if len(out) >= TOP_K:
                    break
            return out

        cA, cC = top20(oA), top20(oC)
        base_list = prod_by_sid[sid]["predicted_track_ids"][:TOP_K]
        base_top1 = base_list[0] if base_list else None
        c0 = cC[0] if cC else None
        # GT-independent selector signals (vs the deployed R92 base)
        c0_gte_cos = gmaps["scores"].get(c0, 0.0) if c0 else 0.0
        c0_present = 1 if (c0 and c0 in gmaps["ranks"]) else 0
        c0_art = artist_of(maps, c0) if c0 else ""
        base_art = artist_of(maps, base_top1) if base_top1 else ""
        c0_diff_artist = int(bool(c0_art) and bool(base_art) and c0_art != base_art)
        c0_base_absent = int(c0 is not None and c0 not in set(base_list))
        a_margin = float(np.sort(sA)[::-1][0] - np.sort(sA)[::-1][1]) if len(sA) > 1 else 0.0
        rows.append({"sid": sid, "turn": case["turn_number"], "n_prior_music": int(case.get("n_prior_music", len(case.get("music_turns", [])))),
                     "base_list": base_list, "cC": cC,
                     "c_top1_gte_cos": c0_gte_cos, "c_top1_gte_present": c0_present,
                     "c_top1_diff_artist": c0_diff_artist, "c_top1_base_absent": c0_base_absent,
                     "a_margin": a_margin})

    # ---- selectors ----
    def sel_rowbudget(r):
        return (r["c_top1_gte_cos"] * (1.0 + r["c_top1_diff_artist"] + 2.0 * r["c_top1_base_absent"])) >= args.threshold

    def sel_hybrid_switch(r):
        return (r["n_prior_music"] != 7 and r["c_top1_gte_present"] == 1 and
                (r["c_top1_diff_artist"] == 0 or
                 (r["c_top1_diff_artist"] == 1 and r["c_top1_base_absent"] == 0 and r["a_margin"] < 2.0)))

    def build(name, select_fn):
        out, changed, overlaps = [], 0, []
        for r in rows:
            base = r["base_list"]
            patched = r["cC"] if (select_fn(r) and r["cC"]) else base
            ch = int(bool(patched) and bool(base) and patched[0] != base[0])
            changed += ch
            overlaps.append(len(set(patched[:TOP_K]) & set(base[:TOP_K])))
            out.append({"session_id": r["sid"], "turn_number": r["turn"],
                        "predicted_track_ids": patched, "_patched": int(patched is r["cC"])})
        audit = {"candidate": name, "n_cases": len(rows), "n_patched": sum(o["_patched"] for o in out),
                 "top1_churn": changed, "top20_overlap_mean": round(float(np.mean(overlaps)), 3)}
        return out, audit

    r103a, audit_a = build("R103a_rowbudget", sel_rowbudget)
    r103b, audit_b = build("R103b_hybrid", sel_hybrid_switch)

    out_a = REPO / "exp/inference/blind_a/r103a_blind_track_lists.json"
    out_b = REPO / "exp/inference/blind_a/r103b_blind_track_lists.json"
    json.dump(r103a, open(out_a, "w"), indent=2)
    json.dump(r103b, open(out_b, "w"), indent=2)
    audit = {"created_at": datetime.now().isoformat(), "threshold": args.threshold,
             "gates_R103a": {"top1_churn_le_30": audit_a["top1_churn"] <= 30,
                             "top20_overlap_ge_16": audit_a["top20_overlap_mean"] >= 16},
             "R103a": audit_a, "R103b": audit_b}
    json.dump(audit, open(REPO / "exp/eval/expR103_blind_audit.json", "w"), indent=2)

    print(f"\n{ts()} === R103 blind audit (vs R92 p11) ===")
    for a in (audit_a, audit_b):
        print(f"  {a['candidate']:18} patched={a['n_patched']:2d}/80  top1_churn={a['top1_churn']:2d}/80  "
              f"overlap={a['top20_overlap_mean']:.2f}/20")
    g = audit["gates_R103a"]
    print(f"  R103a gates: top1<=30 {g['top1_churn_le_30']}  overlap>=16 {g['top20_overlap_ge_16']}")
    print(f"  wrote {out_a.name}, {out_b.name}, expR103_blind_audit.json")
    print(f"  elapsed {(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

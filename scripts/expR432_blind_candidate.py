#!/usr/bin/env python3
"""R432 blind candidate — selective goal-aware patch on R106 A-clean production (first-turn only).

Deployed object: base = R106 A-clean production (tracks == R92 p11; responses = LexDiv A-clean),
patched ONLY on FIRST-TURN (turn_number==1) rows where the validated GT-independent selector picks,
by switching to arm-C (goal-aug LR) ranking. Production kept EXACTLY elsewhere (all multi-turn rows
+ unselected first-turn) -> churn = patched first-turn rows.

Selector (dev Phase 3 winner): on first-turn cases, deploy arm-C if
    goal_top1_cos >= --cos-thr  OR  arm-C top1 == base top1   (i.e. high-confidence injection or no top-1 change)

Feature stack matches the dev integration EXACTLY: 37 = FEAT_R39_ALL(34)+R84(3); +3 goal cols = 40.
Pool = R54-stacked RRF + goal source (w=1.0). arm A = 37-feat LR, arm C = 40-feat LR, both all-data.

Inputs (dev): _R12 payload, R21/R54/ALS, R84 OOF (5 folds), goal OOF (cache/r432_goal/goal_oof_lists.json)
Inputs (blind): cache/blind_a/source_cache.pkl, cache/r54_production/blind_r54_lists.json,
                cache/r84_production/blind_r84_ensemble_lists.json,
                cache/r432_goal/blind_goal_lists.json
Base: exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip (tracks + A-clean responses)
Output: exp/inference/blind_a/r432_blind_candidate_submission.zip (HOLD — preflight only, not submitted)
        exp/eval/expR432_blind_audit.json, exp/eval/expR432_blind_rows.json
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, zipfile
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

GOAL_W = 1.0
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K, POOL_K, TOP_K = 20, 300, 20
N_R39 = len(FEAT_R39_ALL)
FEAT_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
FEAT_GOAL = FEAT_R84 + ["gte_rank_inv", "gte_presence", "gte_cosine"]
LR_PARAMS = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20], "num_leaves": 31,
             "learning_rate": 0.05, "min_data_in_leaf": 10, "verbose": -1, "seed": 0}
LR_ROUNDS = 300

R84_OOF = [REPO / "cache/r84/phase0b_fold0/oof_r84_lists.json"] + \
    [REPO / f"cache/r84/phase1_fold{k}/oof_r84_lists.json" for k in range(1, 5)]
W0_STATS = REPO / "exp/eval/expR68_r54_reference_stats.pkl"
BLIND_SRC = REPO / "cache/blind_a/source_cache.pkl"
R54_BLIND = REPO / "cache/r54_production/blind_r54_lists.json"
R84_BLIND_ENS = REPO / "cache/r84_production/blind_r84_ensemble_lists.json"
BASE_ZIP = REPO / "exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip"
META_JSON = REPO / "cache/metadata/track_metadata_all_tracks.json"
# GOAL_OOF / GOAL_BLIND / OUT_ZIP set from CLI in main()


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


def load_goal_oof(path):
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


def goal_cols(pool, ranks, scores):
    g = np.zeros((len(pool), 3), dtype=np.float32)
    for i, tid in enumerate(pool):
        if tid in ranks:
            g[i, 0] = 1.0 / ranks[tid]; g[i, 1] = 1.0; g[i, 2] = scores.get(tid, 0.0)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos-thr", type=float, default=0.65, help="goal top-1 cosine threshold")
    ap.add_argument("--goal-oof", default=str(REPO / "cache/r432_goal_only/goal_oof_lists.json"))
    ap.add_argument("--goal-blind", default=str(REPO / "cache/r432_goal_only/blind_goal_lists.json"))
    ap.add_argument("--tag", default="goalonly", help="output filename tag")
    args = ap.parse_args()
    GOAL_OOF = Path(args.goal_oof); GOAL_BLIND = Path(args.goal_blind)
    OUT_ZIP = REPO / f"exp/inference/blind_a/r432_{args.tag}_PROVISIONAL.zip"
    MANIFEST = REPO / f"exp/eval/expR432_{args.tag}_repair_manifest.json"
    t0 = datetime.now()
    print(f"{ts()} R432 blind candidate [{args.tag}] — first-turn selective goal patch (cos-thr={args.cos_thr})")
    if not GOAL_BLIND.exists():
        sys.exit(f"MISSING blind goal source: {GOAL_BLIND} (run expR432_blind_goal_encode.py --no-profile)")
    meta = json.load(open(META_JSON))

    def tinfo(tid):
        m = meta.get(tid, {})
        def g(k):
            v = m.get(k, "")
            return (v[0] if isinstance(v, list) and v else v) or ""
        return {"title": g("track_name"), "artist": g("artist_name"), "album": g("album_name"),
                "year": str(g("release_date"))[:10], "tags": (m.get("tag_list") or [])[:12]}

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
    goal = load_goal_oof(GOAL_OOF)

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
        gmaps = goal.get(i, {"ids": [], "ranks": {}, "scores": {}})
        src_aug = {**src, "GTE": gmaps["ids"][:POOL_K]}
        pool = weighted_rrf(src_aug, {**SW_BASELINE, "GTE": GOAL_W}, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]; gp = pool.index(gt) if gt in pool else -1
        r84 = r84_per_fold[case_fold[i]][i]
        fA = sub_r84(feat37(i, src, pool), pool, r84)
        fC = np.concatenate([fA, goal_cols(pool, gmaps["ranks"], gmaps["scores"])], axis=1)
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
                    group=gC, feature_name=FEAT_GOAL), num_boost_round=LR_ROUNDS)
    del XA, yA, XC, yC

    # ---- blind: featurize + score arm A/C ----
    print(f"{ts()} loading blind inputs ...", flush=True)
    blind = pickle.load(open(BLIND_SRC, "rb"))
    r54_blind = json.load(open(R54_BLIND))
    r84_ens = json.load(open(R84_BLIND_ENS))
    goal_raw = json.load(open(GOAL_BLIND))
    sid_by_ci = goal_raw.get("sid", {})
    goal_by_sid = {}
    for ci, lst in goal_raw["lists"].items():
        sid = sid_by_ci.get(str(ci))
        if sid:
            goal_by_sid[sid] = [(p[0], float(p[1])) for p in lst]

    def r84_blind_maps(sid):
        pairs = r84_ens.get(sid, [])
        return {"ranks": {t: r + 1 for r, t in enumerate([p[0] for p in pairs][:POOL_K])},
                "scores": {p[0]: float(p[1]) for p in pairs}}

    z = zipfile.ZipFile(BASE_ZIP)
    base_entries = json.loads(z.read([x for x in z.namelist() if x.endswith(".json")][0]))
    base_by_sid = {e["session_id"]: e for e in base_entries}

    print(f"{ts()} scoring {len(blind)} blind cases ...", flush=True)
    rows = []
    for sid, case in blind.items():
        src = {"A": case.get("src_a", []), "B": case.get("src_b", []), "C": case.get("src_c", []),
               "D": case.get("src_d", []), "F": case.get("src_f", []), "ALS": case.get("als_tracks", []),
               "R21": case.get("r21_list", []),
               "R54": [t for t, _ in r54_blind.get(sid, [])][:POOL_K]}
        gpairs = goal_by_sid.get(sid, [])
        gids = [t for t, _ in gpairs]
        gmaps = {"ranks": {t: r + 1 for r, t in enumerate(gids)}, "scores": {t: float(s) for t, s in gpairs}}
        src_aug = {**src, "GTE": gids[:POOL_K]}
        pool = weighted_rrf(src_aug, {**SW_BASELINE, "GTE": GOAL_W}, topk=POOL_K, k=RRF_K)
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
        fC = np.concatenate([fA, goal_cols(pool, gmaps["ranks"], gmaps["scores"])], axis=1)
        sA = lrA.predict(fA); sC = lrC.predict(fC)
        oC = np.argsort(-sC, kind="mergesort")
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

        cC = top20(oC)
        base_list = base_by_sid[sid]["predicted_track_ids"][:TOP_K]
        base_top1 = base_list[0] if base_list else None
        c0 = cC[0] if cC else None
        c0_goal_cos = gmaps["scores"].get(c0, 0.0) if c0 else 0.0
        rows.append({"sid": sid, "turn": int(case["turn_number"]),
                     "n_prior_music": int(case.get("n_prior_music", len(case.get("music_turns", [])))),
                     "base_list": base_list, "cC": cC,
                     "c_top1_goal_cos": c0_goal_cos,
                     "c_top1_eq_base": int(c0 is not None and c0 == base_top1)})

    json.dump({"n": len(rows), "rows": rows}, open(REPO / "exp/eval/expR432_blind_rows.json", "w"))

    # ---- selector: FIRST-TURN only, dev Phase-3 winner ----
    def select(r):
        return r["turn"] == 1 and bool(r["cC"]) and \
            (r["c_top1_goal_cos"] >= args.cos_thr or r["c_top1_eq_base"])

    out, changed, overlaps, patched = [], 0, [], 0
    for r in rows:
        base = r["base_list"]
        use_c = select(r)
        new_tracks = r["cC"] if use_c else base
        patched += int(use_c)
        ch = int(bool(new_tracks) and bool(base) and new_tracks[0] != base[0])
        changed += ch
        overlaps.append(len(set(new_tracks[:TOP_K]) & set(base[:TOP_K])))
        be = base_by_sid[r["sid"]]
        out.append({"session_id": r["sid"], "turn_number": r["turn"],
                    "predicted_track_ids": new_tracks,
                    "predicted_response": be["predicted_response"],  # R106 A-clean response, unchanged
                    "_patched": int(use_c)})

    # ---- repair manifest: every CHANGED-TOP-1 row needs a response rewrite ----
    blind_items = {str(it["session_id"]): it for it in __import__("datasets").load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
        download_config=__import__("datasets").DownloadConfig(local_files_only=True))}

    def last_user(sid):
        conv = sorted(blind_items[sid]["conversations"], key=lambda x: x["turn_number"])
        us = [c["content"] for c in conv if c["role"] == "user"]
        return us[-1] if us else ""

    manifest = []
    for o, r in zip(out, rows):
        if not o["_patched"]:
            continue
        new_top1 = o["predicted_track_ids"][0]
        old_top1 = r["base_list"][0]
        if new_top1 == old_top1:
            continue  # reorder-only: top-1 unchanged, response still valid
        sid = o["session_id"]
        manifest.append({
            "sid": sid, "turn": o["turn_number"],
            "query": last_user(sid),
            "old_top1": {"tid": old_top1, **tinfo(old_top1)},
            "new_top1": {"tid": new_top1, **tinfo(new_top1)},
            "old_response": base_by_sid[sid]["predicted_response"],
            "new_top20_titles": [tinfo(t)["title"] for t in o["predicted_track_ids"][:5]],
        })
    json.dump({"n_changed_top1": len(manifest), "rows": manifest}, open(MANIFEST, "w"), indent=2, ensure_ascii=False)

    n_ft = sum(1 for r in rows if r["turn"] == 1)
    audit = {"created_at": datetime.now().isoformat(), "cos_thr": args.cos_thr, "tag": args.tag,
             "n_cases": len(rows), "n_first_turn": n_ft, "n_patched": patched,
             "top1_churn": changed, "n_changed_top1_need_repair": len(manifest),
             "top20_overlap_mean": round(float(np.mean(overlaps)), 3),
             "gates": {"top1_churn_le_30": changed <= 30, "overlap_ge_16": float(np.mean(overlaps)) >= 16,
                       "patched_only_first_turn": all(r["turn"] == 1 for r, o in zip(rows, out) if o["_patched"])}}
    json.dump(audit, open(REPO / f"exp/eval/expR432_{args.tag}_blind_audit.json", "w"), indent=2)

    # PROVISIONAL zip (OLD responses — NOT submittable; for churn audit + repair input only)
    prediction = [{k: e[k] for k in ("session_id", "turn_number", "predicted_track_ids", "predicted_response")}
                  for e in out]
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(prediction))

    print(f"\n{ts()} === R432 [{args.tag}] blind preflight (vs R106 A-clean) ===")
    print(f"  first-turn cases: {n_ft}/80   patched: {patched}   top1_churn: {changed}/80   "
          f"overlap: {audit['top20_overlap_mean']:.2f}/20")
    print(f"  CHANGED-TOP-1 needing response repair: {len(manifest)}")
    print(f"  gates: churn<=30 {audit['gates']['top1_churn_le_30']}  overlap>=16 {audit['gates']['overlap_ge_16']}  "
          f"patched_only_first_turn {audit['gates']['patched_only_first_turn']}")
    print(f"  wrote {OUT_ZIP.name} (PROVISIONAL — old responses, NOT submittable)")
    print(f"  repair manifest -> {MANIFEST.name}")
    print(f"  elapsed {(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()

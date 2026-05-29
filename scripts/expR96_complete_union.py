#!/usr/bin/env python3
"""R96 Phase 0a.2: complete-union recall gate + absent-set forensics (CPU).

Phase 0a used only dense pools (R54/R84/R90/R21) and found 24.1% of dev GT
absent. That excluded the orthogonal families (BM25/ALS/CFBPR/qwen3) that drove
R94. This script closes that gap: the production candidate pool
(`case_features.pkl['pool']`, 300 deep) IS the R54-stacked RRF union and already
fuses BM25/ALS/CFBPR/qwen3, so the complete current universe is

    pool  ∪  R54-single  ∪  R84  ∪  R90  ∪  R21   (each top-300)

We measure how much GT is absent from THIS complete union (the true recall
headroom), then dissect the absent set: history depth, same-artist coverage, and
catalog/metadata recoverability — to decide whether the next move is a new
encoder (Phase 0b, A100) or just better admission/routing over existing sources.

CPU only, no Codabench slots. Output: exp/eval/expR96_complete_union.json.
"""
from __future__ import annotations

import ast
import gc
import glob
import json
import pickle
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CASE_FEATS = REPO / "cache" / "r84b" / "case_features.pkl"
REF = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
R84_OOF = sorted(glob.glob(str(REPO / "cache/r84/phase1_fold*/oof_r84_lists.json"))) + \
    [str(REPO / "cache/r84/phase0b_fold0/oof_r84_lists.json")]
R90_OOF = sorted(glob.glob(str(REPO / "cache/r90/phase1_fold*_varA/oof_r84_lists.json")))
R21_OOF = REPO / "cache/r21_production/dev_r21_oof_lists.json"
CATALOG = REPO / "cache" / "metadata" / "track_metadata_all_tracks.json"
PAYLOAD = REPO / "cache" / "r54_phase3_payload_maps.pkl"
OUT = REPO / "exp" / "eval" / "expR96_complete_union.json"

DEPTHS = [20, 30, 100, 300]
LIVE_TOKENS = ("live", "remix", "acoustic", "version", "edit", "mix", "remaster")


def ids(seq, k=None):
    out = [x[0] if isinstance(x, (list, tuple)) else x for x in (seq or [])]
    return out[:k] if k else out


def load_idx_dict(files):
    m = {}
    for f in files:
        d = json.load(open(f))
        inner = d.get("lists", d) if isinstance(d, dict) else None
        if inner is None:
            continue
        for k, v in inner.items():
            m[int(k)] = ids(v)
    return m


def parse_field(raw):
    if raw is None:
        return None
    try:
        v = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return str(raw)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return str(v)


def first(v):
    return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else None)


def main():
    print("loading reference_stats / OOF / catalog ...")
    ref = pickle.load(open(REF, "rb"))
    r84 = load_idx_dict(R84_OOF)
    r90 = load_idx_dict(R90_OOF)
    r21 = {i: ids(v) for i, v in enumerate(json.load(open(R21_OOF)))}
    catalog = json.load(open(CATALOG))
    cat = {}
    for tid, rec in (catalog.items() if isinstance(catalog, dict) else
                     ((r.get("track_id"), r) for r in catalog)):
        key = (parse_field(tid) if isinstance(tid, str) else tid)
        key = key if isinstance(key, str) else (key[0] if isinstance(key, list) else tid)
        cat[str(key).strip("'\"")] = rec
    track_artist = pickle.load(open(PAYLOAD, "rb"))["track_artist"]

    def artist_of(t):
        v = parse_field(track_artist.get(t))
        return (first(v) or "").strip().lower() if v else ""

    print("loading case_features (2.2 GB) ...")
    cf = pickle.load(open(CASE_FEATS, "rb"))
    pool_by_idx = {k: list(r["pool"]) for k, r in cf.items()}
    del cf
    gc.collect()

    N = len(ref)
    cov = {s: Counter() for s in ["pool", "R54", "R84", "R90", "R21", "UNION"]}
    union300 = 0
    absent = []
    for e in ref:
        i = e["case_idx"]
        gt = e["gt_track_id"]
        srcs = {
            "pool": pool_by_idx.get(i, []),
            "R54": ids(e["r54_single_source_top300"]),
            "R84": r84.get(i, []),
            "R90": r90.get(i, []),
            "R21": r21.get(i, []),
        }
        for s, lst in srcs.items():
            pos = lst.index(gt) if gt in lst else None
            for D in DEPTHS:
                if pos is not None and pos < D:
                    cov[s][D] += 1
        for D in DEPTHS:
            u = set()
            for lst in srcs.values():
                u.update(lst[:D])
            if gt in u:
                cov["UNION"][D] += 1
        full = set()
        for lst in srcs.values():
            full.update(lst)
        if gt in full:
            union300 += 1
        else:
            absent.append((i, gt, e.get("history_depth", 0), srcs["pool"]))

    n_abs = len(absent)

    # ---- absent-set forensics ----
    hist_bucket = Counter()
    same_artist_in_pool = 0     # GT artist appears among production-pool tracks
    in_catalog = 0
    has_text = 0                # GT in catalog with nonempty title+artist
    non_ascii_title = 0
    live_remix = 0
    rare_or_no_tags = 0
    target_recoverable = []     # absent + present in catalog with usable text
    for i, gt, hd, pool in absent:
        hist_bucket["h0" if hd == 0 else "h1-2" if hd <= 2 else "h3-6" if hd <= 6 else "h7+"] += 1
        gt_art = artist_of(gt)
        if gt_art and any(artist_of(t) == gt_art for t in pool[:100]):
            same_artist_in_pool += 1
        rec = cat.get(gt)
        if rec:
            in_catalog += 1
            title = first(parse_field(rec.get("track_name")))
            artist = first(parse_field(rec.get("artist_name")))
            album = first(parse_field(rec.get("album_name"))) or ""
            tags = parse_field(rec.get("tag_list")) or []
            if title and artist:
                has_text += 1
                target_recoverable.append((i, gt))
                if any(not c.isascii() for c in title):
                    non_ascii_title += 1
                blob = f"{title} {album}".lower()
                if any(tok in blob for tok in LIVE_TOKENS):
                    live_remix += 1
                if len(tags) <= 1:
                    rare_or_no_tags += 1

    def pct(x, d=N):
        return round(x / d, 4)

    report = {
        "experiment": "R96 Phase 0a.2 complete-union recall gate + forensics",
        "n_dev_cases": N,
        "union_sources": ["prod_pool(RRF: incl BM25/ALS/CFBPR/qwen3)", "R54", "R84", "R90", "R21"],
        "per_source_recall": {
            s: {str(D): pct(cov[s][D]) for D in DEPTHS}
            for s in ["pool", "R54", "R84", "R90", "R21", "UNION"]
        },
        "complete_union_top300_coverage": pct(union300),
        "gt_absent_from_complete_union": pct(n_abs),
        "dense_only_union_absent_was": 0.241,
        "absent_forensics": {
            "n_absent": n_abs,
            "history_depth_split": dict(hist_bucket),
            "gt_artist_present_in_pool": pct(same_artist_in_pool, n_abs),
            "gt_in_catalog": pct(in_catalog, n_abs),
            "gt_has_usable_text(title+artist)": pct(has_text, n_abs),
            "of_text_recoverable__non_english_title": pct(non_ascii_title, max(1, has_text)),
            "of_text_recoverable__live_remix_version": pct(live_remix, max(1, has_text)),
            "of_text_recoverable__rare_or_no_tags": pct(rare_or_no_tags, max(1, has_text)),
            "n_text_recoverable_targets": len(target_recoverable),
        },
        "target_recoverable_sample": [g for _, g in target_recoverable[:25]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)

    print(f"\ndev cases: {N}")
    print("Per-source GT recall by depth:")
    print(f"  {'src':6s}{'@20':>7s}{'@30':>7s}{'@100':>8s}{'@300':>7s}")
    for s in ["pool", "R54", "R84", "R90", "R21", "UNION"]:
        r = report["per_source_recall"][s]
        print(f"  {s:6s}{r['20']:>7.3f}{r['30']:>7.3f}{r['100']:>8.3f}{r['300']:>7.3f}")
    print(f"\nCOMPLETE union top-300 coverage: {report['complete_union_top300_coverage']:.3f}")
    print(f"GT ABSENT from complete union:   {report['gt_absent_from_complete_union']:.3f}  "
          f"(dense-only was 0.241)")
    f = report["absent_forensics"]
    print(f"\nAbsent set: {f['n_absent']} cases")
    print(f"  history split: {f['history_depth_split']}")
    print(f"  GT artist present in pool (alias/coverage): {f['gt_artist_present_in_pool']:.3f}")
    print(f"  GT in catalog: {f['gt_in_catalog']:.3f} | has usable text: {f['gt_has_usable_text(title+artist)']:.3f}")
    print(f"  of text-recoverable: non-English {f['of_text_recoverable__non_english_title']:.3f}, "
          f"live/remix {f['of_text_recoverable__live_remix_version']:.3f}, "
          f"rare/no-tags {f['of_text_recoverable__rare_or_no_tags']:.3f}")
    print(f"  => text-recoverable target set: {f['n_text_recoverable_targets']} cases "
          f"({pct(f['n_text_recoverable_targets']):.3f} of dev)")
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

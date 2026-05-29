#!/usr/bin/env python3
"""R95: forensic analysis of every scored Blind-A oracle probe (R92/R93/R94).

We have spent ~14 Codabench slots on single-row probes and found exactly one
positive (R92 p11, +0.0004). This script assembles the full labeled ledger with
unified offline features and asks: is there ANY feature that separates the lone
win from the many flat/negative probes? If not, single-row oracle probing has
no learnable policy at this metric resolution and the EV of more blind uploads
is poor.

It uploads nothing. Output: exp/eval/expR95_oracle_forensics.json (+ console).
"""
from __future__ import annotations

import ast
import csv
import json
import pickle
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

R84C_ZIP = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
P11_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
R90_TRACKS = REPO / "exp" / "inference" / "blind_a" / "r90_blind_track_lists.json"
R90_COS = REPO / "cache" / "r90_production" / "blind_r90_ensemble_lists.json"
R84_COS = REPO / "cache" / "r84_production" / "blind_r84_ensemble_lists.json"
R54_COS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
PAYLOAD_MAPS = REPO / "cache" / "r54_phase3_payload_maps.pkl"
SOURCE_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"

PROBES = [
    # (scores_csv, manifest, base_zip, base_ndcg, family)
    (REPO / "exp/eval/expR92_probe_scores_template.csv",
     REPO / "exp/eval/expR92_probe_manifest.json", R84C_ZIP, 0.5069, "R92_fullswap"),
    (REPO / "exp/eval/expR93_policy_probe_scores_template.csv",
     REPO / "exp/eval/expR93_policy_probe_manifest.json", P11_ZIP, 0.5073, "R93_reorder"),
    (REPO / "exp/eval/expR94_inject_probe_scores_template.csv",
     REPO / "exp/eval/expR94_inject_probe_manifest.json", P11_ZIP, 0.5073, "R94_inject"),
]
OUT = REPO / "exp" / "eval" / "expR95_oracle_forensics.json"
ORTHO = ["src_b", "src_c", "r21_list", "als_tracks", "src_f", "src_a", "src_d"]
POS, NEG = 0.00005, -0.00005


def zrows(z):
    with zipfile.ZipFile(z) as zf:
        n = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        return json.loads(zf.read(n))


def cos(p):
    d = json.load(open(p))["lists"]
    return {s: [t for t, _ in v] for s, v in d.items()}


def artist_of(ta, t):
    raw = ta.get(t)
    if raw is None:
        return None
    try:
        v = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return str(raw).strip().lower()
    return str(v[0]).strip().lower() if isinstance(v, (list, tuple)) and v else str(v).strip().lower()


def rk(pool, sid, t):
    l = pool.get(sid, [])
    return l.index(t) if t in l else None


def load_scores(csv_path):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = (row.get("ndcg20") or "").strip()
            if v:
                out[row["probe_id"].strip()] = float(v)
    return out


def main():
    base_lists = {
        "R92_fullswap": {r["session_id"]: r["predicted_track_ids"] for r in zrows(R84C_ZIP)},
        "R93_reorder": {r["session_id"]: r["predicted_track_ids"] for r in zrows(P11_ZIP)},
        "R94_inject": {r["session_id"]: r["predicted_track_ids"] for r in zrows(P11_ZIP)},
    }
    r90tr = {r["session_id"]: r["predicted_track_ids"] for r in json.load(open(R90_TRACKS))}
    R90c, R84c, R54c = cos(R90_COS), cos(R84_COS), cos(R54_COS)
    ta = pickle.load(open(PAYLOAD_MAPS, "rb"))["track_artist"]
    src = pickle.load(open(SOURCE_CACHE, "rb"))

    ledger = []
    for csv_path, man_path, base_zip, base_ndcg, family in PROBES:
        scores = load_scores(csv_path)
        man = {p["probe_id"]: p for p in json.load(open(man_path))["probes"]}
        probe_rows = {}
        # R92 p11/p12 are not in the manifest; reconstruct their changed row.
        for pid in scores:
            p = man.get(pid)
            sid = p["session_id"] if p else pid.split("_")[1] + ""  # placeholder
            if p is None:
                # map short sid -> full sid via r90 tracks
                short = pid.split("_")[1]
                full = next((s for s in r90tr if s[:8] == short), None)
                if full is None:
                    continue
                sid = full
            probe_rows[pid] = (p, sid)

        for pid, (p, sid) in probe_rows.items():
            base = base_lists[family].get(sid)
            if base is None:
                # R92 base is r84c keyed by full sid; resolve short ids
                full = next((s for s in base_lists[family] if s[:8] == sid[:8]), None)
                base = base_lists[family].get(full) if full else None
                if full:
                    sid = full
            ndcg = scores[pid]
            delta = round(ndcg - base_ndcg, 4)
            sign = "POS" if delta > POS else ("NEG" if delta < NEG else "flat")

            # Determine the changed action list for this probe.
            if family == "R92_fullswap":
                action_list = r90tr.get(sid, base)
                action = "full_swap"
            elif p is not None and "action_track_ids" in p:
                action_list = p["action_track_ids"]
                action = p.get("action_source", "reorder")
            else:
                # R93/R94 manifest stores changed via injected_track/insert
                action_list = None
                action = p.get("label") or p.get("action_source") or "?"

            # Reconstruct changed list for R93 (action_source) / R94 (inject) from zip if needed.
            zip_rel = p.get("zip") if p else None
            if action_list is None and zip_rel:
                zr = {r["session_id"]: r["predicted_track_ids"] for r in zrows(REPO / zip_rel)}
                action_list = zr.get(sid, base)

            top1_changed = action_list[0] != base[0]
            new_top1 = action_list[0]
            overlap = len(set(base) & set(action_list))
            added = [t for t in action_list if t not in set(base)]
            # The "key new track": new top1 if changed, else the first added tail track.
            key_track = new_top1 if top1_changed else (added[0] if added else None)

            kt_prior = base.index(key_track) if (key_track and key_track in base) else None
            played = [x[0] if isinstance(x, (list, tuple)) else x
                      for x in (src.get(sid, {}).get("music_turns") or [])]
            played_artists = {a for a in (artist_of(ta, t) for t in played) if a}
            kt_artist = artist_of(ta, key_track) if key_track else None
            ortho_hits = 0
            if key_track:
                for fld in ORTHO:
                    ids = [x[0] if isinstance(x, (list, tuple)) else x
                           for x in (src.get(sid, {}).get(fld) or [])][:50]
                    if key_track in ids:
                        ortho_hits += 1

            ledger.append({
                "probe_id": pid, "family": family, "action": action,
                "session_id": sid[:8], "ndcg20": ndcg, "delta": delta, "sign": sign,
                "top1_changed": top1_changed, "overlap_20": overlap,
                "n_added": len(added),
                "key_track": key_track[:8] if key_track else None,
                "key_track_prior_rank_in_base": kt_prior,   # None=absent from base 20
                "key_track_R90cos": rk(R90c, sid, key_track) if key_track else None,
                "key_track_R84cos": rk(R84c, sid, key_track) if key_track else None,
                "key_track_R54cos": rk(R54c, sid, key_track) if key_track else None,
                "key_track_ortho_consensus": ortho_hits,
                "key_track_same_artist_old_top1": bool(kt_artist and kt_artist == artist_of(ta, base[0])),
                "key_track_same_artist_history": bool(kt_artist and kt_artist in played_artists),
            })

    # attach r54_margin from r90 track meta
    r90meta = {r["session_id"][:8]: r.get("_r54_margin") for r in json.load(open(R90_TRACKS))}
    for e in ledger:
        e["r54_margin"] = r90meta.get(e["session_id"])

    ledger.sort(key=lambda e: e["delta"], reverse=True)
    pos = [e for e in ledger if e["sign"] == "POS"]
    flat = [e for e in ledger if e["sign"] == "flat"]
    neg = [e for e in ledger if e["sign"] == "NEG"]

    summary = {
        "n_scored": len(ledger),
        "n_pos": len(pos), "n_flat": len(flat), "n_neg": len(neg),
        "cumulative_gain": round(sum(max(0, e["delta"]) for e in pos), 4),
        "metric_resolution": 0.0001,
        "max_single_row_impact": round(1.0 / 80, 4),
        "by_family": {},
    }
    for fam in ("R92_fullswap", "R93_reorder", "R94_inject"):
        fl = [e for e in ledger if e["family"] == fam]
        summary["by_family"][fam] = {
            "n": len(fl), "pos": sum(e["sign"] == "POS" for e in fl),
            "neg": sum(e["sign"] == "NEG" for e in fl),
            "flat": sum(e["sign"] == "flat" for e in fl),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "ledger": ledger}, open(OUT, "w"), indent=2)

    print(f"Scored probes: {summary['n_scored']}  "
          f"POS={summary['n_pos']} flat={summary['n_flat']} NEG={summary['n_neg']}  "
          f"cumulative gain={summary['cumulative_gain']:+.4f}")
    print(f"Max possible single-row impact on the 80-mean: {summary['max_single_row_impact']:.4f}\n")
    cols = ["probe_id", "family", "sign", "delta", "top1_changed", "overlap_20",
            "key_track_prior_rank_in_base", "key_track_R90cos", "key_track_R84cos",
            "key_track_R54cos", "key_track_ortho_consensus",
            "key_track_same_artist_old_top1", "key_track_same_artist_history", "r54_margin"]
    print(f"{'probe_id':24s}{'sign':5s}{'delta':>8s}{'t1c':>4s}{'ov':>3s}"
          f"{'prior':>6s}{'R90':>5s}{'R84':>5s}{'R54':>5s}{'cons':>5s}{'a@1':>4s}{'a@h':>4s}{'mgn':>6s}")
    for e in ledger:
        pr = e["key_track_prior_rank_in_base"]
        pr = "abs" if pr is None else str(pr)
        def s(x): return "-" if x is None else str(x)
        mgn = f"{e['r54_margin']:.2f}" if e["r54_margin"] is not None else "  -"
        print(f"{e['probe_id']:24s}{e['sign']:5s}{e['delta']:+8.4f}"
              f"{int(e['top1_changed']):>4d}{e['overlap_20']:>3d}{pr:>6s}"
              f"{s(e['key_track_R90cos']):>5s}{s(e['key_track_R84cos']):>5s}{s(e['key_track_R54cos']):>5s}"
              f"{e['key_track_ortho_consensus']:>5d}"
              f"{int(e['key_track_same_artist_old_top1']):>4d}{int(e['key_track_same_artist_history']):>4d}{mgn:>6s}")
    print(f"\nby family: {json.dumps(summary['by_family'])}")
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

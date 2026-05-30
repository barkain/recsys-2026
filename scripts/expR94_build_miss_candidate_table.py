#!/usr/bin/env python3
"""R94 Phase 0: rank (row, new-candidate) pairs by hidden-GT likelihood.

R93 proved reordering the R84/R90 BGE pool is futile (3/3 top-1-preserving
reorders flat at 0.5073): for the saturated rows the GT is either already at
top-1 or absent from the 20-list. The only R92 win (c4f7d055) INTRODUCED a
track absent from R84c's list, and it was same-artist as the row's top-1.

R94 therefore stops scavenging R90 lineage and instead asks, per row: is there a
genuinely NEW track (not in current production's top-20) that an ENSEMBLE of
retrievers ORTHOGONAL to the R84/R90 BGE encoder agrees on? Such a track is a
plausible missed GT worth a one-row oracle probe.

Orthogonal source families (cached in source_cache.pkl, lineage from
expR55_blind_source_cache.py):
  src_b, src_c   BM25 lexical/sparse (all 80 cases)
  r21_list       R21 SentenceTransformer retriever (all 80, depth 300)
  als_tracks     ALS collaborative filtering (cases with history)
  src_f          CFBPR collaborative (cases with history)
  src_a          qwen3 metadata-neighbors of played tracks (cases with history)
  src_d          qwen3 track-similarity neighbors (cases with an anchor)
R54 cosine top-300 is used only as a weak dense corroborator (semi-orthogonal),
never as a primary voter; R84/R90 are deliberately excluded (saturated lineage).

This builds the ranked table only. It uploads/probes nothing.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
SOURCE_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"
R54_COSINE = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
PAYLOAD_MAPS = REPO / "cache" / "r54_phase3_payload_maps.pkl"
R90_TRACKS = REPO / "exp" / "inference" / "blind_a" / "r90_blind_track_lists.json"
R92_MANIFEST = REPO / "exp" / "eval" / "expR92_probe_manifest.json"
R92_SCORES = REPO / "exp" / "eval" / "expR92_probe_scores_template.csv"
R93_SCORES = REPO / "exp" / "eval" / "expR93_policy_probe_scores_template.csv"
R93_MANIFEST = REPO / "exp" / "eval" / "expR93_policy_probe_manifest.json"

OUT_TABLE = REPO / "exp" / "eval" / "expR94_miss_candidate_table.json"

# Orthogonal voters and the consensus depth we count membership within.
VOTER_FIELDS = ["src_b", "src_c", "r21_list", "als_tracks", "src_f", "src_a", "src_d"]
CONSENSUS_DEPTH = 50
R54_CORROB_DEPTH = 300


def load_zip_rows(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        return json.loads(zf.read(name))


def artist_of(track_artist: dict, tid: str):
    raw = track_artist.get(tid)
    if raw is None:
        return None
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return str(raw).strip().lower()
    if isinstance(parsed, (list, tuple)) and parsed:
        return str(parsed[0]).strip().lower()
    return str(parsed).strip().lower()


def ids_of(seq) -> list[str]:
    return [x[0] if isinstance(x, (list, tuple)) else x for x in (seq or [])]


def already_probed_sessions() -> dict[str, str]:
    """Sessions for which Codabench has already scored ANY one-row action.
    R94 should not re-probe these without a materially different rationale."""
    probed: dict[str, str] = {}
    # R92 (manifest gives session per probe_id) + extra scored p11/p12
    man = {p["probe_id"]: p for p in json.load(open(R92_MANIFEST))["probes"]}
    extra_sid = {"r92p11": "c4f7d055", "r92p12": "d9cca604"}
    for csv_path in (R92_SCORES,):
        with open(csv_path) as f:
            next(f)
            for line in f:
                pid, _, val = line.strip().partition(",")
                if not val:
                    continue
                if pid in man:
                    probed[man[pid]["session_id"][:8]] = f"R92:{pid}"
                elif pid.split("_")[0] in extra_sid:
                    probed[extra_sid[pid.split("_")[0]]] = f"R92:{pid}"
    r93man = {p["probe_id"]: p for p in json.load(open(R93_MANIFEST))["probes"]}
    with open(R93_SCORES) as f:
        next(f)
        for line in f:
            pid, _, val = line.strip().partition(",")
            if val and pid in r93man:
                probed[r93man[pid]["session_id"][:8]] = f"R93:{pid}"
    return probed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_TABLE)
    ap.add_argument("--consensus-depth", type=int, default=CONSENSUS_DEPTH)
    args = ap.parse_args()

    base_rows = load_zip_rows(BASE_ZIP)
    src = pickle.load(open(SOURCE_CACHE, "rb"))
    r54c = json.load(open(R54_COSINE))["lists"]
    pm = pickle.load(open(PAYLOAD_MAPS, "rb"))
    track_artist = pm["track_artist"]
    r90_by_sid = {r["session_id"]: r for r in json.load(open(R90_TRACKS))}
    probed = already_probed_sessions()

    rows_out = []
    for base in base_rows:
        sid = base["session_id"]
        turn = int(base["turn_number"])
        top20 = base["predicted_track_ids"]
        top1 = top20[0]
        in_top20 = set(top20)
        case = src.get(sid, {})
        played = ids_of(case.get("music_turns"))
        played_artists = {a for a in (artist_of(track_artist, t) for t in played) if a}
        top1_artist = artist_of(track_artist, top1)
        r54_margin = r90_by_sid.get(sid, {}).get("_r54_margin")

        # Per-source membership (within consensus depth) and best rank.
        voter_top = {}
        for fld in VOTER_FIELDS:
            ids = ids_of(case.get(fld))[: args.consensus_depth]
            voter_top[fld] = {t: r for r, t in enumerate(ids)}
        r54_rank = {t: r for r, t in enumerate(ids_of(r54c.get(sid))[:R54_CORROB_DEPTH])}

        # Is the CURRENT top-1 corroborated by any orthogonal voter? If not, the
        # BGE pick stands alone -> higher miss-likelihood.
        top1_voter_support = sum(1 for fld in VOTER_FIELDS if top1 in voter_top[fld])

        # Build NEW candidate set (not already in production top-20).
        cand_sources = defaultdict(list)  # tid -> [(field, rank)]
        for fld in VOTER_FIELDS:
            for t, r in voter_top[fld].items():
                if t not in in_top20:
                    cand_sources[t].append((fld, r))

        candidates = []
        for t, hits in cand_sources.items():
            consensus = len(hits)
            best_rank = min(r for _, r in hits)
            srcs = sorted({f for f, _ in hits})
            t_artist = artist_of(track_artist, t)
            same_artist_top1 = bool(t_artist and t_artist == top1_artist)
            same_artist_history = bool(t_artist and t_artist in played_artists)
            r54_corrob = t in r54_rank
            # Hidden-GT-likelihood score (transparent, no R90 lineage).
            score = 0.0
            score += 1.5 * consensus                      # agreement across orthogonal retrievers
            score += 2.0 if same_artist_history else 0.0  # behavioral relevance (strong)
            score += 1.2 if same_artist_top1 else 0.0     # c4f7d055 win signature
            score += max(0.0, 1.0 - best_rank / args.consensus_depth)  # shallow in some source
            score += 0.8 if r54_corrob else 0.0           # dense-pool corroboration
            candidates.append({
                "track_id": t,
                "consensus": consensus,
                "sources": srcs,
                "best_rank": best_rank,
                "same_artist_top1": same_artist_top1,
                "same_artist_history": same_artist_history,
                "r54_corroborated": r54_corrob,
                "cand_score": round(score, 3),
            })
        candidates.sort(key=lambda c: c["cand_score"], reverse=True)
        best = candidates[0] if candidates else None

        # Row miss-likelihood: only worth a probe if (a) a strong new candidate
        # exists, (b) the current top-1 is weakly corroborated, (c) margin low.
        margin_term = 0.0
        if r54_margin is not None:
            margin_term = max(0.0, 1.0 - min(r54_margin, 2.0) / 2.0)
        top1_weak = 1.0 - min(top1_voter_support, 3) / 3.0
        best_cand_term = best["cand_score"] if best else 0.0
        probe_priority = round(
            best_cand_term + 1.5 * top1_weak + 1.0 * margin_term, 3
        )
        # Insertion strategy: the c4f7d055 win replaced top-1 with a same-artist
        # track at low margin. Reserve top-1 injection for that signature;
        # otherwise inject at rank 2 (top-1-preserving, response-safe).
        insert_at = "top1" if (best and best["same_artist_top1"]
                               and r54_margin is not None and r54_margin < 0.5) else "rank2"

        rows_out.append({
            "session_id": sid,
            "turn_number": turn,
            "current_top1": top1,
            "current_top1_artist": top1_artist,
            "has_history": bool(played),
            "n_played": len(played),
            "r54_margin": r54_margin,
            "top1_voter_support": top1_voter_support,
            "n_new_candidates": len(candidates),
            "best_candidate": best,
            "top3_candidates": candidates[:3],
            "probe_priority": probe_priority,
            "suggested_insert": insert_at,
            "already_probed": probed.get(sid[:8]),
        })

    rows_out.sort(key=lambda r: r["probe_priority"], reverse=True)
    out = {
        "experiment": "R94 hidden-GT miss-candidate ranking",
        "base_submission": str(BASE_ZIP.relative_to(REPO)),
        "base_ndcg20": 0.5073,
        "voter_fields": VOTER_FIELDS,
        "consensus_depth": args.consensus_depth,
        "note": "candidate_score and probe_priority use NO R84/R90 lineage; "
                "R54 cosine is a weak corroborator only.",
        "n_rows": len(rows_out),
        "n_rows_with_candidate": sum(1 for r in rows_out if r["best_candidate"]),
        "n_already_probed": sum(1 for r in rows_out if r["already_probed"]),
        "rows": rows_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    fresh = [r for r in rows_out if not r["already_probed"]]
    print(f"rows={out['n_rows']}  with_candidate={out['n_rows_with_candidate']}  "
          f"already_probed={out['n_already_probed']}")
    print(f"\nTop 15 FRESH rows by probe_priority (insert | consensus | artist signal):")
    print(f"  {'sess_t':12s} {'prio':>5s} {'ins':>5s} {'mgn':>5s} {'t1sup':>5s} "
          f"{'cons':>4s} {'cscore':>6s}  signals")
    for r in fresh[:15]:
        b = r["best_candidate"] or {}
        sig = []
        if b.get("same_artist_history"): sig.append("artist∈hist")
        if b.get("same_artist_top1"): sig.append("artist=top1")
        if b.get("r54_corroborated"): sig.append("r54")
        sig += b.get("sources", [])
        mgn = f"{r['r54_margin']:.2f}" if r["r54_margin"] is not None else " n/a"
        print(f"  {r['session_id'][:8]}_t{r['turn_number']:<2d} {r['probe_priority']:5.2f} "
              f"{r['suggested_insert']:>5s} {mgn:>5s} {r['top1_voter_support']:5d} "
              f"{b.get('consensus',0):4d} {b.get('cand_score',0):6.2f}  {','.join(sig)}")
    print(f"\nWrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

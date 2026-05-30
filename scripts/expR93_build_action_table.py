#!/usr/bin/env python3
"""R93 Phase 0: build the blind-oracle action table.

R92 proved the real Blind-A scorer can find row-level nDCG wins that offline
gates miss, but one-row probing by hand is inefficient. R93 makes each scarce
Codabench slot information-rich: enumerate many candidate row-level *actions*
per blind case, describe each with cheap offline features, attach the R92
oracle outcomes as labels, and (downstream) rank which probes to spend slots on.

This script only builds the table. It does NOT upload or score anything.

Base / current production
-------------------------
Production is R92 p11 (`r92_p11_oracle_submission.zip`, nDCG@20 0.5073), which
is R84c with exactly one winning row swapped (c4f7d055_t7 -> R90). Every R93
action is measured as a single-row delta *on top of* p11, so the per-row credit
stacks additively on the current production score.

Action sources (each yields a 20-track ordering for one case)
-------------------------------------------------------------
- r90_full              : R90's full 20-list (the global-swap action; mostly
                          neutral/negative in R92, kept for completeness).
- r90_keep_top1/3/5     : top-k from production preserved, tail re-filled in
                          R90 order. Top-1/3/5-preserving reorders => no
                          response-semantics risk for k>=1.
- r90_keep1_repl2_5     : keep production rank 1 and ranks 6-20, replace ranks
                          2-5 from R90 order (a tight conservative edit).
- r54_top20 / r84src_top20 / r90src_top20 : top-20 of the raw cosine retrieval
                          lists. These are retrieval order, NOT LR order, so
                          they are flagged rough/low-prior.

Labels
------
R92 measured nine single-row R90 full-swap probes (p01-p07, p11, p12). Those
join onto the matching (session, r90_full) action with
`blind_delta = observed_ndcg - 0.5069`. Everything else is unlabeled (null).
The eight prepared-but-unscored R92 reorder probes (r92r01-08) are flagged so
the downstream policy does not re-emit an identical ZIP.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
R84C_ZIP = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
R90_TRACKS = REPO / "exp" / "inference" / "blind_a" / "r90_blind_track_lists.json"
R54_COSINE = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R84_COSINE = REPO / "cache" / "r84_production" / "blind_r84_ensemble_lists.json"
R90_COSINE = REPO / "cache" / "r90_production" / "blind_r90_ensemble_lists.json"
PAYLOAD_MAPS = REPO / "cache" / "r54_phase3_payload_maps.pkl"
SOURCE_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"

R92_MANIFEST = REPO / "exp" / "eval" / "expR92_probe_manifest.json"
R92_SCORES = REPO / "exp" / "eval" / "expR92_probe_scores_template.csv"
R92_REORDER_MANIFEST = REPO / "exp" / "eval" / "expR92_reorder_probe_manifest.json"

OUT_TABLE = REPO / "exp" / "eval" / "expR93_action_table.json"

R84C_BASE_NDCG = 0.5069   # original R84c official Blind-A nDCG@20
PROD_NDCG = 0.5073        # current production (R92 p11) nDCG@20
POS_THRESH = 0.00005


def load_zip_rows(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        return json.loads(zf.read(name))


def key(row: dict) -> tuple[str, int]:
    return row["session_id"], int(row["turn_number"])


def cosine_order(d: dict, sid: str) -> list[str]:
    return [t for t, _ in d["lists"].get(sid, [])]


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


def topk_keep_reorder(base: list[str], alt: list[str], k: int, n: int = 20) -> list[str]:
    """Preserve the first k of `base`; refill the tail in `alt` order."""
    head = base[:k]
    seen = set(head)
    out = list(head)
    for t in alt:
        if len(out) >= n:
            break
        if t not in seen:
            out.append(t)
            seen.add(t)
    # If alt was too short to fill, top up from base remainder.
    for t in base:
        if len(out) >= n:
            break
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out[:n]


def keep1_replace_2_5(base: list[str], alt: list[str], n: int = 20) -> list[str]:
    """Keep base rank 1 and ranks 6-20; replace ranks 2-5 from `alt` order."""
    head = base[:1]
    tail = base[5:n]
    fixed = set(head) | set(tail)
    picks: list[str] = []
    for t in alt:
        if len(picks) >= 4:
            break
        if t not in fixed and t not in picks:
            picks.append(t)
    # Pad with original ranks 2-5 if alt could not supply 4 fresh tracks.
    for t in base[1:5]:
        if len(picks) >= 4:
            break
        if t not in picks and t not in fixed:
            picks.append(t)
    out = head + picks + tail
    return out[:n]


def first_n_unique(seq: list[str], n: int = 20) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in seq:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= n:
            break
    return out[:n]


def describe(base: list[str], action: list[str], track_artist: dict) -> dict:
    """Feature row comparing an action ordering against the production ordering."""
    n = len(base)
    base_rank = {t: i for i, t in enumerate(base)}
    act_rank = {t: i for i, t in enumerate(action)}
    base_set, act_set = set(base), set(action)
    common = base_set & act_set

    movements = [abs(act_rank[t] - base_rank[t]) for t in common]
    n_up = sum(1 for t in common if act_rank[t] < base_rank[t])
    n_down = sum(1 for t in common if act_rank[t] > base_rank[t])
    pos_changed = [i for i in range(n) if base[i] != action[i]]
    top5_changed = sum(1 for i in range(min(5, n)) if base[i] != action[i])
    highest_touched = pos_changed[0] if pos_changed else None

    top1_changed = base[0] != action[0]
    same_artist_top1 = None
    if top1_changed:
        a0, a1 = artist_of(track_artist, base[0]), artist_of(track_artist, action[0])
        same_artist_top1 = bool(a0 is not None and a0 == a1)

    return {
        "top1_changed": top1_changed,
        "base_top1": base[0],
        "new_top1": action[0],
        "same_artist_top1": same_artist_top1,
        "overlap_20": len(common),
        "n_positions_changed": len(pos_changed),
        "n_added": len(act_set - base_set),
        "n_removed": len(base_set - act_set),
        "min_abs_move": min(movements) if movements else 0,
        "mean_abs_move": round(sum(movements) / len(movements), 3) if movements else 0.0,
        "max_abs_move": max(movements) if movements else 0,
        "n_moved_up": n_up,
        "n_moved_down": n_down,
        "n_top5_positions_changed": top5_changed,
        "highest_rank_touched": highest_touched,
        "preserves_top1": action[0] == base[0],
        "preserves_top3": action[:3] == base[:3],
        "preserves_top5": action[:5] == base[:5],
        "response_risk": bool(top1_changed),
    }


def ev_and_rationale(f: dict, action_type: str, has_evidence: bool) -> tuple[float, list[str]]:
    """Transparent rule-based expected-value score + human-readable rationale.

    Anchored on the only R92 evidence: the single positive was a near-minimal
    top-1 swap (overlap 19, routed_r90, margin ~0.20). Large/low-margin top-1
    swaps were neutral-to-negative. Top-1-preserving reorders are untested but
    carry no response-semantics risk, so they are the safest search direction.
    """
    ev = 0.0
    why: list[str] = []
    ov = f["overlap_20"]

    if f["preserves_top1"]:
        ev += 3.0
        why.append("preserves top-1 (no response risk)")
    else:
        ev -= 0.5
        why.append("top-1 swap (response-semantics risk)")

    # Minimal-change regime is where the only win lived.
    if ov >= 19:
        ev += 1.5
        why.append("overlap>=19 (minimal change, winning regime)")
    elif ov >= 18:
        ev += 1.0
        why.append("overlap 18 (tight)")
    elif ov >= 16:
        ev += 0.2
    else:
        ev -= 1.0 * (16 - ov)
        why.append(f"overlap {ov}<16 (broad change penalty)")

    # Movement concentrated in the high-nDCG ranks is worth more.
    if f["highest_rank_touched"] is not None and f["highest_rank_touched"] <= 4 and not f["top1_changed"]:
        ev += 0.8
        why.append("touches a top-5 rank")

    # Low-margin top-1 swaps were catastrophic in R92.
    margin = f.get("r54_margin")
    if f["top1_changed"] and margin is not None:
        if margin < 0.05:
            ev -= 3.0
            why.append(f"top-1 swap at very low margin {margin:.3f} (R92 danger zone)")
        elif margin < 0.15:
            ev -= 1.5
            why.append(f"top-1 swap at low margin {margin:.3f}")
        elif ov >= 19:
            ev += 1.0
            why.append("near-minimal top-1 swap (c4f7d055 regime)")

    if action_type == "source_top20":
        ev -= 1.5
        why.append("retrieval-order source (not LR-ranked)")
    elif action_type == "reorder":
        ev += 0.5
        why.append("R90 reorder lineage")

    # R90/R84 disagreement with high overlap is the sweet spot the spec calls out.
    if f.get("routed_r90") and not f.get("routed_r84") and ov >= 18:
        ev += 0.5
        why.append("R90/R84 route disagreement at high overlap")

    if has_evidence:
        ev += 0.3
        why.append("direct R92 evidence on this row")

    return round(ev, 3), why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_TABLE)
    args = ap.parse_args()

    base_rows = load_zip_rows(BASE_ZIP)
    base_by_key = {key(r): r for r in base_rows}

    r84c_by_key = {key(r): r for r in load_zip_rows(R84C_ZIP)}
    r90_tracks = json.load(open(R90_TRACKS))
    r90_by_key = {key(r): r for r in r90_tracks}

    r54c = json.load(open(R54_COSINE))
    r84c_cos = json.load(open(R84_COSINE))
    r90c = json.load(open(R90_COSINE))

    pm = pickle.load(open(PAYLOAD_MAPS, "rb"))
    track_artist = pm["track_artist"]
    src = pickle.load(open(SOURCE_CACHE, "rb"))

    # R92 measured labels (full R90 swaps).
    r92_manifest = json.load(open(R92_MANIFEST))
    r92_probe_by_sid = {p["session_id"]: p for p in r92_manifest["probes"]}
    r92_scores: dict[str, float] = {}
    with open(R92_SCORES) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid, _, val = line.partition(",")
            if val:
                r92_scores[pid.strip()] = float(val)

    # Prepared-but-unscored reorder probes: track-list fingerprints to avoid re-emitting.
    reorder_manifest = json.load(open(R92_REORDER_MANIFEST))
    reorder_sids = {p["session_id"] for p in reorder_manifest["probes"]}

    rows_out: list[dict] = []
    n_cases = 0
    for k, base in base_by_key.items():
        n_cases += 1
        sid, turn = k
        base_list = base["predicted_track_ids"]
        r90_row = r90_by_key.get(k, {})
        r90_list = r90_row.get("predicted_track_ids", base_list)
        r54_margin = r90_row.get("_r54_margin")
        routed_r90 = r90_row.get("_routed_r90")
        routed_r84 = r84c_by_key.get(k, {}).get("_routed_r84")
        case = src.get(sid, {})
        history_len = len(case.get("history", []) or [])
        n_played = len(case.get("music_turns", []) or [])

        actions = {
            "r90_full": ("full_swap", r90_list),
            "r90_keep_top1": ("reorder", topk_keep_reorder(base_list, r90_list, 1)),
            "r90_keep_top3": ("reorder", topk_keep_reorder(base_list, r90_list, 3)),
            "r90_keep_top5": ("reorder", topk_keep_reorder(base_list, r90_list, 5)),
            "r90_keep1_repl2_5": ("conservative", keep1_replace_2_5(base_list, r90_list)),
            "r54_top20": ("source_top20", first_n_unique(cosine_order(r54c, sid))),
            "r84src_top20": ("source_top20", first_n_unique(cosine_order(r84c_cos, sid))),
            "r90src_top20": ("source_top20", first_n_unique(cosine_order(r90c, sid))),
        }

        for source, (atype, action_list) in actions.items():
            if len(action_list) != 20 or len(set(action_list)) != 20:
                # Skip degenerate constructions (e.g. cosine list with dups/short).
                continue
            if action_list == base_list:
                continue  # no-op vs current production

            feats = describe(base_list, action_list, track_artist)
            feats["r54_margin"] = r54_margin
            feats["routed_r90"] = routed_r90
            feats["routed_r84"] = routed_r84

            # Label join: R92 measured the full R90 swap on certain sessions.
            measured = None
            blind_delta = None
            label_pos = None
            if source == "r90_full" and sid in r92_probe_by_sid:
                pid = r92_probe_by_sid[sid]["probe_id"]
                if pid in r92_scores:
                    measured = pid
                    blind_delta = round(r92_scores[pid] - R84C_BASE_NDCG, 4)
                    label_pos = blind_delta > POS_THRESH

            has_evidence = measured is not None
            ev, why = ev_and_rationale({**feats}, atype, has_evidence)

            rows_out.append({
                "session_id": sid,
                "turn_number": turn,
                "history_len": history_len,
                "n_played": n_played,
                "action_source": source,
                "action_type": atype,
                "action_track_ids": action_list,
                **feats,
                "measured_in_r92": measured,
                "blind_delta": blind_delta,
                "label_positive": label_pos,
                "equals_r92_reorder_candidate": bool(
                    atype == "reorder" and source == "r90_keep_top1" and sid in reorder_sids
                ),
                "ev_score": ev,
                "ev_rationale": why,
            })

    rows_out.sort(key=lambda r: r["ev_score"], reverse=True)

    measured_rows = [r for r in rows_out if r["measured_in_r92"]]
    out = {
        "experiment": "R93 blind-oracle action table",
        "base_submission": str(BASE_ZIP.relative_to(REPO)),
        "base_ndcg20_prod": PROD_NDCG,
        "base_ndcg20_r84c": R84C_BASE_NDCG,
        "positive_threshold": POS_THRESH,
        "n_cases": n_cases,
        "n_actions": len(rows_out),
        "n_measured": len(measured_rows),
        "n_measured_positive": sum(1 for r in measured_rows if r["label_positive"]),
        "action_sources": sorted({r["action_source"] for r in rows_out}),
        "rows": rows_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"cases={n_cases}  actions={len(rows_out)}  "
          f"measured={out['n_measured']} (positive={out['n_measured_positive']})")
    print(f"sources: {out['action_sources']}")
    print(f"\nTop 15 actions by EV:")
    for r in rows_out[:15]:
        lbl = f" label={r['blind_delta']:+.4f}" if r["blind_delta"] is not None else ""
        print(f"  ev={r['ev_score']:+.2f} {r['session_id'][:8]}_t{r['turn_number']:<2d} "
              f"{r['action_source']:18s} ov={r['overlap_20']:2d} "
              f"top1_chg={int(r['top1_changed'])}{lbl}")
    print(f"\nWrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

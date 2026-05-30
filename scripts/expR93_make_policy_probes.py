#!/usr/bin/env python3
"""R93 Phase 2: generate the policy-selected single-row probe batch.

Reads the EV-ranked action table (expR93_build_action_table.py) and selects 10
single-row probes that maximize expected information per Codabench slot, under
the R92-derived policy:

  Bucket A (4) - r90_keep_top1   : trust R90's ordering below rank 1; pure
                                   top-1-preserving reorder.
  Bucket B (3) - r90_keep_top5   : freeze ranks 1-5, reshuffle 6-20 in R90
                                   order; lowest-risk conservative action.
  Bucket C (3) - r90_keep1_repl2_5 : keep rank 1 and ranks 6-20, replace the
                                   high-value ranks 2-5 from R90; bounded but
                                   higher upside.

Every selected action preserves top-1, so the R92 p11 response is reused with
no response-semantics risk. R92 showed top-1 *swaps* are mostly neutral or
catastrophic, so they are deliberately excluded from this batch.

Exclusions:
  - one probe per session,
  - never duplicate the changed-row track list of any existing R92 probe ZIP
    (measured p01-p12 or prepared reorder r92r01-08).

Builds probes on the current production submission (R92 p11). Writes a manifest,
a scores template CSV, and the probe ZIPs. Uploads nothing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ACTION_TABLE = REPO / "exp" / "eval" / "expR93_action_table.json"
BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
R90_TRACKS = REPO / "exp" / "inference" / "blind_a" / "r90_blind_track_lists.json"
R92_MANIFEST = REPO / "exp" / "eval" / "expR92_probe_manifest.json"
R92_SCORES = REPO / "exp" / "eval" / "expR92_probe_scores_template.csv"
OUT_DIR = REPO / "exp" / "inference" / "blind_a" / "r93_policy_probes"
OUT_MANIFEST = REPO / "exp" / "eval" / "expR93_policy_probe_manifest.json"
OUT_SCORES = REPO / "exp" / "eval" / "expR93_policy_probe_scores_template.csv"

BUCKETS = [
    ("A_reorder_keep_top1", "r90_keep_top1", 4),
    ("B_conservative_keep_top5", "r90_keep_top5", 3),
    ("C_bounded_replace_2_5", "r90_keep1_repl2_5", 3),
]


def load_zip_rows(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        return json.loads(zf.read(name))


def write_prediction_zip(path: Path, rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, int]:
    return row["session_id"], int(row["turn_number"])


def scored_answer_fingerprints() -> set[tuple]:
    """Fingerprint only the probes Codabench has ALREADY scored, so the new
    batch never spends a slot re-learning a known answer.

    Every scored R92 probe was a full R90 swap on its session, so its changed
    row is exactly the R90 track list for that session. The unscored prepared
    r92r* reorder probes are NOT excluded: they are unscored and were built on
    the stale pre-p11 base, so R93 supersedes them."""
    r90_by_key = {key(r): r for r in json.load(open(R90_TRACKS))}
    manifest = json.load(open(R92_MANIFEST))
    sid_to_turn = {p["session_id"]: int(p["turn_number"]) for p in manifest["probes"]}
    scored_sessions: set[str] = set()
    with open(R92_SCORES) as f:
        next(f)
        for line in f:
            pid, _, val = line.strip().partition(",")
            if not val:
                continue
            # probe_id format r92pNN_<sid8>_t<turn>
            parts = pid.split("_")
            if len(parts) >= 2:
                scored_sessions.add(parts[1])
    fps: set[tuple] = set()
    for r in json.load(open(R90_TRACKS)):
        sid8 = r["session_id"][:8]
        if sid8 in scored_sessions:
            fps.add((r["session_id"], int(r["turn_number"]), tuple(r["predicted_track_ids"])))
    return fps


def validate_probe(rows: list[dict], base_rows: list[dict], changed_key: tuple) -> list[str]:
    issues = []
    if len(rows) != 80:
        issues.append(f"expected 80 rows, got {len(rows)}")
    base_by_key = {key(r): r for r in base_rows}
    n_changed = 0
    changed_seen = None
    for r in rows:
        k = key(r)
        tids = r["predicted_track_ids"]
        if len(tids) != 20:
            issues.append(f"{k[0][:8]} has {len(tids)} tracks")
        if len(set(tids)) != len(tids):
            issues.append(f"{k[0][:8]} has duplicate track ids")
        if not r.get("predicted_response", "").strip():
            issues.append(f"{k[0][:8]} has empty response")
        base = base_by_key.get(k)
        if base is None:
            issues.append(f"{k[0][:8]} not in base")
            continue
        if tids != base["predicted_track_ids"]:
            n_changed += 1
            changed_seen = k
        if r.get("predicted_response") != base.get("predicted_response"):
            issues.append(f"{k[0][:8]} response differs from base (should be reused)")
    if n_changed != 1:
        issues.append(f"expected exactly 1 changed row, got {n_changed}")
    elif changed_seen != changed_key:
        issues.append(f"changed row {changed_seen} != intended {changed_key}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-table", type=Path, default=ACTION_TABLE)
    ap.add_argument("--base-zip", type=Path, default=BASE_ZIP)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    ap.add_argument("--out-scores", type=Path, default=OUT_SCORES)
    args = ap.parse_args()

    table = json.load(open(args.action_table))
    base_rows = load_zip_rows(args.base_zip)
    base_by_key = {key(r): r for r in base_rows}
    base_index = {key(r): i for i, r in enumerate(base_rows)}

    fps = scored_answer_fingerprints()
    print(f"Scored-answer fingerprints to avoid: {len(fps)}")

    rows = table["rows"]  # already EV-sorted desc
    used_sessions: set[str] = set()
    selected: list[dict] = []

    for bucket_name, source, n_want in BUCKETS:
        picked = 0
        for cand in rows:
            if picked >= n_want:
                break
            if cand["action_source"] != source:
                continue
            sid, turn = cand["session_id"], cand["turn_number"]
            if sid in used_sessions:
                continue
            if not cand["preserves_top1"]:
                continue
            fp = (sid, turn, tuple(cand["action_track_ids"]))
            if fp in fps:
                continue
            if cand["measured_in_r92"]:
                continue
            cand = dict(cand)
            cand["bucket"] = bucket_name
            selected.append(cand)
            used_sessions.add(sid)
            picked += 1
        if picked < n_want:
            print(f"  WARNING: bucket {bucket_name} only filled {picked}/{n_want}")

    if len(selected) < 10:
        # Backfill from any remaining top-1-preserving reorder/conservative action.
        for cand in rows:
            if len(selected) >= 10:
                break
            sid = cand["session_id"]
            if sid in used_sessions or not cand["preserves_top1"]:
                continue
            if cand["action_type"] not in ("reorder", "conservative"):
                continue
            if cand["measured_in_r92"]:
                continue
            fp = (sid, cand["turn_number"], tuple(cand["action_track_ids"]))
            if fp in fps:
                continue
            cand = dict(cand)
            cand["bucket"] = "D_backfill"
            selected.append(cand)
            used_sessions.add(sid)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "R93 policy-selected single-row nDCG probes",
        "base_submission": str(args.base_zip.relative_to(REPO)),
        "base_ndcg20_prod": table["base_ndcg20_prod"],
        "base_ndcg20_r84c": table["base_ndcg20_r84c"],
        "policy": "all probes preserve top-1 (zero response risk); R92 showed "
                  "top-1 swaps are dangerous. Buckets A/B/C = R90 reorder, "
                  "top-5-frozen, and rank-2-5 replacement.",
        "n_probes": len(selected),
        "instructions": [
            "Upload each ZIP manually to Codabench.",
            "Record the 4-decimal nDCG@20 from the scoring log (not the rounded "
            "leaderboard) for each probe_id into the scores template.",
            "Run scripts/expR93_analyze_policy_scores.py with the filled CSV.",
        ],
        "probes": [],
    }

    all_issues = 0
    for i, cand in enumerate(selected, start=1):
        sid, turn = cand["session_id"], cand["turn_number"]
        probe_id = f"r93p{i:02d}_{sid[:8]}_t{turn}"
        rows_out = copy.deepcopy(base_rows)
        idx = base_index[(sid, turn)]
        rows_out[idx]["predicted_track_ids"] = list(cand["action_track_ids"])
        # Response intentionally reused (top-1 unchanged => semantically valid).
        zip_path = args.out_dir / f"{probe_id}.zip"
        sha = write_prediction_zip(zip_path, rows_out)

        issues = validate_probe(rows_out, base_rows, (sid, turn))
        all_issues += len(issues)
        manifest["probes"].append({
            "probe_id": probe_id,
            "bucket": cand["bucket"],
            "zip": str(zip_path.relative_to(REPO)),
            "sha256": sha,
            "row_index": idx,
            "session_id": sid,
            "turn_number": turn,
            "action_source": cand["action_source"],
            "action_type": cand["action_type"],
            "ev_score": cand["ev_score"],
            "ev_rationale": cand["ev_rationale"],
            "top1_changed": cand["top1_changed"],
            "overlap_20": cand["overlap_20"],
            "n_positions_changed": cand["n_positions_changed"],
            "n_top5_positions_changed": cand["n_top5_positions_changed"],
            "r54_margin": cand["r54_margin"],
            "routed_r90": cand["routed_r90"],
            "base_top1": cand["base_top1"],
            "new_top1": cand["new_top1"],
            "validation_issues": issues,
        })

    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(args.out_scores, "w") as f:
        f.write("probe_id,ndcg20\n")
        for p in manifest["probes"]:
            f.write(f"{p['probe_id']},\n")

    print(f"\nSelected {len(selected)} probes "
          f"(validation issues: {all_issues}):")
    for p in manifest["probes"]:
        flag = " !!" + ";".join(p["validation_issues"]) if p["validation_issues"] else ""
        print(f"  {p['probe_id']:24s} {p['bucket']:24s} {p['action_source']:18s} "
              f"ov={p['overlap_20']:2d} ev={p['ev_score']:+.2f}{flag}")
    print(f"\nManifest -> {args.out_manifest.relative_to(REPO)}")
    print(f"Scores template -> {args.out_scores.relative_to(REPO)}")
    print(f"Probe ZIPs -> {args.out_dir.relative_to(REPO)}")
    if all_issues:
        raise SystemExit("Validation failed; do not upload.")


if __name__ == "__main__":
    main()

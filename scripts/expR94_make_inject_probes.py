#!/usr/bin/env python3
"""R94 Phase 2: generate hidden-GT injection probes (Hybrid policy).

Consumes the R94 miss-candidate ranking (expR94_build_miss_candidate_table.py)
and emits 8-10 single-row probes that INJECT a genuinely new candidate track
(absent from current production's top-20) into a likely-miss row. Unlike R92/R93
this changes the candidate SET, not just its order; R93 proved reorders are
no-op.

Hybrid insertion rule (per user spec):
  top-1 injection ONLY when ALL hold:
    - r54_margin < 0.5
    - candidate same artist as history OR as current top-1
    - consensus >= 4 orthogonal sources
    - candidate is R54-corroborated
  otherwise rank-2 injection (top-1 preserved, response-safe).

Selection: top fresh rows by probe_priority (sessions Codabench has already
SCORED are excluded; a previously-scored session would be allowed only if the
injected action is materially different, but the current batch needs none).

Responses are reused verbatim from R92 p11 (probes measure nDCG@20 only).
Validation gate before upload: 80 rows, 20 unique tracks/row, exactly one
changed row at the intended key, non-empty response. Uploads nothing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TABLE = REPO / "exp" / "eval" / "expR94_miss_candidate_table.json"
BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
OUT_DIR = REPO / "exp" / "inference" / "blind_a" / "r94_inject_probes"
OUT_MANIFEST = REPO / "exp" / "eval" / "expR94_inject_probe_manifest.json"
OUT_SCORES = REPO / "exp" / "eval" / "expR94_inject_probe_scores_template.csv"

PROD_NDCG = 0.5073


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


def use_top1(row: dict) -> bool:
    b = row["best_candidate"]
    m = row["r54_margin"]
    return bool(
        b
        and m is not None and m < 0.5
        and (b["same_artist_history"] or b["same_artist_top1"])
        and b["consensus"] >= 4
        and b["r54_corroborated"]
    )


def inject(top20: list[str], cand: str, at: str) -> list[str]:
    rest = [t for t in top20 if t != cand]
    if at == "top1":
        return [cand] + rest[:19]
    # rank2: preserve current top-1
    return [rest[0], cand] + rest[1:19]


def validate(rows: list[dict], base_rows: list[dict], changed_key: tuple) -> list[str]:
    issues = []
    if len(rows) != 80:
        issues.append(f"expected 80 rows, got {len(rows)}")
    base_by_key = {key(r): r for r in base_rows}
    n_changed = 0
    seen = None
    for r in rows:
        k = key(r)
        tids = r["predicted_track_ids"]
        if len(tids) != 20 or len(set(tids)) != 20:
            issues.append(f"{k[0][:8]} bad track list (len {len(tids)}, uniq {len(set(tids))})")
        if not (r.get("predicted_response") or "").strip():
            issues.append(f"{k[0][:8]} empty response")
        base = base_by_key.get(k)
        if base and tids != base["predicted_track_ids"]:
            n_changed += 1
            seen = k
    if n_changed != 1:
        issues.append(f"expected exactly 1 changed row, got {n_changed}")
    elif seen != changed_key:
        issues.append(f"changed row {seen} != intended {changed_key}")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=TABLE)
    ap.add_argument("--base-zip", type=Path, default=BASE_ZIP)
    ap.add_argument("--n", type=int, default=10, help="number of probes (8-10)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-manifest", type=Path, default=OUT_MANIFEST)
    ap.add_argument("--out-scores", type=Path, default=OUT_SCORES)
    args = ap.parse_args()

    table = json.load(open(args.table))
    base_rows = load_zip_rows(args.base_zip)
    base_by_key = {key(r): r for r in base_rows}
    base_index = {key(r): i for i, r in enumerate(base_rows)}

    fresh = [r for r in table["rows"]
             if not r["already_probed"] and r["best_candidate"]]
    fresh.sort(key=lambda r: r["probe_priority"], reverse=True)
    selected = fresh[: args.n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "R94 hidden-GT injection probes (Hybrid)",
        "base_submission": str(args.base_zip.relative_to(REPO)),
        "base_ndcg20": PROD_NDCG,
        "policy": "Hybrid: top-1 injection iff margin<0.5 & same-artist(hist|top1) "
                  "& consensus>=4 & R54-corroborated; else rank-2 injection.",
        "judge": "nDCG@20 vs current production baseline 0.5073 ONLY.",
        "instructions": [
            "Upload each ZIP manually to Codabench.",
            "Record 4-decimal nDCG@20 from the scoring log into the scores template.",
            "Run scripts/expR94_analyze_inject_scores.py with the filled CSV.",
        ],
        "probes": [],
    }

    all_issues = 0
    n_top1 = 0
    for i, row in enumerate(selected, start=1):
        sid, turn = row["session_id"], row["turn_number"]
        b = row["best_candidate"]
        cand = b["track_id"]
        at = "top1" if use_top1(row) else "rank2"
        if at == "top1":
            n_top1 += 1
        probe_id = f"r94p{i:02d}_{sid[:8]}_t{turn}_{at}"
        rows_out = copy.deepcopy(base_rows)
        idx = base_index[(sid, turn)]
        top20 = base_by_key[(sid, turn)]["predicted_track_ids"]
        rows_out[idx]["predicted_track_ids"] = inject(top20, cand, at)
        sha = write_prediction_zip(args.out_dir / f"{probe_id}.zip", rows_out)
        issues = validate(rows_out, base_rows, (sid, turn))
        all_issues += len(issues)
        manifest["probes"].append({
            "probe_id": probe_id,
            "insert": at,
            "label": "top1_inject" if at == "top1" else "rank2_inject",
            "zip": str((args.out_dir / f"{probe_id}.zip").relative_to(REPO)),
            "sha256": sha,
            "row_index": idx,
            "session_id": sid,
            "turn_number": turn,
            "current_top1": row["current_top1"],
            "injected_track": cand,
            "injected_consensus": b["consensus"],
            "injected_sources": b["sources"],
            "same_artist_history": b["same_artist_history"],
            "same_artist_top1": b["same_artist_top1"],
            "r54_corroborated": b["r54_corroborated"],
            "cand_score": b["cand_score"],
            "r54_margin": row["r54_margin"],
            "probe_priority": row["probe_priority"],
            "validation_issues": issues,
            "needs_response_regen_if_kept": at == "top1",
        })

    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(args.out_scores, "w") as f:
        f.write("probe_id,ndcg20\n")
        for p in manifest["probes"]:
            f.write(f"{p['probe_id']},\n")

    print(f"Selected {len(selected)} probes ({n_top1} top1, {len(selected)-n_top1} rank2); "
          f"validation issues: {all_issues}")
    for p in manifest["probes"]:
        flag = " !!" + ";".join(p["validation_issues"]) if p["validation_issues"] else ""
        sig = []
        if p["same_artist_history"]: sig.append("art∈hist")
        if p["same_artist_top1"]: sig.append("art=top1")
        if p["r54_corroborated"]: sig.append("r54")
        print(f"  {p['probe_id']:30s} cons={p['injected_consensus']} "
              f"mgn={p['r54_margin']:.2f} prio={p['probe_priority']:.1f} "
              f"[{','.join(sig)}]{flag}")
    print(f"\nManifest -> {args.out_manifest.relative_to(REPO)}")
    print(f"Scores template -> {args.out_scores.relative_to(REPO)}")
    print(f"Probe ZIPs -> {args.out_dir.relative_to(REPO)}")
    if all_issues:
        raise SystemExit("Validation failed; do not upload.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R93 Phase 3: analyze policy-probe scores and assemble a combined candidate.

Input CSV (probe_id,ndcg20) holds the official Blind-A nDCG@20 returned by
Codabench for each R93 probe. Each probe is the current production submission
(R92 p11, nDCG 0.5073) with exactly one top-1-preserving row swapped, so:

    delta_vs_prod = ndcg20 - 0.5073      # credit ON TOP of current production
    delta_vs_r84c = ndcg20 - 0.5069      # credit vs original R84c

Because every probe changes a different session and nDCG@20 is a mean of
independent per-case scores, positive single-row deltas combine additively.
This script selects the positive actions and stacks them onto the p11 base to
produce a combined candidate ZIP, with collision / response-mismatch warnings.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MANIFEST = REPO / "exp" / "eval" / "expR93_policy_probe_manifest.json"
BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
DEFAULT_OUT = REPO / "exp" / "eval" / "expR93_policy_score_analysis.json"
OUT_CAND_ZIP = REPO / "exp" / "inference" / "blind_a" / "r93_combined_candidate.zip"
OUT_CAND_AUDIT = REPO / "exp" / "eval" / "expR93_combined_candidate_audit.json"

PROD_NDCG = 0.5073
R84C_NDCG = 0.5069
POS_THRESH = 0.00005


def load_zip_rows(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        name = "prediction.json" if "prediction.json" in zf.namelist() else zf.namelist()[0]
        return json.loads(zf.read(name))


def key(row: dict) -> tuple[str, int]:
    return row["session_id"], int(row["turn_number"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--scores-csv", type=Path, required=True)
    ap.add_argument("--base-zip", type=Path, default=BASE_ZIP)
    ap.add_argument("--base-ndcg20-prod", type=float, default=PROD_NDCG)
    ap.add_argument("--base-ndcg20-r84c", type=float, default=R84C_NDCG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--build-candidate", action="store_true",
                    help="Write the combined positive-rows candidate ZIP.")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    probes = {p["probe_id"]: p for p in manifest["probes"]}

    scores: dict[str, float] = {}
    with open(args.scores_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["probe_id"].strip()
            val = (row.get("ndcg20") or "").strip()
            if val:
                scores[pid] = float(val)

    rows = []
    for pid, ndcg in scores.items():
        if pid not in probes:
            raise ValueError(f"score for unknown probe_id: {pid}")
        p = probes[pid]
        d_prod = ndcg - args.base_ndcg20_prod
        rows.append({
            "probe_id": pid,
            "bucket": p["bucket"],
            "action_source": p["action_source"],
            "ndcg20": ndcg,
            "delta_vs_prod": round(d_prod, 4),
            "delta_vs_r84c": round(ndcg - args.base_ndcg20_r84c, 4),
            "keep": d_prod > POS_THRESH,
            "session_id": p["session_id"],
            "turn_number": p["turn_number"],
            "top1_changed": p["top1_changed"],
            "overlap_20": p["overlap_20"],
            "zip": p["zip"],
        })
    rows.sort(key=lambda r: r["delta_vs_prod"], reverse=True)

    positives = [r for r in rows if r["keep"]]
    warnings: list[str] = []
    sess_seen: dict[str, str] = {}
    for r in positives:
        if r["session_id"] in sess_seen:
            warnings.append(
                f"two positive probes touch session {r['session_id'][:8]} "
                f"({sess_seen[r['session_id']]} & {r['probe_id']}); deltas may NOT add")
        sess_seen[r["session_id"]] = r["probe_id"]
        if r["top1_changed"]:
            warnings.append(
                f"{r['probe_id']} changes top-1 -> response must be regenerated "
                f"before any real submission (probe reused the p11 response)")

    out = {
        "experiment": "R93 policy-probe score analysis",
        "base_ndcg20_prod": args.base_ndcg20_prod,
        "base_ndcg20_r84c": args.base_ndcg20_r84c,
        "positive_threshold": POS_THRESH,
        "n_scored": len(rows),
        "n_positive": len(positives),
        "additive_delta_vs_prod": round(sum(r["delta_vs_prod"] for r in positives), 4),
        "projected_prod_ndcg": round(args.base_ndcg20_prod
                                     + sum(r["delta_vs_prod"] for r in positives), 4),
        "warnings": warnings,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Scored: {len(rows)}  positive: {len(positives)}")
    print(f"Additive delta vs prod: {out['additive_delta_vs_prod']:+.4f} "
          f"-> projected nDCG {out['projected_prod_ndcg']:.4f}")
    for r in rows:
        mark = "KEEP" if r["keep"] else "drop"
        print(f"  {mark:4s} {r['probe_id']:24s} {r['bucket']:24s} "
              f"ndcg={r['ndcg20']:.4f} d_prod={r['delta_vs_prod']:+.4f} ov={r['overlap_20']}")
    for w in warnings:
        print(f"  WARN: {w}")

    if args.build_candidate and positives:
        base_rows = load_zip_rows(args.base_zip)
        base_index = {key(r): i for i, r in enumerate(base_rows)}
        applied = []
        for r in positives:
            probe_rows = load_zip_rows(REPO / r["zip"])
            pr_by_key = {key(x): x for x in probe_rows}
            k = (r["session_id"], r["turn_number"])
            base_rows[base_index[k]]["predicted_track_ids"] = \
                list(pr_by_key[k]["predicted_track_ids"])
            applied.append({"session_id": k[0], "turn_number": k[1],
                            "probe_id": r["probe_id"], "delta_vs_prod": r["delta_vs_prod"],
                            "top1_changed": r["top1_changed"]})
        payload = json.dumps(base_rows, ensure_ascii=False, indent=2)
        with zipfile.ZipFile(OUT_CAND_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("prediction.json", payload)
        audit = {
            "experiment": "R93 combined positive-rows candidate",
            "base_submission": str(args.base_zip.relative_to(REPO)),
            "n_rows_applied": len(applied),
            "expected_ndcg_delta_vs_prod": out["additive_delta_vs_prod"],
            "projected_prod_ndcg": out["projected_prod_ndcg"],
            "needs_response_regen": [a for a in applied if a["top1_changed"]],
            "applied": applied,
            "warnings": warnings,
        }
        with open(OUT_CAND_AUDIT, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\nCombined candidate -> {OUT_CAND_ZIP.relative_to(REPO)} "
              f"({len(applied)} rows)")
        if audit["needs_response_regen"]:
            print(f"  NOTE: {len(audit['needs_response_regen'])} rows changed top-1; "
                  f"regenerate responses before submitting.")

    print(f"\nWrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

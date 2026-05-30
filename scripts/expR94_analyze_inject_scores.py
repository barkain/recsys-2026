#!/usr/bin/env python3
"""R94 Phase 3: analyze injection-probe scores; judge nDCG@20 ONLY.

Each probe is current production (R92 p11, nDCG@20 0.5073) with exactly one row
changed by injecting a new candidate, so:

    delta_vs_prod = ndcg20 - 0.5073

Positive rows on DIFFERENT sessions add. Selects positives and stacks them onto
the p11 base into a combined candidate. top-1 injections that win need a
response regen before any real submission (the probe reused the p11 response);
this is flagged, never auto-applied. Responses are otherwise reused, so do NOT
read composite/LLM here — nDCG@20 only.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MANIFEST = REPO / "exp" / "eval" / "expR94_inject_probe_manifest.json"
BASE_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
DEFAULT_OUT = REPO / "exp" / "eval" / "expR94_inject_score_analysis.json"
OUT_CAND_ZIP = REPO / "exp" / "inference" / "blind_a" / "r94_combined_candidate.zip"
OUT_CAND_AUDIT = REPO / "exp" / "eval" / "expR94_combined_candidate_audit.json"

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
    ap.add_argument("--base-ndcg20", type=float, default=PROD_NDCG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--build-candidate", action="store_true")
    args = ap.parse_args()

    probes = {p["probe_id"]: p for p in json.load(open(args.manifest))["probes"]}
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
        d = ndcg - args.base_ndcg20
        rows.append({
            "probe_id": pid,
            "label": p["label"],
            "ndcg20": ndcg,
            "delta_vs_prod": round(d, 4),
            "delta_vs_r84c": round(ndcg - R84C_NDCG, 4),
            "keep": d > POS_THRESH,
            "session_id": p["session_id"],
            "turn_number": p["turn_number"],
            "injected_track": p["injected_track"],
            "insert": p["insert"],
            "injected_consensus": p["injected_consensus"],
            "needs_response_regen_if_kept": p["needs_response_regen_if_kept"],
            "zip": p["zip"],
        })
    rows.sort(key=lambda r: r["delta_vs_prod"], reverse=True)
    positives = [r for r in rows if r["keep"]]

    warnings = []
    seen = {}
    for r in positives:
        if r["session_id"] in seen:
            warnings.append(f"two positives touch session {r['session_id'][:8]}; deltas may not add")
        seen[r["session_id"]] = r["probe_id"]
        if r["insert"] == "top1":
            warnings.append(f"{r['probe_id']} is a winning top-1 injection -> regenerate "
                            f"its response before any real submission")

    out = {
        "experiment": "R94 injection-probe score analysis",
        "judge": "nDCG@20 only",
        "base_ndcg20_prod": args.base_ndcg20,
        "base_ndcg20_r84c": R84C_NDCG,
        "n_scored": len(rows),
        "n_positive": len(positives),
        "n_positive_top1": sum(1 for r in positives if r["insert"] == "top1"),
        "n_positive_rank2": sum(1 for r in positives if r["insert"] == "rank2"),
        "additive_delta_vs_prod": round(sum(r["delta_vs_prod"] for r in positives), 4),
        "projected_prod_ndcg": round(args.base_ndcg20
                                     + sum(r["delta_vs_prod"] for r in positives), 4),
        "warnings": warnings,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Scored: {len(rows)}  positive: {len(positives)} "
          f"(top1={out['n_positive_top1']}, rank2={out['n_positive_rank2']})")
    print(f"Additive delta vs prod: {out['additive_delta_vs_prod']:+.4f} "
          f"-> projected nDCG {out['projected_prod_ndcg']:.4f}")
    for r in rows:
        mark = "KEEP" if r["keep"] else "drop"
        print(f"  {mark:4s} {r['probe_id']:30s} ndcg={r['ndcg20']:.4f} "
              f"d_prod={r['delta_vs_prod']:+.4f} [{r['label']}]")
    for w in warnings:
        print(f"  WARN: {w}")

    if args.build_candidate and positives:
        base_rows = load_zip_rows(args.base_zip)
        bidx = {key(r): i for i, r in enumerate(base_rows)}
        applied = []
        for r in positives:
            pr = {key(x): x for x in load_zip_rows(REPO / r["zip"])}
            k = (r["session_id"], r["turn_number"])
            base_rows[bidx[k]]["predicted_track_ids"] = list(pr[k]["predicted_track_ids"])
            applied.append({"session_id": k[0], "turn_number": k[1], "probe_id": r["probe_id"],
                            "insert": r["insert"], "delta_vs_prod": r["delta_vs_prod"]})
        with zipfile.ZipFile(OUT_CAND_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("prediction.json", json.dumps(base_rows, ensure_ascii=False, indent=2))
        audit = {
            "experiment": "R94 combined positive-injection candidate",
            "n_rows_applied": len(applied),
            "projected_prod_ndcg": out["projected_prod_ndcg"],
            "needs_response_regen": [a for a in applied if a["insert"] == "top1"],
            "applied": applied,
            "warnings": warnings,
        }
        with open(OUT_CAND_AUDIT, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\nCombined candidate -> {OUT_CAND_ZIP.relative_to(REPO)} ({len(applied)} rows)")
        if audit["needs_response_regen"]:
            print(f"  NOTE: {len(audit['needs_response_regen'])} winning top-1 injections "
                  f"need response regen before submitting.")
    print(f"\nWrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

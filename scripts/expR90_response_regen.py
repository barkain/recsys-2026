"""R90 response regeneration and Blind-A submission packaging.

R90 changes only the recommendation tracks. Reuse the current production R84c
responses when the top-1 track is unchanged; regenerate R78-style responses for
rows whose top-1 changed vs R84c, then package a Codabench ZIP.

Run:
  ANTHROPIC_RECSYS_API_KEY=sk-... uv run python scripts/expR90_response_regen.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR84c_response_regen import (  # noqa: E402
    MODEL_ID,
    build_prompt,
    call_opus,
    load_catalog_meta,
    short_ref,
    validate,
)

R90_TRACKS_JSON = REPO / "exp" / "inference" / "blind_a" / "r90_blind_track_lists.json"
R84C_SUB = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
R90_AUDIT = REPO / "exp" / "eval" / "expR90_blind_audit.json"
BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_ZIP = OUT_DIR / "r90_submission.zip"
OUT_METADATA = OUT_DIR / "r90_submission.metadata.json"
OUT_REGEN_ROWS = OUT_DIR / "r90_regen_rows.jsonl"
OUT_DIFF_MD = REPO / "docs" / "r90_blind_diff.md"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        return "no-git"
    return subprocess.check_output(
        [git_bin, "rev-parse", "HEAD"], cwd=str(REPO)
    ).decode().strip()


def load_prediction_zip(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-on-issues", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R90 response regeneration")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not args.dry_run and not api_key:
        print("ERROR: ANTHROPIC_API_KEY or ANTHROPIC_RECSYS_API_KEY is not set.")
        sys.exit(1)

    with open(R90_TRACKS_JSON) as f:
        r90_tracks = json.load(f)
    r84c_items = load_prediction_zip(R84C_SUB)
    r84c_by_key = {
        (item["session_id"], int(item["turn_number"])): item for item in r84c_items
    }
    with open(R90_AUDIT) as f:
        audit = json.load(f)
    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)

    print(f"  R90 tracks: {len(r90_tracks)} cases")
    print(f"  R84c production responses: {len(r84c_items)} cases")
    print(
        "  audit: "
        f"churn={audit.get('top1_churn_per_80')}/80, "
        f"overlap={audit.get('top20_overlap_mean'):.2f}/20, "
        f"routes R90/R54={audit.get('n_routed_r90')}/{audit.get('n_routed_r54')}, "
        f"passes={audit.get('audit_passes')}"
    )
    if not audit.get("audit_passes"):
        raise RuntimeError("R90 blind audit did not pass packaging gates.")

    changed_indices = []
    for idx, r90 in enumerate(r90_tracks):
        key = (r90["session_id"], int(r90["turn_number"]))
        r84c = r84c_by_key.get(key)
        if r84c is None:
            raise KeyError(f"R84c base response missing for {key}")
        r90_top1 = r90["predicted_track_ids"][0]
        r84c_top1 = r84c["predicted_track_ids"][0]
        if r90_top1 != r84c_top1:
            changed_indices.append(idx)

    print(f"\n  Cases needing regeneration: {len(changed_indices)}/80")
    for idx in changed_indices:
        r90 = r90_tracks[idx]
        key = (r90["session_id"], int(r90["turn_number"]))
        r84c = r84c_by_key[key]
        print(
            f"    {r90['session_id'][:8]} t{r90['turn_number']} "
            f"R84c_top1={r84c['predicted_track_ids'][0][:8]} -> "
            f"R90_top1={r90['predicted_track_ids'][0][:8]} "
            f"(R90={r90.get('_routed_r90')}, margin={r90.get('_r54_margin'):.3f})"
        )

    if args.dry_run:
        print(f"\n  [dry-run] would call Opus {len(changed_indices)} times.")
        return

    print(f"\n{ts()} Loading catalog metadata...")
    catalog_meta = load_catalog_meta()
    print(f"  {len(catalog_meta)} tracks")

    played_refs_per_case = {}
    for r90 in r90_tracks:
        sid = r90["session_id"]
        if sid not in blind:
            continue
        played_refs_per_case[sid] = {
            tid: short_ref(tid, catalog_meta)
            for tid in blind[sid]["music_turns"]
            if tid in catalog_meta
        }

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    regen_rows = []
    regen_responses = {}
    print(f"\n{ts()} Regenerating {len(changed_indices)} responses with {MODEL_ID}...")
    for n, idx in enumerate(changed_indices, 1):
        r90 = r90_tracks[idx]
        sid = r90["session_id"]
        case = blind[sid]
        new_top1 = r90["predicted_track_ids"][0]
        top_meta = catalog_meta.get(new_top1)
        if top_meta is None:
            raise KeyError(f"Missing catalog metadata for R90 top1 {new_top1}")

        prompt = build_prompt(case, top_meta, played_refs_per_case.get(sid, {}))
        t_call = time.time()
        resp, _msg = call_opus(client, prompt)
        issues = validate(resp)
        if issues and args.retry_on_issues:
            retry_prompt = (
                prompt
                + "\n\nIssues with prior attempt: "
                + ", ".join(issues)
                + ". Rewrite addressing those issues exactly."
            )
            resp2, _msg2 = call_opus(client, retry_prompt)
            issues2 = validate(resp2)
            if len(issues2) <= len(issues):
                resp = resp2
                issues = issues2

        regen_responses[sid] = resp
        row = {
            "session_id": sid,
            "turn_number": int(r90["turn_number"]),
            "new_top1": new_top1,
            "response": resp,
            "issues_after_regen": issues,
            "_routed_r90": r90.get("_routed_r90"),
            "_r54_margin": r90.get("_r54_margin"),
            "n_words": len(resp.split()),
            "elapsed_s": time.time() - t_call,
        }
        regen_rows.append(row)
        print(
            f"  {n}/{len(changed_indices)} {sid[:8]} "
            f"({row['n_words']} words, issues={issues}, {row['elapsed_s']:.1f}s)"
        )

    if len(regen_responses) != len(changed_indices):
        raise RuntimeError(
            f"Only regenerated {len(regen_responses)} of {len(changed_indices)} changed rows."
        )

    with open(OUT_REGEN_ROWS, "w") as f:
        for row in regen_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\n  Saved regen rows -> {OUT_REGEN_ROWS.relative_to(REPO)}")

    final_items = []
    n_reused = 0
    n_regen = 0
    for r90 in r90_tracks:
        sid = r90["session_id"]
        key = (sid, int(r90["turn_number"]))
        if sid in regen_responses:
            response = regen_responses[sid]
            n_regen += 1
        else:
            response = r84c_by_key[key]["predicted_response"]
            n_reused += 1
        final_items.append(
            {
                "session_id": sid,
                "turn_number": int(r90["turn_number"]),
                "predicted_track_ids": r90["predicted_track_ids"],
                "predicted_response": response,
            }
        )

    payload = json.dumps(final_items, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    sha = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()
    print(f"\n  Final assembly: {n_regen} regen responses + {n_reused} reused R84c")
    print(f"  Wrote {OUT_ZIP.relative_to(REPO)} ({OUT_ZIP.stat().st_size} bytes)")
    print(f"  sha256: {sha}")

    metadata = {
        "experiment": "R90 Blind-A candidate",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "model": MODEL_ID,
        "n_cases": len(final_items),
        "n_responses_regenerated": n_regen,
        "n_responses_reused_from_r84c": n_reused,
        "tracks_from": "R90 5-fold ensemble + production R90-LR, routed by R54c margin < 0.25 OR >= 2.0",
        "responses_changed_when": "R90 top-1 != R84c top-1",
        "audit": {
            "top1_churn_per_80": audit.get("top1_churn_per_80"),
            "top20_overlap_mean": audit.get("top20_overlap_mean"),
            "n_routed_r90": audit.get("n_routed_r90"),
            "n_routed_r54": audit.get("n_routed_r54"),
            "sessions_changed_6plus": audit.get("sessions_changed_6plus"),
            "audit_passes": audit.get("audit_passes"),
            "route_low": audit.get("route_low"),
            "route_high": audit.get("route_high"),
        },
        "submission_sha256": sha,
        "submission_path": str(OUT_ZIP.relative_to(REPO)),
    }
    with open(OUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata -> {OUT_METADATA.relative_to(REPO)}")

    md = [
        "# R90 Blind-A Candidate",
        "",
        f"HEAD: `{metadata['head_sha'][:10]}`",
        f"Submission: `{OUT_ZIP.name}` ({OUT_ZIP.stat().st_size} bytes)",
        f"sha256: `{sha}`",
        "",
        "## Composition",
        "",
        f"- 80 cases total",
        f"- Tracks from R90 5-fold ensemble + production R90-LR",
        f"- Routing: R54c margin `<0.25` or `>=2.0` -> R90, else R54c",
        f"- Responses: **{n_regen} regenerated** (changed top-1 vs R84c), "
        f"**{n_reused} reused from R84c**",
        "",
        "## Audit",
        "",
        f"- Top-1 churn: {audit.get('top1_churn_per_80')}/80",
        f"- Top-20 overlap mean: {audit.get('top20_overlap_mean'):.2f}/20",
        f"- Route counts: R90={audit.get('n_routed_r90')}, R54={audit.get('n_routed_r54')}",
        f"- Sessions with >=6 changed tracks: {audit.get('sessions_changed_6plus')}/80",
        f"- Packaging gate: {'PASS' if audit.get('audit_passes') else 'FAIL'}",
        "",
        "## Regenerated sessions",
        "",
        "| session_id | turn | new top-1 | routed | margin | words | issues |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in regen_rows:
        md.append(
            f"| `{row['session_id'][:12]}` | {row['turn_number']} | "
            f"`{row['new_top1'][:12]}` | "
            f"{'R90' if row['_routed_r90'] else 'R54'} | "
            f"{row['_r54_margin']:.3f} | {row['n_words']} | "
            f"{','.join(row['issues_after_regen']) or '-'} |"
        )
    OUT_DIFF_MD.write_text("\n".join(md) + "\n")
    print(f"  Diff report -> {OUT_DIFF_MD.relative_to(REPO)}")
    print(f"\n{ts()} Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

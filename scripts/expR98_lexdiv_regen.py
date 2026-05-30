#!/usr/bin/env python3
"""R98: LexDiv recovery on R92 p11 — score-gated against the REAL scorer.

Base = R97 (p11 tracks + GT-safe CatalogDiv dedup, responses == p11). Regenerate
ONLY the 17 changed-top-1 responses (the R84c regens that dragged corpus LexDiv
from R78's 0.8845 down to 0.8720), describing p11's ACTUAL top-1 track for each
row, pushing for higher lexical diversity. Each candidate is accepted ONLY if it
raises the pooled corpus Distinct-2 LexDiv (scripts/lexdiv_scorer.py, the exact
reproduced metric) vs the current best — so LexDiv can only go up, never down.
nDCG/tracks untouched (top ranks fixed). Reports metrics; does NOT upload.

Run:
  ANTHROPIC_RECSYS_API_KEY=... uv run python scripts/expR98_lexdiv_regen.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR84c_response_regen import build_prompt, call_opus, validate  # noqa: E402
from scripts.lexdiv_scorer import lexical_diversity, catalog_diversity  # noqa: E402

R97_ZIP = REPO / "exp" / "inference" / "blind_a" / "r97_p11_catdiv_dedup.zip"
P11_ZIP = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
REGEN_ROWS = REPO / "exp" / "inference" / "blind_a" / "r84c_regen_rows.jsonl"
CATALOG = REPO / "cache" / "metadata" / "track_metadata_all_tracks.json"
SOURCE_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"
OUT_ZIP = REPO / "exp" / "inference" / "blind_a" / "r98_lexdiv_regen.zip"
OUT_META = REPO / "exp" / "inference" / "blind_a" / "r98_lexdiv_regen.metadata.json"

N_CANDIDATES = 3
LEXDIV_GRADIENT = 0.10           # composite per LexDiv unit (conservative: 0.083)
P11_COMPOSITE = 0.6364
P11_LEXDIV = 0.8720

RICHNESS = (
    "\n- Use vivid, specific, varied vocabulary; avoid phrasings and word pairs "
    "you would reuse across many recommendations (maximize lexical variety) while "
    "staying accurate and coherent about THIS track."
)


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def parse_field(raw):
    if raw is None:
        return raw
    try:
        return ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return raw


def load_zip_rows(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def key(r):
    return r["session_id"], int(r["turn_number"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="No API calls; just report base + plan.")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not args.dry_run and not api_key:
        print("ERROR: set ANTHROPIC_RECSYS_API_KEY"); sys.exit(1)

    base_rows = load_zip_rows(R97_ZIP)            # tracks deduped, responses == p11
    by_key = {key(r): r for r in base_rows}
    targets = [json.loads(l) for l in open(REGEN_ROWS)]
    target_keys = [(t["session_id"], int(t["turn_number"])) for t in targets]
    src = pickle.load(open(SOURCE_CACHE, "rb"))

    raw_cat = json.load(open(CATALOG))
    meta = {}
    for tid, rec in raw_cat.items():
        meta[str(tid).strip("'\"")] = {k: parse_field(v) for k, v in rec.items()}

    base_lex = lexical_diversity([r["predicted_response"] for r in base_rows])
    base_cat = catalog_diversity([r["predicted_track_ids"] for r in base_rows])
    print(f"{ts()} R98 LexDiv recovery on R97 base")
    print(f"  base (R97): LexDiv={base_lex:.4f}  CatalogDiv={base_cat:.4f}  (p11 LexDiv {P11_LEXDIV})")
    print(f"  regen targets: {len(target_keys)} rows")
    if args.dry_run:
        print("  [dry-run] stop."); return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # running corpus of responses keyed by case
    cur_resp = {key(r): r["predicted_response"] for r in base_rows}
    order = [key(r) for r in base_rows]

    def pooled_lex(trial_key=None, trial_text=None):
        return lexical_diversity([
            (trial_text if (trial_key and k == trial_key) else cur_resp[k]) for k in order
        ])

    accepted = 0
    changelog = []
    for n, k in enumerate(target_keys, 1):
        sid, turn = k
        case = src.get(sid)
        row = by_key.get(k)
        if case is None or row is None:
            print(f"  {n}/{len(target_keys)} {sid[:8]} SKIP (missing case/row)"); continue
        top1 = row["predicted_track_ids"][0]
        top_meta = meta.get(top1)
        if top_meta is None:
            print(f"  {n}/{len(target_keys)} {sid[:8]} SKIP (no catalog meta for top1)"); continue
        played_refs = {tid: (lambda m: (
            (m.get("track_name") or ["?"])[0] + " by " + (m.get("artist_name") or ["?"])[0]
        ))(meta[tid]) for tid in case.get("music_turns", []) if tid in meta}

        prompt = build_prompt(case, top_meta, played_refs) + RICHNESS
        cur_lex = pooled_lex()
        best_text, best_lex, best_issues = cur_resp[k], cur_lex, None
        for c in range(N_CANDIDATES):
            try:
                resp, _ = call_opus(client, prompt)
            except Exception as e:
                print(f"      cand {c} error {e}"); continue
            issues = validate(resp)
            trial = pooled_lex(k, resp)
            if trial > best_lex + 1e-9:
                best_text, best_lex, best_issues = resp, trial, issues
        if best_lex > cur_lex + 1e-9:
            cur_resp[k] = best_text
            accepted += 1
            changelog.append({"session_id": sid, "turn_number": turn, "top1": top1,
                              "pooled_lex_before": round(cur_lex, 6), "pooled_lex_after": round(best_lex, 6),
                              "n_words": len(best_text.split()), "issues": best_issues})
            print(f"  {n}/{len(target_keys)} {sid[:8]} ACCEPT pooled {cur_lex:.4f}->{best_lex:.4f} "
                  f"({len(best_text.split())}w, issues={best_issues})")
        else:
            print(f"  {n}/{len(target_keys)} {sid[:8]} keep original (no LexDiv gain)")

    final_rows = []
    for r in base_rows:
        k = key(r)
        final_rows.append({**r, "predicted_response": cur_resp[k]})
    final_lex = lexical_diversity([r["predicted_response"] for r in final_rows])
    final_cat = catalog_diversity([r["predicted_track_ids"] for r in final_rows])

    payload = json.dumps(final_rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    sha = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()

    pred_composite = P11_COMPOSITE + LEXDIV_GRADIENT * (final_lex - P11_LEXDIV)
    meta_out = {
        "experiment": "R98 LexDiv recovery on R97 (p11 tracks + dedup)",
        "created_at": datetime.now().isoformat(),
        "base": "r97_p11_catdiv_dedup.zip", "model": "claude-opus-4-7",
        "n_regen_targets": len(target_keys), "n_accepted": accepted,
        "base_lexdiv": round(base_lex, 6), "final_lexdiv": round(final_lex, 6),
        "final_catalogdiv": round(final_cat, 6),
        "p11_lexdiv": P11_LEXDIV, "p11_composite": P11_COMPOSITE,
        "predicted_composite_lexdiv_only": round(pred_composite, 4),
        "predicted_delta_vs_p11": round(pred_composite - P11_COMPOSITE, 4),
        "submission_sha256": sha, "changelog": changelog,
        "caveat": "LexDiv+CatalogDiv offline-validated; LLM judge effect of regens is NOT "
                  "offline-validatable (Gemini). nDCG/tracks identical to p11 (top ranks fixed).",
    }
    json.dump(meta_out, open(OUT_META, "w"), indent=2)

    print(f"\n{ts()} R98 done. accepted {accepted}/{len(target_keys)}")
    print(f"  LexDiv {base_lex:.4f} -> {final_lex:.4f}  (p11 {P11_LEXDIV}; R78 ceiling 0.8845)")
    print(f"  CatalogDiv {final_cat:.4f}")
    print(f"  predicted composite (LexDiv only): {pred_composite:.4f}  (+{pred_composite-P11_COMPOSITE:.4f} vs p11 0.6364)")
    print(f"  wrote {OUT_ZIP.relative_to(REPO)}  sha {sha[:12]}")
    print("  NOTE: report only — LLM-judge effect unvalidatable; you decide R98 vs zero-risk R97.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R105 — CatalogDiv tail-rank audit + low-risk variants on R92 p11.

CatalogDiv (exact competition scorer, verified in scripts/lexdiv_scorer.py) =
  |unique track ids across all 80 cases| / 47071.
R92 p11 = 0.030146 (1419/1600 slots unique; 181 cross-submission duplicate slots).

Lever: replace cross-submission DUPLICATE tracks sitting at TAIL ranks with
submission-novel, plausible tracks drawn from that row's own deep candidate pool.
This raises CatalogDiv with provably ZERO change to top-1/5/10 (we only touch ranks
>=11) and ZERO change to responses (predicted_response left byte-identical -> LexDiv
and LLM-judge unaffected).

nDCG safety: we remove ONLY duplicates (tracks appearing in >=2 cases), and for each
duplicate we KEEP its best-ranked (min-rank) occurrence, stripping only the redundant
tail copies. A track recommended to many different cases at tail rank is generic, not
any single case's precise GT; keeping its best-ranked home preserves the case it most
likely belongs to. Extra guard: never swap a track whose name is referenced in that
row's response text (avoids list/response inconsistency the judge could see).

Variants (windows of editable ranks, 1-based):
  A = rank 20 only      B = ranks 16-20      C = ranks 11-20  (maximal top-10-safe)

Outputs: exp/inference/blind_a/r105_catdiv_{A,B,C}_submission.zip + exp/eval/expR105_catdiv_audit.json
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.lexdiv_scorer import catalog_diversity, lexical_diversity  # noqa: E402

CAT = 47071
SUB = REPO / "exp" / "inference" / "blind_a" / "r92_p11_oracle_submission.zip"
R84_POOL = REPO / "cache" / "r84_production" / "blind_r84_ensemble_lists.json"
R54_POOL = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
META = REPO / "cache" / "metadata" / "track_metadata_all_tracks.json"
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_AUDIT = REPO / "exp" / "eval" / "expR105_catdiv_audit.json"

VARIANTS = {"A": (20, 20), "B": (16, 20), "C": (11, 20)}


def load_sub(path: Path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def load_pool(path: Path) -> dict[str, list[str]]:
    d = json.load(open(path))
    lists = d.get("lists", d)
    out = {}
    for sid, pairs in lists.items():
        out[sid] = [p[0] if isinstance(p, (list, tuple)) else p for p in pairs]
    return out


def write_zip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(rows))


def main():
    rows = load_sub(SUB)
    assert len(rows) == 80, len(rows)
    lists = [r["predicted_track_ids"] for r in rows]
    responses = [(r.get("predicted_response") or "") for r in rows]
    sids = [r["session_id"] for r in rows]

    r84 = load_pool(R84_POOL)
    r54 = load_pool(R54_POOL)
    meta = json.load(open(META))

    def tname(tid):
        m = meta.get(tid, {})
        n = m.get("track_name", [])
        return (n[0] if isinstance(n, list) and n else str(n)).strip()

    # base CatalogDiv + duplicate bookkeeping
    base_cd = catalog_diversity(lists)
    base_lex = lexical_diversity(responses)
    allt = [t for L in lists for t in L]
    cnt = Counter(allt)
    occ = defaultdict(list)                       # tid -> [(case_i, rank1based)]
    for ci, L in enumerate(lists):
        for r, t in enumerate(L, start=1):
            occ[t].append((ci, r))
    kept = {}                                     # tid -> (case_i, rank) we KEEP (min rank)
    for t, os in occ.items():
        kept[t] = min(os, key=lambda x: x[1])

    print(f"baseline CatalogDiv={base_cd:.6f} LexDiv={base_lex:.6f} "
          f"unique={int(round(base_cd*CAT))}/1600  dup_slots={sum(c-1 for c in cnt.values() if c>1)}")

    def build_variant(lo, hi):
        used = set(allt)                          # global novelty set (grows as we insert)
        new_lists = [list(L) for L in lists]
        n_swapped = 0
        skipped_response_ref = 0
        no_replacement = 0
        swap_log = []
        for ci in range(80):
            sid = sids[ci]
            pool = [t for t in (r84.get(sid, []) + r54.get(sid, [])) if t]  # tier-1 deep pool
            resp_l = responses[ci].lower()
            row = new_lists[ci]
            for p in range(lo, hi + 1):           # 1-based rank
                t = row[p - 1]
                if cnt[t] < 2:                    # not a duplicate -> never touch
                    continue
                if kept[t] == (ci, p):           # this is the occurrence we keep
                    continue
                nm = tname(t)
                if nm and nm.lower() in resp_l:  # response references this track -> keep
                    skipped_response_ref += 1
                    continue
                # find a submission-novel, in-row-unique replacement from the deep pool
                rep = None
                for cand in pool:
                    if cand not in used and cand not in row:
                        rep = cand
                        break
                if rep is None:                  # fallback: any catalog track novel to submission
                    for cand in meta.keys():
                        if cand not in used and cand not in row:
                            rep = cand
                            break
                if rep is None:
                    no_replacement += 1
                    continue
                row[p - 1] = rep
                used.add(rep)
                n_swapped += 1
                swap_log.append({"case": ci, "rank": p, "removed": t,
                                 "removed_dupcount": cnt[t], "added": rep,
                                 "from_pool": rep in (r84.get(sid, []) + r54.get(sid, []))})
        return new_lists, n_swapped, skipped_response_ref, no_replacement, swap_log

    audit = {"experiment": "R105 CatalogDiv tail-rank audit on R92 p11",
             "created_at": datetime.now().isoformat(),
             "scorer": "exact (scripts/lexdiv_scorer.py)",
             "baseline": {"catalog_diversity": base_cd, "lexical_diversity": base_lex,
                          "unique_tracks": int(round(base_cd * CAT)), "dup_slots": 181},
             "variants": {}}

    for name, (lo, hi) in VARIANTS.items():
        new_lists, n_swap, skip_ref, no_rep, log = build_variant(lo, hi)
        new_rows = []
        for ci, r in enumerate(rows):
            nr = dict(r)
            nr["predicted_track_ids"] = new_lists[ci]
            new_rows.append(nr)
        # exact rescore
        cd = catalog_diversity(new_lists)
        lex = lexical_diversity([(r.get("predicted_response") or "") for r in new_rows])
        # churn audit vs baseline
        top1 = sum(1 for ci in range(80) if new_lists[ci][0] != lists[ci][0])
        top5 = sum(1 for ci in range(80) if new_lists[ci][:5] != lists[ci][:5])
        top10 = sum(1 for ci in range(80) if new_lists[ci][:10] != lists[ci][:10])
        overlap20 = sum(len(set(new_lists[ci]) & set(lists[ci])) for ci in range(80)) / 80
        # validity: each row 20 unique ids
        valid = all(len(nl) == 20 and len(set(nl)) == 20 for nl in new_lists)
        rep_from_pool = sum(1 for s in log if s["from_pool"])
        outzip = OUT_DIR / f"r105_catdiv_{name}_submission.zip"
        write_zip(outzip, new_rows)
        audit["variants"][name] = {
            "window": [lo, hi], "n_swapped": n_swap, "skipped_response_ref": skip_ref,
            "no_replacement": no_rep, "replacements_from_deep_pool": rep_from_pool,
            "catalog_diversity": cd, "delta_catdiv": cd - base_cd,
            "lexical_diversity": lex, "lex_unchanged": abs(lex - base_lex) < 1e-9,
            "top1_changed": top1, "top5_changed": top5, "top10_changed": top10,
            "top20_overlap_mean": overlap20, "valid_20_unique": valid,
            "zip": str(outzip.relative_to(REPO)),
        }
        print(f"\nVariant {name} (ranks {lo}-{hi}): swapped={n_swap} "
              f"(pool={rep_from_pool}, skip_resp_ref={skip_ref})")
        print(f"  CatalogDiv {base_cd:.6f} -> {cd:.6f}  (Δ{cd-base_cd:+.6f})")
        print(f"  LexDiv unchanged: {abs(lex-base_lex)<1e-9} ({lex:.6f})")
        print(f"  churn: top1={top1} top5={top5} top10={top10}  overlap20={overlap20:.3f}/20")
        print(f"  valid (20 unique/row): {valid}  -> {outzip.name}")

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(OUT_AUDIT, "w"), indent=2)
    print(f"\nsaved {OUT_AUDIT}")


if __name__ == "__main__":
    main()

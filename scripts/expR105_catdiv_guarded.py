#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R105 — GT-RISK-GUARDED CatalogDiv variant 'Cg' on the blind R92 p11 submission.

Variant C edits ranks 11-20, swapping every cross-submission DUPLICATE tail track
(count>=2, non-min-rank occurrence) for a submission-novel pool track: 112 swaps,
+0.002379 CatalogDiv, but dev-estimated p_gt ~0.6% => ~0.68 GTs clipped on blind.

Guard 'Cg' (combined, from dev simulation expR105_catdiv_devsim.py):
  (a) dup-count D: only remove tracks appearing in >= D distinct submission lists
      (more generic => less likely to be any single row's precise GT).
  (b) pool-rank PR: only remove a tail track whose BEST rank in this row's own deep
      retrieval pool (R84 ensemble 300 + R54 300) is > PR, or which is absent from
      both pools (= retrieval filler, not a near-miss GT the system scored high).
  + inherited: never swap a track whose name is referenced in this row's response.

Pool-rank is the lever that actually moves dev p_gt (dup-count is non-discriminating
on 8000 dev lists where everything is a cross-case dup). We keep dup-count as a
belt-and-suspenders generality filter and pool-rank as the primary GT-risk guard.

Output: exp/inference/blind_a/r105_catdiv_Cg_submission.zip + exp/eval/expR105_catdiv_guarded.json
"""
from __future__ import annotations

import json
import math
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
OUT_AUDIT = REPO / "exp" / "eval" / "expR105_catdiv_guarded.json"
DEVSIM = REPO / "exp" / "eval" / "expR105_catdiv_devsim.json"

EDIT_LO, EDIT_HI = 11, 20

# Chosen guard (set after gain/risk comparison below). p_gt_dev keyed by (D, PR).
GUARDS = [
    {"name": "C_baseline", "D": 2, "PR": None},   # reproduce variant C for control
    {"name": "Cg_D2_PR50", "D": 2, "PR": 50},
    {"name": "Cg_D3_PR50", "D": 3, "PR": 50},
    {"name": "Cg_D3_PR30", "D": 3, "PR": 30},
]
CHOSEN = "Cg_D2_PR50"  # primary deliverable (see report)


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
    devsim = json.load(open(DEVSIM)) if DEVSIM.exists() else {}
    dev_pgt = devsim.get("p_gt_guards", {})

    def tname(tid):
        m = meta.get(tid, {})
        n = m.get("track_name", [])
        return (n[0] if isinstance(n, list) and n else str(n)).strip()

    # best rank of track t in row sid's deep pools (None if absent from both)
    r84_rank = {sid: {t: i + 1 for i, t in enumerate(L)} for sid, L in r84.items()}
    r54_rank = {sid: {t: i + 1 for i, t in enumerate(L)} for sid, L in r54.items()}

    def best_pool_rank(sid, t):
        ranks = []
        rk = r84_rank.get(sid, {}).get(t)
        if rk:
            ranks.append(rk)
        rk = r54_rank.get(sid, {}).get(t)
        if rk:
            ranks.append(rk)
        return min(ranks) if ranks else None

    base_cd = catalog_diversity(lists)
    base_lex = lexical_diversity(responses)
    allt = [t for L in lists for t in L]
    cnt = Counter(allt)
    occ = defaultdict(list)
    for ci, L in enumerate(lists):
        for r, t in enumerate(L, start=1):
            occ[t].append((ci, r))
    kept = {t: min(os, key=lambda x: x[1]) for t, os in occ.items()}

    print(f"baseline CatalogDiv={base_cd:.6f} LexDiv={base_lex:.6f} "
          f"unique={int(round(base_cd*CAT))}/1600  "
          f"dup_slots={sum(c-1 for c in cnt.values() if c>1)}")

    def build_variant(D, PR):
        used = set(allt)
        new_lists = [list(L) for L in lists]
        n_swapped = 0
        skipped_response_ref = 0
        skipped_pool_high = 0      # kept because retriever ranked it high (<=PR)
        skipped_not_dup = 0        # kept because count < D
        no_replacement = 0
        swap_log = []
        for ci in range(80):
            sid = sids[ci]
            pool = [t for t in (r84.get(sid, []) + r54.get(sid, [])) if t]
            resp_l = responses[ci].lower()
            row = new_lists[ci]
            for p in range(EDIT_LO, EDIT_HI + 1):
                t = row[p - 1]
                if cnt[t] < D:                       # dup-count guard (a)
                    if cnt[t] >= 2:
                        skipped_not_dup += 1
                    continue
                if kept[t] == (ci, p):               # keep best-ranked occurrence
                    continue
                if PR is not None:                   # pool-rank guard (b)
                    bpr = best_pool_rank(sid, t)
                    if bpr is not None and bpr <= PR:  # retriever scored it high -> keep
                        skipped_pool_high += 1
                        continue
                nm = tname(t)
                if nm and nm.lower() in resp_l:      # response-referenced -> keep
                    skipped_response_ref += 1
                    continue
                rep = None
                for cand in pool:                    # submission-novel pool replacement
                    if cand not in used and cand not in row:
                        rep = cand
                        break
                if rep is None:
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
                swap_log.append({
                    "case": ci, "rank": p, "removed": t, "removed_dupcount": cnt[t],
                    "removed_best_pool_rank": best_pool_rank(sid, t),
                    "added": rep,
                    "from_pool": rep in (r84.get(sid, []) + r54.get(sid, [])),
                })
        return (new_lists, n_swapped, skipped_response_ref, skipped_pool_high,
                skipped_not_dup, no_replacement, swap_log)

    audit = {
        "experiment": "R105 GT-risk-guarded CatalogDiv (Cg) on R92 p11",
        "created_at": datetime.now().isoformat(),
        "scorer": "exact (scripts/lexdiv_scorer.py)",
        "baseline": {"catalog_diversity": base_cd, "lexical_diversity": base_lex,
                     "unique_tracks": int(round(base_cd * CAT))},
        "dev_p_gt_guards": dev_pgt,
        "chosen": CHOSEN,
        "variants": {},
    }

    for g in GUARDS:
        name, D, PR = g["name"], g["D"], g["PR"]
        (new_lists, n_swap, skip_ref, skip_pool, skip_notdup, no_rep,
         log) = build_variant(D, PR)
        new_rows = []
        for ci, r in enumerate(rows):
            nr = dict(r)
            nr["predicted_track_ids"] = new_lists[ci]
            new_rows.append(nr)
        cd = catalog_diversity(new_lists)
        lex = lexical_diversity([(r.get("predicted_response") or "") for r in new_rows])
        top1 = sum(1 for ci in range(80) if new_lists[ci][0] != lists[ci][0])
        top10 = sum(1 for ci in range(80) if new_lists[ci][:10] != lists[ci][:10])
        overlap20 = sum(len(set(new_lists[ci]) & set(lists[ci])) for ci in range(80)) / 80
        valid = all(len(nl) == 20 and len(set(nl)) == 20 for nl in new_lists)
        rep_from_pool = sum(1 for s in log if s["from_pool"])

        # dev p_gt + expected blind GTs clipped under this guard
        pr_key = (f"combined_D{D}_PR{PR}" if (PR is not None and D >= 3) else
                  (f"poolrank_PR{PR}" if PR is not None else f"dupcount_D{D}"))
        pgt = dev_pgt.get(pr_key, {}).get("p_gt")
        exp_clipped = (pgt * n_swap) if pgt is not None else None
        exp_ndcg_loss_ub = (-exp_clipped / 80.0) if exp_clipped is not None else None
        # Realistic rank-weighted nDCG@20 loss: a GT clipped from slot at rank p (11-20)
        # loses its nDCG@20 contribution 1/log2(p+1). Expected loss over 80 cases =
        # p_gt * sum_over_removed_slots(1/log2(rank+1)) / 80.
        swap_ranks = [s["rank"] for s in log]
        exp_ndcg_loss = (
            -pgt * sum(1.0 / math.log2(p + 1) for p in swap_ranks) / 80.0
            if pgt is not None else None
        )

        is_chosen = (name == CHOSEN)
        outzip = OUT_DIR / ("r105_catdiv_Cg_submission.zip" if is_chosen
                            else f"r105_catdiv_{name}_submission.zip")
        write_zip(outzip, new_rows)
        audit["variants"][name] = {
            "guard": {"dup_count_ge": D, "pool_rank_gt": PR, "window": [EDIT_LO, EDIT_HI]},
            "n_swapped": n_swap, "skipped_response_ref": skip_ref,
            "skipped_pool_rank_high": skip_pool, "skipped_below_dupcount": skip_notdup,
            "no_replacement": no_rep, "replacements_from_deep_pool": rep_from_pool,
            "catalog_diversity": cd, "delta_catdiv": cd - base_cd,
            "lexical_diversity": lex, "lex_unchanged": abs(lex - base_lex) < 1e-9,
            "top1_changed": top1, "top10_changed": top10,
            "top20_overlap_mean": overlap20, "valid_20_unique": valid,
            "dev_p_gt": pgt, "dev_p_gt_key": pr_key,
            "expected_blind_gts_clipped": exp_clipped,
            "expected_ndcg20_loss_realistic": exp_ndcg_loss,
            "expected_ndcg20_loss_upper": exp_ndcg_loss_ub,
            "zip": str(outzip.relative_to(REPO)),
        }
        tag = " <-- CHOSEN (Cg)" if is_chosen else ""
        print(f"\n{name}  (D>={D}, PR>{PR}){tag}")
        print(f"  swaps={n_swap}  (pool={rep_from_pool}, skip[resp_ref={skip_ref}, "
              f"pool_high={skip_pool}, below_D={skip_notdup}])")
        print(f"  CatalogDiv {base_cd:.6f} -> {cd:.6f}  (Δ{cd-base_cd:+.6f})")
        print(f"  LexDiv unchanged: {abs(lex-base_lex)<1e-9}  top1={top1} top10={top10} "
              f"overlap20={overlap20:.2f}/20  valid={valid}")
        if pgt is not None:
            print(f"  dev p_gt={pgt*100:.4f}%  -> exp blind GTs clipped={exp_clipped:.3f}"
                  f"  (nDCG@20 loss: realistic {exp_ndcg_loss:+.5f}, "
                  f"upper {exp_ndcg_loss_ub:+.5f})")
        print(f"  -> {outzip.name}")

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(OUT_AUDIT, "w"), indent=2)
    print(f"\nsaved {OUT_AUDIT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""R432s — targeted-safe reorder subset (one probe).

Hypothesis: the reorders-only LLM drop (4.90->4.85, deterministic) is caused by the 2 visibly
incoherent reorders (ab8bc8fa: P!nk into a K-Pop query; 4831a3b4: eclectic Noisia/Cage/post-rock
mix), not by all goal-aware reorders. This drops those 2 and keeps the other 6 reorder-only patches.

Base: R106 A-clean. For the 6 KEEP sids, use the goal-C reordered tracks (from reorders-only zip);
everything else (incl. the 2 dropped) stays exactly R106. Constraints: 0 top-1 changes, 0 response
changes, responses byte-identical, row order identical to R106.
"""
import json, zipfile, hashlib, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from lexdiv_scorer import lexical_diversity, catalog_diversity  # noqa: E402

BASE = REPO / "exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip"
REORDERS = REPO / "exp/inference/blind_a/r432_goalonly_reorders_only_submission.zip"
OUT = REPO / "exp/inference/blind_a/r432s_targeted_subset_submission.zip"

KEEP = {"25cc9533", "31bf71ab", "f321d88d", "fc6ba76a", "db11666f", "3afb9f67"}  # 6
DROP = {"ab8bc8fa", "4831a3b4"}  # 2 high-risk -> revert to R106


def load(p):
    z = zipfile.ZipFile(p); fn = [x for x in z.namelist() if x.endswith(".json")][0]
    return {e["session_id"]: e for e in json.loads(z.read(fn))}, json.loads(z.read(fn))


def main():
    base, base_list = load(BASE)
    ro, _ = load(REORDERS)
    final = []
    track_changes = top1_changes = resp_changes = 0
    overlaps, changed_sids = [], []
    for e in base_list:  # iterate in R106 order -> row order identical
        s = e["session_id"]
        if s[:8] in KEEP:
            tracks = ro[s]["predicted_track_ids"]
        else:  # untouched + the 2 DROP -> exactly R106
            tracks = e["predicted_track_ids"]
        resp = e["predicted_response"]  # always R106, byte-identical
        if tracks != e["predicted_track_ids"]:
            track_changes += 1
            changed_sids.append(s[:8])
            overlaps.append(len(set(tracks[:20]) & set(e["predicted_track_ids"][:20])))
            if tracks[0] != e["predicted_track_ids"][0]:
                top1_changes += 1
        if resp != e["predicted_response"]:
            resp_changes += 1
        final.append({"session_id": s, "turn_number": e["turn_number"],
                      "predicted_track_ids": tracks, "predicted_response": resp})

    # checks
    checks = {
        "80_rows": len(final) == 80,
        "20_unique_each": all(len(x["predicted_track_ids"]) == 20 and len(set(x["predicted_track_ids"])) == 20 for x in final),
        "track_changes_eq_6": track_changes == 6,
        "top1_changes_eq_0": top1_changes == 0,
        "resp_changes_eq_0": resp_changes == 0,
        "responses_byte_identical_R106": all(final[i]["predicted_response"] == base_list[i]["predicted_response"] for i in range(80)),
        "row_order_identical_R106": [x["session_id"] for x in final] == [x["session_id"] for x in base_list],
        "dropped_reverted_to_R106": all(
            next(x for x in final if x["session_id"][:8] == d)["predicted_track_ids"]
            == base[next(s for s in base if s[:8] == d)]["predicted_track_ids"] for d in DROP),
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(final))
    lexdiv = lexical_diversity([x["predicted_response"] for x in final])
    catdiv = catalog_diversity([x["predicted_track_ids"] for x in final])
    base_cat = catalog_diversity([x["predicted_track_ids"] for x in base_list])
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

    print("=== R432s targeted-safe reorder subset (6 keep / 2 dropped) ===")
    print(f"  changed cases: {track_changes}  -> {changed_sids}")
    print(f"  top1_changes={top1_changes}  resp_changes={resp_changes}")
    print(f"  overlap@20 on changed cases: {overlaps}  (mean {sum(overlaps)/len(overlaps):.2f})")
    print("  --- checks ---")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"  LexDiv: {lexdiv:.6f}  (R106 0.885861)")
    print(f"  CatDiv: {catdiv:.6f}  (R106 {base_cat:.6f})")
    print(f"  zip: {OUT.relative_to(REPO)}")
    print(f"  sha256: {sha}")
    print(f"\n  ALL CHECKS {'PASS -> ready (HELD for upload)' if all(checks.values()) else 'FAIL'}")
    json.dump({"checks": checks, "track_changes": track_changes, "changed_sids": changed_sids,
               "overlap20": overlaps, "lexdiv": lexdiv, "catdiv": catdiv, "sha256": sha,
               "keep": sorted(KEEP), "drop": sorted(DROP), "all_pass": all(checks.values())},
              open(REPO / "exp/eval/expR432s_targeted_subset_audit.json", "w"), indent=2)


if __name__ == "__main__":
    main()

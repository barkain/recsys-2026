#!/usr/bin/env python3
"""R432 goal-only FINAL conservative candidate (Option 1).

From R106 A-clean base:
  - apply goal-C track lists to the 8 reorder-only patches + 2 valid top-1 swaps (10 cases)
  - REVERT the 3 query-conflicting top-1 swaps (78487c90 Robert Johnson, 77faef85 'beyond Santa',
    7905bb71 happy->melancholic) -> keep R106
  - repair responses for ONLY the 2 valid swaps (60a6fd69 Camp Lo, c0d4e758 Jim Guthrie)
Runs all preflight checks; writes final zip HELD for upload.
"""
import json, zipfile, hashlib, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from lexdiv_scorer import lexical_diversity  # noqa: E402

BASE = REPO / "exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip"
PROV = REPO / "exp/inference/blind_a/r432_goalonly_PROVISIONAL.zip"
OUT = REPO / "exp/inference/blind_a/r432_goalonly_FINAL_submission.zip"

REVERT = {"78487c90", "77faef85", "7905bb71"}        # query-conflicting -> keep R106
SWAP = {"60a6fd69", "c0d4e758"}                        # valid top-1 swaps -> repair response

REPAIRED = {
"60a6fd69": "Where your memory points toward smooth flows and laid-back warmth, Sparkle by Camp Lo lands right in that pocket. The Bronx duo drape unhurried, conversational rhymes over a jazzy, soul-tinged boom-bap groove — dusty drums laced with mellow, horn-flecked loops. A 1997 cut steeped in the East Coast Golden Age sound, it carries exactly the soulful, half-remembered hip-hop feel you're chasing: understated, jazzy, and easy to sink into.",
"c0d4e758": "The Whirling Infinite by Jim Guthrie is the closest match to that moonlit, retro-futuristic memory you're tracing. Drawn from his Sword & Sworcery LP — a soundtrack built around the game's haunting lunar imagery — it's instrumental and atmospheric, layering folk-tinged guitar over warm synth drones and a slow, hypnotic pulse. That fusion of organic and electronic gives it the unique, almost otherworldly vibe you're after, with the moon sitting right at the center of its world.",
}
# (title, artist) that the repaired response MUST contain (new) / MUST NOT contain (old)
CHECK = {
"60a6fd69": {"new": ["Sparkle", "Camp Lo"], "old": ["On and On", "Pete Rock", "C.L. Smooth"]},
"c0d4e758": {"new": ["The Whirling Infinite", "Jim Guthrie"], "old": ["Miami", "Jasper Byrne", "Hotline Miami"]},
}


def load(p):
    z = zipfile.ZipFile(p); fn = [x for x in z.namelist() if x.endswith(".json")][0]
    return {e["session_id"]: e for e in json.loads(z.read(fn))}, json.loads(z.read(fn))


def pre(s): return s[:8]


def main():
    base, base_list = load(BASE)
    prov, _ = load(PROV)
    # patched (goal-C track-changed) cases
    patched = [s for s in prov if prov[s]["predicted_track_ids"] != base[s]["predicted_track_ids"]]
    changed_top1 = [s for s in patched if prov[s]["predicted_track_ids"][0] != base[s]["predicted_track_ids"][0]]
    reorder = [s for s in patched if s not in changed_top1]
    swap_sids = [s for s in patched if pre(s) in SWAP]
    revert_sids = [s for s in patched if pre(s) in REVERT]
    keep_track_sids = set(reorder) | set(swap_sids)   # 8 + 2 = 10

    # build final, preserving original order
    final = []
    track_changes = resp_changes = top1_churn = 0
    for e in base_list:
        s = e["session_id"]
        tracks = prov[s]["predicted_track_ids"] if s in keep_track_sids else e["predicted_track_ids"]
        resp = REPAIRED[pre(s)] if pre(s) in SWAP else e["predicted_response"]
        if tracks != e["predicted_track_ids"]:
            track_changes += 1
            if tracks[0] != e["predicted_track_ids"][0]:
                top1_churn += 1
        if resp != e["predicted_response"]:
            resp_changes += 1
        final.append({"session_id": s, "turn_number": e["turn_number"],
                      "predicted_track_ids": tracks, "predicted_response": resp})

    # ---- checks ----
    checks = {}
    checks["80_rows"] = len(final) == 80
    checks["20_unique_tracks_each"] = all(len(e["predicted_track_ids"]) == 20 and
                                          len(set(e["predicted_track_ids"])) == 20 for e in final)
    checks["track_changes_eq_10"] = track_changes == 10
    checks["top1_churn_eq_2"] = top1_churn == 2
    checks["resp_changes_eq_2"] = resp_changes == 2
    checks["reverts_match_R106"] = all(
        next(e for e in final if e["session_id"] == s)["predicted_track_ids"] == base[s]["predicted_track_ids"]
        and next(e for e in final if e["session_id"] == s)["predicted_response"] == base[s]["predicted_response"]
        for s in revert_sids)
    # response consistency: each repaired names NEW title+artist, not OLD
    cons = {}
    for s in swap_sids:
        r = REPAIRED[pre(s)].lower(); c = CHECK[pre(s)]
        cons[pre(s)] = {
            "has_new": all(x.lower() in r for x in c["new"]),
            "no_old": all(x.lower() not in r for x in c["old"]),
        }
    checks["repaired_name_new_not_old"] = all(v["has_new"] and v["no_old"] for v in cons.values())

    # ---- write + score ----
    pred_json = json.dumps(final)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", pred_json)
    lexdiv = lexical_diversity([e["predicted_response"] for e in final])
    base_lexdiv = lexical_diversity([e["predicted_response"] for e in base_list])
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()

    print("=== R432 goal-only FINAL conservative candidate ===")
    print(f"  patched={len(patched)}  reorder_only={len(reorder)}  swaps_kept={len(swap_sids)}  reverted={len(revert_sids)}")
    print(f"  track_changes={track_changes}  top1_churn={top1_churn}  resp_changes={resp_changes}")
    print("  --- checks ---")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"  response consistency: {json.dumps(cons)}")
    print(f"  LexDiv: candidate={lexdiv:.4f}  (R106 base={base_lexdiv:.4f})")
    print(f"  zip: {OUT.relative_to(REPO)}")
    print(f"  sha256: {sha}")
    all_pass = all(checks.values())
    print(f"\n  ALL CHECKS {'PASS -> submit-worthy (HELD for upload)' if all_pass else 'FAIL -> do not submit'}")
    json.dump({"checks": checks, "consistency": cons, "lexdiv": lexdiv, "base_lexdiv": base_lexdiv,
               "track_changes": track_changes, "top1_churn": top1_churn, "resp_changes": resp_changes,
               "sha256": sha, "all_pass": all_pass,
               "reorder_sids": reorder, "swap_sids": swap_sids, "revert_sids": revert_sids},
              open(REPO / "exp/eval/expR432_goalonly_FINAL_audit.json", "w"), indent=2)


if __name__ == "__main__":
    main()

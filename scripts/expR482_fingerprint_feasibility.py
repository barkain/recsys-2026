#!/usr/bin/env python3
# ruff: noqa: T201
"""R482 — Reconstruction fingerprint feasibility / coverage estimate (Blind-A, provided data only).

HARMLESS FEASIBILITY ONLY (per user's permitted scope): uses ONLY the provided Blind-A
dataset. No external data, no hidden-GT lookup, no candidate generation for upload.

Question it answers: IF source-session reconstruction were permitted AND the session data
existed, how strong is the per-row fingerprint we could match on? The fingerprint = the
tracks the user actually played in the conversation (real session tracks) + demographics +
session_date. A row is only plausibly reconstructable if it carries enough *played tracks*
to uniquely pin a listening session; turn-1 / demographics-only rows are not.

Output: per-row fingerprint strength + an aggregate coverage estimate (what fraction of the
80 Blind-A rows could even in principle be reconstructed). This gates whether the
organizer-clearance path is worth pursuing.
"""
from __future__ import annotations
import json
from collections import Counter


def main():
    from datasets import load_dataset
    d = load_dataset("talkpl-ai/talk_play_data-challenge-blind-a")["test"]
    n = len(d)
    print(f"Blind-A rows: {n}\n")

    rows = []
    for r in d:
        conv = r["conversations"]
        # played tracks = 'music' role turns carry a real track_id (a session/pool track)
        played = [t["content"] for t in conv if t.get("role") == "music"]
        # user turns = the conversation text
        user_turns = [t for t in conv if t.get("role") == "user"]
        prof = r["user_profile"]
        demo_fields = [prof.get(k) for k in ("age", "country_code", "gender",
                                             "preferred_language", "preferred_musical_culture")]
        demo_complete = sum(1 for v in demo_fields if v not in (None, "", "unknown"))
        rows.append({
            "session_id": r["session_id"],
            "session_date": r["session_date"],
            "turns": len(conv),
            "user_turns": len(user_turns),
            "n_played": len(played),          # the fingerprint strength driver
            "demo_complete": demo_complete,   # out of 5
        })

    # fingerprint strength classification
    def strength(x):
        if x["n_played"] >= 3:
            return "STRONG (>=3 played tracks -> likely pins a session)"
        if x["n_played"] in (1, 2):
            return "WEAK (1-2 played tracks -> ambiguous)"
        return "NONE (cold/turn-1: demographics+date only -> not reconstructable)"

    for x in rows:
        x["fp"] = strength(x)
    cnt = Counter(x["fp"].split(" ")[0] for x in rows)
    played_dist = Counter(x["n_played"] for x in rows)
    yr = Counter(x["session_date"][:4] for x in rows)

    print("played-track count per row (fingerprint driver):")
    for k in sorted(played_dist):
        print(f"  {k} played: {played_dist[k]:>2} rows")
    print("\nfingerprint strength (coverage estimate):")
    for k in ("STRONG", "WEAK", "NONE"):
        print(f"  {k:7s}: {cnt.get(k,0):>2}/{n}  ({cnt.get(k,0)/n:.0%})")
    print("\nsession_date years:", dict(sorted(yr.items())))
    strong = cnt.get("STRONG", 0)
    print(f"\nCEILING: at most {strong}/{n} ({strong/n:.0%}) rows could be reconstructed even with "
          f"perfect data+policy (the rest lack enough played-track fingerprint).")
    print("Each reconstructed row that flips a miss->rank-1 = +1/80 = +0.0125 nDCG.")
    print(f"So the WHOLE reconstruction lever caps at ~+{strong/n*0.0125*80/80:.4f}... "
          f"i.e. <= {strong} rows * 0.0125 = +{strong*0.0125:.4f} nDCG IF every strong row is a "
          f"recoverable miss AND uniquely resolves. Realistic yield is a fraction of that.")
    json.dump(rows, open("exp/eval/r482_fingerprint_feasibility.json", "w"), indent=1)
    print("\nwrote exp/eval/r482_fingerprint_feasibility.json")


if __name__ == "__main__":
    main()

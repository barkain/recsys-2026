#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Track B2 Stage-0 prep: build fold-0 h7 mis-ranked probe cases (arc + candidates)
for arc-conditioned LLM-relevance scoring. Dumps .scratch/probe_cases.json."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.exp_goal65_eval import load_dev

META = json.load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))
N_CAND = 60      # candidates per case (keeps the scoring prompt tractable)
N_CASES = 60


def track_str(tid: str) -> str:
    m = META.get(tid, {})
    def g(k):
        v = m.get(k, [])
        return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
    name = g("track_name"); art = g("artist_name"); alb = g("album_name")
    tg = m.get("tag_list", [])
    tags = ", ".join(str(t) for t in tg[:8]) if isinstance(tg, list) else ""
    return f"{name} — {art} | album: {alb} | tags: {tags}"


def short(tid: str) -> str:
    m = META.get(tid, {})
    def g(k):
        v = m.get(k, [])
        return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
    return f"{g('track_name')} by {g('artist_name')}"


def build_arc(case) -> str:
    users = [str(h.get("content", "")) for h in case["history"] if h.get("role") == "user"]
    played = [short(t) for t in case["music_turns"] if t in META]
    s = f'User\'s current request: "{case["user_query"]}".'
    if users[-4:]:
        s += "\nEarlier user turns: " + " | ".join(users[-4:])
    if played:
        s += "\nTracks played so far this session (in order): " + "; ".join(played)
    return s


def main():
    dev = load_dev()
    payload = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))
    cases = payload["cases"]
    gt, fold, npri, lr20 = dev["gt"], dev["fold"], dev["n_prior"], dev["lr_top20"]
    out = []
    for i in range(dev["n"]):
        if fold[i] != 0 or npri[i] != 7:
            continue
        pool = list(dict.fromkeys(dev["r54pool"][i][:100] + dev["r84pool"][i][:100]))
        g = gt[i]
        if g not in pool[:100]:
            continue                       # need GT recoverable in the candidate window
        prod1 = lr20[i][0] if lr20[i] else None
        if prod1 == g or prod1 is None:
            continue                       # only MIS-ranked cases (production top-1 is a false positive)
        cand = pool[:N_CAND]
        if g not in cand:
            cand = cand[:-1] + [g]         # ensure GT present
        if prod1 not in cand:
            cand = cand[:-1] + [prod1]     # ensure the production FP present
        out.append({
            "case_idx": i,
            "arc": build_arc(cases[i]),
            "candidates": [{"id": t, "text": track_str(t)} for t in cand],
            "prod_top1": prod1,            # the false positive to beat (eval only)
            "gt": g,                       # eval only — NOT shown to scorer
        })
        if len(out) >= N_CASES:
            break
    Path(REPO / ".scratch").mkdir(exist_ok=True)
    json.dump(out, open(REPO / ".scratch/probe_cases.json", "w"))
    print(f"prepared {len(out)} fold-0 h7 mis-ranked probe cases -> .scratch/probe_cases.json")
    print(f"  avg candidates/case: {sum(len(c['candidates']) for c in out)/len(out):.0f}")
    print(f"  sample arc:\n{out[0]['arc'][:400]}")
    print(f"  sample candidate: {out[0]['candidates'][0]['text'][:100]}")


if __name__ == "__main__":
    main()

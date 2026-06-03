#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R421 Phase 0 — cross-encoder reranker data prep (conversation -> GT).

Builds the training corpus + dev OOF eval sets for a candidate-conditional reranker that
READS the conversation against each candidate (the signal our text-blind LR lacks). Shared
query builder for train/dev consistency. Hard negatives = same-artist + popular + random.
Outputs to .scratch/r421/ (sync to Drive for the Colab A100 trainer).

  train_examples.jsonl : {query, pos, negs:[...]}     (~121k transitions)
  dev_eval.jsonl       : {case_idx, query, pool:[tids], gt, fold}   (8000 dev OOF cases)
  track_docs.json      : tid -> doc text (catalog)
"""
from __future__ import annotations
import json, pickle, random, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / ".scratch" / "r421"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(0)


def g1(m, k):
    v = m.get(k, [])
    return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")


def track_doc(tid, meta):
    m = meta.get(tid, {})
    tags = m.get("tag_list", [])
    tags = ", ".join(str(t) for t in (tags[:10] if isinstance(tags, list) else []))
    yr = str(g1(m, "release_date"))[:4]
    return f"{g1(m,'track_name')} — {g1(m,'artist_name')} | {g1(m,'album_name')} ({yr}) | {tags}".strip()


def build_query(profile, turns, meta, max_hist=8):
    """turns: list of {role, content}; content of 'music' role is a track_id. Shared train/dev."""
    head = []
    if profile:
        cult = profile.get("preferred_musical_culture", "")
        head.append(f"Listener taste: {cult}." if cult else "")
    lines = []
    for t in turns[-max_hist:]:
        role, c = t.get("role"), str(t.get("content", ""))
        if role == "user" and c:
            lines.append(f"User: {c}")
        elif role == "assistant" and c:
            lines.append(f"Assistant: {c}")
        elif role == "music" and c:
            nm = g1(meta.get(c, {}), "track_name"); ar = g1(meta.get(c, {}), "artist_name")
            if nm:
                lines.append(f"[played: {nm} — {ar}]")
    return " ".join(x for x in head if x) + "\n" + "\n".join(lines)


def main():
    from datasets import load_dataset
    meta = json.load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))
    catalog = set(meta.keys())
    artist_tracks = defaultdict(list)
    for tid in catalog:
        a = g1(meta[tid], "artist_name")
        if a:
            artist_tracks[a].append(tid)

    # ---- train transitions + popularity ----
    print("loading train sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]
    pop = Counter()
    transitions = []  # (profile, prefix_turns, gt)
    for r in train:
        prof = r.get("user_profile", {})
        conv = r["conversations"]
        for j, t in enumerate(conv):
            if t.get("role") == "music":
                gt = t.get("content")
                if gt in catalog:
                    pop[gt] += 1
                    transitions.append((prof, conv[:j], gt))
    pop_top = [t for t, _ in pop.most_common(2000)]
    print(f"  {len(transitions)} train transitions, {len(pop)} distinct played tracks", flush=True)

    # ---- write train examples with hard negatives ----
    n_neg_sa, n_neg_pop, n_neg_rand = 4, 8, 8
    catalog_list = list(catalog)
    with open(OUT / "train_examples.jsonl", "w") as f:
        for prof, prefix, gt in transitions:
            q = build_query(prof, prefix, meta)
            ga = g1(meta[gt], "artist_name")
            negs = set()
            # same-artist confounders
            sa = [t for t in artist_tracks.get(ga, []) if t != gt]
            negs.update(random.sample(sa, min(n_neg_sa, len(sa))))
            # popular confounders
            negs.update(t for t in random.sample(pop_top, min(n_neg_pop * 3, len(pop_top))) if t != gt)
            # random
            while len(negs) < (n_neg_sa + n_neg_pop + n_neg_rand):
                negs.add(random.choice(catalog_list))
            negs.discard(gt)
            f.write(json.dumps({"query": q, "pos": gt, "negs": list(negs)[:20]}) + "\n")
    print(f"  wrote {OUT/'train_examples.jsonl'}", flush=True)

    # ---- dev OOF eval sets (reuse production pools + R12 conversation) ----
    from scripts.exp_goal65_eval import load_dev
    dev = load_dev()
    cases = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))["cases"]
    POOL_K = 200
    with open(OUT / "dev_eval.jsonl", "w") as f:
        for i in range(dev["n"]):
            c = cases[i]
            prof = c.get("user_profile") if isinstance(c.get("user_profile"), dict) else {}
            turns = list(c.get("history", []))
            if c.get("user_query"):
                turns = turns + [{"role": "user", "content": c["user_query"]}]
            q = build_query(prof, turns, meta)
            pool = list(dict.fromkeys((dev["r84pool"][i] or [])[:POOL_K] + (dev["r54pool"][i] or [])[:POOL_K]))
            pool = [t for t in pool if t in catalog][:POOL_K]
            f.write(json.dumps({"case_idx": i, "query": q, "pool": pool,
                                "gt": dev["gt"][i], "fold": dev["fold"][i]}) + "\n")
    print(f"  wrote {OUT/'dev_eval.jsonl'} (pool_k={POOL_K})", flush=True)

    # ---- track docs (only tracks that appear in train negs/pos or dev pools, to keep small) ----
    used = set()
    for line in open(OUT / "train_examples.jsonl"):
        d = json.loads(line); used.add(d["pos"]); used.update(d["negs"])
    for line in open(OUT / "dev_eval.jsonl"):
        d = json.loads(line); used.add(d["gt"]); used.update(d["pool"])
    docs = {t: track_doc(t, meta) for t in used if t in catalog}
    json.dump(docs, open(OUT / "track_docs.json", "w"), ensure_ascii=False)
    print(f"  wrote {OUT/'track_docs.json'} ({len(docs)} tracks)")
    print("\nPhase 0 done. Sync .scratch/r421/ to Drive for the Colab A100 trainer.")


if __name__ == "__main__":
    main()

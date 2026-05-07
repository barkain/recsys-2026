#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S603,S607
"""R26: Structured intent retrieval — extract music intent, generate query variants,
test as new retrieval sources.

Phase 0 (extract): LLM-based structured intent extraction -> cache/r26/intents_dev.json
Phase 1 (retrieve): Build new sources (intent BM25, artist-boost BM25, intent dense),
                     measure standalone hit@K and unique GTs vs R21/V3.
Phase 2 (fuse):     Fuse with existing sources, LambdaRank rerank, measure pool_hit/nDCG.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CACHE_DIR = REPO / "cache" / "r26"
INTENTS_PATH = CACHE_DIR / "intents_dev.json"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_LISTS_PATH = REPO / "cache" / "r21_production" / "dev_r21_lists.json"
V3_POOLS_PATH = REPO / "cache" / "r21_production" / "v3_pools.json"
R21_MODEL_PATH = REPO / "cache" / "r21_production" / "model"
R21_TRACK_IDS_PATH = REPO / "cache" / "r21_production" / "track_ids.json"
R21_TRACK_EMBS_PATH = REPO / "cache" / "r21_production" / "track_embeddings.npy"

EXTRACT_MODEL = "claude-haiku-4-5-20251001"
EXTRACT_MAX_TOKENS = 1024


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# ---------------------------------------------------------------------------
# Phase 0: Structured Intent Extraction
# ---------------------------------------------------------------------------

INTENT_SCHEMA = """\
{
  "summary": "short natural-language music request (1-2 sentences)",
  "positive_artists": ["artist names explicitly desired or referenced"],
  "negative_artists": ["artist names explicitly rejected"],
  "similarity_anchors": ["tracks/artists/albums/soundtracks mentioned as style references"],
  "genres": ["genre and subgenre terms"],
  "moods": ["mood/vibe adjectives"],
  "energy": "low|medium|high|unknown",
  "era": "decade or scene if stated, else empty string",
  "context": "activity or listening situation if stated, else empty string",
  "must_have": ["hard constraints the user insists on"],
  "avoid": ["things the user explicitly rejects"],
  "query_variants": [
    "variant 1: artist/style focused query for dense retrieval",
    "variant 2: mood/genre focused query for dense retrieval",
    "variant 3: keyword-rich query for BM25 search over track metadata"
  ]
}"""

EXTRACT_SYSTEM = f"""\
You are a music recommendation intent parser. Given a conversation between a user and a music assistant, extract the user's current music request as structured JSON.

Focus on the LAST user message and its immediate context. Parse the conversation to identify what the user wants RIGHT NOW, not what they asked earlier.

Output ONLY valid JSON matching this schema:
{INTENT_SCHEMA}

Rules:
- summary should capture the essence of the current request
- positive_artists: only artists the user WANTS to hear. Include artists from similarity anchors only if the user wants more from them
- negative_artists: artists the user explicitly said to AVOID or is tired of
- similarity_anchors: tracks/artists/albums/games/movies referenced as style touchpoints, even if the user doesn't want more of that exact artist
- genres/moods: extract from both explicit mentions and implied context
- must_have/avoid: hard constraints like "must be instrumental", "no vocals", "has to be from the 90s"
- query_variants: generate 3 diverse search queries that a music search engine could use to find relevant tracks. Make them specific and grounded in the conversation, not generic
- If a field has no information, use an empty list [] or empty string ""
- Output raw JSON only, no markdown fences"""


def build_conversation_text(case):
    """Build readable conversation text from a case."""
    parts = []
    for h in case["history"]:
        role = h["role"]
        content = str(h["content"])
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content[:300]}")
        elif role == "music":
            parts.append(f"[Track played: {content}]")
    parts.append(f"User: {case['user_query']}")
    return "\n".join(parts)


def extract_single_intent(conversation_text):
    """Extract structured intent from a conversation using LLM API."""
    from mcrs.utils import call_llm_api
    result = call_llm_api(
        system=EXTRACT_SYSTEM,
        user=conversation_text,
        model=EXTRACT_MODEL,
        max_tokens=EXTRACT_MAX_TOKENS,
    )
    if result is None:
        return None
    text = result.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_one(case_tuple):
    """Worker function for parallel extraction."""
    i, case = case_tuple
    conv_text = build_conversation_text(case)
    intent = extract_single_intent(conv_text)
    if intent is None:
        intent = extract_single_intent(conv_text)
    return i, case, intent


def phase_extract(args):
    """Phase 0: Extract intents for all dev cases (parallel)."""
    print(f"{ts()} PHASE 0: Structured intent extraction (workers={args.workers})", flush=True)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    print(f"  {n} dev cases loaded", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if INTENTS_PATH.exists():
        with open(INTENTS_PATH) as f:
            existing_list = json.load(f)
        for item in existing_list:
            existing[item["session_id"] + "_" + str(item["turn_number"])] = item
        print(f"  {len(existing)} existing intents loaded from cache", flush=True)

    # Separate cached vs todo
    results_by_idx: dict[int, dict] = {}
    todo = []
    for i, case in enumerate(cases):
        key = case["session_id"] + "_" + str(case["turn_number"])
        if key in existing and existing[key].get("intent"):
            results_by_idx[i] = existing[key]
        else:
            todo.append((i, case))

    n_cached = len(results_by_idx)
    print(f"  {n_cached} cached, {len(todo)} to extract", flush=True)

    n_done = 0
    n_failed = 0
    save_lock = __import__("threading").Lock()

    def save_checkpoint():
        ordered = [results_by_idx[i] for i in sorted(results_by_idx)]
        with open(INTENTS_PATH, "w") as f:
            json.dump(ordered, f, indent=1)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, case, intent in pool.map(_extract_one, todo):
            with save_lock:
                if intent is None:
                    n_failed += 1
                    print(f"  FAIL case {i} (session={case['session_id'][:8]})", flush=True)
                results_by_idx[i] = {
                    "session_id": case["session_id"],
                    "turn_number": case["turn_number"],
                    "user_query": case["user_query"],
                    "intent": intent,
                }
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{ts()}   {n_done}/{len(todo)} extracted ({n_failed} failed, {n_cached} cached)",
                          flush=True)
                if n_done % 200 == 0:
                    save_checkpoint()

    save_checkpoint()

    n_valid = sum(1 for r in results_by_idx.values() if r.get("intent"))
    print(f"\n{ts()} DONE: {n_valid}/{n} valid intents ({n_done} new, {n_cached} cached, {n_failed} failed)",
          flush=True)
    print(f"  Saved to {INTENTS_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Retrieval Source Diagnostics
# ---------------------------------------------------------------------------

def build_intent_bm25_query(intent):
    """Q1: Build a BM25 query from intent fields."""
    parts = []
    if intent.get("summary"):
        parts.append(intent["summary"])
    parts.extend(intent.get("genres", []))
    parts.extend(intent.get("moods", []))
    if intent.get("era"):
        parts.append(intent["era"])
    if intent.get("context"):
        parts.append(intent["context"])
    parts.extend(intent.get("positive_artists", []))
    return " ".join(parts)


def build_artist_boost_bm25_query(intent):
    """Q2: BM25 query with artist/genre repetition for term weighting."""
    parts = []
    for artist in intent.get("positive_artists", []):
        parts.extend([artist] * 3)
    for genre in intent.get("genres", []):
        parts.extend([genre] * 2)
    parts.extend(intent.get("moods", []))
    if intent.get("era"):
        parts.append(intent["era"])
    if intent.get("context"):
        parts.append(intent["context"])
    parts.extend(intent.get("similarity_anchors", []))
    return " ".join(parts) if parts else intent.get("summary", "")


def phase_retrieve(args):
    """Phase 1: Build intent-based retrieval sources and evaluate standalone."""
    print(f"{ts()} PHASE 1: Intent retrieval diagnostics", flush=True)

    # Load intents
    if not INTENTS_PATH.exists():
        print("ERROR: No intents cache found. Run --phase extract first.", flush=True)
        sys.exit(1)
    with open(INTENTS_PATH) as f:
        intents_list = json.load(f)
    intent_map = {(r["session_id"], r["turn_number"]): r.get("intent") for r in intents_list}
    n_valid = sum(1 for v in intent_map.values() if v)
    print(f"  {n_valid}/{len(intent_map)} valid intents", flush=True)

    # Load dev payload
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    print(f"  {n} dev cases", flush=True)

    # Load R21 baseline lists
    with open(R21_LISTS_PATH) as f:
        r21_lists = json.load(f)
    print(f"  R21 lists: {len(r21_lists)} cases", flush=True)

    # Load V3 pools
    with open(V3_POOLS_PATH) as f:
        v3_pools = json.load(f)
    print(f"  V3 pools: {len(v3_pools)} cases", flush=True)

    # -- Q1/Q2: Intent BM25 retrieval --
    print(f"\n{ts()} Building BM25 queries from intents...", flush=True)
    q1_queries = []
    q2_queries = []
    q3_variants = []

    for case in cases:
        intent = intent_map.get((case["session_id"], case["turn_number"]))
        if intent:
            q1_queries.append(build_intent_bm25_query(intent))
            q2_queries.append(build_artist_boost_bm25_query(intent))
            variants = intent.get("query_variants", [])
            q3_variants.append(variants if variants else [intent.get("summary", case["user_query"])])
        else:
            q1_queries.append(case["user_query"])
            q2_queries.append(case["user_query"])
            q3_variants.append([case["user_query"]])

    print(f"  Q1 sample: {q1_queries[0][:120]}", flush=True)
    print(f"  Q2 sample: {q2_queries[0][:120]}", flush=True)
    print(f"  Q3 sample variants: {len(q3_variants[0])}", flush=True)

    # BM25 retrieval
    print(f"\n{ts()} Loading BM25 index...", flush=True)
    from mcrs.retrieval_modules.bm25 import BM25Retriever
    bm25 = BM25Retriever(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
        cache_dir=str(REPO / "cache"),
    )

    topk_bm25 = 300
    print(f"{ts()} Q1: Intent BM25 retrieval (top-{topk_bm25})...", flush=True)
    q1_results = bm25.batch_text_to_item_retrieval(q1_queries, topk=topk_bm25)

    print(f"{ts()} Q2: Artist-boost BM25 retrieval (top-{topk_bm25})...", flush=True)
    q2_results = bm25.batch_text_to_item_retrieval(q2_queries, topk=topk_bm25)

    # -- Q3: Intent Dense Retrieval (BGE/R21 on query variants) --
    print(f"\n{ts()} Q3: Intent dense retrieval (R21 model on query variants)...", flush=True)
    q3_results = intent_dense_retrieval(cases, q3_variants)

    # -- Diagnostics --
    print(f"\n{ts()} Computing diagnostics...", flush=True)
    train_tracks = load_train_track_ids()

    sources = {
        "Q1_intent_bm25": q1_results,
        "Q2_artist_boost_bm25": q2_results,
        "Q3_intent_dense": q3_results,
    }

    report: dict[str, dict] = {}
    for src_name, src_lists in sources.items():
        report[src_name] = evaluate_source(src_name, src_lists, cases, r21_lists, v3_pools, train_tracks)

    # Save
    out_path = REPO / "exp" / "eval" / "expR26_stage1_source_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Saved diagnostics to {out_path}", flush=True)

    # Print summary
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print("R26 STAGE 1 SOURCE DIAGNOSTICS", flush=True)
    print(sep, flush=True)
    for src_name, src_metrics in report.items():
        print(f"\n  {src_name}:", flush=True)
        for depth in [20, 50, 100, 200, 300]:
            key = f"hit@{depth}"
            if key in src_metrics:
                print(f"    {key}: {src_metrics[key]} ({src_metrics[key]/n*100:.1f}%)", flush=True)
        print(f"    unique vs R21@200: {src_metrics.get('unique_vs_r21', 0)}", flush=True)
        print(f"    lost vs R21@200:   {src_metrics.get('lost_vs_r21', 0)}", flush=True)
        print(f"    unique vs V3 pool: {src_metrics.get('unique_vs_v3', 0)}", flush=True)
        print(f"    unseen hit@200:    {src_metrics.get('unseen_hit200', 0)}/{src_metrics.get('unseen_total', 0)}",
              flush=True)

    # Gate check
    print(f"\n{sep}", flush=True)
    print("GATE CHECK", flush=True)
    best_unique = max(m.get("unique_vs_r21", 0) for m in report.values())
    print(f"  Best unique GTs vs R21@200: {best_unique} (gate >= 150: {'PASS' if best_unique >= 150 else 'FAIL'})",
          flush=True)


def load_train_track_ids() -> set[str]:
    """Load track IDs that appear in the official training sessions.

    Do not derive this from dev cases: adding dev GTs makes every GT look
    "seen" and destroys the R26 seen/unseen diagnostics.
    """
    from datasets import DownloadConfig, load_dataset

    train_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        split="train",
        download_config=DownloadConfig(local_files_only=True),
    )
    train_tracks: set[str] = set()
    for row in train_ds:
        for turn in row["conversations"]:
            if turn.get("role") == "music":
                content = str(turn.get("content", "")).strip()
                if content:
                    train_tracks.add(content)
    print(f"  Train tracks for seen/unseen: {len(train_tracks)}", flush=True)
    return train_tracks


def intent_dense_retrieval(cases, q3_variants):
    """Encode query variants with R21 model and retrieve."""
    dense_cache = CACHE_DIR / "q3_dense_results.json"
    if dense_cache.exists():
        print(f"  Loading cached Q3 results from {dense_cache}", flush=True)
        with open(dense_cache) as f:
            return json.load(f)

    # Flatten all variants and track which case they belong to
    flat_queries = []
    case_indices = []
    for i, variants in enumerate(q3_variants):
        for v in variants:
            flat_queries.append(v)
            case_indices.append(i)

    print(f"  {len(flat_queries)} total variant queries across {len(cases)} cases", flush=True)

    # Save queries to temp file for subprocess
    queries_path = CACHE_DIR / "q3_flat_queries.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(queries_path, "w") as f:
        json.dump({"queries": flat_queries, "case_indices": case_indices}, f)

    # Run dense retrieval in isolated subprocess (avoids loky/torch issues)
    print(f"{ts()}   Launching dense retrieval subprocess...", flush=True)
    script = f'''
import json, numpy as np, time
from pathlib import Path
from sentence_transformers import SentenceTransformer

REPO = Path("{REPO}")
with open(REPO / "cache/r26/q3_flat_queries.json") as f:
    data = json.load(f)
queries = data["queries"]
case_indices = data["case_indices"]

track_ids = json.loads((REPO / "cache/r21_production/track_ids.json").read_text())
track_embs = np.load(REPO / "cache/r21_production/track_embeddings.npy")

print("Loading R21 model...", flush=True)
model = SentenceTransformer(str(REPO / "cache/r21_production/model"))

print(f"Encoding {{len(queries)}} variant queries...", flush=True)
t0 = time.time()
query_embs = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
query_embs = query_embs.astype(np.float32)
print(f"  {{time.time()-t0:.1f}}s", flush=True)

n_cases = max(case_indices) + 1
topk = 300
results = [[] for _ in range(n_cases)]

print(f"Retrieving top-{{topk}} per variant, merging per case...", flush=True)
t0 = time.time()
for qi in range(len(queries)):
    ci = case_indices[qi]
    scores = track_embs @ query_embs[qi]
    top_idx = np.argpartition(-scores, topk)[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    for j in top_idx:
        results[ci].append((track_ids[j], float(scores[j])))

# Merge variants per case: take best score per track, sort, keep top-300
merged = []
for ci in range(n_cases):
    track_scores = {{}}
    for tid, sc in results[ci]:
        if tid not in track_scores or sc > track_scores[tid]:
            track_scores[tid] = sc
    ranked = sorted(track_scores, key=track_scores.__getitem__, reverse=True)[:topk]
    merged.append(ranked)

print(f"  {{time.time()-t0:.1f}}s", flush=True)
out_path = REPO / "cache/r26/q3_dense_results.json"
with open(out_path, "w") as f:
    json.dump(merged, f)
print(f"Saved to {{out_path}}", flush=True)
'''
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=3600,
        cwd=str(REPO),
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:2000]}", flush=True)
        raise RuntimeError(f"Dense retrieval subprocess failed with code {result.returncode}")

    with open(dense_cache) as f:
        return json.load(f)


def evaluate_source(name, results_lists, cases, r21_lists, v3_pools, train_tracks):
    """Evaluate a single retrieval source."""
    hit_counts: dict[str, int] = defaultdict(int)
    gt_ranks: list[int] = []

    unique_vs_r21 = 0
    lost_vs_r21 = 0
    unique_vs_v3 = 0
    unseen_hit200 = 0
    unseen_total = 0
    seen_hit200 = 0
    seen_total = 0
    hist0_hit200 = 0
    hist0_total = 0
    hist7_hit200 = 0
    hist7_total = 0

    for i, case in enumerate(cases):
        gt = case["gt"]
        r = results_lists[i] if i < len(results_lists) else []
        r21 = r21_lists[i] if i < len(r21_lists) else []
        v3 = v3_pools[i] if i < len(v3_pools) else []

        r_set = set(r[:200])
        r21_set = set(r21[:200])
        v3_set = set(v3[:300])

        is_unseen = gt not in train_tracks

        for depth in [20, 50, 100, 200, 300]:
            if gt in set(r[:depth]):
                hit_counts[f"hit@{depth}"] += 1

        if gt in r_set and gt not in r21_set:
            unique_vs_r21 += 1
        if gt not in r_set and gt in r21_set:
            lost_vs_r21 += 1
        if gt in r_set and gt not in v3_set:
            unique_vs_v3 += 1

        if is_unseen:
            unseen_total += 1
            if gt in r_set:
                unseen_hit200 += 1
        else:
            seen_total += 1
            if gt in r_set:
                seen_hit200 += 1

        n_prior = case["n_prior_music"]
        if n_prior == 0:
            hist0_total += 1
            if gt in r_set:
                hist0_hit200 += 1
        elif n_prior >= 7:
            hist7_total += 1
            if gt in r_set:
                hist7_hit200 += 1

        if gt in r:
            gt_ranks.append(r.index(gt) + 1)

    result: dict[str, int | float | None] = dict(hit_counts)
    result["unique_vs_r21"] = unique_vs_r21
    result["lost_vs_r21"] = lost_vs_r21
    result["net_vs_r21"] = unique_vs_r21 - lost_vs_r21
    result["unique_vs_v3"] = unique_vs_v3
    result["unseen_hit200"] = unseen_hit200
    result["unseen_total"] = unseen_total
    result["seen_hit200"] = seen_hit200
    result["seen_total"] = seen_total
    result["hist0_hit200"] = hist0_hit200
    result["hist0_total"] = hist0_total
    result["hist7_hit200"] = hist7_hit200
    result["hist7_total"] = hist7_total
    result["median_gt_rank"] = float(np.median(gt_ranks)) if gt_ranks else None
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R26 Structured Intent Retrieval")
    parser.add_argument("--phase", required=True, choices=["extract", "retrieve", "fuse"],
                        help="Phase to run")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers for extraction")
    args = parser.parse_args()

    if args.phase == "extract":
        phase_extract(args)
    elif args.phase == "retrieve":
        phase_retrieve(args)
    elif args.phase == "fuse":
        print("Phase 2 (fuse) not yet implemented.", flush=True)


if __name__ == "__main__":
    main()

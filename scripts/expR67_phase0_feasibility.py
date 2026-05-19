"""R67 Wave 1 Phase 0 feasibility — Opus 4.7 candidate reranker on stratified sample.

Two prompt styles (A: concise expert, B: strict rubric) x ~150 stratified cases
from exp/eval/expR67_top30_candidates.pkl. Computes per-prompt phase-0 gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import pickle
import random
import re
import string
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = REPO / "exp/eval/expR67_top30_candidates.pkl"
R12_PATH = REPO / "exp/eval/_R12_all_turns_payload.pkl"
TRACK_META_PATH = REPO / "cache/metadata/track_metadata_all_tracks.json"
CACHE_DIR = REPO / "cache/r67/llm_calls"
OUT_SAMPLE = REPO / "exp/eval/expR67_phase0_sample.json"
OUT_RESULT = REPO / "exp/eval/expR67_phase0_feasibility.json"
OUT_DOC = REPO / "docs/r67_phase0_feasibility_result.md"

MODEL_ID = "claude-opus-4-7"
PER_STRATUM = 30
SEED = 0
MAX_CALLS = 350
MAX_PARALLEL = 4
MAX_TOKENS = 1024
TOP_N = 20

PROMPT_A_SYSTEM = (
    "You are an expert music recommender. Given a user conversation and 30 candidate tracks, "
    "output the 20 tracks most appropriate to play next, ordered from most to least appropriate. "
    "Use only the conversation context, conversation_goal (if provided), played-track metadata, "
    "and candidate metadata. Output strict JSON only — no preface, no markdown fences:\n"
    "{\"ranking\": [\"<alias>\", \"<alias>\", ...20 items], \"confidence\": <integer 0-10>, "
    "\"rationale_one_line\": \"<<=120 chars>\"}"
)

PROMPT_B_SYSTEM = (
    "You are an expert music recommender. Score each of the 30 candidate tracks against this rubric:\n"
    "(1) user intent fit — does the track match what the user explicitly asked for?\n"
    "(2) conversational continuity vs requested pivot — does the track maintain the flow, or pivot if requested?\n"
    "(3) artist/genre fit — does the artist/genre match the conversation context?\n"
    "(4) novelty without overfitting played artists — avoid stacking the same artist\n"
    "(5) album/era coherence — does the release era fit?\n"
    "Output the top 20 candidates by overall fit, ordered from best to worst. Output strict JSON only — "
    "no preface, no markdown fences:\n"
    "{\"ranking\": [\"<alias>\", ...20 items], "
    "\"confidence\": <integer 0-10>, \"rationale_one_line\": \"<<=120 chars>\"}"
)

PROMPTS = {"A": PROMPT_A_SYSTEM, "B": PROMPT_B_SYSTEM}


def ts() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_artist_list(raw: Any) -> list[str]:
    """gt_artist is stored as a string-repr of a python list (e.g. \"['nirvana']\")."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    # try literal_eval
    try:
        import ast
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x).strip().lower() for x in v]
        if isinstance(v, str):
            return [v.strip().lower()]
    except Exception:
        pass
    return [s.strip().lower()]


def build_alias_pool() -> list[str]:
    """Aliases A..Z, AA..AD (30 total)."""
    letters = list(string.ascii_uppercase)
    aliases = letters[:26] + [f"A{c}" for c in letters[:4]]
    assert len(aliases) == 30
    return aliases


def load_data():
    print(f"{ts()} loading candidates pkl")
    cand = pickle.load(CANDIDATES_PATH.open("rb"))
    print(f"{ts()} loading R12 payload pkl")
    r12 = pickle.load(R12_PATH.open("rb"))
    print(f"{ts()} loading track metadata json")
    meta_raw = json.load(TRACK_META_PATH.open())
    return cand, r12, meta_raw


def case_index_from_r12(r12) -> dict[tuple[str, int], dict]:
    return {(c["session_id"], c["turn_number"]): c for c in r12["cases"]}


def get_track_meta(meta_raw: dict, tid: str) -> dict:
    m = meta_raw.get(tid, {})
    def first(x):
        if isinstance(x, list) and x:
            return x[0]
        return x
    title = first(m.get("track_name")) or ""
    artist = first(m.get("artist_name")) or ""
    album = first(m.get("album_name")) or ""
    tags = m.get("tag_list") or []
    if isinstance(tags, list):
        tags = tags[:5]
    rd = m.get("release_date") or ""
    year = ""
    if isinstance(rd, str) and len(rd) >= 4:
        year = rd[:4]
    return {
        "title": title,
        "artist": artist,
        "album": album,
        "tags": tags,
        "release_year": year,
    }


def conversation_text_list(r12_case: dict) -> list[dict]:
    """Turn texts: user/assistant strings (skip music role since track_id is opaque).
    Use history field for prior turns + current user_query.
    """
    out: list[dict] = []
    for h in r12_case.get("history", []):
        role = h.get("role")
        if role in ("user", "assistant"):
            out.append({"role": role, "text": h.get("content", "")})
    # append current user query as final user turn
    out.append({"role": "user", "text": r12_case.get("user_query", "")})
    return out


def played_track_meta(r12_case: dict, meta_raw: dict) -> list[dict]:
    out = []
    for tid in r12_case.get("music_turns", []) or []:
        m = get_track_meta(meta_raw, tid)
        out.append(m)
    return out


def played_artists_lower(r12_case: dict, meta_raw: dict) -> set[str]:
    arts: set[str] = set()
    for tid in r12_case.get("music_turns", []) or []:
        a = get_track_meta(meta_raw, tid).get("artist") or ""
        if a:
            arts.add(a.strip().lower())
    return arts


def build_sample(records, r12_idx, meta_raw) -> dict:
    rng = random.Random(SEED)
    s1 = [r for r in records if r["history_depth"] == 7]
    # S2: diff_artist — gt_artist not in any played artist for that case
    s2_pool = []
    for r in records:
        gt_arts = parse_artist_list(r.get("gt_artist"))
        if not gt_arts:
            continue
        rcase = r12_idx.get((r["case_id"], r["history_depth"] + 1))
        if not rcase:
            continue
        played = played_artists_lower(rcase, meta_raw)
        if any(a in played for a in gt_arts):
            continue
        s2_pool.append(r)
    # S3 small margin — relax: smallest-30 by margin (literal <0.05 yields only 6)
    margins = [(abs(r["lr_rank1_score"] - r["lr_rank5_score"]), r) for r in records]
    margins.sort(key=lambda x: x[0])
    s3_pool = [r for _, r in margins[: max(PER_STRATUM * 3, 100)]]  # large pool of small-margin
    s4_pool = [r for r in records if r.get("gt_in_top20_rank") and 2 <= r["gt_in_top20_rank"] <= 20]
    s5_pool = [r for r in records if r.get("gt_in_top20_rank") is None]

    def pick(pool, name):
        # sample 30 unique by (case_id, history_depth)
        local = list(pool)
        rng.shuffle(local)
        seen = set()
        picked = []
        for r in local:
            key = (r["case_id"], r["history_depth"])
            if key in seen:
                continue
            seen.add(key)
            picked.append(r)
            if len(picked) == PER_STRATUM:
                break
        if len(picked) < PER_STRATUM:
            print(f"WARN: {name} pool size {len(local)} only yielded {len(picked)} picks")
        return picked

    picks = {
        "S1": pick(s1, "S1"),
        "S2": pick(s2_pool, "S2"),
        "S3": pick(s3_pool, "S3"),
        "S4": pick(s4_pool, "S4"),
        "S5": pick(s5_pool, "S5"),
    }
    # dedupe union; first stratum to claim a case wins the label
    label = {}
    order = ["S1", "S2", "S3", "S4", "S5"]
    union_records = []
    for s in order:
        for r in picks[s]:
            key = (r["case_id"], r["history_depth"])
            if key in label:
                continue
            label[key] = s
            union_records.append(r)

    strata_counts = {s: 0 for s in order}
    for key, s in label.items():
        strata_counts[s] += 1

    return {
        "records": union_records,
        "label": label,  # (case_id, hd) -> stratum
        "strata_counts": strata_counts,
    }


def build_packet(rec: dict, r12_case: dict, meta_raw: dict, aliases: list[str]) -> tuple[dict, dict, dict]:
    """Return (payload_dict_for_llm, alias_map alias->tid, alias_map tid->alias)."""
    conv = conversation_text_list(r12_case)
    played = played_track_meta(r12_case, meta_raw)
    cands = rec["lr_top30"]
    alias_to_tid = {}
    tid_to_alias = {}
    cand_packet = []
    for a, c in zip(aliases, cands):
        tid = c["track_id"]
        alias_to_tid[a] = tid
        tid_to_alias[tid] = a
        cand_packet.append({
            "candidate_id": a,
            "title": c.get("title", "") or "",
            "artist": c.get("artist", "") or "",
            "album": c.get("album", "") or "",
            "tags": (c.get("tags") or [])[:5],
            "release_year": c.get("release_year", "") or "",
        })
    payload = {
        "conversation": conv,
        "conversation_goal": None,
        "played_tracks": played,
        "candidates": cand_packet,
    }
    return payload, alias_to_tid, tid_to_alias


def cache_key(case_id: str, hd: int, model: str, prompt_hash: str, payload_hash: str) -> str:
    return sha256_str(f"{case_id}|{hd}|{model}|{prompt_hash}|{payload_hash}")


def validate_response(text: str, aliases_in_input: set[str], n_required: int = TOP_N):
    """Return (parsed_dict, error_str). parsed_dict has ranking, confidence, rationale."""
    # strip markdown fences if present
    t = text.strip()
    if t.startswith("```"):
        # remove leading fence line
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    # extract first {...} block
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None, "no_json_object"
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except Exception as e:
        return None, f"json_decode:{e}"
    if not isinstance(obj, dict):
        return None, "not_object"
    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        return None, "ranking_not_list"
    if len(ranking) != n_required:
        return None, f"ranking_len_{len(ranking)}"
    if len(set(ranking)) != n_required:
        return None, "ranking_dup"
    if not all(isinstance(x, str) for x in ranking):
        return None, "ranking_nonstring"
    if any(x not in aliases_in_input for x in ranking):
        return None, "ranking_unknown_alias"
    conf = obj.get("confidence")
    rat = obj.get("rationale_one_line", "")
    return {
        "ranking": ranking,
        "confidence": conf,
        "rationale_one_line": rat,
    }, None


_telemetry_lock = threading.Lock()


class Telemetry:
    def __init__(self):
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_cache_misses = 0
        self.n_retries = 0
        self.n_malformed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_made_live = 0  # for hard cap

    def add(self, **kw):
        with _telemetry_lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)


def call_llm(client, system_prompt: str, payload: dict, telem: Telemetry, max_retries: int = 1):
    """Returns (parsed_dict_or_none, response_text, n_in, n_out, retries, malformed_bool, last_error)."""
    user_content = json.dumps(payload, ensure_ascii=False)
    aliases_in_input = {c["candidate_id"] for c in payload["candidates"]}
    retries = 0
    last_text = ""
    last_err = None
    parsed = None
    n_in_total = 0
    n_out_total = 0
    for attempt in range(max_retries + 1):
        try:
            # Opus 4.7 deprecates the temperature parameter — omit it.
            kw = dict(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content if attempt == 0 else (
                    user_content + "\n\nSTRICT REPROMPT: Output ONLY a JSON object with keys 'ranking' (list of exactly 20 alias strings drawn from candidates.candidate_id, no duplicates), 'confidence' (integer 0-10), and 'rationale_one_line' (<=120 chars). No prose, no markdown fences."
                )}],
            )
            resp = client.messages.create(**kw)
        except Exception as e:
            msg = str(e)
            last_err = msg
            # handle 429 / rate-limit
            if "429" in msg or "rate" in msg.lower():
                time.sleep(2 ** attempt)
                retries += 1
                continue
            # other error: retry once
            if attempt < max_retries:
                retries += 1
                time.sleep(1)
                continue
            return None, "", 0, 0, retries, True, msg
        n_in = getattr(resp.usage, "input_tokens", 0) if hasattr(resp, "usage") else 0
        n_out = getattr(resp.usage, "output_tokens", 0) if hasattr(resp, "usage") else 0
        n_in_total += n_in
        n_out_total += n_out
        text = resp.content[0].text if resp.content else ""
        last_text = text
        parsed, err = validate_response(text, aliases_in_input)
        if parsed is not None:
            return parsed, text, n_in_total, n_out_total, retries, False, None
        last_err = err
        if attempt < max_retries:
            retries += 1
            continue
        # final attempt failed
        return None, text, n_in_total, n_out_total, retries, True, err
    return parsed, last_text, n_in_total, n_out_total, retries, parsed is None, last_err


def process_one(client, rec, r12_case, meta_raw, aliases, prompt_style, system_prompt, telem):
    case_id = rec["case_id"]
    hd = rec["history_depth"]
    payload, alias_to_tid, tid_to_alias = build_packet(rec, r12_case, meta_raw, aliases)
    prompt_hash = sha256_str(system_prompt)
    payload_hash = sha256_str(json.dumps(payload, ensure_ascii=False, sort_keys=False))
    key = cache_key(case_id, hd, MODEL_ID, prompt_hash, payload_hash)
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        try:
            d = json.load(cache_path.open())
            telem.add(n_cache_hits=1)
            return {
                "rec": rec,
                "alias_to_tid": alias_to_tid,
                "tid_to_alias": tid_to_alias,
                "parsed": d.get("parsed_ranking") and {
                    "ranking": d["parsed_ranking"],
                    "confidence": d.get("parsed_confidence"),
                    "rationale_one_line": d.get("parsed_rationale", ""),
                },
                "malformed": bool(d.get("malformed", False)),
                "prompt_style": prompt_style,
                "from_cache": True,
            }
        except Exception:
            pass
    # cache miss; respect hard cap
    with _telemetry_lock:
        if telem.calls_made_live >= MAX_CALLS:
            telem.n_cache_misses += 1
            return {
                "rec": rec, "alias_to_tid": alias_to_tid, "tid_to_alias": tid_to_alias,
                "parsed": None, "malformed": True, "prompt_style": prompt_style,
                "from_cache": False, "cap_hit": True,
            }
        telem.calls_made_live += 1
    telem.add(n_cache_misses=1, n_calls=1)
    parsed, text, n_in, n_out, retries, malformed, err = call_llm(client, system_prompt, payload, telem)
    telem.add(n_retries=retries, total_input_tokens=n_in, total_output_tokens=n_out,
              n_malformed=1 if malformed else 0)
    record = {
        "case_id": case_id,
        "history_depth": hd,
        "prompt_template_hash": prompt_hash,
        "payload_hash": payload_hash,
        "model": MODEL_ID,
        "response_text": text,
        "parsed_ranking": parsed["ranking"] if parsed else None,
        "parsed_confidence": parsed["confidence"] if parsed else None,
        "parsed_rationale": parsed["rationale_one_line"] if parsed else "",
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "retries": retries,
        "malformed": bool(malformed),
        "error": err,
        "prompt_style": prompt_style,
    }
    try:
        with cache_path.open("w") as f:
            json.dump(record, f)
    except Exception:
        pass
    return {
        "rec": rec, "alias_to_tid": alias_to_tid, "tid_to_alias": tid_to_alias,
        "parsed": parsed, "malformed": malformed, "prompt_style": prompt_style,
        "from_cache": False,
    }


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranking: list[str], gt: str, k: int = TOP_N) -> float:
    rels = [1 if x == gt else 0 for x in ranking[:k]]
    ideal = sorted(rels, reverse=True)
    d = dcg(rels)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return d / idcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="build sample, no LLM calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)

    cand, r12, meta_raw = load_data()
    r12_idx = case_index_from_r12(r12)
    records = cand["records"]
    aliases = build_alias_pool()

    sample = build_sample(records, r12_idx, meta_raw)
    sample_recs = sample["records"]
    labels = sample["label"]
    print(f"{ts()} sample size={len(sample_recs)} strata={sample['strata_counts']}")

    sample_payload = {
        "seed": SEED,
        "n": len(sample_recs),
        "strata_counts": sample["strata_counts"],
        "cases": [
            {
                "case_id": r["case_id"],
                "history_depth": r["history_depth"],
                "stratum": labels[(r["case_id"], r["history_depth"])],
                "gt_track_id": r["gt_track_id"],
                "gt_in_top20_rank": r["gt_in_top20_rank"],
                "lr_rank1_score": r["lr_rank1_score"],
                "lr_rank5_score": r["lr_rank5_score"],
            }
            for r in sample_recs
        ],
    }
    OUT_SAMPLE.write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False))
    print(f"{ts()} wrote sample to {OUT_SAMPLE}")

    if args.dry_run:
        print(f"{ts()} --dry-run requested; stopping after sample build")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    telem = Telemetry()
    t0 = time.time()

    # Build task list: (prompt_style, rec)
    tasks = []
    for r in sample_recs:
        for p in ("A", "B"):
            tasks.append((p, r))
    print(f"{ts()} total tasks: {len(tasks)} (cap={MAX_CALLS})")

    results: dict[str, list[dict]] = {"A": [], "B": []}
    completed = 0
    log_every = 20
    with cf.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        fut_to_meta = {}
        for prompt_style, rec in tasks:
            r12_case = r12_idx[(rec["case_id"], rec["history_depth"] + 1)]
            fut = ex.submit(process_one, client, rec, r12_case, meta_raw, aliases,
                            prompt_style, PROMPTS[prompt_style], telem)
            fut_to_meta[fut] = (prompt_style, rec["case_id"], rec["history_depth"])
        for fut in cf.as_completed(fut_to_meta):
            prompt_style, cid, hd = fut_to_meta[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"{ts()} EXC {prompt_style} {cid[:8]} hd={hd}: {e}")
                continue
            results[prompt_style].append(out)
            completed += 1
            if completed % log_every == 0:
                print(f"{ts()} done={completed}/{len(tasks)} "
                      f"hits={telem.n_cache_hits} miss={telem.n_cache_misses} "
                      f"malformed={telem.n_malformed} retries={telem.n_retries} "
                      f"calls_live={telem.calls_made_live}")

    elapsed = time.time() - t0
    print(f"{ts()} all done. elapsed={elapsed:.1f}s n_calls={telem.n_calls} "
          f"hits={telem.n_cache_hits} miss={telem.n_cache_misses} malformed={telem.n_malformed}")

    # Compute metrics
    per_prompt_metrics = {}
    for ps in ("A", "B"):
        rs = results[ps]
        n_total = len(rs)
        n_valid = sum(1 for r in rs if r["parsed"] and not r["malformed"])
        # per-case stats
        rows = []
        for r in rs:
            rec = r["rec"]
            stratum = labels[(rec["case_id"], rec["history_depth"])]
            gt = rec["gt_track_id"]
            lr_top20 = [c["track_id"] for c in rec["lr_top30"][:TOP_N]]
            row = {
                "case_id": rec["case_id"],
                "hd": rec["history_depth"],
                "stratum": stratum,
                "ndcg_lr": ndcg_at_k(lr_top20, gt),
                "ndcg_llm": None,
                "valid": bool(r["parsed"] and not r["malformed"]),
                "recovered": False,
                "lost": False,
                "top1_changed": False,
            }
            if row["valid"]:
                ranking_aliases = r["parsed"]["ranking"]
                llm_top20_tids = [r["alias_to_tid"][a] for a in ranking_aliases]
                row["ndcg_llm"] = ndcg_at_k(llm_top20_tids, gt)
                in_lr = gt in lr_top20
                in_llm = gt in llm_top20_tids
                row["recovered"] = (in_llm and not in_lr)
                row["lost"] = (in_lr and not in_llm)
                row["top1_changed"] = (llm_top20_tids[0] != lr_top20[0])
                # same_artist: gt artist appears in candidates' played artist set
                # diff_artist: per S2 definition — gt_artist not in played artist set
                r12_case = r12_idx[(rec["case_id"], rec["history_depth"] + 1)]
                gt_arts = parse_artist_list(rec.get("gt_artist"))
                played_a = played_artists_lower(r12_case, meta_raw)
                same_artist = bool(gt_arts) and any(a in played_a for a in gt_arts)
                row["same_artist"] = same_artist
                row["diff_artist"] = bool(gt_arts) and not same_artist
            else:
                row["ndcg_llm"] = 0.0
                row["same_artist"] = False
                row["diff_artist"] = False
            rows.append(row)
        # aggregates over valid only
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else 0.0

        valid_rows = [r for r in rows if r["valid"]]
        ndcg_sample_llm = mean([r["ndcg_llm"] for r in valid_rows])
        ndcg_sample_lr = mean([r["ndcg_lr"] for r in valid_rows])
        h7_rows = [r for r in valid_rows if r["stratum"] == "S1"]
        ndcg_h7_llm = mean([r["ndcg_llm"] for r in h7_rows])
        ndcg_h7_lr = mean([r["ndcg_lr"] for r in h7_rows])
        sa_rows = [r for r in valid_rows if r["same_artist"]]
        da_rows = [r for r in valid_rows if r["diff_artist"]]
        ndcg_sa_llm = mean([r["ndcg_llm"] for r in sa_rows]) if sa_rows else 0.0
        ndcg_sa_lr = mean([r["ndcg_lr"] for r in sa_rows]) if sa_rows else 0.0
        ndcg_da_llm = mean([r["ndcg_llm"] for r in da_rows]) if da_rows else 0.0
        ndcg_da_lr = mean([r["ndcg_lr"] for r in da_rows]) if da_rows else 0.0

        recovered = sum(1 for r in valid_rows if r["recovered"])
        lost = sum(1 for r in valid_rows if r["lost"])
        net = recovered - lost
        top1_changed_count = sum(1 for r in valid_rows if r["top1_changed"])
        churn_per_80 = (top1_changed_count / max(len(valid_rows), 1)) * 80
        validity_rate = n_valid / n_total if n_total else 0.0

        gate_h7 = (ndcg_h7_llm - ndcg_h7_lr) >= 0.005
        gate_sample = (ndcg_sample_llm - ndcg_sample_lr) >= 0.0
        gate_rec = recovered > lost
        gate_churn = churn_per_80 <= 25.0
        gate_validity = validity_rate >= 0.95
        passes = all([gate_h7, gate_sample, gate_rec, gate_churn, gate_validity])

        per_prompt_metrics[ps] = {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_h7": len(h7_rows),
            "n_same_artist": len(sa_rows),
            "n_diff_artist": len(da_rows),
            "ndcg20_sample": ndcg_sample_llm,
            "ndcg20_sample_lr": ndcg_sample_lr,
            "delta_sample_vs_lr": ndcg_sample_llm - ndcg_sample_lr,
            "ndcg20_h7": ndcg_h7_llm,
            "ndcg20_h7_lr": ndcg_h7_lr,
            "delta_h7_vs_lr": ndcg_h7_llm - ndcg_h7_lr,
            "ndcg20_same_artist": ndcg_sa_llm,
            "delta_same_artist": ndcg_sa_llm - ndcg_sa_lr,
            "ndcg20_diff_artist": ndcg_da_llm,
            "delta_diff_artist": ndcg_da_llm - ndcg_da_lr,
            "recovered": recovered,
            "lost": lost,
            "net": net,
            "top1_changed_count": top1_changed_count,
            "churn_per_80": churn_per_80,
            "validity_rate": validity_rate,
            "gate_h7_delta_ge_0005": gate_h7,
            "gate_sample_delta_ge_0": gate_sample,
            "gate_recovered_gt_lost": gate_rec,
            "gate_churn_le_25_80": gate_churn,
            "gate_validity_ge_095": gate_validity,
            "passes_phase0_gate": passes,
        }

    # winner / verdict
    pass_A = per_prompt_metrics["A"]["passes_phase0_gate"]
    pass_B = per_prompt_metrics["B"]["passes_phase0_gate"]
    if pass_A and pass_B:
        # pick winner by higher h7 delta then sample delta
        winner = "A" if per_prompt_metrics["A"]["delta_h7_vs_lr"] >= per_prompt_metrics["B"]["delta_h7_vs_lr"] else "B"
        verdict = "PROCEED"
    elif pass_A:
        winner = "A"; verdict = "PROCEED"
    elif pass_B:
        winner = "B"; verdict = "PROCEED"
    else:
        winner = None; verdict = "ARCHIVE_PHASE_0"

    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    out = {
        "experiment": "R67 Phase 0 LLM feasibility",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha,
        "verdict": verdict,
        "winner_prompt_style": winner,
        "model": MODEL_ID,
        "sample_size": len(sample_recs),
        "strata_counts": sample["strata_counts"],
        "telemetry": {
            "n_calls": telem.n_calls,
            "n_cache_hits": telem.n_cache_hits,
            "n_cache_misses": telem.n_cache_misses,
            "n_retries": telem.n_retries,
            "n_malformed": telem.n_malformed,
            "total_input_tokens": telem.total_input_tokens,
            "total_output_tokens": telem.total_output_tokens,
            "elapsed_s": elapsed,
        },
        "prompts": per_prompt_metrics,
    }
    OUT_RESULT.write_text(json.dumps(out, indent=2))
    print(f"{ts()} wrote {OUT_RESULT}")

    # markdown doc
    def fmt(v, p=4):
        if isinstance(v, bool):
            return "PASS" if v else "FAIL"
        if isinstance(v, float):
            return f"{v:.{p}f}"
        return str(v)
    lines = [
        f"# R67 Phase 0 Feasibility — {verdict}",
        "",
        f"- HEAD: `{head_sha}`",
        f"- model: `{MODEL_ID}`",
        f"- sample size: {len(sample_recs)}",
        f"- strata: {sample['strata_counts']}",
        f"- winner: **{winner if winner else 'NONE'}**",
        "",
        "## Telemetry",
        f"- n_calls (live): {telem.n_calls}",
        f"- cache_hits: {telem.n_cache_hits}, cache_misses: {telem.n_cache_misses}",
        f"- retries: {telem.n_retries}",
        f"- malformed: {telem.n_malformed}",
        f"- total input tokens: {telem.total_input_tokens}",
        f"- total output tokens: {telem.total_output_tokens}",
        f"- elapsed_s: {elapsed:.1f}",
        "",
        "## Metrics",
        "",
        "| metric | Style A | Style B |",
        "|---|---|---|",
    ]
    keys = [
        ("n_valid","int"),
        ("ndcg20_sample","f"),
        ("ndcg20_sample_lr","f"),
        ("delta_sample_vs_lr","f"),
        ("ndcg20_h7","f"),
        ("ndcg20_h7_lr","f"),
        ("delta_h7_vs_lr","f"),
        ("ndcg20_same_artist","f"),
        ("delta_same_artist","f"),
        ("ndcg20_diff_artist","f"),
        ("delta_diff_artist","f"),
        ("recovered","int"),
        ("lost","int"),
        ("net","int"),
        ("top1_changed_count","int"),
        ("churn_per_80","f"),
        ("validity_rate","f"),
        ("gate_h7_delta_ge_0005","bool"),
        ("gate_sample_delta_ge_0","bool"),
        ("gate_recovered_gt_lost","bool"),
        ("gate_churn_le_25_80","bool"),
        ("gate_validity_ge_095","bool"),
        ("passes_phase0_gate","bool"),
    ]
    for k, _ in keys:
        a = per_prompt_metrics["A"][k]
        b = per_prompt_metrics["B"][k]
        lines.append(f"| {k} | {fmt(a)} | {fmt(b)} |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if verdict == "PROCEED":
        lines.append(f"PROCEED to W2 with prompt style **{winner}**.")
    else:
        lines.append("ARCHIVE Phase 0 — neither prompt style cleared the gate.")
    OUT_DOC.write_text("\n".join(lines))
    print(f"{ts()} wrote {OUT_DOC}")


if __name__ == "__main__":
    main()

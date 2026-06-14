#!/usr/bin/env python3
"""R498: aggressive listwise LLM reranker.

This is deliberately not another safe rank-2 tweak. It tests whether a modern
instruction-following model can choose the hidden GT from a joint slate:

  current production top-20 + challenger candidates from the retrieval pool.

The script has four modes:

  pack-dev    build a dev JSONL prompt pack with hidden GT for offline scoring
  run         call an API model for a prompt pack, cached via mcrs.utils
  eval-dev    evaluate model outputs under aggressive deployment policies
  pack-blind  build the analogous Blind-A prompt pack
  build-blind build a Blind-A submission from model outputs and a policy

No Codabench slot should be spent until eval-dev shows a real full-dev gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mcrs.utils import call_llm_api  # noqa: E402


PAYLOAD = REPO / "exp/eval/_R12_all_turns_payload.pkl"
HARNESS = REPO / "exp/eval/expR436_integrate_eval_percase.json"
META = REPO / "cache/metadata/track_metadata_all_tracks.json"
BASE_BLIND_ZIP = REPO / "exp/inference/blind_a/r432s_targeted_subset_submission.zip"
BLIND_SOURCE_DIR = REPO / "cache/blind_a/source_cache"
BLIND_SCAN_POOL = REPO / "cache/r460_blind_scan/scan_pool.json"
OUT_DIR = REPO / "exp/eval/r498_listwise_llm"
OUT_INF_DIR = REPO / "exp/inference/blind_a/r498_listwise_llm"
SRC_KEYS = ("src_a", "src_b", "src_c", "src_d", "src_f", "src_g")


SYSTEM_PROMPT = """\
You are predicting the ACTUAL hidden ground-truth track for a music
recommendation benchmark.

This is not a normal "best recommendation" task. The benchmark target was
sampled from a real Last.fm listening session, then a model produced a response
while constrained to recommend from a hidden session pool. Therefore:
- the correct track can be a session-continuation or same-artist/listening-habit
  continuation, not only the most semantically obvious track;
- do not over-diversify away from played artists;
- exact named artist/title intent matters, but real-session continuity also
  matters;
- you may keep the current production rank-1 if it is the best hidden-GT guess.

Rank the candidates by probability of being the hidden ground truth. Output
strict JSON only:
{"ranking":["C01","C02",...],"confidence":0.0,"rationale":"short"}.
The ranking should include candidate labels only, no track IDs.
"""


USER_TEMPLATE = """\
Conversation:
{conversation}

Current production top-20 is included in the candidate list. Challenger tracks
come from deeper retrieval pools. Choose the track most likely to be the hidden
ground truth, not merely the safest user-facing recommendation.

Candidates:
{candidates}

Return strict JSON:
{{"ranking":["Cxx", "..."], "confidence":0.0-1.0, "rationale":"short"}}
"""


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def load_meta() -> dict[str, dict[str, Any]]:
    raw = _read_json(META)
    out: dict[str, dict[str, Any]] = {}
    for tid, m in raw.items():
        tags = m.get("tag_list") or []
        if isinstance(tags, str):
            tags = [tags]
        out[str(tid)] = {
            "title": _first(m.get("track_name")),
            "artist": _first(m.get("artist_name")),
            "album": _first(m.get("album_name")),
            "year": _first(m.get("release_date"))[:4],
            "tags": [str(t) for t in tags[:10]],
        }
    return out


def label_track(tid: str, meta: dict[str, dict[str, Any]]) -> str:
    m = meta.get(tid, {})
    title = m.get("title") or "?"
    artist = m.get("artist") or "?"
    album = m.get("album") or "?"
    year = f" ({m.get('year')})" if m.get("year") else ""
    tags = ", ".join(m.get("tags") or [])
    return f"{title} - {artist}{year} | album: {album} | tags: {tags}"


def conv_text_from_case(case: dict[str, Any], meta: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for h in (case.get("history") or [])[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "music":
            lines.append(f"Played: {label_track(str(content), meta)}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif content and content != "Unknown message":
            lines.append(f"Assistant: {content}")
    lines.append(f"User: {case.get('user_query', '')}")
    return "\n".join(lines)


def conv_text_from_blind_cache(cache: dict[str, Any], meta: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for h in (cache.get("history") or [])[-10:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "music":
            lines.append(f"Played: {label_track(str(content), meta)}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif content and content != "Unknown message":
            lines.append(f"Assistant: {content}")
    lines.append(f"User: {cache.get('user_query', '')}")
    return "\n".join(lines)


def tid(value: Any) -> str:
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def rrf_pool(source_lists: dict[str, list[Any]], keys: tuple[str, ...] = SRC_KEYS, k: float = 60.0) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for key in keys:
        for rank, item in enumerate(source_lists.get(key, [])):
            score[tid(item)] += 1.0 / (k + rank + 1)
    return [t for t, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def format_candidates(candidate_ids: list[str], base_top20: list[str], meta: dict[str, dict[str, Any]]) -> str:
    lines = []
    base_rank = {t: i + 1 for i, t in enumerate(base_top20)}
    for i, t in enumerate(candidate_ids, 1):
        if t in base_rank:
            origin = f"current_rank={base_rank[t]}"
        else:
            origin = "challenger"
        lines.append(f"C{i:02d} [{origin}] track_id={t} | {label_track(t, meta)}")
    return "\n".join(lines)


def make_prompt(record: dict[str, Any]) -> str:
    return USER_TEMPLATE.format(
        conversation=record["conversation"],
        candidates=record["candidates_text"],
    )


def build_record(
    *,
    row_id: str,
    case_idx: int | None,
    session_id: str,
    turn_number: int,
    conversation: str,
    base_top20: list[str],
    challengers: list[str],
    meta: dict[str, dict[str, Any]],
    gt: str | None = None,
    fold: int | None = None,
) -> dict[str, Any]:
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for t in base_top20 + challengers:
        if t and t not in seen:
            seen.add(t)
            candidate_ids.append(t)
    rec = {
        "row_id": row_id,
        "case_idx": case_idx,
        "session_id": session_id,
        "turn_number": turn_number,
        "fold": fold,
        "base_top20": base_top20,
        "candidate_ids": candidate_ids,
        "conversation": conversation,
        "candidates_text": format_candidates(candidate_ids, base_top20, meta),
    }
    if gt is not None:
        rec["gt"] = gt
        rec["base_rank"] = (base_top20.index(gt) + 1) if gt in base_top20 else 0
        rec["gt_in_slate"] = gt in candidate_ids
    rec["system"] = SYSTEM_PROMPT
    rec["prompt"] = make_prompt(rec)
    return rec


def pack_dev(args: argparse.Namespace) -> None:
    meta = load_meta()
    payload = pickle.load(open(PAYLOAD, "rb"))
    cases = payload["cases"]
    harness = _read_json(HARNESS)["results"]

    rows: list[dict[str, Any]] = []
    for h in harness:
        ci = int(h["case_idx"])
        case = cases[ci]
        base_top20 = [str(t) for t in h["topA"][:20]]
        source_lists = {key: payload[key][ci] for key in SRC_KEYS}
        pool = rrf_pool(source_lists)
        played = {str(t) for t in case.get("music_turns", [])}
        challengers = [t for t in pool if t not in base_top20 and t not in played]
        row = build_record(
            row_id=f"dev_{ci}",
            case_idx=ci,
            session_id=str(case["session_id"]),
            turn_number=int(case["turn_number"]),
            conversation=conv_text_from_case(case, meta),
            base_top20=base_top20,
            challengers=challengers[: max(0, args.slate_size - 20)],
            meta=meta,
            gt=str(case["gt"]),
            fold=int(h.get("fold", -1)),
        )
        if args.filter == "misses" and row["base_rank"] > 0:
            continue
        if args.filter == "recoverable" and not (row["base_rank"] == 0 and row["gt_in_slate"]):
            continue
        if args.filter == "hits" and row["base_rank"] == 0:
            continue
        rows.append(row)

    if args.sample_mode == "fold_balanced" and args.limit:
        by_bucket: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            fold = int(r.get("fold", -1))
            if r.get("base_rank") == 0 and r.get("gt_in_slate"):
                bucket = "recoverable"
            elif r.get("base_rank", 0) > 0:
                bucket = "hit"
            else:
                bucket = "other"
            by_bucket[(fold, bucket)].append(r)

        folds = sorted({int(r.get("fold", -1)) for r in rows})
        target_pairs = [(fold, "recoverable") for fold in folds] + [(fold, "hit") for fold in folds]
        per_pair = max(1, args.limit // max(1, len(target_pairs)))
        selected: list[dict[str, Any]] = []
        selected_ids: set[tuple[int | None, str]] = set()
        for pair in target_pairs:
            for r in by_bucket.get(pair, [])[:per_pair]:
                key = (r.get("case_idx"), r["row_id"])
                selected.append(r)
                selected_ids.add(key)

        # Fill any remainder round-robin from the same buckets, then from all
        # rows. This keeps the sample deterministic while avoiding fold skew.
        cursor = per_pair
        while len(selected) < args.limit:
            added = False
            for pair in target_pairs:
                bucket_rows = by_bucket.get(pair, [])
                if cursor < len(bucket_rows):
                    r = bucket_rows[cursor]
                    key = (r.get("case_idx"), r["row_id"])
                    if key not in selected_ids:
                        selected.append(r)
                        selected_ids.add(key)
                        added = True
                        if len(selected) >= args.limit:
                            break
            if not added:
                break
            cursor += 1
        if len(selected) < args.limit:
            for r in rows:
                key = (r.get("case_idx"), r["row_id"])
                if key not in selected_ids:
                    selected.append(r)
                    selected_ids.add(key)
                    if len(selected) >= args.limit:
                        break
        rows = sorted(selected, key=lambda r: (int(r.get("fold", -1)), r["case_idx"]))
    elif args.sample_mode == "balanced" and args.limit:
        recoverable = [r for r in rows if r.get("base_rank") == 0 and r.get("gt_in_slate")]
        hits = [r for r in rows if r.get("base_rank", 0) > 0]
        other = [r for r in rows if r not in recoverable and r not in hits]
        n_recoverable = min(len(recoverable), args.limit // 2)
        n_hits = min(len(hits), args.limit - n_recoverable)
        rows = recoverable[:n_recoverable] + hits[:n_hits]
        if len(rows) < args.limit:
            rows.extend(other[: args.limit - len(rows)])
        rows.sort(key=lambda r: r["case_idx"])
    else:
        # Deterministic high-information ordering: recoverable misses first,
        # then current hits as displacement controls.
        rows.sort(key=lambda r: (0 if (r.get("base_rank") == 0 and r.get("gt_in_slate")) else 1, r["case_idx"]))
        if args.limit:
            rows = rows[: args.limit]

    out = Path(args.out)
    _write_jsonl(out, rows)
    manifest = {
        "experiment": "R498 aggressive listwise LLM reranker",
        "kind": "dev_prompt_pack",
        "rows": len(rows),
        "slate_size": args.slate_size,
        "filter": args.filter,
        "sample_mode": args.sample_mode,
        "output": str(out.relative_to(REPO) if out.is_relative_to(REPO) else out),
        "recoverable_misses": sum(1 for r in rows if r.get("base_rank") == 0 and r.get("gt_in_slate")),
        "hits": sum(1 for r in rows if r.get("base_rank", 0) > 0),
    }
    _write_json(out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


def read_base_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def pack_blind(args: argparse.Namespace) -> None:
    meta = load_meta()
    base_rows = read_base_zip(Path(args.base_zip))
    scan_pool = _read_json(BLIND_SCAN_POOL) if BLIND_SCAN_POOL.exists() else {}
    rows: list[dict[str, Any]] = []
    for row_idx, base_row in enumerate(base_rows):
        sid = str(base_row["session_id"])
        cache_path = BLIND_SOURCE_DIR / f"{sid}.pkl"
        cache = pickle.load(open(cache_path, "rb"))
        base_top20 = [str(t) for t in base_row["predicted_track_ids"][:20]]
        played = {str(t) for t in cache.get("music_turns", [])}
        if sid in scan_pool and scan_pool[sid].get("scan_pool"):
            pool = [str(t) for t in scan_pool[sid]["scan_pool"]]
        else:
            source_lists = {key: cache.get(key, []) for key in ("src_a", "src_b", "src_c", "src_d", "src_f")}
            source_lists["src_g"] = cache.get("r21_list", [])
            pool = rrf_pool(source_lists, keys=("src_a", "src_b", "src_c", "src_d", "src_f", "src_g"))
        challengers = [t for t in pool if t not in base_top20 and t not in played]
        rows.append(build_record(
            row_id=f"blind_{row_idx:02d}_{sid[:8]}",
            case_idx=None,
            session_id=sid,
            turn_number=int(base_row["turn_number"]),
            conversation=conv_text_from_blind_cache(cache, meta),
            base_top20=base_top20,
            challengers=challengers[: max(0, args.slate_size - 20)],
            meta=meta,
        ))

    out = Path(args.out)
    _write_jsonl(out, rows)
    manifest = {
        "experiment": "R498 aggressive listwise LLM reranker",
        "kind": "blind_prompt_pack",
        "rows": len(rows),
        "slate_size": args.slate_size,
        "base_zip": str(Path(args.base_zip).relative_to(REPO)),
        "output": str(out.relative_to(REPO) if out.is_relative_to(REPO) else out),
    }
    _write_json(out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


def run_model(args: argparse.Namespace) -> None:
    rows = _read_jsonl(Path(args.prompts))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and not args.overwrite:
        for row in _read_jsonl(out_path):
            done.add(row["row_id"])

    n = 0
    with open(out_path, "a" if out_path.exists() and not args.overwrite else "w", encoding="utf-8") as f:
        for row in rows:
            if row["row_id"] in done:
                continue
            if args.limit and n >= args.limit:
                break
            text = call_llm_api(
                row["system"],
                row["prompt"],
                model=args.model,
                max_tokens=args.max_tokens,
                strict_no_truncation=True,
            )
            payload = {
                "row_id": row["row_id"],
                "case_idx": row.get("case_idx"),
                "session_id": row["session_id"],
                "model": args.model,
                "raw": text or "",
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print(f"{n}: {row['row_id']} ({'ok' if text else 'empty'})", flush=True)


def parse_ranking(raw: str, candidate_count: int) -> tuple[list[int], float | None]:
    raw = raw.strip()
    obj = None
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    ranking_raw: Any = None
    confidence = None
    if isinstance(obj, dict):
        ranking_raw = obj.get("ranking")
        try:
            confidence = float(obj.get("confidence")) if obj.get("confidence") is not None else None
        except (TypeError, ValueError):
            confidence = None
    if ranking_raw is None:
        arrays = re.findall(r"\[[\s\S]*?\]", raw)
        if arrays:
            try:
                ranking_raw = json.loads(arrays[-1])
            except json.JSONDecodeError:
                ranking_raw = None
    out: list[int] = []
    seen: set[int] = set()
    if isinstance(ranking_raw, list):
        for item in ranking_raw:
            idx: int | None = None
            if isinstance(item, str):
                m2 = re.fullmatch(r"C?(\d{1,3})", item.strip(), flags=re.I)
                if m2:
                    idx = int(m2.group(1)) - 1
            elif isinstance(item, int) and not isinstance(item, bool):
                # Accept either C01-style ordinal after JSON coercion or 0-based
                # only if the value cannot be a valid 1-based candidate label.
                idx = item - 1 if 1 <= item <= candidate_count else item
            if idx is not None and 0 <= idx < candidate_count and idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out, confidence


def ndcg_for_top(top: list[str], gt: str) -> float:
    for i, t in enumerate(top[:20]):
        if t == gt:
            return 1.0 / math.log2(i + 2)
    return 0.0


def apply_policy(record: dict[str, Any], ranking: list[int], policy: str, confidence: float | None, threshold: float) -> list[str]:
    base = list(record["base_top20"])
    cands = list(record["candidate_ids"])
    if confidence is not None and confidence < threshold:
        return base
    ranked_tracks = [cands[i] for i in ranking if 0 <= i < len(cands)]
    ranked_tracks = [t for i, t in enumerate(ranked_tracks) if t not in ranked_tracks[:i]]
    if not ranked_tracks:
        return base

    if policy == "full20":
        out = ranked_tracks + [t for t in base if t not in ranked_tracks]
        return out[:20]
    if policy == "full20_keep_top1":
        top1 = base[0]
        body = [t for t in ranked_tracks if t != top1]
        out = [top1] + body + [t for t in base[1:] if t not in body]
        return out[:20]
    if policy.startswith("top") and policy.endswith("_keep_top1"):
        k = int(policy[3:].split("_", 1)[0])
        top1 = base[0]
        # Preserve the response-aligned top recommendation; use the listwise
        # model to choose the remaining visible head positions.
        head = [t for t in ranked_tracks if t != top1][: max(0, k - 1)]
        out = [top1] + head + [t for t in base[1:] if t not in head]
        return out[:20]
    if policy.startswith("top"):
        k = int(policy[3:])
        head = ranked_tracks[:k]
        out = head + [t for t in base if t not in head]
        return out[:20]
    if policy == "challenger_top1":
        winner = ranked_tracks[0]
        if winner in base:
            return base
        return [winner] + base[:19]
    raise ValueError(f"unknown policy: {policy}")


def eval_dev(args: argparse.Namespace) -> None:
    records = {r["row_id"]: r for r in _read_jsonl(Path(args.prompts))}
    outputs = _read_jsonl(Path(args.outputs))
    out_by_id = {r["row_id"]: r for r in outputs}
    policies = args.policies.split(",")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    results = []

    for policy in policies:
        for threshold in thresholds:
            base_scores = []
            new_scores = []
            recovered = lost = improved = worsened = changed = parse_fail = 0
            by_fold: dict[int, list[float]] = defaultdict(list)
            for row_id, rec in records.items():
                gt = rec.get("gt")
                if not gt:
                    continue
                base_top = rec["base_top20"]
                base_nd = ndcg_for_top(base_top, gt)
                out = out_by_id.get(row_id)
                if not out:
                    new_top = base_top
                    parse_fail += 1
                else:
                    ranking, conf = parse_ranking(out.get("raw", ""), len(rec["candidate_ids"]))
                    if not ranking:
                        new_top = base_top
                        parse_fail += 1
                    else:
                        new_top = apply_policy(rec, ranking, policy, conf, threshold)
                new_nd = ndcg_for_top(new_top, gt)
                base_scores.append(base_nd)
                new_scores.append(new_nd)
                if new_top[:20] != base_top[:20]:
                    changed += 1
                if base_nd == 0.0 and new_nd > 0.0:
                    recovered += 1
                elif base_nd > 0.0 and new_nd == 0.0:
                    lost += 1
                if new_nd > base_nd:
                    improved += 1
                elif new_nd < base_nd:
                    worsened += 1
                by_fold[int(rec.get("fold", -1))].append(new_nd - base_nd)
            n = max(1, len(base_scores))
            fold_delta = {str(k): sum(v) / max(1, len(v)) for k, v in sorted(by_fold.items())}
            results.append({
                "policy": policy,
                "threshold": threshold,
                "rows": len(base_scores),
                "changed": changed,
                "parse_fail_or_missing": parse_fail,
                "base_ndcg": sum(base_scores) / n,
                "new_ndcg": sum(new_scores) / n,
                "delta_ndcg": (sum(new_scores) - sum(base_scores)) / n,
                "recovered": recovered,
                "lost": lost,
                "improved": improved,
                "worsened": worsened,
                "fold_delta": fold_delta,
            })

    results.sort(key=lambda r: r["delta_ndcg"], reverse=True)
    out_path = Path(args.out)
    _write_json(out_path, {"results": results})
    print(json.dumps({"best": results[:10], "out": str(out_path)}, indent=2))


def write_zip(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("prediction.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_blind(args: argparse.Namespace) -> None:
    records = {r["row_id"]: r for r in _read_jsonl(Path(args.prompts))}
    outputs = {r["row_id"]: r for r in _read_jsonl(Path(args.outputs))}
    base_rows = read_base_zip(Path(args.base_zip))
    by_sid = {r["session_id"]: r for r in records.values()}
    changed = []
    new_rows = []
    for row in base_rows:
        sid = row["session_id"]
        rec = by_sid[sid]
        out = outputs.get(rec["row_id"])
        if out:
            ranking, conf = parse_ranking(out.get("raw", ""), len(rec["candidate_ids"]))
            new_top = apply_policy(rec, ranking, args.policy, conf, args.threshold) if ranking else rec["base_top20"]
        else:
            new_top = rec["base_top20"]
        new_row = dict(row)
        old_extra = [t for t in row["predicted_track_ids"][20:] if t not in new_top]
        new_row["predicted_track_ids"] = new_top[:20] + old_extra
        if new_row["predicted_track_ids"][:20] != row["predicted_track_ids"][:20]:
            changed.append({
                "session_id": sid,
                "old_top1": row["predicted_track_ids"][0],
                "new_top1": new_row["predicted_track_ids"][0],
                "top20_overlap": len(set(row["predicted_track_ids"][:20]) & set(new_top[:20])),
            })
        new_rows.append(new_row)

    out_zip = Path(args.out)
    sha = write_zip(out_zip, new_rows)
    manifest = {
        "experiment": "R498 aggressive listwise LLM blind candidate",
        "base_zip": str(Path(args.base_zip).relative_to(REPO)),
        "prompts": str(Path(args.prompts)),
        "outputs": str(Path(args.outputs)),
        "policy": args.policy,
        "threshold": args.threshold,
        "changed_rows": len(changed),
        "changed": changed,
        "zip": str(out_zip),
        "sha256": sha,
    }
    _write_json(out_zip.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack-dev")
    p.add_argument("--out", default=str(OUT_DIR / "r498_dev_prompts.jsonl"))
    p.add_argument("--slate-size", type=int, default=60)
    p.add_argument("--filter", choices=["all", "misses", "recoverable", "hits"], default="all")
    p.add_argument("--sample-mode", choices=["default", "balanced", "fold_balanced"], default="default")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=pack_dev)

    p = sub.add_parser("pack-blind")
    p.add_argument("--base-zip", default=str(BASE_BLIND_ZIP))
    p.add_argument("--out", default=str(OUT_DIR / "r498_blind_prompts.jsonl"))
    p.add_argument("--slate-size", type=int, default=60)
    p.set_defaults(func=pack_blind)

    p = sub.add_parser("run")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=run_model)

    p = sub.add_parser("eval-dev")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--out", default=str(OUT_DIR / "r498_eval.json"))
    p.add_argument("--policies", default="top1,top3,top5,full20,challenger_top1")
    p.add_argument("--thresholds", default="0.0,0.55,0.65,0.75,0.85")
    p.set_defaults(func=eval_dev)

    p = sub.add_parser("build-blind")
    p.add_argument("--base-zip", default=str(BASE_BLIND_ZIP))
    p.add_argument("--prompts", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--policy", default="top3")
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--out", default=str(OUT_INF_DIR / "r498_listwise_llm_submission.zip"))
    p.set_defaults(func=build_blind)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

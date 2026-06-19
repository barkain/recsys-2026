#!/usr/bin/env python3
"""Build Blind-A source-fusion candidates on top of R510.

This is deliberately retrieval-only: keep each R510 top-1 and response
unchanged, then rerank positions 2-20 with deeper source lists.
"""

from __future__ import annotations

import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ZIP = (
    ROOT
    / "exp/inference/blind_a/r510_stack_r498_official_positives/"
    / "r510_r498_plus_official_positive_rows_submission.zip"
)
SOURCE_DIR = ROOT / "cache/blind_a/source_cache"
R54_PATH = ROOT / "cache/r54_production/blind_r54_lists.json"
R84_PATH = ROOT / "cache/r84_production/blind_r84_ensemble_lists.json"
OUT_DIR = ROOT / "exp/inference/blind_a/r524_source_rrf_keep1"
AUDIT_PATH = ROOT / "exp/eval/expR524_source_rrf_keep1_audit.json"


@dataclass(frozen=True)
class Config:
    name: str
    weights: dict[str, float]
    k: int = 30
    depth: int = 100


CONFIGS = [
    Config(
        name="r524_current_r21_src_keep1",
        weights={"current": 2.0, "r21": 1.0, "src_b": 1.0, "src_c": 1.0, "src_f": 0.5},
    ),
    Config(
        name="r524_current_r54_r21_src_keep1",
        weights={
            "current": 2.0,
            "r54": 1.0,
            "r21": 1.0,
            "src_b": 1.0,
            "src_c": 1.0,
            "src_f": 0.5,
        },
    ),
    Config(
        name="r524_current_r54_r84_r21_src_keep1",
        weights={
            "current": 2.0,
            "r54": 1.0,
            "r84": 1.0,
            "r21": 1.0,
            "src_b": 1.0,
            "src_c": 1.0,
            "src_f": 0.5,
        },
    ),
    Config(
        name="r524_bold_current_r54_r21_src_keep1",
        weights={
            "current": 1.0,
            "r54": 1.0,
            "r21": 1.0,
            "src_b": 1.0,
            "src_c": 1.0,
            "src_f": 1.0,
        },
        k=60,
    ),
    Config(
        name="r524_ultra_current_r54_r21_src_keep1",
        weights={
            "current": 0.5,
            "r54": 1.0,
            "r21": 1.0,
            "src_b": 1.0,
            "src_c": 1.0,
            "src_f": 1.0,
        },
        k=60,
    ),
]

CHOSEN = "r524_ultra_current_r54_r21_src_keep1"


def load_prediction(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def write_prediction_zip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)


def normalize_list(items, depth: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not items:
        return out
    for item in items:
        tid = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(tid, str) and tid not in seen:
            seen.add(tid)
            out.append(tid)
            if depth is not None and len(out) >= depth:
                break
    return out


def load_keyed_lists(path: Path) -> dict[str, list[str]]:
    obj = json.loads(path.read_text())
    lists = obj["lists"] if isinstance(obj, dict) and "lists" in obj else obj
    return {sid: normalize_list(lst, 300) for sid, lst in lists.items()}


def load_source_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in SOURCE_DIR.glob("*.pkl"):
        with path.open("rb") as f:
            obj = pickle.load(f)
        out[obj["session_id"]] = obj
    return out


def rrf(parts: list[tuple[str, list[str]]], weights: dict[str, float], k: int, depth: int) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for name, ranked in parts:
        weight = weights.get(name)
        if weight is None:
            continue
        for rank, tid in enumerate(ranked[:depth], start=1):
            if tid not in first_seen:
                first_seen[tid] = order
                order += 1
            scores[tid] = scores.get(tid, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda tid: (-scores[tid], first_seen[tid]))


def build_candidate(
    base_rows: list[dict],
    source_cache: dict[str, dict],
    r54: dict[str, list[str]],
    r84: dict[str, list[str]],
    cfg: Config,
) -> tuple[list[dict], dict]:
    new_rows: list[dict] = []
    audit_rows: list[dict] = []
    top1_churn = 0
    response_changes = 0
    changed_rows = 0
    overlap_sum = 0

    for row in base_rows:
        sid = row["session_id"]
        current = normalize_list(row["predicted_track_ids"], 20)
        cache = source_cache[sid]
        parts = [
            ("current", current),
            ("r54", r54.get(sid, [])),
            ("r84", r84.get(sid, [])),
            ("r21", normalize_list(cache.get("r21_list"), 300)),
            ("src_b", normalize_list(cache.get("src_b"), 300)),
            ("src_c", normalize_list(cache.get("src_c"), 300)),
            ("src_f", normalize_list(cache.get("src_f"), 300)),
        ]
        ranked = rrf(parts, cfg.weights, cfg.k, cfg.depth)

        # Preserve the response contract: top-1 remains R510's top recommendation.
        final = [current[0]] + [tid for tid in ranked if tid != current[0]]
        deduped: list[str] = []
        seen: set[str] = set()
        for tid in final:
            if tid not in seen:
                seen.add(tid)
                deduped.append(tid)
            if len(deduped) >= 20:
                break
        for tid in current:
            if len(deduped) >= 20:
                break
            if tid not in seen:
                seen.add(tid)
                deduped.append(tid)

        out_row = dict(row)
        out_row["predicted_track_ids"] = deduped
        new_rows.append(out_row)

        changed = deduped != current
        if changed:
            changed_rows += 1
        if deduped[0] != current[0]:
            top1_churn += 1
        if out_row["predicted_response"] != row["predicted_response"]:
            response_changes += 1
        overlap = len(set(deduped) & set(current))
        overlap_sum += overlap
        audit_rows.append(
            {
                "session_id": sid,
                "turn_number": row["turn_number"],
                "changed": changed,
                "overlap20": overlap,
                "old_top5": current[:5],
                "new_top5": deduped[:5],
            }
        )

    return new_rows, {
        "name": cfg.name,
        "k": cfg.k,
        "depth": cfg.depth,
        "weights": cfg.weights,
        "rows": len(base_rows),
        "changed_rows": changed_rows,
        "top1_churn": top1_churn,
        "response_changes": response_changes,
        "mean_overlap20": overlap_sum / len(base_rows),
        "min_overlap20": min(r["overlap20"] for r in audit_rows),
        "max_overlap20": max(r["overlap20"] for r in audit_rows),
        "row_audit": audit_rows,
    }


def main() -> None:
    base_rows = load_prediction(BASE_ZIP)
    source_cache = load_source_cache()
    r54 = load_keyed_lists(R54_PATH)
    r84 = load_keyed_lists(R84_PATH)

    missing = [r["session_id"] for r in base_rows if r["session_id"] not in source_cache]
    if missing:
        raise RuntimeError(f"Missing source cache rows: {missing[:5]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audits = []
    chosen_zip = None

    for cfg in CONFIGS:
        rows, audit = build_candidate(base_rows, source_cache, r54, r84, cfg)
        zip_path = OUT_DIR / f"{cfg.name}_submission.zip"
        write_prediction_zip(zip_path, rows)
        audit["zip_path"] = str(zip_path.relative_to(ROOT))
        audit["chosen"] = cfg.name == CHOSEN
        audits.append(audit)
        if cfg.name == CHOSEN:
            chosen_zip = zip_path

    AUDIT_PATH.write_text(json.dumps({"base_zip": str(BASE_ZIP.relative_to(ROOT)), "chosen": CHOSEN, "candidates": audits}, indent=2))
    print(f"wrote audit: {AUDIT_PATH.relative_to(ROOT)}")
    for audit in audits:
        mark = " *" if audit["chosen"] else "  "
        print(
            f"{mark} {audit['name']}: changed={audit['changed_rows']} "
            f"top1={audit['top1_churn']} resp={audit['response_changes']} "
            f"overlap={audit['mean_overlap20']:.2f} "
            f"zip={audit['zip_path']}"
        )
    print(f"chosen_zip={chosen_zip.relative_to(ROOT) if chosen_zip else None}")


if __name__ == "__main__":
    main()

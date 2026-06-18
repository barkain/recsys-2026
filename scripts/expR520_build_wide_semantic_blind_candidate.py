#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R520: deploy the R518 wide semantic residual reranker on Blind-A.

This is an nDCG-first candidate.  It uses the strongest leak-free offline
signal currently available:

  R510 base top20 + R480-style source insertion candidates + R54/R84 depth 100

Then it trains the R518 LightGBM LambdaRank model on all dev rows and applies
the best R518 policy (`blend_bw0.02_keep0`) to Blind-A.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_goal65_eval import evaluate, load_dev  # noqa: E402
from scripts.expR498_listwise_llm_reranker import read_base_zip, rrf_pool  # noqa: E402
from scripts.expR516_semantic_residual_reranker import (  # noqa: E402
    FEATURE_NAMES,
    NUM_BOOST_ROUND,
    _case_semantics,
    _track_idx_map,
    build_matrix,
    load_payload_bits,
    load_r480_rows,
    rank_features,
    unique_keep_order,
)
from scripts.expR518_wide_semantic_residual import build_wide_pools  # noqa: E402

BASE_ZIP = REPO / "exp/inference/blind_a/r510_stack_r498_official_positives/r510_r498_plus_official_positive_rows_submission.zip"
BLIND_SOURCE_DIR = REPO / "cache/blind_a/source_cache"
R54_BLIND = REPO / "cache/r54_production/blind_r54_lists.json"
R84_BLIND = REPO / "cache/r84_production/blind_r84_ensemble_lists.json"
R21_MODEL = REPO / "cache/r21_production/model"
TRACK_EMBS = REPO / "cache/r21_production/track_embeddings.npy"
OUT_DIR = REPO / "exp/inference/blind_a/r520_wide_semantic_r510"
OUT_ZIP = OUT_DIR / "r520_r510_wide_semantic_blend002_submission.zip"
OUT_JSON = REPO / "exp/eval/expR520_wide_semantic_blind_candidate.json"
OUT_MD = REPO / "docs/r520_wide_semantic_blind_candidate.md"
BLIND_QUERY_EMBS = REPO / "cache/r520_blind_wide_semantic/blind_query_embs.npy"

TOP_K = 20
BASE_WEIGHT = 0.02
R480_DEPTH = 80
R54_DEPTH = 100
R84_DEPTH = 100


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def write_zip(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("prediction.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_ranked_entries(entries: list[Any]) -> list[str]:
    out: list[str] = []
    for item in entries:
        if isinstance(item, (list, tuple)):
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def load_blind_source_maps() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    r54_raw = json.load(open(R54_BLIND))
    r84_raw = json.load(open(R84_BLIND))
    r54_lists = r54_raw.get("lists", r54_raw)
    r84_lists = r84_raw.get("lists", r84_raw)
    r54 = {str(sid): normalize_ranked_entries(vals) for sid, vals in r54_lists.items()}
    r84 = {str(sid): normalize_ranked_entries(vals) for sid, vals in r84_lists.items()}
    return r54, r84


def build_query_text_from_cache(cache: dict[str, Any]) -> str:
    parts = [
        str(h.get("content", ""))
        for h in cache.get("history", [])
        if h.get("role") == "user" and h.get("content")
    ]
    parts.append(str(cache.get("user_query", "")))
    return " ".join(parts[-3:])


def load_or_encode_blind_queries(base_rows: list[dict[str, Any]]) -> np.ndarray:
    if BLIND_QUERY_EMBS.exists():
        arr = np.load(BLIND_QUERY_EMBS)
        if arr.shape[0] == len(base_rows):
            print(f"{ts()} loaded blind query embeddings {arr.shape}", flush=True)
            return arr.astype(np.float32, copy=False)

    print(f"{ts()} encoding blind queries with frozen R21 model on CPU", flush=True)
    from sentence_transformers import SentenceTransformer

    texts = []
    for row in base_rows:
        sid = str(row["session_id"])
        cache = pickle.load(open(BLIND_SOURCE_DIR / f"{sid}.pkl", "rb"))
        texts.append(build_query_text_from_cache(cache))

    model = SentenceTransformer(str(R21_MODEL), device="cpu")
    arr = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    BLIND_QUERY_EMBS.parent.mkdir(parents=True, exist_ok=True)
    np.save(BLIND_QUERY_EMBS, arr)
    del model
    gc.collect()
    print(f"{ts()} wrote blind query embeddings {BLIND_QUERY_EMBS}", flush=True)
    return arr


def train_all_dev_model() -> tuple[lgb.Booster, dict[str, Any]]:
    print(f"{ts()} loading dev and building R518 wide pools", flush=True)
    dev = load_dev()
    r480_rows = load_r480_rows()
    played, track_artist = load_payload_bits()
    cfg = {"name": "r48080_r54r84_100", "r480_depth": R480_DEPTH, "r54_depth": R54_DEPTH, "r84_depth": R84_DEPTH}
    pools, pool_stats = build_wide_pools(dev, r480_rows, cfg)
    base_metrics = evaluate(dev, dev["lr_top20"])
    X, y, _starts, counts = build_matrix(dev, pools, r480_rows, played, track_artist)
    print(f"{ts()} train matrix {X.shape}, positives={int(y.sum())}, pool_hit={pool_stats['pool_hit']:.4f}", flush=True)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "label_gain": [0, 1],
        "num_leaves": 31,
        "learning_rate": 0.035,
        "min_data_in_leaf": 35,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": 4,
        "force_col_wise": True,
        "deterministic": True,
        "seed": 520,
    }
    dtrain = lgb.Dataset(X, label=y, group=[int(c) for c in counts], feature_name=FEATURE_NAMES)
    model = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, callbacks=[lgb.log_evaluation(0)])
    report = {
        "params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "pool_stats": pool_stats,
        "base_dev_metrics": base_metrics,
        "feature_importance": {
            name: float(val)
            for name, val in sorted(
                zip(FEATURE_NAMES, model.feature_importance(importance_type="gain"), strict=True),
                key=lambda x: -x[1],
            )
        },
    }
    del X, y, dtrain, pools, r480_rows, played
    gc.collect()
    return model, report


def build_blind_pools(
    base_rows: list[dict[str, Any]],
    r54: dict[str, list[str]],
    r84: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]], list[list[str]], list[int]]:
    pools: list[list[str]] = []
    base_top20s: list[list[str]] = []
    ins_lists: list[list[str]] = []
    r54_lists: list[list[str]] = []
    r84_lists: list[list[str]] = []
    n_prior: list[int] = []
    for row in base_rows:
        sid = str(row["session_id"])
        cache = pickle.load(open(BLIND_SOURCE_DIR / f"{sid}.pkl", "rb"))
        base = [str(t) for t in row["predicted_track_ids"][:TOP_K]]
        source_lists = {key: cache.get(key, []) for key in ("src_a", "src_b", "src_c", "src_d", "src_f")}
        source_lists["src_g"] = cache.get("r21_list", [])
        full_rrf = rrf_pool(source_lists, keys=("src_a", "src_b", "src_c", "src_d", "src_f", "src_g"))
        ins = [t for t in full_rrf if t not in set(base)][:R480_DEPTH]
        r54_top = list(r54.get(sid, []))[:R54_DEPTH]
        r84_top = list(r84.get(sid, []))[:R84_DEPTH]
        pool = unique_keep_order(base + ins + r54_top + r84_top)
        pools.append(pool)
        base_top20s.append(base)
        ins_lists.append(ins)
        r54_lists.append(r54_top)
        r84_lists.append(r84_top)
        n_prior.append(len(cache.get("music_turns", [])))
    return pools, base_top20s, ins_lists, r54_lists, r84_lists, n_prior


def build_blind_matrix(
    base_rows: list[dict[str, Any]],
    pools: list[list[str]],
    base_top20s: list[list[str]],
    ins_lists: list[list[str]],
    r54_lists: list[list[str]],
    r84_lists: list[list[str]],
    n_prior: list[int],
    query_embs: np.ndarray,
    track_artist: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track_embs = np.load(TRACK_EMBS, mmap_mode="r")
    tid_to_idx = _track_idx_map()
    n_rows = sum(len(p) for p in pools)
    X = np.zeros((n_rows, len(FEATURE_NAMES)), dtype=np.float32)
    starts = np.zeros(len(pools), dtype=np.int64)
    counts = np.zeros(len(pools), dtype=np.int32)
    name_to_idx = {name: j for j, name in enumerate(FEATURE_NAMES)}
    row_pos = 0
    for i, pool in enumerate(pools):
        sid = str(base_rows[i]["session_id"])
        cache = pickle.load(open(BLIND_SOURCE_DIR / f"{sid}.pkl", "rb"))
        starts[i] = row_pos
        counts[i] = len(pool)
        base_rank = {t: r for r, t in enumerate(base_top20s[i], 1)}
        ins_rank = {t: r for r, t in enumerate(ins_lists[i], 1)}
        r54_rank = {t: r for r, t in enumerate(r54_lists[i], 1)}
        r84_rank = {t: r for r, t in enumerate(r84_lists[i], 1)}
        played = [str(t) for t in cache.get("music_turns", [])]
        hist_artists = [track_artist.get(t, "") for t in played]
        hist_counter = Counter(a for a in hist_artists if a)
        last_artist = hist_artists[-1] if hist_artists else ""
        semantic = _case_semantics(i, pool, played, tid_to_idx, query_embs, track_embs)
        for tid in pool:
            vals: dict[str, float] = {}
            br = base_rank.get(tid, -1)
            ir = ins_rank.get(tid, -1)
            r54r = r54_rank.get(tid, -1)
            r84r = r84_rank.get(tid, -1)
            for k, v in rank_features(br, "base").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(ir, "ins").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(r54r, "r54").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(r84r, "r84").items():
                if k in name_to_idx:
                    vals[k] = v
            present = [r for r in (br, ir, r54r, r84r) if r > 0]
            best = min(present) if present else 999
            worst = max(present) if present else 999
            vals.update({
                "n_sources_present": float(len(present)),
                "best_rank": float(best),
                "best_rank_inv": float(1.0 / best if best > 0 and best < 999 else 0.0),
                "rank_spread": float(worst - best if present else 0.0),
                "not_in_base_any_source": float(br <= 0 and any(r > 0 for r in (ir, r54r, r84r))),
                "n_prior_music": float(n_prior[i]),
            })
            artist = track_artist.get(tid, "")
            vals["last_artist_match"] = float(bool(artist and artist == last_artist))
            vals["any_artist_match"] = float(bool(artist and artist in hist_counter))
            vals["artist_history_count"] = float(hist_counter.get(artist, 0)) if artist else 0.0
            vals.update(semantic[tid])
            for name, val in vals.items():
                if name in name_to_idx:
                    X[row_pos, name_to_idx[name]] = val
            row_pos += 1
    return X, starts, counts


def rank_blind(
    pools: list[list[str]],
    scores: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    base_top20s: list[list[str]],
) -> list[list[str]]:
    out: list[list[str]] = []
    for i, pool in enumerate(pools):
        start = int(starts[i])
        count = int(counts[i])
        case_scores = scores[start:start + count]
        base_rank = {t: r for r, t in enumerate(base_top20s[i], 1)}
        vals = np.asarray(
            [
                float(case_scores[j]) + (BASE_WEIGHT / base_rank[tid] if tid in base_rank else 0.0)
                for j, tid in enumerate(pool)
            ],
            dtype=np.float32,
        )
        ranked = [pool[int(j)] for j in np.argsort(-vals, kind="mergesort")]
        out.append(ranked[:TOP_K])
    return out


def preflight(base_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], pools: list[list[str]]) -> dict[str, Any]:
    changed_rows = 0
    top1_churn = 0
    overlaps = []
    duplicate_rows = []
    for i, (old, new) in enumerate(zip(base_rows, new_rows, strict=True)):
        old_tracks = [str(t) for t in old["predicted_track_ids"][:TOP_K]]
        new_tracks = [str(t) for t in new["predicted_track_ids"][:TOP_K]]
        if old_tracks != new_tracks:
            changed_rows += 1
        if old_tracks[0] != new_tracks[0]:
            top1_churn += 1
        overlaps.append(len(set(old_tracks) & set(new_tracks)))
        if len(set(new_tracks)) != TOP_K:
            duplicate_rows.append(i)
    return {
        "rows": len(new_rows),
        "changed_rows": changed_rows,
        "top1_churn": top1_churn,
        "mean_overlap20_vs_base": float(np.mean(overlaps)),
        "min_overlap20_vs_base": int(min(overlaps)),
        "duplicate_rows": duplicate_rows,
        "mean_pool_size": float(np.mean([len(p) for p in pools])),
        "response_changes": sum(
            1
            for old, new in zip(base_rows, new_rows, strict=True)
            if old.get("predicted_response") != new.get("predicted_response")
        ),
    }


def main() -> None:
    print(f"{ts()} R520 wide semantic blind candidate", flush=True)
    model, train_report = train_all_dev_model()
    base_rows = read_base_zip(BASE_ZIP)
    r54, r84 = load_blind_source_maps()
    pools, base_top20s, ins_lists, r54_lists, r84_lists, n_prior = build_blind_pools(base_rows, r54, r84)
    _, track_artist = load_payload_bits()
    query_embs = load_or_encode_blind_queries(base_rows)
    print(f"{ts()} building blind matrix", flush=True)
    Xb, starts, counts = build_blind_matrix(
        base_rows, pools, base_top20s, ins_lists, r54_lists, r84_lists, n_prior, query_embs, track_artist
    )
    print(f"{ts()} blind matrix {Xb.shape}; predicting", flush=True)
    scores = model.predict(Xb).astype(np.float32)
    rankings = rank_blind(pools, scores, starts, counts, base_top20s)

    new_rows: list[dict[str, Any]] = []
    for row, ranking in zip(base_rows, rankings, strict=True):
        new_row = dict(row)
        new_row["predicted_track_ids"] = ranking
        new_rows.append(new_row)

    pf = preflight(base_rows, new_rows, pools)
    if pf["rows"] != 80 or pf["duplicate_rows"]:
        raise RuntimeError(f"preflight failed: {pf}")
    sha = write_zip(OUT_ZIP, new_rows)
    audit = {
        "experiment": "R520 wide semantic residual blind candidate",
        "created_at": datetime.now().isoformat(),
        "base_zip": str(BASE_ZIP),
        "output_zip": str(OUT_ZIP),
        "sha256": sha,
        "policy": {
            "dev_source": "R518 best policy",
            "config": "r48080_r54r84_100",
            "base_weight": BASE_WEIGHT,
            "keep_top1": False,
            "r480_depth": R480_DEPTH,
            "r54_depth": R54_DEPTH,
            "r84_depth": R84_DEPTH,
        },
        "preflight": pf,
        "train_report": train_report,
        "note": "nDCG-first candidate; responses are preserved from R510, so LLM may move if top-1 changes.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2))
    OUT_MD.write_text(
        "\n".join([
            "# R520 - Wide Semantic Blind Candidate",
            "",
            f"**Date:** {datetime.now():%Y-%m-%d}",
            "**Intent:** nDCG-first deployment of the R518 wide semantic residual reranker on top of R510.",
            "",
            "## Candidate",
            "",
            f"- zip: `{OUT_ZIP}`",
            f"- sha256: `{sha}`",
            f"- policy: `blend_bw0.02_keep0`, R480 depth `{R480_DEPTH}`, R54/R84 depth `{R54_DEPTH}`",
            "",
            "## Preflight",
            "",
            f"- rows: `{pf['rows']}`",
            f"- changed rows: `{pf['changed_rows']}`",
            f"- top-1 churn: `{pf['top1_churn']}/80`",
            f"- mean overlap@20 vs R510: `{pf['mean_overlap20_vs_base']:.3f}`",
            f"- response changes: `{pf['response_changes']}`",
            "",
            "This candidate is not composite-safe by design; it keeps R510 responses while changing rankings.",
            f"Full audit JSON: `{OUT_JSON}`",
        ])
        + "\n"
    )
    print(json.dumps({"zip": str(OUT_ZIP), "sha256": sha, "preflight": pf}, indent=2), flush=True)


if __name__ == "__main__":
    main()

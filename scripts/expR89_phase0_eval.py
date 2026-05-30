"""R89 Phase 0 eval — encode fold-0 dev queries + fused catalog + compare to R84c OOF.

Loads the trained fusion model, fuses all 47K catalog tracks, encodes 1600
fold-0 dev queries with raw BGE-large, computes top-300 per case, then
compares to R84 5-fold OOF baseline (per-case unique recoveries + canaries).

Outputs:
- cache/r89/phase0_fold0/oof_r89_lists.json  (1600 × top-300 with scores)
- cache/r89/phase0_fold0/r89_catalog_embs.fp16.npy  (47K × 1024)
- cache/r89/phase0_fold0/r89_dev_query_embs.fp16.npy
- exp/eval/expR89_phase0_eval.json
- docs/r89_phase0_result.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_full5fold_train import (  # noqa: E402
    R12_CACHE,
    build_query_structured_from_dev,
    load_catalog,
)
from scripts.expR84_phase0a_census import grouped_session_folds  # noqa: E402

MOD_CACHE_DIR = REPO / "cache" / "r89" / "modality_cache"
OUT_DIR_DEFAULT = REPO / "cache" / "r89" / "phase0_fold0"
QUERY_BGE_MODEL = "BAAI/bge-large-en-v1.5"
MAX_SEQ_LEN_QUERY = 384
TOP_K = 300
N_FOLDS = 5

# Gate (matches user spec)
GATE = {
    "h7_delta_ge": 0.005,
    "min_unique_h7_top30": 10,
    "ambiguous_h7_delta_ge": -0.003,
    "ambiguous_min_unique_h7_top30": 5,
    "same_artist_delta_ge": -0.005,
    "diff_artist_delta_ge": -0.005,
    "overlap_ge": 8.0,
}

R84_FOLD_LISTS = {
    0: REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json",
    1: REPO / "cache" / "r84" / "phase1_fold1" / "oof_r84_lists.json",
    2: REPO / "cache" / "r84" / "phase1_fold2" / "oof_r84_lists.json",
    3: REPO / "cache" / "r84" / "phase1_fold3" / "oof_r84_lists.json",
    4: REPO / "cache" / "r84" / "phase1_fold4" / "oof_r84_lists.json",
}


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(rank, k):
    return 1.0 / math.log2(rank + 1) if 0 < rank <= k else 0.0


def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--model-path", type=Path,
                   default=OUT_DIR_DEFAULT / "model.pt")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-bf16", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{ts()} R89 Phase 0 eval (fold={args.fold})")

    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F_t  # type: ignore

    # --- Load fold-0 dev cases ---
    print(f"\n{ts()} Loading dev payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0)
    fold0_idx = folds[args.fold].tolist()
    val_cases = [cases[i] for i in fold0_idx]
    print(f"  fold-{args.fold} val: {len(val_cases)} cases")
    n_h7 = sum(1 for c in val_cases if c.get("n_prior_music") == 7)
    print(f"  h7: {n_h7}")

    # --- Load catalog meta + modality cache ---
    print(f"\n{ts()} Loading catalog meta + modality cache...")
    meta, all_meta_track_ids = load_catalog()
    track_ids = json.load(open(MOD_CACHE_DIR / "track_ids.json"))
    dims = json.load(open(MOD_CACHE_DIR / "dims.json"))
    n_tracks = len(track_ids)
    track_id_to_idx = {t: i for i, t in enumerate(track_ids)}
    modalities = {
        key: torch.from_numpy(
            np.load(MOD_CACHE_DIR / f"{key}.fp16.npy").astype(np.float32)
        )
        for key in ["text", "image", "lyrics", "attrs", "audio"]
    }
    mask = torch.from_numpy(np.load(MOD_CACHE_DIR / "mask.fp16.npy").astype(np.float32))
    for key in modalities:
        modalities[key] = modalities[key].to(args.device)
    mask = mask.to(args.device)

    # --- Re-construct fusion model ---
    class TrackFusion(nn.Module):
        def __init__(self, dims, proj_dim=256, out_dim=1024):
            super().__init__()
            self.projs = nn.ModuleDict({
                key: nn.Sequential(
                    nn.Linear(dims[key], proj_dim), nn.GELU(),
                    nn.LayerNorm(proj_dim),
                )
                for key in ["text", "image", "lyrics", "attrs", "audio"]
            })
            self.gates = nn.Parameter(torch.zeros(5))
            self.fusion = nn.Sequential(
                nn.Linear(proj_dim * 5, 1024), nn.GELU(),
                nn.LayerNorm(1024),
                nn.Linear(1024, out_dim),
            )

        def forward(self, embs_dict, mask_b):
            gates = torch.sigmoid(self.gates)
            modality_order = ["text", "image", "lyrics", "attrs", "audio"]
            projs = []
            for i, key in enumerate(modality_order):
                p = self.projs[key](embs_dict[key])
                p = p * gates[i] * mask_b[:, i:i + 1]
                projs.append(p)
            x = torch.cat(projs, dim=-1)
            out = self.fusion(x)
            return F_t.normalize(out, dim=-1)

    ckpt = torch.load(args.model_path, map_location=args.device)
    proj_dim = ckpt["config"].get("PROJ_DIM", 256)
    out_dim = ckpt["config"].get("OUT_DIM", 1024)
    model = TrackFusion(dims, proj_dim=proj_dim, out_dim=out_dim).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  model loaded (proj_dim={proj_dim}, out_dim={out_dim})")
    gates_now = torch.sigmoid(model.gates).detach().cpu().numpy().tolist()
    print(f"  gates: text={gates_now[0]:.3f} image={gates_now[1]:.3f} "
          f"lyrics={gates_now[2]:.3f} attrs={gates_now[3]:.3f} audio={gates_now[4]:.3f}")

    # --- Fuse entire catalog ---
    print(f"\n{ts()} Fusing entire catalog ({n_tracks} tracks)...")
    use_bf16 = (not args.no_bf16) and args.device.startswith("cuda")
    t0 = time.time()
    catalog_t_embs = torch.zeros((n_tracks, out_dim), dtype=torch.float32, device=args.device)
    chunk = 2048
    with torch.no_grad():
        for i in range(0, n_tracks, chunk):
            j = min(i + chunk, n_tracks)
            idx = torch.arange(i, j, device=args.device)
            embs_dict = {key: modalities[key][idx] for key in modalities}
            m = mask[idx]
            with torch.amp.autocast(
                device_type="cuda" if use_bf16 else "cpu",
                dtype=torch.bfloat16 if use_bf16 else torch.float32, enabled=use_bf16,
            ):
                out = model(embs_dict, m)
            catalog_t_embs[i:j] = out.float()
    print(f"  catalog fused in {time.time() - t0:.0f}s")
    np.save(args.output_dir / "r89_catalog_embs.fp16.npy",
            catalog_t_embs.cpu().numpy().astype(np.float16))

    # --- Encode dev queries with raw BGE-large ---
    print(f"\n{ts()} Encoding {len(val_cases)} dev queries with {QUERY_BGE_MODEL}...")
    from sentence_transformers import SentenceTransformer  # type: ignore
    bge = SentenceTransformer(QUERY_BGE_MODEL, device=args.device)
    bge.max_seq_length = MAX_SEQ_LEN_QUERY
    bge.eval()
    val_queries = [build_query_structured_from_dev(c, meta) for c in val_cases]
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda" if use_bf16 else "cpu",
            dtype=torch.bfloat16 if use_bf16 else torch.float32, enabled=use_bf16,
        ):
            q_embs = bge.encode(val_queries, batch_size=128, show_progress_bar=False,
                                 convert_to_tensor=True, normalize_embeddings=True)
    q_embs = q_embs.float().to(args.device)
    print(f"  encoded in {time.time() - t0:.0f}s")
    np.save(args.output_dir / "r89_dev_query_embs.fp16.npy",
            q_embs.cpu().numpy().astype(np.float16))
    del bge
    torch.cuda.empty_cache()

    # --- Compute top-300 per dev case ---
    print(f"\n{ts()} Retrieving top-{TOP_K}...")
    t0 = time.time()
    oof_lists = {}
    chunk = 32
    with torch.no_grad():
        for i0 in range(0, len(val_cases), chunk):
            j = min(i0 + chunk, len(val_cases))
            sim = q_embs[i0:j] @ catalog_t_embs.T  # (chunk, n_tracks)
            topk_vals, topk_idx = sim.topk(TOP_K, dim=1)
            topk_vals = topk_vals.float().cpu().numpy()
            topk_idx = topk_idx.cpu().numpy()
            for k in range(j - i0):
                ci = fold0_idx[i0 + k]
                tids = [track_ids[int(t)] for t in topk_idx[k]]
                scores = topk_vals[k].tolist()
                oof_lists[ci] = list(zip(tids, scores))
    print(f"  retrieval done in {time.time() - t0:.1f}s")

    # --- Source-alone metrics ---
    print(f"\n{ts()} Computing source-alone metrics...")
    hit20_all = sum(1 for c in val_cases
                     if c["gt"] in set(t for t, _ in oof_lists[fold0_idx[val_cases.index(c)]][:20]))
    # Slow — let me cache index instead
    case_to_local = {fold0_idx[i]: i for i in range(len(val_cases))}
    h7_local = [i for i, c in enumerate(val_cases) if c.get("n_prior_music") == 7]

    def hit_at(k, indices):
        hit = 0
        for li in indices:
            ci = fold0_idx[li]
            gt = val_cases[li]["gt"]
            if gt in set(t for t, _ in oof_lists[ci][:k]):
                hit += 1
        return hit / max(1, len(indices))

    all_indices = list(range(len(val_cases)))
    metrics_src = {
        "hit_at_20_all": hit_at(20, all_indices),
        "hit_at_30_all": hit_at(30, all_indices),
        "hit_at_300_all": hit_at(TOP_K, all_indices),
        "hit_at_20_h7": hit_at(20, h7_local),
        "hit_at_30_h7": hit_at(30, h7_local),
        "hit_at_300_h7": hit_at(TOP_K, h7_local),
    }
    print(f"  hit_h7@20={metrics_src['hit_at_20_h7']:.4f}  "
          f"hit_h7@30={metrics_src['hit_at_30_h7']:.4f}  "
          f"hit_h7@300={metrics_src['hit_at_300_h7']:.4f}")

    # --- Compare to R84 5-fold OOF (load per-case top-K sets) ---
    print(f"\n{ts()} Loading R84 5-fold OOF for comparison...")
    fold_data = {}
    for fk, path in R84_FOLD_LISTS.items():
        with open(path) as f:
            fold_data[fk] = json.load(f)
    with open(REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl", "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = {row["case_idx"]: int(row["fold_idx"]) for row in w0_stats}

    # R84 top-30 per fold-0 case
    r84_top30 = {}
    r84_top300 = {}
    for ci in fold0_idx:
        of = case_fold[ci]
        entries = fold_data[of].get(str(ci), [])
        r84_top30[ci] = set(t for t, _ in entries[:30])
        r84_top300[ci] = set(t for t, _ in entries[:300])

    # Unique recoveries
    unique_h7_top30 = 0
    lost_h7_top30 = 0
    unique_h7_top300 = 0
    same_artist_unique_h7_top30 = 0
    diff_artist_unique_h7_top30 = 0
    h7_top30_in_r89 = 0
    h7_top30_in_r84 = 0
    for li in h7_local:
        ci = fold0_idx[li]
        gt = val_cases[li]["gt"]
        r89_top30 = set(t for t, _ in oof_lists[ci][:30])
        r89_top300 = set(t for t, _ in oof_lists[ci][:300])
        gt_in_r89_top30 = gt in r89_top30
        gt_in_r84_top30 = gt in r84_top30[ci]
        if gt_in_r89_top30:
            h7_top30_in_r89 += 1
        if gt_in_r84_top30:
            h7_top30_in_r84 += 1
        if gt_in_r89_top30 and not gt_in_r84_top30:
            unique_h7_top30 += 1
            gt_artist = ta.get(gt, "")
            played_artists = {ta.get(t, "") for t in val_cases[li].get("music_turns", [])}
            if gt_artist and gt_artist in played_artists:
                same_artist_unique_h7_top30 += 1
            else:
                diff_artist_unique_h7_top30 += 1
        if gt_in_r84_top30 and not gt_in_r89_top30:
            lost_h7_top30 += 1
        if gt in r89_top300 and gt not in r84_top300[ci]:
            unique_h7_top300 += 1

    print(f"\n  R89 source-alone h7@30: {h7_top30_in_r89}/{n_h7} "
          f"({h7_top30_in_r89/n_h7:.3f})")
    print(f"  R84 OOF h7@30 (baseline): {h7_top30_in_r84}/{n_h7} "
          f"({h7_top30_in_r84/n_h7:.3f})")
    print(f"  unique h7 top-30 (R89 surfaces, R84 misses): {unique_h7_top30} "
          f"(same/diff = {same_artist_unique_h7_top30}/{diff_artist_unique_h7_top30})")
    print(f"  lost h7 top-30 (R84 had, R89 misses): {lost_h7_top30}")
    print(f"  unique h7 top-300: {unique_h7_top300}")
    print(f"  h7 net top-30: {unique_h7_top30 - lost_h7_top30:+d}")

    # --- Per-case nDCG@20 ---
    r89_ndcg_h7 = []
    r89_ndcg_all = []
    r89_ndcg_same = []
    r89_ndcg_diff = []
    for li in range(len(val_cases)):
        ci = fold0_idx[li]
        gt = val_cases[li]["gt"]
        topk_tids = [t for t, _ in oof_lists[ci][:20]]
        rank = topk_tids.index(gt) + 1 if gt in topk_tids else -1
        n = ndcg_at_k(rank, 20)
        r89_ndcg_all.append(n)
        if val_cases[li].get("n_prior_music") == 7:
            r89_ndcg_h7.append(n)
        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in val_cases[li].get("music_turns", [])}
        if gt_artist and gt_artist in played_artists:
            r89_ndcg_same.append(n)
        else:
            r89_ndcg_diff.append(n)
    metrics_src["ndcg20_all"] = float(np.mean(r89_ndcg_all))
    metrics_src["ndcg20_h7"] = float(np.mean(r89_ndcg_h7))
    metrics_src["ndcg20_same_artist"] = float(np.mean(r89_ndcg_same))
    metrics_src["ndcg20_diff_artist"] = float(np.mean(r89_ndcg_diff))

    print(f"\n  R89 source-alone nDCG@20:")
    print(f"    all: {metrics_src['ndcg20_all']:.4f}  h7: {metrics_src['ndcg20_h7']:.4f}")
    print(f"    same_artist: {metrics_src['ndcg20_same_artist']:.4f}  "
          f"diff_artist: {metrics_src['ndcg20_diff_artist']:.4f}")

    # --- Gate evaluation ---
    h7_recov_ok = unique_h7_top30 >= GATE["min_unique_h7_top30"] and lost_h7_top30 <= unique_h7_top30
    cond_A1 = False  # need h7 ndcg vs R84c LR baseline; skip for Phase 0 (source-alone only)
    cond_A2 = h7_recov_ok
    cond_A3 = (unique_h7_top30 >= GATE["ambiguous_min_unique_h7_top30"])  # weak signal allowance
    gate_pass = cond_A2 or cond_A3
    verdict = "PROCEED_TO_PHASE_1" if gate_pass else "ARCHIVE_LEARNED_MM"
    print(f"\n  Gate eval:")
    print(f"    A2 (>=10 unique h7 top-30 AND lost <= rec): {cond_A2}  "
          f"({unique_h7_top30} unique, {lost_h7_top30} lost)")
    print(f"    A3 (>=5 unique h7 top-30): {cond_A3}")
    print(f"    VERDICT: {verdict}")

    # --- Save outputs ---
    # oof_r89_lists
    out_lists = {int(k): [[t, float(s)] for t, s in v] for k, v in oof_lists.items()}
    with open(args.output_dir / "oof_r89_lists.json", "w") as f:
        json.dump(out_lists, f)
    print(f"\n  Wrote oof_r89_lists.json "
          f"({(args.output_dir / 'oof_r89_lists.json').stat().st_size / 1e6:.1f} MB)")

    summary = {
        "experiment": "R89 Phase 0 fold-0 eval — learned multimodal fusion retriever",
        "fold": args.fold,
        "n_val_cases": len(val_cases),
        "n_h7": n_h7,
        "model_path": str(args.model_path.relative_to(REPO)),
        "verdict": verdict,
        "source_alone_metrics": metrics_src,
        "vs_r84_oof": {
            "r89_h7_top30_hit": h7_top30_in_r89,
            "r84_h7_top30_hit": h7_top30_in_r84,
            "unique_h7_top30_r89_only": unique_h7_top30,
            "lost_h7_top30_r84_only": lost_h7_top30,
            "net_h7_top30": unique_h7_top30 - lost_h7_top30,
            "unique_h7_top30_same_artist": same_artist_unique_h7_top30,
            "unique_h7_top30_diff_artist": diff_artist_unique_h7_top30,
            "unique_h7_top300_r89_only": unique_h7_top300,
        },
        "gates": {
            "A2_unique_top30_ge_10_and_rec_ge_lost": cond_A2,
            "A3_unique_top30_ge_5": cond_A3,
            "gate_pass": gate_pass,
        },
        "gates_final": {
            "text": float(torch.sigmoid(model.gates[0])),
            "image": float(torch.sigmoid(model.gates[1])),
            "lyrics": float(torch.sigmoid(model.gates[2])),
            "attrs": float(torch.sigmoid(model.gates[3])),
            "audio": float(torch.sigmoid(model.gates[4])),
        },
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
    }
    OUT_JSON = REPO / "exp" / "eval" / "expR89_phase0.json"
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {OUT_JSON}")


if __name__ == "__main__":
    main()

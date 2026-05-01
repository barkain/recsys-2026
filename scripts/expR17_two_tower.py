#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R17: learned session-to-track two-tower retriever diagnostic.

This is intentionally local-only: no blind inference, no response generation.

Goal
----
Test whether a learned score over frozen metadata embeddings can make hard
candidate tracks rankable. Prior R13-R16 experiments admitted candidates into
the pool, but LambdaRank could not consistently promote them. R17 trains a
session tower and a candidate tower directly on next-track supervision.

Stage-1 gates are diagnostic:
  * top-200 hit lift over V3
  * unique GT hits beyond V3 pool
  * unique unreachable GT hits
  * pop=0 / different-artist / hist_0 recovery
  * GT rank distribution under the learned score
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
QWEN_DIR = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"
R13_QUERY_DIR = REPO_ROOT / "cache" / "r13_query_emb"
OUT_PATH = REPO_ROOT / "exp" / "eval" / "expR17_two_tower.json"

RRF_K = 20
SOURCE_WEIGHTS = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "D": 0.5,
    "F": 1.0,
    "ALS": 1.0,
}


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def load_qwen_catalog() -> tuple[list[str], np.ndarray, dict[str, int]]:
    track_ids = json.load(open(QWEN_DIR / "track_ids.json"))
    matrix = np.load(QWEN_DIR / "vectors.npy").astype(np.float32)
    matrix = l2_normalize_np(matrix)
    return track_ids, matrix, {tid: i for i, tid in enumerate(track_ids)}


def load_catalog_scalars(track_ids: list[str]) -> np.ndarray:
    """Return lightweight scalar metadata for candidate tower.

    Columns:
      0 catalog popularity / 100
      1 train log-count normalized
      2 release year centered/scaled
      3 duration minutes / 10
      4 tag count / 50
    """
    from datasets import Dataset

    track_to_pos = {tid: i for i, tid in enumerate(track_ids)}
    scalars = np.zeros((len(track_ids), 5), dtype=np.float32)

    train_pop = build_train_popularity_stats()
    max_log = math.log1p(max(train_pop.values()) if train_pop else 1)
    for tid, cnt in train_pop.items():
        idx = track_to_pos.get(tid)
        if idx is not None and max_log > 0:
            scalars[idx, 1] = math.log1p(cnt) / max_log

    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(
        hf_cache.glob(
            "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            "talk_play_data-challenge-track-metadata-all_tracks.arrow"
        )
    )
    if not matches:
        return scalars

    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    for i, tid in enumerate(cols["track_id"]):
        idx = track_to_pos.get(str(tid))
        if idx is None:
            continue
        pop = cols.get("popularity", [0])[i] or 0
        scalars[idx, 0] = float(pop) / 100.0

        release = cols.get("release_date", [""])[i] or ""
        try:
            year = int(str(release)[:4])
            scalars[idx, 2] = (year - 1990.0) / 40.0
        except Exception:
            scalars[idx, 2] = 0.0

        dur = cols.get("duration", [0])[i] or 0
        scalars[idx, 3] = min(float(dur) / 600_000.0, 2.0)

        tags = cols.get("tag_list", [[]])[i] or []
        scalars[idx, 4] = min(len(tags), 50) / 50.0

    return scalars


def load_train_dataset_from_cache():
    """Load train split directly from the Arrow cache without HF lock files."""
    from datasets import Dataset

    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(
        hf_cache.glob(
            "talkpl-ai___talk_play_data-challenge-dataset/default/*/*/"
            "talk_play_data-challenge-dataset-train.arrow"
        )
    )
    if not matches:
        raise FileNotFoundError("Cached train arrow file not found")
    return Dataset.from_file(str(matches[-1]))


def build_train_popularity_stats() -> Counter:
    train = load_train_dataset_from_cache()
    counts = Counter()
    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                counts[str(c["content"]).strip()] += 1
    return counts


def build_als_from_cache():
    """Train ALS from cached train Arrow without invoking HF load_dataset."""
    train = load_train_dataset_from_cache()
    track_set = set()
    session_tracks = []
    for item in train:
        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                track_set.add(tid)
        session_tracks.append(tracks)

    track_ids = sorted(track_set)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}
    rows, cols, vals = [], [], []
    for si, tracks in enumerate(session_tracks):
        for tid in tracks:
            rows.append(si)
            cols.append(track_to_idx[tid])
            vals.append(1.0)
    matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(session_tracks), len(track_ids)),
        dtype=np.float32,
    )

    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=128,
        alpha=100,
        regularization=0.05,
        iterations=20,
        random_state=42,
        use_gpu=False,
    )
    model.fit(matrix)
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    elif not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    return factors, track_ids, track_to_idx


def build_als_source(cases: list[dict], topk: int = 200):
    factors, track_ids, track_to_idx = build_als_from_cache()
    out: list[list[str]] = []
    vecs = []
    for c in cases:
        sv = als_session_vector(c["music_turns"], track_to_idx, factors)
        vecs.append(sv)
        if sv is None:
            out.append([])
            continue
        scores = factors @ sv
        for tid in c["music_turns"]:
            idx = track_to_idx.get(tid)
            if idx is not None:
                scores[idx] = -np.inf
        k = min(topk, len(scores) - 1)
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out.append([track_ids[j] for j in top_idx])
    return out, vecs


def build_v3_pools(payload: dict, als_source: list[list[str]], pool_k: int = 200) -> list[list[str]]:
    pools = []
    for i in range(len(payload["cases"])):
        src_lists = {
            "A": payload["src_a"][i],
            "B": payload["src_b"][i],
            "C": payload["src_c"][i],
            "D": payload["src_d"][i],
            "F": payload["src_f"][i],
            "ALS": als_source[i],
        }
        pools.append(weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=pool_k, k=RRF_K))
    return pools


def build_session_arrays(
    cases: list[dict],
    q_current: np.ndarray,
    q_context: np.ndarray,
    track_matrix: np.ndarray,
    track_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-size session-side inputs.

    played_mean and last_track are zeros when no valid played embedding exists.
    """
    n = len(cases)
    dim = track_matrix.shape[1]
    played_mean = np.zeros((n, dim), dtype=np.float32)
    last_track = np.zeros((n, dim), dtype=np.float32)
    scalars = np.zeros((n, 4), dtype=np.float32)

    for i, c in enumerate(cases):
        played = c["music_turns"]
        valid = [track_to_idx[t] for t in played if t in track_to_idx]
        if valid:
            # Recency-weighted mean over last 5 valid tracks.
            valid_recent = valid[-5:]
            m = len(valid_recent)
            w = np.array([0.8 ** (m - 1 - j) for j in range(m)], dtype=np.float32)
            w /= w.sum()
            played_mean[i] = (track_matrix[valid_recent] * w[:, None]).sum(axis=0)
            last_track[i] = track_matrix[valid_recent[-1]]

        n_hist = len(played)
        scalars[i, 0] = min(n_hist, 8) / 8.0
        scalars[i, 1] = 1.0 if n_hist == 0 else 0.0
        scalars[i, 2] = 1.0 if n_hist == 1 else 0.0
        scalars[i, 3] = min(len(str(c.get("user_query", "")).split()), 40) / 40.0

    # q_current/q_context are already normalized from R13, but normalize defensively.
    q_current = l2_normalize_np(q_current.astype(np.float32))
    q_context = l2_normalize_np(q_context.astype(np.float32))
    return played_mean, last_track, q_current, np.concatenate([q_context, scalars], axis=1)


@dataclass
class R17Data:
    cases: list[dict]
    sessions: list[str]
    y_idx: np.ndarray
    played_mean: np.ndarray
    last_track: np.ndarray
    q_current: np.ndarray
    q_context_scalars: np.ndarray
    hard_negs: np.ndarray
    v3_pools: list[list[str]]
    track_ids: list[str]
    track_matrix: np.ndarray
    track_scalars: np.ndarray
    track_to_idx: dict[str, int]


class R17Dataset(Dataset):
    def __init__(self, data: R17Data, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[j])
        return {
            "played_mean": torch.from_numpy(self.data.played_mean[i]),
            "last_track": torch.from_numpy(self.data.last_track[i]),
            "q_current": torch.from_numpy(self.data.q_current[i]),
            "q_context_scalars": torch.from_numpy(self.data.q_context_scalars[i]),
            "pos_idx": torch.tensor(self.data.y_idx[i], dtype=torch.long),
            "neg_idx": torch.from_numpy(self.data.hard_negs[i]),
        }


class TwoTower(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        scalar_dim: int = 5,
        hidden: int = 512,
        out_dim: int = 256,
        candidate_mode: str = "learned",
    ):
        super().__init__()
        self.candidate_mode = candidate_mode
        if candidate_mode == "identity":
            out_dim = emb_dim
        session_in = emb_dim * 4 + 4
        cand_in = emb_dim + scalar_dim
        self.session = nn.Sequential(
            nn.Linear(session_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, out_dim),
        )
        if candidate_mode == "learned":
            self.candidate = nn.Sequential(
                nn.Linear(cand_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.candidate = None

    def encode_session(self, played_mean, last_track, q_current, q_context_scalars):
        x = torch.cat([played_mean, last_track, q_current, q_context_scalars], dim=-1)
        return F.normalize(self.session(x), dim=-1)

    def encode_candidate(self, track_emb, track_scalars):
        if self.candidate_mode == "identity":
            return F.normalize(track_emb, dim=-1)
        x = torch.cat([track_emb, track_scalars], dim=-1)
        return F.normalize(self.candidate(x), dim=-1)


def choose_hard_negs(
    cases: list[dict],
    v3_pools: list[list[str]],
    track_to_idx: dict[str, int],
    n_hard: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    valid_indices = np.arange(len(track_to_idx), dtype=np.int64)
    hard = np.zeros((len(cases), n_hard), dtype=np.int64)
    for i, c in enumerate(cases):
        banned = set(c["music_turns"]) | {c["gt"]}
        chosen: list[int] = []
        for tid in v3_pools[i]:
            if tid in banned:
                continue
            idx = track_to_idx.get(tid)
            if idx is not None:
                chosen.append(idx)
            if len(chosen) >= n_hard:
                break
        while len(chosen) < n_hard:
            idx = int(rng.choice(valid_indices))
            tid = list(track_to_idx.keys())[idx]  # deterministic order follows insertion.
            if tid not in banned:
                chosen.append(idx)
        hard[i] = np.asarray(chosen[:n_hard], dtype=np.int64)
    return hard


def prepare_data(max_cases: int | None, pool_k: int, n_hard: int, seed: int) -> R17Data:
    print(f"{ts()} Loading payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    if max_cases:
        cases = cases[:max_cases]
        payload = {k: (v[:max_cases] if isinstance(v, list) and len(v) >= max_cases else v)
                   for k, v in payload.items()}
    sessions = [c["session_id"] for c in cases]
    print(f"  cases={len(cases)} sessions={len(set(sessions))}", flush=True)

    print(f"{ts()} Loading Qwen catalog and query embeddings...", flush=True)
    track_ids, track_matrix, track_to_idx = load_qwen_catalog()
    q_current = np.load(R13_QUERY_DIR / "emb_current.npy").astype(np.float32)[:len(cases)]
    q_context = np.load(R13_QUERY_DIR / "emb_context.npy").astype(np.float32)[:len(cases)]
    track_scalars = load_catalog_scalars(track_ids)

    print(f"{ts()} Building ALS and V3 pools...", flush=True)
    als_source, _ = build_als_source(cases, topk=200)
    v3_pools = build_v3_pools(payload, als_source, pool_k=pool_k)

    played_mean, last_track, q_current, q_context_scalars = build_session_arrays(
        cases, q_current, q_context, track_matrix, track_to_idx
    )

    y_idx = np.full(len(cases), -1, dtype=np.int64)
    missing_gt = 0
    for i, c in enumerate(cases):
        idx = track_to_idx.get(c["gt"])
        if idx is None:
            missing_gt += 1
        else:
            y_idx[i] = idx
    if missing_gt:
        print(f"  WARNING: {missing_gt} GTs missing from Qwen catalog; they count as misses.", flush=True)

    hard = choose_hard_negs(cases, v3_pools, track_to_idx, n_hard, np.random.RandomState(seed))

    return R17Data(
        cases=cases,
        sessions=sessions,
        y_idx=y_idx,
        played_mean=played_mean,
        last_track=last_track,
        q_current=q_current,
        q_context_scalars=q_context_scalars,
        hard_negs=hard,
        v3_pools=v3_pools,
        track_ids=track_ids,
        track_matrix=track_matrix,
        track_scalars=track_scalars,
        track_to_idx=track_to_idx,
    )


def train_one_fold(
    data: R17Data,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TwoTower, dict]:
    model = TwoTower(hidden=args.hidden, out_dim=args.out_dim,
                     candidate_mode=args.candidate_mode).to(device)
    track_t = torch.from_numpy(data.track_matrix).float().to(device)
    scalar_t = torch.from_numpy(data.track_scalars).float().to(device)

    loader = DataLoader(
        R17Dataset(data, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"epoch": 0, "loss": float("inf")}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            played_mean = batch["played_mean"].to(device)
            last_track = batch["last_track"].to(device)
            q_current = batch["q_current"].to(device)
            q_context_scalars = batch["q_context_scalars"].to(device)
            pos_idx = batch["pos_idx"].to(device)
            neg_idx = batch["neg_idx"].to(device)

            keep = pos_idx >= 0
            if keep.sum() < 2:
                continue
            played_mean = played_mean[keep]
            last_track = last_track[keep]
            q_current = q_current[keep]
            q_context_scalars = q_context_scalars[keep]
            pos_idx = pos_idx[keep]
            neg_idx = neg_idx[keep]

            s = model.encode_session(played_mean, last_track, q_current, q_context_scalars)
            pos = model.encode_candidate(track_t[pos_idx], scalar_t[pos_idx])
            in_batch = s @ pos.T

            neg_emb = track_t[neg_idx]
            neg_scalars = scalar_t[neg_idx]
            B, H, D = neg_emb.shape
            neg = model.encode_candidate(neg_emb.reshape(B * H, D), neg_scalars.reshape(B * H, -1))
            neg = neg.reshape(B, H, -1)
            hard_logits = torch.einsum("bd,bhd->bh", s, neg)

            logits = torch.cat([in_batch, hard_logits], dim=1) / args.tau
            targets = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"    epoch {epoch:02d} loss={mean_loss:.4f}", flush=True)
        if mean_loss < best["loss"]:
            best = {"epoch": epoch, "loss": mean_loss}

    return model, best


@torch.no_grad()
def evaluate_model(
    model: TwoTower,
    data: R17Data,
    eval_idx: np.ndarray,
    device: torch.device,
    topk: int = 500,
    chunk: int = 512,
) -> dict:
    model.eval()
    track_t = torch.from_numpy(data.track_matrix).float().to(device)
    scalar_t = torch.from_numpy(data.track_scalars).float().to(device)
    cand = model.encode_candidate(track_t, scalar_t).cpu().numpy().astype(np.float32)
    cand_t = torch.from_numpy(cand).to(device)

    ranks = []
    ndcg20 = []
    hit50 = hit100 = hit200 = hit500 = 0
    unique_vs_v3 = unique_unreachable = 0
    pop0_rec = diff_rec = hist0_rec = 0
    v3_hit200 = 0
    total_valid_gt = 0

    # Reconstruct train popularity for pop=0 slice.
    train_pop = build_train_popularity_stats()
    ta = None
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
        ta = payload["track_artist"]

    idx_to_track = data.track_ids
    for start in range(0, len(eval_idx), chunk):
        case_ids = eval_idx[start:start + chunk]
        pm = torch.from_numpy(data.played_mean[case_ids]).float().to(device)
        lt = torch.from_numpy(data.last_track[case_ids]).float().to(device)
        qc = torch.from_numpy(data.q_current[case_ids]).float().to(device)
        qs = torch.from_numpy(data.q_context_scalars[case_ids]).float().to(device)
        sess = model.encode_session(pm, lt, qc, qs)
        scores = (sess @ cand_t.T).cpu().numpy()

        for row, ci in enumerate(case_ids):
            c = data.cases[int(ci)]
            gt_idx = int(data.y_idx[int(ci)])
            if c["gt"] in data.v3_pools[int(ci)]:
                v3_hit200 += 1
            if gt_idx < 0:
                ndcg20.append(0.0)
                continue
            total_valid_gt += 1
            for tid in c["music_turns"]:
                pidx = data.track_to_idx.get(tid)
                if pidx is not None:
                    scores[row, pidx] = -np.inf
            # topk via argpartition; rank of GT by score.
            gt_score = scores[row, gt_idx]
            rank0 = int(np.sum(scores[row] > gt_score))
            rank = rank0 + 1
            ranks.append(rank)
            if rank <= 50:
                hit50 += 1
            if rank <= 100:
                hit100 += 1
            if rank <= 200:
                hit200 += 1
            if rank <= 500:
                hit500 += 1
            ndcg20.append(1.0 / math.log2(rank + 1) if rank <= 20 else 0.0)

            if rank <= 200 and c["gt"] not in data.v3_pools[int(ci)]:
                unique_vs_v3 += 1
                in_any = False
                # Current V3 includes these sources at depth 500/200 where available.
                for src_name in ["src_a", "src_b", "src_c", "src_d", "src_f"]:
                    src = payload[src_name][int(ci)]
                    if c["gt"] in src[:500]:
                        in_any = True
                        break
                if not in_any:
                    unique_unreachable += 1

            if rank <= 200:
                if train_pop.get(c["gt"], 0) == 0:
                    pop0_rec += 1
                if len(c["music_turns"]) == 0:
                    hist0_rec += 1
                if c["music_turns"]:
                    last_artist = ta.get(c["music_turns"][-1], "")
                    gt_artist = ta.get(c["gt"], "")
                    if last_artist and gt_artist and last_artist != gt_artist:
                        diff_rec += 1

    n = len(eval_idx)
    return {
        "n": n,
        "valid_gt": total_valid_gt,
        "hit50": hit50 / n,
        "hit100": hit100 / n,
        "hit200": hit200 / n,
        "hit500": hit500 / n,
        "ndcg20": float(np.mean(ndcg20)),
        "median_gt_rank": float(np.median(ranks)) if ranks else None,
        "mean_gt_rank": float(np.mean(ranks)) if ranks else None,
        "unique_vs_v3": unique_vs_v3,
        "unique_unreachable": unique_unreachable,
        "pop0_rec": pop0_rec,
        "diff_artist_rec": diff_rec,
        "hist0_rec": hist0_rec,
        "v3_hit200": v3_hit200 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--candidate_mode", choices=["learned", "identity"], default="learned")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_hard", type=int, default=32)
    parser.add_argument("--pool_k", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overfit", action="store_true",
                        help="Train and evaluate on the same cases as a data-path sanity check.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"{ts()} R17 two-tower diagnostic on {device}", flush=True)

    data = prepare_data(args.max_cases, args.pool_k, args.n_hard, args.seed)
    valid_cases = np.asarray([i for i, y in enumerate(data.y_idx) if y >= 0], dtype=np.int64)
    if args.smoke or args.overfit:
        # Tiny grouped split for data-flow validation.
        rng = np.random.RandomState(args.seed)
        rng.shuffle(valid_cases)
        if args.overfit:
            train_idx = valid_cases
            val_idx = valid_cases
            mode = "overfit"
        else:
            split = max(1, int(0.8 * len(valid_cases)))
            train_idx, val_idx = valid_cases[:split], valid_cases[split:]
            mode = "smoke"
        print(f"{ts()} {mode} train={len(train_idx)} val={len(val_idx)}", flush=True)
        model, best = train_one_fold(data, train_idx, val_idx, args, device)
        metrics = evaluate_model(model, data, val_idx, device, topk=500)
        out = {"mode": mode, "best": best, "metrics": metrics, "args": vars(args)}
        print(json.dumps(out, indent=2), flush=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Artifact: {OUT_PATH}")
        return

    print(f"{ts()} Grouped CV folds={args.folds}", flush=True)
    folds = grouped_session_folds(data.sessions, args.seed, k=args.folds)
    fold_metrics = []
    for fi, val_idx in enumerate(folds):
        val_set = set(val_idx.tolist())
        train_idx = np.asarray([i for i in valid_cases if i not in val_set], dtype=np.int64)
        val_idx = np.asarray([i for i in val_idx if data.y_idx[i] >= 0], dtype=np.int64)
        print(f"\n{ts()} Fold {fi}: train={len(train_idx)} val={len(val_idx)}", flush=True)
        model, best = train_one_fold(data, train_idx, val_idx, args, device)
        metrics = evaluate_model(model, data, val_idx, device, topk=500)
        metrics["fold"] = fi
        metrics["best_train"] = best
        print(f"  fold {fi} metrics: {json.dumps(metrics, sort_keys=True)}", flush=True)
        fold_metrics.append(metrics)

    summary = {}
    for key in [
        "hit50", "hit100", "hit200", "hit500", "ndcg20",
        "median_gt_rank", "unique_vs_v3", "unique_unreachable",
        "pop0_rec", "diff_artist_rec", "hist0_rec", "v3_hit200",
    ]:
        vals = [m[key] for m in fold_metrics if m.get(key) is not None]
        summary[key] = float(np.mean(vals)) if vals else None
    summary["n_folds"] = len(fold_metrics)
    summary["args"] = vars(args)
    summary["gate_pool_lift_003"] = (
        summary["hit200"] is not None
        and summary["v3_hit200"] is not None
        and summary["hit200"] - summary["v3_hit200"] >= 0.03
    )
    summary["gate_unique_unreachable_150"] = summary["unique_unreachable"] >= 150

    out = {"summary": summary, "folds": fold_metrics}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} SUMMARY")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Artifact: {OUT_PATH}")


if __name__ == "__main__":
    main()

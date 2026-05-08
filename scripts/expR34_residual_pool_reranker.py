#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R34: residual neural pool reranker.

This is a fold-0 clean diagnostic. It reuses the clean R33c tensor cache
and trains a small neural residual on top of OOF LambdaRank scores:

    final_score = zscore(lambdarank_score) + beta * neural_delta

The goal is to test whether neural features can make bounded corrections to
LambdaRank without replacing it.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expS2_lambdarank_grouped import grouped_session_folds

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R33_TENSORS = REPO / "cache" / "r33c_clean" / "tensors.npz"
R34_DIR = REPO / "cache" / "r34_residual"
OUT_JSON = REPO / "exp" / "eval" / "expR34_residual_pool_reranker.json"

POOL_K = 300
N_LR_FEAT = 29
N_INTERACT = 3


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


class ResidualPoolRanker(nn.Module):
    """Low-capacity candidate scorer that predicts only a residual delta."""

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 64,
        lr_dim: int = N_LR_FEAT,
        inter_dim: int = N_INTERACT,
        hidden: int = 128,
        dropout: float = 0.1,
        use_proj: bool = False,
    ) -> None:
        super().__init__()
        self.use_proj = use_proj
        if use_proj:
            self.q_proj = nn.Linear(emb_dim, proj_dim)
            self.c_proj = nn.Linear(emb_dim, proj_dim)
            extra_dim = 1
        else:
            self.q_proj = None
            self.c_proj = None
            extra_dim = 0
        self.mlp = nn.Sequential(
            nn.Linear(lr_dim + inter_dim + extra_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        q_emb: torch.Tensor,
        c_emb: torch.Tensor,
        lr_feat: torch.Tensor,
        interact: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual delta scores with shape [B, K]."""
        if self.use_proj:
            assert self.q_proj is not None
            assert self.c_proj is not None
            qh = F.normalize(self.q_proj(q_emb), dim=-1)
            ch = F.normalize(self.c_proj(c_emb), dim=-1)
            emb_score = (qh * ch).sum(dim=-1, keepdim=True)
            x = torch.cat([lr_feat, interact, emb_score], dim=-1)
        else:
            x = torch.cat([lr_feat, interact], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_cases() -> list[dict]:
    with open(R12_CACHE, "rb") as f:
        return pickle.load(f)["cases"]


def zscore_rows(scores: np.ndarray, pool_sizes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    for i, s_raw in enumerate(pool_sizes):
        s = int(s_raw)
        if s <= 0:
            continue
        row = scores[i, :s]
        mu = float(np.mean(row))
        sd = float(np.std(row))
        if sd < 1e-6:
            out[i, :s] = row - mu
        else:
            out[i, :s] = (row - mu) / sd
    return out


def compute_lambdarank_scores(
    lr_feat: np.ndarray,
    gt_pos: np.ndarray,
    pool_sizes: np.ndarray,
    force: bool = False,
) -> np.ndarray:
    """Train CV5 LambdaRank and return OOF scores for every case/candidate."""
    R34_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = R34_DIR / "lr_scores.npy"
    if cache_path.exists() and not force:
        print(f"{ts()} Loading cached LambdaRank scores: {cache_path}", flush=True)
        return np.load(cache_path)

    import lightgbm as lgb

    print(f"{ts()} Computing CV5 LambdaRank scores...", flush=True)
    cases = load_cases()
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0)
    n = len(cases)
    lr_scores = np.zeros((n, POOL_K), dtype=np.float32)

    for fi in range(5):
        val_set = set(folds[fi].tolist())
        train_idx = [j for j in range(n) if j not in val_set]
        val_idx = sorted(val_set)

        x_tr: list[np.ndarray] = []
        y_tr: list[float] = []
        g_tr: list[int] = []
        x_va: list[np.ndarray] = []
        y_va: list[float] = []
        g_va: list[int] = []

        for idx in train_idx:
            s = int(pool_sizes[idx])
            if s <= 0:
                continue
            x_tr.extend(lr_feat[idx, :s])
            y_tr.extend(1.0 if k == gt_pos[idx] else 0.0 for k in range(s))
            g_tr.append(s)

        for idx in val_idx:
            s = int(pool_sizes[idx])
            if s <= 0:
                continue
            x_va.extend(lr_feat[idx, :s])
            y_va.extend(1.0 if k == gt_pos[idx] else 0.0 for k in range(s))
            g_va.append(s)

        ds_tr = lgb.Dataset(np.asarray(x_tr), label=np.asarray(y_tr), group=g_tr)
        ds_va = lgb.Dataset(np.asarray(x_va), label=np.asarray(y_va), group=g_va, reference=ds_tr)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [20],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 10,
            "verbose": -1,
            "seed": 0,
        }
        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=300,
            valid_sets=[ds_va],
            callbacks=[lgb.log_evaluation(0)],
        )

        pred = model.predict(np.asarray(x_va))
        offset = 0
        for idx in val_idx:
            s = int(pool_sizes[idx])
            if s <= 0:
                continue
            lr_scores[idx, :s] = pred[offset : offset + s]
            offset += s
        print(f"  Fold {fi} done", flush=True)

    np.save(cache_path, lr_scores)
    print(f"{ts()} Saved LambdaRank scores: {cache_path}", flush=True)
    return lr_scores


def ndcg_from_rank(rank0: int) -> float:
    return float(1.0 / np.log2(rank0 + 2)) if rank0 < 20 else 0.0


def evaluate_scores(
    final_scores: np.ndarray,
    baseline_scores: np.ndarray,
    gt_pos: np.ndarray,
    pool_sizes: np.ndarray,
    val_idx: np.ndarray,
    same_artist: np.ndarray,
) -> dict:
    base_ndcg = 0.0
    model_ndcg = 0.0
    base_pool = 0.0
    model_pool = 0.0
    same_base = 0.0
    same_model = 0.0
    diff_base = 0.0
    diff_model = 0.0
    n_same = 0
    n_diff = 0
    n_in_pool = 0
    recovered = 0
    lost = 0
    base_ranks: list[int] = []
    model_ranks: list[int] = []

    for i in val_idx:
        s = int(pool_sizes[i])
        gp = int(gt_pos[i])
        is_same = bool(same_artist[i])
        if is_same:
            n_same += 1
        else:
            n_diff += 1

        base_v = 0.0
        model_v = 0.0
        base_hit = False
        model_hit = False
        if s > 0 and gp >= 0 and gp < s:
            n_in_pool += 1
            b_ranked = np.argsort(-baseline_scores[i, :s])
            m_ranked = np.argsort(-final_scores[i, :s])
            b_pos = np.where(b_ranked == gp)[0]
            m_pos = np.where(m_ranked == gp)[0]
            if len(b_pos):
                br = int(b_pos[0])
                base_ranks.append(br + 1)
                base_v = ndcg_from_rank(br)
                base_hit = br < 20
            if len(m_pos):
                mr = int(m_pos[0])
                model_ranks.append(mr + 1)
                model_v = ndcg_from_rank(mr)
                model_hit = mr < 20
            base_pool += base_v
            model_pool += model_v

        base_ndcg += base_v
        model_ndcg += model_v
        if is_same:
            same_base += base_v
            same_model += model_v
        else:
            diff_base += base_v
            diff_model += model_v
        if (not base_hit) and model_hit:
            recovered += 1
        if base_hit and (not model_hit):
            lost += 1

    n_val = len(val_idx)
    return {
        "baseline_h7": base_ndcg / max(n_val, 1),
        "model_h7": model_ndcg / max(n_val, 1),
        "delta_h7": (model_ndcg - base_ndcg) / max(n_val, 1),
        "baseline_h7_in_pool": base_pool / max(n_in_pool, 1),
        "model_h7_in_pool": model_pool / max(n_in_pool, 1),
        "n_val": int(n_val),
        "n_in_pool": int(n_in_pool),
        "same_baseline": same_base / max(n_same, 1),
        "same_model": same_model / max(n_same, 1),
        "diff_baseline": diff_base / max(n_diff, 1),
        "diff_model": diff_model / max(n_diff, 1),
        "n_same": int(n_same),
        "n_diff": int(n_diff),
        "recovered": int(recovered),
        "lost": int(lost),
        "net": int(recovered - lost),
        "baseline_median_rank": float(np.median(base_ranks)) if base_ranks else -1.0,
        "model_median_rank": float(np.median(model_ranks)) if model_ranks else -1.0,
    }


def batch_scores(
    model: ResidualPoolRanker,
    q_emb: np.ndarray,
    track_emb: torch.Tensor,
    lr_feat: np.ndarray,
    interact: np.ndarray,
    cand_idx: np.ndarray,
    pool_sizes: np.ndarray,
    lr_z: np.ndarray,
    beta: float,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out = np.zeros((len(q_emb), POOL_K), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            b = indices[start : start + batch_size]
            bs = len(b)
            q = torch.from_numpy(q_emb[b]).unsqueeze(1).expand(-1, POOL_K, -1)
            if model.use_proj:
                cidx = cand_idx[b]
                cidx_safe = np.where(cidx >= 0, cidx, 0)
                c = track_emb[cidx_safe.reshape(-1)].reshape(bs, POOL_K, q_emb.shape[1])
            else:
                c = torch.empty(bs, POOL_K, 0)
            lr = torch.from_numpy(lr_feat[b])
            inter = torch.from_numpy(interact[b])
            delta = model(q, c, lr, inter).numpy()
            score = lr_z[b] + beta * delta
            for row, idx in enumerate(b):
                s = int(pool_sizes[idx])
                if s < POOL_K:
                    score[row, s:] = -1e9
            out[b] = score
    return out


def run_one_beta(
    beta: float,
    data: dict[str, np.ndarray],
    lr_z: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    query_embs = data["query_embs"]
    track_embs = data["track_embs"]
    lr_feat = data["lr_feat"]
    interact = data["interact"]
    cand_idx = data["cand_idx"]
    gt_pos = data["gt_pos"]
    pool_sizes = data["pool_sizes"]
    same_artist = data["gt_same_artist"]
    emb_dim = query_embs.shape[1]

    model = ResidualPoolRanker(
        emb_dim=emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        use_proj=args.use_proj,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    track_t = torch.from_numpy(track_embs)
    best: dict | None = None
    beta_tag = str(beta).replace(".", "p")
    run_tag = args.run_tag.replace("/", "_")

    print(f"\n{ts()} Training beta={beta}...", flush=True)
    print(
        f"  model params={sum(p.numel() for p in model.parameters())/1e6:.3f}M "
        f"use_proj={args.use_proj}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        shuffled = train_idx.copy()
        np.random.shuffle(shuffled)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(shuffled), args.batch_size):
            b = shuffled[start : start + args.batch_size]
            bs = len(b)
            q = torch.from_numpy(query_embs[b]).unsqueeze(1).expand(-1, POOL_K, -1)
            if args.use_proj:
                cidx = cand_idx[b]
                cidx_safe = np.where(cidx >= 0, cidx, 0)
                c = track_t[cidx_safe.reshape(-1)].reshape(bs, POOL_K, emb_dim)
            else:
                c = torch.empty(bs, POOL_K, 0)
            lr = torch.from_numpy(lr_feat[b])
            inter = torch.from_numpy(interact[b])
            delta = model(q, c, lr, inter)
            final = torch.from_numpy(lr_z[b]) + beta * delta

            pad_mask = torch.from_numpy(np.arange(POOL_K)[None, :] >= pool_sizes[b][:, None])
            final = final.masked_fill(pad_mask, -1e9)
            targets = torch.from_numpy(gt_pos[b]).long()
            ce = F.cross_entropy(final / args.tau, targets)
            reg = (delta.masked_fill(pad_mask, 0.0) ** 2).sum() / torch.clamp((~pad_mask).sum(), min=1)
            loss = ce + args.delta_l2 * reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scores = batch_scores(
            model, query_embs, track_t, lr_feat, interact, cand_idx,
            pool_sizes, lr_z, beta, val_idx, args.batch_size,
        )
        metrics = evaluate_scores(scores, lr_z, gt_pos, pool_sizes, val_idx, same_artist)
        metrics.update({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "beta": beta})

        if best is None or metrics["model_h7"] > best["model_h7"]:
            best = dict(metrics)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metrics": best,
                    "beta": beta,
                    "args": vars(args),
                },
                R34_DIR / f"best_{run_tag}_beta_{beta_tag}.pt",
            )

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"  ep={epoch:02d} loss={metrics['loss']:.4f} "
                f"h7={metrics['model_h7']:.5f} "
                f"delta={metrics['delta_h7']:+.5f} "
                f"rec/lost={metrics['recovered']}/{metrics['lost']}",
                flush=True,
            )

    assert best is not None
    print(
        f"  BEST beta={beta}: h7={best['model_h7']:.5f} "
        f"delta={best['delta_h7']:+.5f} epoch={best['epoch']}"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--delta-l2", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--force-lr", action="store_true")
    parser.add_argument("--use-proj", action="store_true")
    parser.add_argument("--run-tag", default="default")
    args = parser.parse_args()

    t0 = time.time()
    R34_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} R34 residual pool reranker", flush=True)
    print("=" * 70, flush=True)
    if not R33_TENSORS.exists():
        raise FileNotFoundError(f"Missing {R33_TENSORS}. Run expR33c_clean.py --phase build first.")

    npz = np.load(R33_TENSORS)
    data = {k: npz[k] for k in npz.files}
    query_embs = data["query_embs"]
    lr_feat = data["lr_feat"]
    gt_pos = data["gt_pos"]
    pool_sizes = data["pool_sizes"]
    hist_depth = data["hist_depth"]
    is_fold0 = data["is_fold0"]

    lr_scores = compute_lambdarank_scores(lr_feat, gt_pos, pool_sizes, force=args.force_lr)
    lr_z = zscore_rows(lr_scores, pool_sizes)

    train_idx = np.asarray(
        [i for i in range(len(query_embs)) if hist_depth[i] >= 5 and not is_fold0[i] and gt_pos[i] >= 0],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [i for i in range(len(query_embs)) if hist_depth[i] == 7 and is_fold0[i]],
        dtype=np.int64,
    )

    baseline = evaluate_scores(lr_z, lr_z, gt_pos, pool_sizes, val_idx, data["gt_same_artist"])
    print(f"  train={len(train_idx)} hist5+ non-fold0 GT-in-pool", flush=True)
    print(f"  val={len(val_idx)} fold0 hist7, in_pool={baseline['n_in_pool']}", flush=True)
    print(f"  LambdaRank baseline h7={baseline['baseline_h7']:.5f}", flush=True)

    results: dict[str, dict] = {"baseline": baseline, "configs": {}}
    best_name = ""
    best_delta = -999.0

    for beta in args.betas:
        metrics = run_one_beta(beta, data, lr_z, train_idx, val_idx, args)
        name = f"beta_{beta:g}"
        results["configs"][name] = metrics
        if metrics["delta_h7"] > best_delta:
            best_delta = metrics["delta_h7"]
            best_name = name

    results["best"] = best_name
    results["gate_pass"] = bool(best_delta >= 0.005)
    results["elapsed_s"] = time.time() - t0
    out_json = OUT_JSON if args.run_tag == "default" else OUT_JSON.with_name(
        f"expR34_residual_pool_reranker_{args.run_tag}.json"
    )
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("R34 SUMMARY")
    print("=" * 70)
    for name, m in results["configs"].items():
        print(
            f"  {name:10s} h7={m['model_h7']:.5f} "
            f"delta={m['delta_h7']:+.5f} "
            f"same={m['same_model']:.5f} diff={m['diff_model']:.5f} "
            f"rec/lost={m['recovered']}/{m['lost']}"
        )
    print(f"  best={best_name} delta={best_delta:+.5f} {'PASS' if best_delta >= 0.005 else 'FAIL'}")
    print(f"  saved={out_json}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

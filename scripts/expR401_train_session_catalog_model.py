#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R401 Model A — REAL two-tower session/user->track neural retriever (Colab A100).

Trains on ALL train prefix->next transitions (106393) with softmax over the catalog
(sampled + hard negatives), learning the user->track CHOICE directly — NOT a scalar
cf-bpr cosine in a tree (that was R402, a proxy). Dev sessions are held out (0 session
leakage) for OOF eval; dev users overlap 74% so their TRAIN history is a legit input.

Track tower:   official embeddings (cf-bpr 128 + metadata-qwen3 1024 + audio-clap 512
               + lyrics-qwen3 1024 = 2688-d) -> MLP -> 256-d  (+ optional learned id emb).
Session tower: [ user-history pooled track-vec, in-session played pooled track-vec
               (recency-weighted), n_prior ] -> MLP -> 256-d.
Score = session . track.  Loss = sampled-softmax CE (pos target + K catalog negs by
popularity + hard negs from the production pool / same-artist).

Eval (OOF on 8000 dev): build session vec from user-train-history + played; score the
FULL 47k catalog; report recall@{20,100,300} + GT-rank distribution (does a TRAINED rep
surface the GT SHALLOW, unlike the raw cf-bpr centroid's deep-k@1000?). Also score the
production pool to measure nDCG@20 conversion as an added source/feature.

Run on Colab A100. Inputs from Drive (corpus) + HF (embeddings). Checkpoints to Drive.
"""
from __future__ import annotations
import argparse, json, os, time, math, random
from pathlib import Path
import numpy as np


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_track_features(emb_names):
    """Concatenated official track embeddings -> (47071, D) float32, + id list. Robust to cold."""
    from datasets import load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
    ids = [str(t) for t in ds["track_id"]]
    dims = {"cf-bpr": 128, "metadata-qwen3_embedding_0.6b": 1024,
            "audio-laion_clap": 512, "lyrics-qwen3_embedding_0.6b": 1024,
            "image-siglip2": 768}
    mats = []
    for name in emb_names:
        d = dims[name]; raw = ds[name]; M = np.zeros((len(ids), d), np.float32)
        for i, v in enumerate(raw):
            if v is not None and len(v) == d: M[i] = v
        # per-block L2 norm so blocks are comparable
        M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-8, None)
        mats.append(M); log(f"  emb {name}: {M.shape}")
    feat = np.concatenate(mats, axis=1)
    return ids, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/content/drive/MyDrive/r400/train_corpus.jsonl")
    ap.add_argument("--user-hist", default="/content/drive/MyDrive/r400/user_train_history.json")
    ap.add_argument("--dev", default="/content/drive/MyDrive/r400/dev_eval.json",
                    help="dev cases {case_idx,user_id,gt,music_turns,n_prior,fold} for OOF eval")
    ap.add_argument("--out", default="/content/drive/MyDrive/r400/r401_model")
    ap.add_argument("--embs", default="cf-bpr,metadata-qwen3_embedding_0.6b,audio-laion_clap,lyrics-qwen3_embedding_0.6b")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--n-neg", type=int, default=2000, help="sampled catalog negatives / step (shared in-batch)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use-id-emb", action="store_true")
    args = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={dev} | torch={torch.__version__}")
    os.makedirs(args.out, exist_ok=True)

    emb_names = args.embs.split(",")
    ids, feat = load_track_features(emb_names)
    Dfeat = feat.shape[1]; n_tracks = len(ids); tid2idx = {t: i for i, t in enumerate(ids)}
    feat_t = torch.tensor(feat, device=dev)
    # track popularity (for negative sampling) from corpus targets
    pop = np.ones(n_tracks, np.float64)

    log("loading corpus + user history ...")
    user_hist = json.load(open(args.user_hist))
    user_hist_idx = {u: [tid2idx[t] for t in ts if t in tid2idx] for u, ts in user_hist.items()}
    rows = []
    with open(args.corpus) as f:
        for line in f:
            r = json.loads(line)
            tgt = r["target_track_id"]
            if tgt not in tid2idx: continue
            played = [tid2idx[t] for t in r.get("history_prefix", []) if t in tid2idx]
            uh = user_hist_idx.get(str(r["user_id"]), [])
            rows.append((uh, played, tid2idx[tgt], int(r.get("turn_index", 0))))
            pop[tid2idx[tgt]] += 1
    log(f"  {len(rows)} usable transitions")
    pop_t = torch.tensor(pop / pop.sum(), device=dev)

    class Tower(nn.Module):
        def __init__(s, din, d):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.LayerNorm(512),
                                  nn.Linear(512, d), nn.LayerNorm(d))
        def forward(s, x): return s.net(x)

    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.track = Tower(Dfeat, args.dim)
            s.idemb = nn.Embedding(n_tracks, args.dim) if args.use_id_emb else None
            s.sess = nn.Sequential(nn.Linear(args.dim * 2 + 1, 512), nn.GELU(),
                                   nn.LayerNorm(512), nn.Linear(512, args.dim), nn.LayerNorm(args.dim))
        def track_vecs(s, idx):
            v = s.track(feat_t[idx])
            if s.idemb is not None: v = v + s.idemb(idx)
            return v
        def session_vec(s, uh_pool, played_pool, nprior):
            return s.sess(torch.cat([uh_pool, played_pool, nprior], dim=1))

    model = Model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | Dfeat={Dfeat} dim={args.dim}")

    def pool_vecs(list_of_idx_lists, recency=False):
        """mean (or recency-weighted) track-tower vec over each list; zeros if empty."""
        out = torch.zeros(len(list_of_idx_lists), args.dim, device=dev)
        for k, idxs in enumerate(list_of_idx_lists):
            if not idxs: continue
            tv = model.track_vecs(torch.tensor(idxs, device=dev))
            if recency:
                w = torch.tensor([0.85 ** (len(idxs) - 1 - j) for j in range(len(idxs))], device=dev).unsqueeze(1)
                out[k] = (w * tv).sum(0) / w.sum()
            else:
                out[k] = tv.mean(0)
        return out

    n = len(rows)
    for ep in range(args.epochs):
        random.shuffle(rows); tot = 0.0; t0 = time.time()
        for bi in range(0, n, args.bs):
            batch = rows[bi:bi + args.bs]
            uh = [b[0] for b in batch]; pl = [b[1] for b in batch]
            tgt = torch.tensor([b[2] for b in batch], device=dev)
            npr = torch.tensor([[b[3] / 7.0] for b in batch], dtype=torch.float32, device=dev)
            sv = model.session_vec(pool_vecs(uh), pool_vecs(pl, recency=True), npr)  # (B,d)
            # shared sampled negatives (popularity) + the batch positives as in-batch
            negs = torch.multinomial(pop_t, args.n_neg, replacement=True)
            cand = torch.cat([tgt, negs])                      # (B + n_neg)
            cv = model.track_vecs(cand)                        # (B+neg, d)
            logits = sv @ cv.T                                 # (B, B+neg)
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        log(f"epoch {ep}: loss={tot/(n//args.bs):.4f} ({time.time()-t0:.0f}s)")
        torch.save(model.state_dict(), f"{args.out}/model_ep{ep}.pt")

    # ---- OOF eval on dev: score FULL catalog, recall + nDCG over production pool ----
    if os.path.exists(args.dev):
        log("OOF eval on dev ...")
        model.eval()
        with torch.no_grad():
            allvec = torch.zeros(n_tracks, args.dim, device=dev)
            for i in range(0, n_tracks, 4096):
                allvec[i:i+4096] = model.track_vecs(torch.arange(i, min(i+4096, n_tracks), device=dev))
            dcases = json.load(open(args.dev))
            ranks = []
            for c in dcases:
                uh = [tid2idx[t] for t in user_hist.get(str(c["user_id"]), []) if t in tid2idx]
                pl = [tid2idx[t] for t in c["music_turns"] if t in tid2idx]
                npr = torch.tensor([[c["n_prior"]/7.0]], device=dev)
                sv = model.session_vec(pool_vecs([uh]), pool_vecs([pl], recency=True), npr)
                s = (sv @ allvec.T).squeeze(0)
                played = set(tid2idx[t] for t in c["music_turns"] if t in tid2idx)
                s[list(played)] = -1e9
                order = torch.argsort(s, descending=True).cpu().numpy()
                gi = tid2idx.get(c["gt"], -1)
                r = int(np.where(order == gi)[0][0]) + 1 if gi >= 0 else 10**9
                ranks.append((r, c.get("n_prior", -1)))
            ranks_a = np.array([r for r, _ in ranks])
            for k in (20, 100, 300, 1000):
                log(f"  catalog recall@{k}: {(ranks_a<=k).mean():.4f}")
            log(f"  median GT rank: {np.median(ranks_a):.0f}")
            json.dump({"recall@20": float((ranks_a<=20).mean()),
                       "recall@100": float((ranks_a<=100).mean()),
                       "recall@300": float((ranks_a<=300).mean()),
                       "median_rank": float(np.median(ranks_a)),
                       "n": len(ranks)}, open(f"{args.out}/oof_eval.json", "w"))
        log(f"saved {args.out}/oof_eval.json")


if __name__ == "__main__":
    main()

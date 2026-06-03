#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R421 Phase 1 — fine-tune a cross-encoder reranker on conversation->GT (Colab A100).

Candidate-conditional conversation reading: scores (conversation query, candidate track doc)
jointly — the signal the production text-blind LR lacks. Trained on the 15199 train sessions
(~121k transitions, disjoint from the 8000 dev sessions -> dev eval is naturally OOF).
Listwise softmax CE over {1 positive (GT) + N hard negatives (same-artist/popular/random)}.

After training, scores the dev pools and writes dev_scores.json (case_idx -> {tid: score})
for Phase 2 (local blend + gate). Run on Colab A100 with inputs synced from Drive.

  pip install -U transformers accelerate
  python expR421_phase1_train_crossencoder.py --data /content/drive/MyDrive/r421 \
       --model BAAI/bge-reranker-v2-m3 --epochs 2 --bs 16 --neg 15
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

random.seed(0)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/drive/MyDrive/r421")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="/content/drive/MyDrive/r421/model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16, help="queries per step")
    ap.add_argument("--neg", type=int, default=15, help="negatives per query")
    ap.add_argument("--maxlen", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    docs = json.load(open(data / "track_docs.json"))
    examples = [json.loads(l) for l in open(data / "train_examples.jsonl")]
    log(f"{len(examples)} train examples | {len(docs)} docs | device={dev}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    def score_pairs(queries, docs_list):
        enc = tok(queries, docs_list, padding=True, truncation=True, max_length=args.maxlen,
                  return_tensors="pt").to(dev)
        return model(**enc).logits.squeeze(-1)

    n = len(examples)
    steps_per_ep = n // args.bs
    for ep in range(args.epochs):
        random.shuffle(examples); model.train(); tot = 0.0; t0 = time.time()
        for bi in range(0, n, args.bs):
            batch = examples[bi:bi + args.bs]
            if not batch:
                continue
            q, d, group = [], [], []
            for ex in batch:
                negs = ex["negs"][:args.neg]
                cands = [ex["pos"]] + negs
                for t in cands:
                    q.append(ex["query"]); d.append(docs.get(t, ""))
                group.append(len(cands))
            with torch.cuda.amp.autocast():
                s = score_pairs(q, d)
                # listwise CE per group: positive is index 0 within each group
                loss = 0.0; off = 0
                for gl in group:
                    logits = s[off:off + gl].unsqueeze(0)
                    loss = loss + F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=dev))
                    off += gl
                loss = loss / len(group)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
            if (bi // args.bs) % 200 == 0:
                log(f"  ep{ep} step {bi//args.bs}/{steps_per_ep} loss={tot/max(1,(bi//args.bs)+1):.4f} "
                    f"({(time.time()-t0)/60:.1f}m)")
        log(f"epoch {ep}: loss={tot/steps_per_ep:.4f} ({(time.time()-t0)/60:.1f}m)")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out); tok.save_pretrained(args.out)

    # ---- score dev pools (OOF) ----
    log("scoring dev pools...")
    model.eval()
    dev_cases = [json.loads(l) for l in open(data / "dev_eval.jsonl")]
    out = {}
    with torch.no_grad():
        for ci, c in enumerate(dev_cases):
            pool = c["pool"]; q = c["query"]
            scores = {}
            for k in range(0, len(pool), 256):
                chunk = pool[k:k + 256]
                with torch.cuda.amp.autocast():
                    s = score_pairs([q] * len(chunk), [docs.get(t, "") for t in chunk])
                for t, sv in zip(chunk, s.float().cpu().tolist()):
                    scores[t] = sv
            out[str(c["case_idx"])] = scores
            if (ci + 1) % 1000 == 0:
                log(f"  scored {ci+1}/{len(dev_cases)} dev cases")
    json.dump(out, open(data / "dev_scores.json", "w"))
    log(f"saved {data/'dev_scores.json'} — copy back to repo for Phase 2 blend+gate")


if __name__ == "__main__":
    main()

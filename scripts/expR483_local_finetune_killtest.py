#!/usr/bin/env python3
# ruff: noqa: T201
"""R483 — LOCAL completion of the R481 fine-tune kill-test (the Colab eval crashed mid-run).

Finishes the open training-quest thread on the Mac's MPS GPU (Colab MCP dropped + A100 down):
LoRA fine-tune Qwen3-Reranker-0.6B on dev folds 1-4, then measure whether fine-tuning lifts
rec@1 on held-out fold-0 recoverable misses (paired adapter on/off, same rows). Gate to
"convert": rec@1 must jump ~10x from ~0.012 to >=0.10. Leak-free deployment-matched slate
(no GT injection), hard negs from the natural slate.

Inputs (local): exp/eval/r480_alldev_sim.json (8000 rows: conv + RRF-pool-minus-top20 cands + gt + rA + fold).
Output: cache/r483_lora/ (adapter) + exp/eval/r483_killtest_result.json (rec@k FT vs base).
"""
from __future__ import annotations
import json
import os
import random
import time

import numpy as np
import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
if DEV != "cuda" and os.environ.get("R483_ALLOW_LOCAL") != "1":
    raise SystemExit(
        f"Refusing to run R483 on local {DEV} by default: the previous local attempt caused an "
        "OOM/restart. Use Colab/A100/CUDA, or set R483_ALLOW_LOCAL=1 only after reducing "
        "batch/sequence/model memory and accepting the local risk."
    )
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICE = os.path.join(REPO, "exp/eval/r480_alldev_sim.json")
ADP = os.path.join(REPO, "cache/r483_lora")
OUT = os.path.join(REPO, "exp/eval/r483_killtest_result.json")
RM = "Qwen/Qwen3-Reranker-0.6B"
MAXLEN = 448


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    log(f"device={DEV}  loading slice {SLICE}")
    sim = json.load(open(SLICE))
    tk = AutoTokenizer.from_pretrained(RM, padding_side="left")
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
              'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    PT = tk.encode(prefix, add_special_tokens=False)
    ST = tk.encode(suffix, add_special_tokens=False)
    T_NO, T_YES = tk.convert_tokens_to_ids("no"), tk.convert_tokens_to_ids("yes")
    INSTRUCT = ("Given a user's multi-turn music chat (their stated tastes, constraints, and the tracks "
                "already played), decide whether this candidate track is the single best next track to recommend.")

    def build_ids(conv, doc):
        body = f"<Instruct>: {INSTRUCT}\n<Query>: {conv[-900:]}\n<Document>: {doc[:170]}"
        ids = tk.encode(body, add_special_tokens=False)[: MAXLEN - len(PT) - len(ST)]
        return PT + ids + ST

    # ---- training pairs (folds != 0), leak-free ----
    random.seed(0)
    pairs = []
    for r in sim:
        if r["fold"] == 0:
            continue
        cands = r["cands"][:60]
        gt = r["gt"]
        ct = r["cand_text"]
        if not cands:
            continue
        if r["rA"] > 20 and gt in cands:           # recoverable miss -> positive + hard negs
            pairs.append((r["conv"], ct[gt], 1))
            for c in [c for c in cands[:12] if c != gt][:6]:
                pairs.append((r["conv"], ct[c], 0))
        else:                                       # hit/absent -> teach LOW score (1 hard neg)
            pairs.append((r["conv"], ct[cands[0]], 0))
    random.shuffle(pairs)
    npos = sum(p[2] for p in pairs)
    pw = (len(pairs) - npos) / max(1, npos)
    log(f"train pairs={len(pairs)} pos={npos} pos_weight={pw:.1f}")

    log(f"loading {RM} on {DEV} (fp32 for MPS stability)")
    model = AutoModelForCausalLM.from_pretrained(RM, dtype=torch.float32).to(DEV)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    posw = torch.tensor(pw, device=DEV)

    def collate(batch):
        seqs = [build_ids(c, d) for c, d, _ in batch]
        mx = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), mx), tk.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, mx - len(s):] = torch.tensor(s)
            am[i, mx - len(s):] = 1
        lab = torch.tensor([float(l) for _, _, l in batch])
        return ids.to(DEV), am.to(DEV), lab.to(DEV)

    EP, BS = 2, 4
    t0 = time.time()
    step = 0
    for ep in range(EP):
        for i in range(0, len(pairs), BS):
            ids, am, lab = collate(pairs[i:i + BS])
            lg = model(input_ids=ids, attention_mask=am, logits_to_keep=1).logits[:, -1, :]
            score = (lg[:, T_YES] - lg[:, T_NO]).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, lab, pos_weight=posw)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                log(f"ep{ep} step{step}/{(len(pairs)//BS)*EP} loss={loss.item():.3f} {el:.0f}s ({step/el:.1f} it/s)")
    os.makedirs(ADP, exist_ok=True)
    model.save_pretrained(ADP)
    log(f"trained in {time.time()-t0:.0f}s, adapter -> {ADP}")

    # ---- eval: fold-0 recoverable misses, paired adapter on/off ----
    model.eval()

    @torch.no_grad()
    def score(conv, cand_texts, bs=16):
        out = []
        for i in range(0, len(cand_texts), bs):
            b = cand_texts[i:i + bs]
            seqs = [build_ids(conv, d) for d in b]
            mx = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), mx), tk.pad_token_id or 0, dtype=torch.long)
            am = torch.zeros_like(ids)
            for k, s in enumerate(seqs):
                ids[k, mx - len(s):] = torch.tensor(s)
                am[k, mx - len(s):] = 1
            lg = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV), logits_to_keep=1).logits[:, -1, :]
            out.append((lg[:, T_YES] - lg[:, T_NO]).float().cpu())
        return torch.cat(out).numpy()

    ev = [r for r in sim if r["fold"] == 0 and r["rA"] > 20 and r["gt"] in r["cands"][:60]]
    log(f"eval: fold-0 recoverable misses = {len(ev)}")

    def rank_gt(conv, cands, ct, gt):
        sc = score(conv, [ct[c] for c in cands])
        order = np.argsort(-sc)
        return [cands[i] for i in order].index(gt) + 1

    ft_r, base_r = [], []
    t1 = time.time()
    for n, r in enumerate(ev):
        cands = r["cands"][:60]
        ct, gt = r["cand_text"], r["gt"]
        ft_r.append(rank_gt(r["conv"], cands, ct, gt))
        with model.disable_adapter():
            base_r.append(rank_gt(r["conv"], cands, ct, gt))
        if (n + 1) % 30 == 0:
            log(f"  eval {n+1}/{len(ev)} {time.time()-t1:.0f}s")
    ft, bs = np.array(ft_r), np.array(base_r)
    ra = lambda a, k: float((a <= k).mean())
    res = {"n": len(ev), "rec": {k: {"base": ra(bs, k), "ft": ra(ft, k)} for k in (1, 3, 5, 10, 20)},
           "median_rank": {"base": float(np.median(bs)), "ft": float(np.median(ft))}}
    json.dump(res, open(OUT, "w"), indent=1)
    log("=" * 60)
    log("R483 LOCAL FINE-TUNE KILL-TEST — fold-0 recoverable misses")
    for k in (1, 3, 5, 10, 20):
        log(f"  rec@{k:<2} base={ra(bs,k):.3f}  FT={ra(ft,k):.3f}  delta={ra(ft,k)-ra(bs,k):+.3f}")
    log(f"  median rank base={np.median(bs):.0f} FT={np.median(ft):.0f}")
    g = ra(ft, 1)
    log(f"GATE rec@1>=0.10 (10x from ~0.012): FT rec@1={g:.3f} -> {'PASS — fine-tuning converts!' if g >= 0.10 else 'FAIL — fine-tuning does not lift rec@1 (cap confirmed)'}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()

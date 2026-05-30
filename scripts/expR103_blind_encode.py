#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R103 blind retrieval — reuse the 5 saved GTE-7B LoRA fold adapters (Drive) to
encode the 80 Blind-A cases, then 5-fold ENSEMBLE (memory: ensemble beats single
all-data on blind). No retraining. Run on Colab A100.

Per fold: load base + saved adapter, encode catalog (47071) + 80 blind queries
(conversational _q + Instruct wrapper, identical to training), retrieve top-300
excluding played. Ensemble = mean cosine across folds where present, top-300.

Inputs:
  --blind-cases  /content/r103_blind_cases.json  (80 cases: case_idx, session_id, user_query, history, music_turns)
  --adapters     /content/drive/MyDrive/r103_gte  (fold_{k}/lora_adapter)
Output:
  --out          /content/drive/MyDrive/r103_gte/blind_r103_ensemble_lists.json
                 {"lists": {case_idx: [[tid, avg_cos], ...]}, "sid": {case_idx: session_id}}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

MID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
TASK = "Given a music listening conversation, retrieve the track the user will most likely play next."
EMB_MAXLEN = 160
ENCODE_BS = 16
TOPK = 300
N_FOLDS = 5


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def prep_env():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)
    for m in [x for x in list(sys.modules) if x.startswith(("torchao", "torchvision"))]:
        del sys.modules[m]


def load_meta():
    from datasets import load_dataset
    try:
        from datasets import Dataset
        hf = Path.home() / ".cache" / "huggingface" / "datasets"
        m = sorted(hf.glob("talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
                           "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
        ds = (Dataset.from_file(str(m[-1])) if m
              else load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"])
    except Exception:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
    cols = ds.to_dict()
    meta = {str(cols["track_id"][i]): {k: cols[k][i] for k in cols} for i in range(len(cols["track_id"]))}
    return meta


def make_builders(meta):
    def _tt(tid):
        m = meta.get(tid, {})
        n = m.get("track_name", []); a = m.get("artist_name", [])
        al = m.get("album_name", []); tg = m.get("tag_list", [])
        name = n[0] if isinstance(n, list) and n else str(n)
        art = ", ".join(a) if isinstance(a, list) else str(a)
        alb = al[0] if isinstance(al, list) and al else str(al)
        tags = ", ".join(str(t) for t in tg[:10]) if isinstance(tg, list) else str(tg)
        return f"{name} by {art}. Album: {alb}. Tags: {tags}"

    def _short(tid):
        m = meta.get(tid, {})
        n = m.get("track_name", []); a = m.get("artist_name", [])
        return f"{(n[0] if isinstance(n, list) and n else n)} by {(a[0] if isinstance(a, list) and a else a)}"

    def _q(case):
        users, played = [], []
        for h in case["history"]:
            if h.get("role") == "user":
                users.append(str(h.get("content", "")))
            elif h.get("role") == "music":
                t = str(h.get("content", "")).strip()
                if t in meta:
                    played.append(_short(t))
        s = f'The user is looking for music and says: "{case["user_query"]}".'
        if users[-3:]:
            s += " Earlier they said: " + " ".join(users[-3:]) + "."
        if played[-5:]:
            s += " Recently played: " + "; ".join(played[-5:]) + "."
        return s

    return _tt, _short, _q


def gq(q):
    return f"Instruct: {TASK}\nQuery: {q}"


def load_base(device):
    import torch
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, n, layer_idx=0: self.get_seq_length(layer_idx)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    base = AutoModel.from_pretrained(MID, trust_remote_code=True,
                                     torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    return base, tok


def make_embed(model, tok, device):
    import torch

    def embed(texts, max_len=EMB_MAXLEN):
        e = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = model(**e).last_hidden_state
        last = e["attention_mask"].sum(1) - 1
        pooled = out[torch.arange(out.size(0), device=out.device), last]
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    return embed


def encode_catalog(embed, _tt, ids, device):
    import torch
    print(f"{ts()} encoding catalog {len(ids)} (bs={ENCODE_BS})...", flush=True)
    t0 = time.time(); chunks = []
    with torch.no_grad():
        for i in range(0, len(ids), ENCODE_BS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = embed([_tt(t) for t in ids[i:i + ENCODE_BS]])
            chunks.append(emb.detach().float().cpu())
            if (i // ENCODE_BS) % 400 == 0 and i:
                print(f"    {i}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    print(f"  catalog encoded {time.time()-t0:.0f}s", flush=True)
    return torch.cat(chunks, 0)


def retrieve(embed, _q, cases, cat_cpu, ids, device, topk=TOPK):
    import torch
    cat = cat_cpu.to(device).float()
    out = {}
    with torch.no_grad():
        for c in cases:
            qtext = c["q"] if c.get("q") else _q(c)          # pre-built query or build from history
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = embed([gq(qtext)])
            sims = (q.float() @ cat.T).squeeze(0).cpu().numpy()
            played = set(str(t) for t in (c.get("played") or c.get("music_turns") or []))
            order = np.argsort(-sims)
            top = []
            for j in order:
                tid = ids[int(j)]
                if tid in played:
                    continue
                top.append((tid, float(sims[int(j)])))
                if len(top) >= topk:
                    break
            out[c["case_idx"]] = top
    del cat
    gc.collect()
    try:
        import torch as _t; _t.cuda.empty_cache()
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-cases", default="/content/r103_blind_cases.json")
    ap.add_argument("--adapters", default="/content/drive/MyDrive/r103_gte")
    ap.add_argument("--out", default="/content/drive/MyDrive/r103_gte/blind_r103_ensemble_lists.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{ts()} R103 blind encode | device={args.device}", flush=True)
    if args.device == "cuda":
        prep_env()

    cases = json.load(open(args.blind_cases))
    print(f"  blind cases: {len(cases)}", flush=True)
    meta = load_meta()
    ids = list(meta.keys())
    _tt, _short, _q = make_builders(meta)
    sample_q = cases[0]["q"] if cases[0].get("q") else _q(cases[0])
    print(f"  catalog {len(ids)} | sample q: {sample_q[:90]}", flush=True)

    from peft import PeftModel
    import torch
    per_fold = {}                                   # case_idx -> {tid: [cos,...]}
    for c in cases:
        per_fold[c["case_idx"]] = defaultdict(list)
    adir = Path(args.adapters)
    for k in range(N_FOLDS):
        ad = adir / f"fold_{k}" / "lora_adapter"
        if not (ad / "adapter_config.json").exists():
            print(f"  fold {k}: adapter MISSING at {ad} — skip", flush=True); continue
        t0 = time.time()
        print(f"\n{ts()} === fold {k}: load adapter + encode ===", flush=True)
        base, tok = load_base(args.device)
        model = PeftModel.from_pretrained(base, str(ad)); model.eval()
        embed = make_embed(model, tok, args.device)
        cat = encode_catalog(embed, _tt, ids, args.device)
        lists = retrieve(embed, _q, cases, cat, ids, args.device)
        for ci, lst in lists.items():
            for tid, cos in lst:
                per_fold[ci][tid].append(cos)
        del base, model, embed, cat
        gc.collect(); torch.cuda.empty_cache()
        print(f"  fold {k} done {time.time()-t0:.0f}s", flush=True)

    # ensemble: mean cosine across folds where present, top-300
    ens = {}
    for c in cases:
        ci = c["case_idx"]
        agg = [(tid, float(np.mean(v))) for tid, v in per_fold[ci].items()]
        agg.sort(key=lambda x: -x[1])
        ens[str(ci)] = [[tid, s] for tid, s in agg[:TOPK]]
    sidmap = {str(c["case_idx"]): c["session_id"] for c in cases}
    json.dump({"experiment": "R103 blind 5-fold GTE ensemble", "model": MID,
               "n_cases": len(cases), "lists": ens, "sid": sidmap,
               "created_at": datetime.now().isoformat()}, open(args.out, "w"))
    print(f"\n{ts()} wrote ensemble -> {args.out} ({len(ens)} cases)", flush=True)
    print(f"  sample case0 top3: {ens['0'][:3]}", flush=True)


if __name__ == "__main__":
    main()

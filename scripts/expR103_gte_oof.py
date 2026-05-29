#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R103 — GTE-Qwen2-7B-instruct + LoRA, 5-fold OOF retriever (A100 / Colab).

GTE-7B was the strongest NEW retrieval signal of the cycle (R102 fold-0: ~31-32
top-30 GTs the complete union MISSES). It is NOT a better primary retriever
(recall@300 0.59 < fine-tuned BGE-large 0.68). Its only value is ORTHOGONAL
recall, the same signal R84 converted into a blind win. R103 produces clean
5-fold OOF GTE lists so the LOCAL sibling-LR (expR103_integrate_eval.py) can test
whether those uniques convert to nDCG@20 — the R84 playbook.

This script REPRODUCES the R102 recipe EXACTLY (so the OOF measurement matches the
fold-0 estimate that motivated the run):
  - model: Alibaba-NLP/gte-Qwen2-7B-instruct, bf16, sdpa, trust_remote_code
  - LoRA r=16 alpha=32 dropout=0.05 on {q,k,v,o,gate,up,down}_proj, FEATURE_EXTRACTION
  - last-token pooling, L2-normalize; query wrapped "Instruct: {TASK}\nQuery: {q}"
  - conversational query builder `_q`, track text `_tt` (identical to R100/R102 cells)
  - contrastive: BATCH=8 LR=1e-4 TEMP=0.05 2 epochs, positive=GT, one union-candidate
    negative per case + in-batch negatives (R102 Cell B). Union-candidate negatives are
    plausible (not mined hard-negs); R102 confirmed no collapse (cf feedback_no_hardneg_aux_first_run
    which concerns from-scratch pretraining, not this LoRA-FT).

OOF cleanliness: for held-out fold k, the LoRA is trained ONLY on cases with
fold_idx != k. Retrieval is over the FULL catalog (47071 tracks) so uniques absent
from the union can be surfaced. The union-candidate negatives used in TRAINING come
from training-fold cases only (their R84/R90/R54/R21 lists are themselves OOF), so
no held-out leakage.

Per-fold checkpoint/skip (robust to Colab disconnects): a fold whose
oof_lists.json already exists is skipped, so re-running resumes.

Inputs (defaults = Colab Drive layout used by R100/R102):
  --dev-cases  expR96_dev_cases_export.json  (case_idx, fold_idx, history, user_query, gt, music_turns)
  --union-cand expR100_union_candidates.json (case_idx -> {gt, candidates[]}) for training negatives
  catalog: HF dataset talkpl-ai/TalkPlayData-Challenge-Track-Metadata (all_tracks)

Output:
  <out-dir>/fold_{k}/oof_lists.json   per fold
  <out-dir>/oof_gte_lists.json        aggregated: {"lists":{case_idx:[[tid,score],...]}, ...}

Run (Colab A100, repo cloned + git pull):
  python scripts/expR103_gte_oof.py --device cuda \
      --dev-cases  /content/drive/MyDrive/r96_e5/expR96_dev_cases_export.json \
      --union-cand /content/drive/MyDrive/r100/expR100_union_candidates.json \
      --out-dir    /content/drive/MyDrive/r103_gte
  # or one fold at a time:  ... --fold 0
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # type: ignore[reportMissingImports]

MID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
TASK = "Given a music listening conversation, retrieve the track the user will most likely play next."

# --- training / retrieval hyperparameters (R102 Cell B, verbatim) ---
BATCH = 8
LR = 1e-4
TEMP = 0.05
EPOCHS = 2
EMB_MAXLEN = 160
ENCODE_BS = 16          # catalog/query encode batch (7B, bf16) — conservative for VRAM
TOPK = 300
N_FOLDS = 5


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def prep_env():
    """R102 env pairing: ensure peft present and remove stale torchao 0.10 BEFORE any
    peft import (a stale torchao breaks peft's LoRA dispatch -> import crash). Safe to
    call once at startup; harmless if torchao absent."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    # torchvision (even the torch-matched wheel) can fail to register its C++ ops on some
    # Colab torch builds -> transformers' model loader crashes ("operator torchvision::nms
    # does not exist"). We never need it for a text retriever; remove so transformers'
    # is_torchvision_available() guard skips it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"], check=False)
    for m in [x for x in list(sys.modules) if x.startswith(("torchao", "torchvision"))]:
        del sys.modules[m]


# ----------------------------------------------------------------------------
# Data: catalog + query/track text builders (identical to R100/R102 Colab cells)
# ----------------------------------------------------------------------------
def load_meta():
    """Track metadata dict {track_id: row}. HF cache if present, else download."""
    from datasets import load_dataset  # type: ignore[reportMissingImports]
    try:
        from datasets import Dataset  # type: ignore[reportMissingImports]
        hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
        matches = sorted(hf_cache.glob(
            "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
        ds = (Dataset.from_file(str(matches[-1])) if matches
              else load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"])
    except Exception:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
    cols = ds.to_dict()
    meta = {}
    for i in range(len(cols["track_id"])):
        meta[str(cols["track_id"][i])] = {k: cols[k][i] for k in cols}
    return meta


def make_text_builders(meta):
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


def gq(q: str) -> str:
    return f"Instruct: {TASK}\nQuery: {q}"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_base(device):
    import torch  # type: ignore[reportMissingImports]
    # gte-Qwen2 remote code calls DynamicCache.get_usable_length (removed in new transformers)
    from transformers.cache_utils import DynamicCache  # type: ignore[reportMissingImports]
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, n, layer_idx=0: self.get_seq_length(layer_idx)
    from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        MID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    base.config.use_cache = False
    try:
        base.gradient_checkpointing_enable()
    except Exception as e:
        print(f"  grad ckpt unavailable: {e}", flush=True)
    return base, tok


def make_lora(base):
    import sys as _sys
    for m in [x for x in _sys.modules if x.startswith("peft")]:
        del _sys.modules[m]
    from peft import LoraConfig, get_peft_model  # type: ignore[reportMissingImports]
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION")
    model = get_peft_model(base, lcfg)
    model.train()
    model.print_trainable_parameters()
    return model


def make_embed(model, tok, device):
    import torch  # type: ignore[reportMissingImports]

    def embed(texts, max_len=EMB_MAXLEN):
        e = tok(texts, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(device)
        out = model(**e).last_hidden_state
        last = e["attention_mask"].sum(1) - 1          # last non-pad idx (right padding)
        pooled = out[torch.arange(out.size(0), device=out.device), last]
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    return embed


# ----------------------------------------------------------------------------
# Train / encode / retrieve
# ----------------------------------------------------------------------------
def train_fold(model, embed, dev_trip, device, epochs=EPOCHS):
    import torch  # type: ignore[reportMissingImports]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    rng = np.random.RandomState(0)
    t0 = time.time(); step = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for ep in range(epochs):
        perm = rng.permutation(len(dev_trip)); nb = len(perm) // BATCH; run = 0.0
        for bi in range(nb):
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            qs = [gq(dev_trip[i][0]) for i in idx]
            pos = [dev_trip[i][1] for i in idx]
            neg = [dev_trip[i][2] for i in idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = embed(qs); d = embed(pos + neg)                # (B,H),(2B,H)
                scores = (q @ d.T) / TEMP                           # (B,2B); positive=i
                loss = torch.nn.functional.cross_entropy(
                    scores, torch.arange(len(idx), device=scores.device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            run += float(loss.item()); step += 1
            if step % 100 == 0:
                print(f"      ep{ep} {bi + 1}/{nb} loss={loss.item():.4f} "
                      f"avg={run / (bi + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        print(f"    epoch {ep} avg_loss={run / max(nb, 1):.4f}", flush=True)
    model.eval()
    print(f"  trained {epochs}ep in {time.time() - t0:.0f}s", flush=True)


def encode_catalog(embed, _tt, all_track_ids, device):
    import torch  # type: ignore[reportMissingImports]
    print(f"{ts()} encoding {len(all_track_ids)} tracks (bs={ENCODE_BS})...", flush=True)
    t0 = time.time(); chunks = []
    with torch.no_grad():
        for i in range(0, len(all_track_ids), ENCODE_BS):
            batch = [_tt(t) for t in all_track_ids[i:i + ENCODE_BS]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = embed(batch)
            chunks.append(emb.detach().float().cpu())              # embed() already returns f32-normalized
            if (i // ENCODE_BS) % 200 == 0 and i:
                print(f"    {i}/{len(all_track_ids)} {time.time() - t0:.0f}s", flush=True)
    cat = torch.cat(chunks, 0)                                     # (N,H) float32 cpu
    print(f"  catalog encoded {tuple(cat.shape)} in {time.time() - t0:.0f}s", flush=True)
    return cat


def retrieve(embed, _q, val_cases, cat_cpu, all_track_ids, device, topk=TOPK):
    import torch  # type: ignore[reportMissingImports]
    print(f"{ts()} retrieving top-{topk} for {len(val_cases)} held-out queries...", flush=True)
    t0 = time.time()
    cat = cat_cpu.to(device).float()                               # (N,H) f32 on gpu
    out = {}
    with torch.no_grad():
        for n, c in enumerate(val_cases):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = embed([gq(_q(c))])                             # (1,H) f32 (embed upcasts)
            sims = (q.float() @ cat.T).squeeze(0).cpu().numpy()    # (N,) float32 cosine
            played = set(str(t) for t in (c.get("music_turns") or []))
            order = np.argsort(-sims)
            top = []
            for j in order:
                tid = all_track_ids[int(j)]
                if tid in played:
                    continue
                top.append([tid, float(sims[int(j)])])
                if len(top) >= topk:
                    break
            out[c["case_idx"]] = top
            if (n + 1) % 400 == 0:
                print(f"    {n + 1}/{len(val_cases)} {time.time() - t0:.0f}s", flush=True)
    del cat
    gc.collect()
    try:
        import torch as _t; _t.cuda.empty_cache()
    except Exception:
        pass
    print(f"  retrieved in {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------
def build_dev_trip(train_cases, cand_by_idx, meta, _tt, _q):
    """R102 Cell B: one union-candidate negative per training case."""
    dev_trip = []
    for c in train_cases:
        uc = cand_by_idx.get(c["case_idx"])
        if uc is None or uc["gt"] not in meta:
            continue
        negs = [t for t in uc["candidates"] if t != uc["gt"] and t in meta]
        if not negs:
            continue
        nt = negs[np.random.RandomState(c["case_idx"]).randint(len(negs))]
        dev_trip.append((_q(c), _tt(uc["gt"]), _tt(nt)))
    return dev_trip


def run_one_fold(fold_i, cases_by_fold, cand_by_idx, meta, _tt, _q,
                 all_track_ids, out_dir, device, epochs):
    import torch  # type: ignore[reportMissingImports]
    fold_dir = out_dir / f"fold_{fold_i}"
    lists_path = fold_dir / "oof_lists.json"
    if lists_path.exists():
        print(f"{ts()} fold {fold_i}: lists exist — skip", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n{ts()} ===== R103 GTE FOLD {fold_i} (held out) =====", flush=True)

    val_cases = cases_by_fold[fold_i]
    adapter_dir = fold_dir / "lora_adapter"

    # Fresh base per fold (no adapter carryover -> clean OOF)
    base, tok = load_base(device)
    if (adapter_dir / "adapter_config.json").exists():
        # durable resume: the trained adapter is already on Drive -> skip retraining
        from peft import PeftModel  # type: ignore[reportMissingImports]
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.eval()
        print(f"  resumed: loaded saved LoRA adapter {adapter_dir} (skip training)", flush=True)
    else:
        train_cases = [c for k, v in cases_by_fold.items() if k != fold_i for c in v]
        dev_trip = build_dev_trip(train_cases, cand_by_idx, meta, _tt, _q)
        print(f"  train cases={len(train_cases)} dev_trip={len(dev_trip)} "
              f"val cases={len(val_cases)}", flush=True)
        model = make_lora(base)
        embed = make_embed(model, tok, device)
        train_fold(model, embed, dev_trip, device, epochs=epochs)
        try:
            model.save_pretrained(str(adapter_dir))      # durable: adapter -> Drive
            print(f"  saved LoRA adapter -> {adapter_dir}", flush=True)
        except Exception as e:
            print(f"  WARN adapter save failed: {e}", flush=True)
    embed = make_embed(model, tok, device)
    cat = encode_catalog(embed, _tt, all_track_ids, device)
    fold_lists = retrieve(embed, _q, val_cases, cat, all_track_ids, device)

    # quick recall report + rough uniques vs union candidates (authoritative
    # orthogonal-recall vs the complete R54-stacked pool is computed locally)
    hit20 = hit300 = uniq30 = 0
    for c in val_cases:
        gt = c["gt"]
        lst = [t for t, _ in fold_lists[c["case_idx"]]]
        if gt in lst[:20]:
            hit20 += 1
        if gt in lst[:300]:
            hit300 += 1
        uc = cand_by_idx.get(c["case_idx"])
        union = set(uc["candidates"]) if uc else set()
        if gt in lst[:30] and gt not in union:
            uniq30 += 1
    nv = len(val_cases)
    manifest = {
        "fold": fold_i, "model": MID, "epochs": epochs,
        "n_train_trip": len(dev_trip), "n_val": nv,
        "query_format": "_q conversational + Instruct wrapper",
        "negatives": "1 union-candidate neg + in-batch (R102 Cell B)",
        "recall20": round(hit20 / nv, 4), "recall300": round(hit300 / nv, 4),
        "unique30_vs_union_cands": uniq30,
        "elapsed_s": round(time.time() - t0, 1),
        "created_at": datetime.now().isoformat(),
    }
    json.dump({"lists": {str(c["case_idx"]): fold_lists[c["case_idx"]] for c in val_cases},
               "manifest": manifest}, open(lists_path, "w"))
    print(f"  fold {fold_i}: recall@20={hit20}/{nv} ({hit20/nv:.4f}) "
          f"recall@300={hit300}/{nv} ({hit300/nv:.4f}) "
          f"unique@30(vs union cands)={uniq30}  elapsed={time.time()-t0:.0f}s", flush=True)

    del model, base, embed, cat
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def aggregate(out_dir):
    merged = {}
    mans = {}
    for fi in range(N_FOLDS):
        p = out_dir / f"fold_{fi}" / "oof_lists.json"
        if not p.exists():
            print(f"  fold {fi} missing — partial aggregate", flush=True)
            continue
        d = json.load(open(p))
        merged.update(d["lists"])
        mans[fi] = d["manifest"]
    agg_path = out_dir / "oof_gte_lists.json"
    json.dump({"experiment": "R103 GTE-Qwen2-7B+LoRA 5-fold OOF",
               "model": MID, "n_cases": len(merged),
               "lists": merged, "fold_manifests": mans,
               "created_at": datetime.now().isoformat()}, open(agg_path, "w"))
    tot20 = np.mean([m["recall20"] for m in mans.values()]) if mans else 0
    tot300 = np.mean([m["recall300"] for m in mans.values()]) if mans else 0
    totu = sum(m["unique30_vs_union_cands"] for m in mans.values())
    print(f"\n{ts()} aggregated {len(merged)} cases -> {agg_path}", flush=True)
    print(f"  mean recall@20={tot20:.4f}  recall@300={tot300:.4f}  "
          f"total unique@30(vs union cands)={totu}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="single fold (default: all)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dev-cases", default="/content/drive/MyDrive/r96_e5/expR96_dev_cases_export.json")
    ap.add_argument("--union-cand", default="/content/drive/MyDrive/r100/expR100_union_candidates.json")
    ap.add_argument("--out-dir", default="/content/drive/MyDrive/r103_gte")
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch  # type: ignore[reportMissingImports]
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"{ts()} R103 GTE 5-fold OOF | device={args.device} | model={MID}", flush=True)
    if args.device == "cuda":
        prep_env()                       # peft present + stale torchao removed before peft import
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dev_cases = json.load(open(args.dev_cases))
    uni = json.load(open(args.union_cand))
    cand_by_idx = {c["case_idx"]: c for c in uni["cases"]}
    print(f"  dev cases={len(dev_cases)} | union cand cases={len(cand_by_idx)}", flush=True)

    meta = load_meta()
    all_track_ids = list(meta.keys())
    _tt, _short, _q = make_text_builders(meta)
    print(f"  catalog={len(all_track_ids)} tracks | sample q: {_q(dev_cases[0])[:90]}", flush=True)

    cases_by_fold = {k: [] for k in range(N_FOLDS)}
    for c in dev_cases:
        cases_by_fold[int(c["fold_idx"])].append(c)
    for k in range(N_FOLDS):
        print(f"  fold {k}: {len(cases_by_fold[k])} cases", flush=True)

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    for fi in folds:
        run_one_fold(fi, cases_by_fold, cand_by_idx, meta, _tt, _q,
                     all_track_ids, out_dir, args.device, args.epochs)

    if not args.no_aggregate:
        aggregate(out_dir)
    print(f"\n{ts()} done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())

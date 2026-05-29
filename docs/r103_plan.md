# R103 — GTE-7B as an ADDED retrieval source (R84 playbook)

**Hypothesis.** GTE-Qwen2-7B+LoRA is the strongest NEW retrieval signal of the cycle
(R102 fold-0: recall@300 0.59, **~32/1600 top-30 GTs the complete union misses**). It is
NOT a better primary retriever (BGE-large 0.68@300). Its only value is **orthogonal
recall** — the same signal R84 converted into a blind win. R103 tests whether those
uniques convert to nDCG@20 through the sibling LR. Marginal + conversion-uncertain, but
the only live positive nDCG lever.

**Production stays R92 p11 (composite 0.6364, nDCG 0.5073) until 5-fold OOF passes the gates.
No fold-0 blind packaging.**

## Stage 1 — 5-fold OOF GTE lists (A100) — `scripts/expR103_gte_oof.py`
Reproduces the R102 recipe EXACTLY (model/LoRA/last-token pooling/`_q`/`_tt`/contrastive
hyperparams — extracted verbatim from the R102 cells). For held-out fold k, LoRA trains
only on cases with `fold_idx != k`; retrieval is over the full 47071-track catalog so
uniques can surface. Per-fold checkpoint/skip (robust to Colab disconnects). float32
cosine (faithful to recipe). Fold map = export `fold_idx` (verified 8000/8000 == W0_STATS).

Run (Colab A100, after `git pull`):
```
python scripts/expR103_gte_oof.py --device cuda \
  --dev-cases  /content/drive/MyDrive/r96_e5/expR96_dev_cases_export.json \
  --union-cand /content/drive/MyDrive/r100/expR100_union_candidates.json \
  --out-dir    /content/drive/MyDrive/r103_gte
```
~2h (dev-only 2ep/fold; R102 showed full-data did not beat it). Output:
`oof_gte_lists.json` (case_idx → top-300 [(tid,score)]). Download to `cache/r103_gte/`.

## Stage 2 — integration + OOF eval (local) — `scripts/expR103_integrate_eval.py`
Three OOF sibling-LR arms (LR trained on the other 4 folds, scores held-out fold):
- **A base_r84** — pool = R54-stacked RRF (no GTE), feats R39+r84 (37). Current-best OOF analog.
- **B base_r84 + aug pool** — pool = R54-stacked + GTE source, feats 37 (isolates pool effect).
- **C r103_gte** — pool = R54-stacked + GTE source, feats 40 (+gte_rank_inv/presence/cosine).

GTE enters the pool as an RRF source so its uniques can actually be ranked (fixes the
R84-Phase-1 fixed-pool wall). Primary comparison **C vs A**; decompose with C-vs-B
(feature effect) and B-vs-A (pool effect). Compared OOF-vs-OOF, never vs frozen in-sample R54c.

## Gates (spec)
- **(A1 h7 nDCG@20 Δ ≥ +0.005) OR (A2 h7 rec > lost)**
- **B1 same-artist Δ ≥ −0.005**, **B2 diff-artist Δ ≥ −0.005**, **B3 top-20 overlap ≥ 8/20**
- GTE uniques must CONVERT to top-20 (not sit deep) — tracked explicitly.
- Stop conditions: uniques don't survive 5-fold OOF; LR ignores GTE features (gain < 1%);
  same-artist canary fires; lift < +0.002 with rec/lost not positive.

`PROCEED_TO_BLIND` only if (A1 or A2) and B1 and B2 and B3 and GTE-features-used.
Reviewed adversarially (OOF leakage / R102 faithfulness / ablation fairness / numerics) before launch.

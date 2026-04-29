# Sequence-Aware Target-Embedding Model — Design & Execution Plan

> **Scope:** Add a new retrieval source `S` to the existing deterministic fusion pipeline. Source `S` is produced by a small transformer trained to predict the next track's embedding from the conversational history.
> **Constraint:** Training runs on Apple Silicon Mac with 16 GB unified memory. No CUDA. MPS only.

---

## Why this exists

The current pipeline (`run_inference_blind_f1.py`) fuses five retrieval sources A'/B/C/D/F. All five are **stateless with respect to dialog trajectory** — they anchor on `played[-1]` or `played[-5:]` symmetrically, ignoring whether the user *accepted* or *pivoted away from* recent picks. The conversation goal can shift mid-session ("now something more upbeat"), and none of A'/B/C/D/F can model that pivot.

Source `S` adds a **trajectory-aware retriever**: given the full conversation up to turn `t-1`, it predicts an embedding in the same space as `metadata-qwen3` track embeddings, and the catalog is scored by cosine similarity to that prediction.

R-series experiments (R7 Haiku rerank, R11 Sonnet rerank, R1 BGE retrieval, R2 LambdaRank) all came back negative when used as **replacements** for parts of the existing pipeline. Source `S` is different: it is **additive**, with a CV5 gate. Worst case is `weight=0` in Powell's solution. No regression risk.

---

## Phase A — gate before any model work

**Do not start Phase B until Phase A is complete.** Phase A is pure infrastructure; it produces no model artifacts but determines whether everything that follows is interpretable.

### A1. CV5 leakage audit

The current CV5 protocol may have leakage in three places. Quantify each.

#### A1.1 Provided-embedding leakage

The provided embeddings (`audio-laion_clap`, `cf-bpr`, all qwen3 variants) were trained by the organizers. The exact training data is unknown. Run **ablation-based bound** instead:

1. Run full CV5 with current pipeline (A'+B+C+D+F + Powell). Record `cv5_full`.
2. Run CV5 dropping F (CF-BPR) only. Record `cv5_no_F`.
3. Run CV5 dropping A' (Qwen3 max-recent) only. Record `cv5_no_A`.
4. Run CV5 dropping D (Qwen3 neighbors) only. Record `cv5_no_D`.
5. Run CV5 with B+C only (BM25 sources, lowest leakage risk). Record `cv5_bm25_only`.

Expected: each embedding-based source contributes a small additive amount. If any single source's removal drops CV5 by more than ~0.02, that source is suspiciously strong — flag for further audit.

**Output:** table in `docs/leakage_audit.md`.

#### A1.2 CV5 fold construction

Read the CV5 fold-building code. Verify:

- [ ] Folds split at the **session level**, not turn level.
- [ ] When fold `k` holds out sessions `S_k`, those sessions' tracks are **not** added to the BM25 corpus build, FAISS indices, or any other index that the held-out evaluation queries.
- [ ] Per-user aggregation (if any) excludes held-out sessions when computing user-side features.

**Output:** documented confirmation in `docs/leakage_audit.md`.

#### A1.3 BM25 corpus check

Read `mcrs/retrieval_modules/bm25.py`. Confirm the index is built **only** over `track_metadata` text fields (track_name, artist_name, album_name, tags, release_date). It must not include any conversation text from the train split.

**Output:** one-line confirmation in `docs/leakage_audit.md`.

### A2. F1 Blind A submission

Submit the current `run_inference_blind_f1.py` config (cfg0209 + CF-BPR F_max_recent5) to Blind A.

**Record in `VERSIONS.md`:**

| Field | Value |
|---|---|
| Git commit (HEAD at submit time) | `<hash>` |
| CV5 nDCG@20 | `<value>` |
| Blind A nDCG@20 | `<value from leaderboard>` |
| Multiplier | `Blind / CV5` |
| LB rank | `<rank>` |

### Gate A → B

Both must hold before proceeding to Phase B:

- [ ] Leakage audit complete with documented bounds. No source contributes >0.02 CV5 alone unless that contribution is structurally explainable.
- [ ] F1 submitted to Blind A. CV5→Blind A multiplier recorded.

---

## Phase B — sequence model

### B1. Confirm baseline embedding identity

Confirm which embedding column each source uses:

| Source | Embedding column | Dim |
|---|---|---|
| A' (max-recent) | metadata-qwen3_embedding_0.6b | 1024 |
| D (track neighbors) | metadata-qwen3_embedding_0.6b | 1024 |
| F (CF-BPR) | cf-bpr | 128 |
| **S (this design)** | predicts target in **same space as D's index** | 1024 |

### B2. Architecture spec

#### B2.1 Inputs

For each prior turn `t ∈ {1, ..., T-1}`, build a single input token:

| Component | Source | Dim | Projection |
|---|---|---|---|
| `track_token_t` | metadata-qwen3 emb of GT track at turn t | 1024 | Linear(1024 → 256) |
| `utt_token_t` | sentence-encoder emb of user utterance at turn t | 384 (E5-small) | Linear(384 → 256) |
| `accept_emb_t` | learned embedding lookup, vocab=2 | 256 | — |
| `turn_emb_t` | learned embedding lookup, vocab=8 | 256 | — |

Combine: `fused_t = LayerNorm(track_token_t + utt_token_t + accept_emb_t + turn_emb_t)`.

For the **current** turn `T` (no track played yet):

| Component | Source | Dim | Projection |
|---|---|---|---|
| `utt_token_T` | sentence-encoder emb of current user utterance | 384 | Linear(384 → 256) |
| `query_marker` | learned single vector | 256 | — |
| `turn_emb_T` | turn position embedding | 256 | — |

Combine: `query_token = LayerNorm(utt_token_T + query_marker + turn_emb_T)`.

**Sequence:** `[fused_1, ..., fused_{T-1}, query_token]`. Pad to length 8 with attention masking.

**`accept_t` heuristic for V0:** treat all played tracks as accepted (label=1).

#### B2.2 Encoder

```
Encoder: nn.TransformerEncoder
  num_layers: 4
  d_model: 256
  nhead: 4
  dim_feedforward: 1024
  dropout: 0.1
  norm_first: True   # pre-norm, more stable on MPS
  batch_first: True
```

Causal mask + padding mask for variable-length sequences (max=8).

#### B2.3 Readout

Take the hidden state at `query_token` position. Project:

```
target = Linear(256 → 1024)(hidden_state)
target = F.normalize(target, dim=-1)
```

Output is a unit vector in metadata-qwen3 space.

#### B2.4 Inference scoring

```
scores = target_emb @ catalog_metadata_qwen3_matrix.T   # (1, 1024) @ (1024, N_catalog)
top_K = topk(scores, K).indices
```

`K=200` matches existing source depths.

#### B2.5 Parameter count target: ~2.7M trainable

### B3. Training data

- **Positives:** ~106k examples (15.2k sessions × 7 turns, skip t=1)
- **5% validation slice:** session-level holdout for early stopping
- **Negatives per positive:** 32 in-batch + 16 BM25 hard + 16 random catalog
- **Utterance encoder:** `intfloat/e5-small-v2` (384d), frozen, cached

### B4. Loss

InfoNCE with τ=0.05, plus anti-collapse auxiliary:

```python
anti_collapse_loss = lam * (query_emb * last_track_emb).sum(-1).mean()
total = info_nce + anti_collapse_loss
```

Start `lam=0.05`. Increase if top-1 == played[-1] rate > 30%.

### B5. Training procedure

| Param | Value |
|---|---|
| Optimizer | AdamW, weight_decay=0.01 |
| Base LR | 3e-4 |
| LR schedule | warmup 500 steps + cosine decay |
| Batch size | 32 |
| Epochs | up to 10, early stop on val nDCG@20 |
| Grad clip | 1.0 |
| Precision | fp32 (MPS, no bf16) |

#### MPS pitfalls:
- No `pin_memory=True` in DataLoader
- No `num_workers > 0`
- Save checkpoints every epoch
- Keep RAM under 10 GB

#### Expected wall-clock: 1–2 hours/epoch, 5 epochs ≈ overnight

### B6. Integration

Mirror `CFBPRMaxRecent` interface. Add to fusion:

```python
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "S": 1.0}
```

### Gate B → submit

All three must hold:

1. **Unique-hit gate:** Source S alone (top-200) produces ≥ 20 unique GT hits not in A'/B/C/D/F's top-200.
2. **CV5 lift gate:** Adding S lifts CV5 nDCG@20 by ≥ +0.005.
3. **Anti-collapse gate:** Top-1 == played[-1] in < 30% of val cases.

---

## Phase C — only if B passes

1. Powell weight sweep over A/B/C/D/F/S
2. Ablation row in VERSIONS.md
3. Submit ensemble to Blind A

---

## File-system conventions

```
cache/seq_model/
  utt_embeddings.npy
  utt_embedding_index.json
  hard_negatives.pkl
  catalog_metadata_qwen3.npy
  runs/<run_id>/
    config.yaml
    epoch_*.pt
    val_metrics.json
    final.pt
  predictions.pkl

docs/
  leakage_audit.md
  sources_inventory.md
  seq_model_design.md          # this file
  seq_model_run_log.md

mcrs/retrieval_modules/seq_model.py
scripts/build_utt_cache.py
scripts/mine_hard_negatives.py
scripts/train_seq_model.py
```

---

## Deliberate exclusions

- Goal/specificity conditioning (train-only field, not available at inference)
- Multi-modal track inputs in V0 (complexity vs. diminishing returns)
- Fine-tuning utterance encoder (blows 16GB budget)
- End-to-end catalog encoder training (catalog side is fixed)
- LLM reranking on top of S (R7/R11 showed this loses to Powell)

# Sequence Model S — Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         SEQUENCE MODEL S             │
                    │    (~2.7M params, MPS training)      │
                    └─────────────────────────────────────┘

INPUT CONSTRUCTION (per turn)
═══════════════════════════════════════════════════════════

Turn 1          Turn 2          Turn 3        ...  Turn T (current)
┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐
│ track emb │  │ track emb │  │ track emb │       │  (none)  │
│ qwen3     │  │ qwen3     │  │ qwen3     │       │          │
│ 1024d     │  │ 1024d     │  │ 1024d     │       │          │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │          │
      │Linear         │              │              │          │
      │1024→256       │              │              │          │
      ▼              ▼              ▼              │          │
  ┌───────┐      ┌───────┐      ┌───────┐         │          │
  │trk 256│      │trk 256│      │trk 256│         │          │
  └───┬───┘      └───┬───┘      └───┬───┘         │          │
      │              │              │              │          │
┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐
│ user utt │  │ user utt │  │ user utt │       │ user utt │
│ E5-small │  │ E5-small │  │ E5-small │       │ E5-small │
│ 384d     │  │ 384d     │  │ 384d     │       │ 384d     │
│ (frozen) │  │ (frozen) │  │ (frozen) │       │ (frozen) │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘
      │Linear         │              │                    │Linear
      │384→256        │              │                    │384→256
      ▼              ▼              ▼                    ▼
  ┌───────┐      ┌───────┐      ┌───────┐           ┌───────┐
  │utt 256│      │utt 256│      │utt 256│           │utt 256│
  └───┬───┘      └───┬───┘      └───┬───┘           └───┬───┘
      │              │              │                    │
      │   + accept_emb (learned, vocab=2)                │   + query_marker
      │   + turn_emb  (learned, vocab=8)                 │   + turn_emb
      ▼              ▼              ▼                    ▼
  ┌───────┐      ┌───────┐      ┌───────┐           ┌───────┐
  │LayerNm│      │LayerNm│      │LayerNm│           │LayerNm│
  │fused_1│      │fused_2│      │fused_3│           │query_T│
  └───┬───┘      └───┬───┘      └───┬───┘           └───┬───┘
      │              │              │                    │
      ▼              ▼              ▼                    ▼

TRANSFORMER ENCODER (causal attention)
═══════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │  [fused_1]  [fused_2]  [fused_3]  ...  [query_T]   │
  │       ↓          ↓          ↓              ↓        │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  Transformer Layer 1  (d=256, 4 heads)       │   │
  │  │  pre-norm, causal mask, padding mask         │   │
  │  └──────────────────────────────────────────────┘   │
  │       ↓          ↓          ↓              ↓        │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  Transformer Layer 2                         │   │
  │  └──────────────────────────────────────────────┘   │
  │       ↓          ↓          ↓              ↓        │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  Transformer Layer 3                         │   │
  │  └──────────────────────────────────────────────┘   │
  │       ↓          ↓          ↓              ↓        │
  │  ┌──────────────────────────────────────────────┐   │
  │  │  Transformer Layer 4                         │   │
  │  └──────────────────────────────────────────────┘   │
  │                                        ↓            │
  │                                   [hidden_T]        │
  │                                     256d            │
  └─────────────────────────────────────────────────────┘
                                        │
                                        ▼

READOUT
═══════════════════════════════════════════════════════════

                                   ┌──────────┐
                                   │ Linear   │
                                   │ 256→1024 │
                                   └────┬─────┘
                                        │
                                        ▼
                                   ┌──────────┐
                                   │L2 normize│
                                   └────┬─────┘
                                        │
                                        ▼
                                ┌──────────────┐
                                │ target_emb   │
                                │ 1024d unit   │
                                │ vector in    │
                                │ qwen3 space  │
                                └──────┬───────┘
                                       │
                                       ▼

SCORING (inference)
═══════════════════════════════════════════════════════════

     target_emb (1, 1024)
          │
          │  dot product
          ▼
    ┌─────────────────────────────────────┐
    │  catalog_qwen3_matrix (47K, 1024)   │
    │  (pre-computed, L2-normalized)      │
    └──────────────────┬──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ cosine scores  │
              │ (47K,)         │
              │                │
              │ top-200 → S    │
              │ candidates     │
              └────────┬───────┘
                       │
                       ▼

FUSION (existing pipeline)
═══════════════════════════════════════════════════════════

    ┌─────┬─────┬─────┬─────┬─────┬─────┐
    │  A' │  B  │  C  │  D  │  F  │  S  │
    │qwen3│BM25 │BM25 │qwen3│cfbpr│ seq │
    │ sim │meta │full │nbrs │     │model│
    └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
       │     │     │     │     │     │
       └─────┴─────┴──┬──┴─────┴─────┘
                      │
                      ▼
              ┌───────────────┐
              │ Weighted RRF  │
              │ k=20, top-50  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ 8-feat Powell │
              │ postrank      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   top-20      │
              │  predictions  │
              └───────────────┘


TRAINING (InfoNCE + anti-collapse)
═══════════════════════════════════════════════════════════

    target_emb ──┬── cos(target, GT_track_emb)     → positive
                 │
                 ├── cos(target, in_batch_tracks)   → 32 negatives
                 │
                 ├── cos(target, BM25_hard_negs)    → 16 negatives
                 │
                 └── cos(target, random_catalog)    → 16 negatives

    Loss = InfoNCE(τ=0.05) + λ·cos(target, played[-1])
                                    ↑
                              anti-collapse
                              (prevent copying
                               last track emb)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Output space | metadata-qwen3 (1024d) | Same as source D's index; no new catalog index needed |
| Utterance encoder | E5-small-v2 (384d), frozen | 16GB RAM budget; fine-tuning adds 30M params |
| Model dim | 256 | ~2.7M total params; fits in memory with batch=32 |
| Layers | 4 | Enough for 8-turn sequences; more layers overfit on 106K examples |
| Anti-collapse | λ·cos(pred, last_played) | Prevents degenerate copy of source A' |
| Attention | Causal + padding mask | Each turn only sees past; variable-length sequences |
| Training device | MPS (Apple Silicon) | No CUDA available; fp32 only |

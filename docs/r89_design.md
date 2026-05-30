# R89 — Learned Multimodal Dual-Encoder Retriever (Design)

**Status:** Design only. Phase 0 launch pending user approval.
**Hypothesis:** Train a fused track representation end-to-end on full-corpus
pairs, so query text → joint(text + image + lyrics + attrs + audio) track.
R85/R88 closed multimodal-as-features at the LR layer. R89 tests whether
learning fusion *at the encoder* produces signal that converts.

## Why R89 is structurally different from R85/R88

- R85: kept R84 retriever + LR fixed; added multimodal candidates/features
  after retrieval. → **failed because LR isn't tuned for cross-class features**.
- R88: same as R85 but with monotone / boost / quota constraints.
  → **failed for the same reason — features bolted onto an LR that doesn't
  know how to weight them**.
- **R89: replaces the track representation itself.** The query side is still
  text (R84 BGE-large). The track side becomes a learned fusion of all
  modalities trained against query text. R84/R54 LR features for this new
  retriever would be retrieved differently — its top-K would already encode
  multimodal evidence in the rank ordering.

## Architecture

```
Query side (text):
  user_query + history + played_context
        │
        ▼
  R84 BGE-large query encoder
        │  (frozen Phase 0; LoRA fine-tune Phase 1 if Phase 0 passes)
        ▼
  q_emb ∈ R^1024  (L2-normalized)

Track side (multimodal):
  ┌─────────────────────────────────────────────┐
  │  BGE-large text emb (1024)                   │
  │  image SigLIP (768)                          │
  │  lyrics Qwen (1024)                          │
  │  attributes Qwen (1024)                      │
  │  audio CLAP (512)                            │
  └─────────────────────────────────────────────┘
        │ (each modality, frozen Phase 0)
        ▼
  per-modality projection MLP → 256-dim each
  + per-modality gate scalar (learned, sigmoid)
        │
        ▼
  Concatenate gated 256-dim projections (1280-dim)
        │
        ▼
  Fusion MLP → 1024-dim (matches query)
        │
        ▼
  t_emb ∈ R^1024  (L2-normalized)

Loss:
  in-batch InfoNCE on (q_emb, t_emb) pairs
  + k=64 random catalog negatives per step (sampled fresh)
  + temperature τ = 0.05
```

## Phase 0 constraints (variable isolation)

- **Query encoder FROZEN** (use R84 fold-0 query embeddings, precomputed once).
- **All base modality embeddings FROZEN** (already precomputed).
- **Only the projections + gates + fusion MLP are trained.** ~2-3M trainable params.
- **Modalities included Phase 0**: text + image + lyrics + attributes.
  Audio CLAP requires coverage check first (R85 inventory showed only
  46579/47071 = 99% valid; gate-able but reduces signal density).
- **Loss is in-batch InfoNCE + random negs only.** NO hard-negative
  auxiliary (per [[feedback_no_hardneg_aux_first_run]]).
- **Train fold-0 only** initially, eval against R84c sibling-R84 OOF on fold-0.

If Phase 0 passes the gate, escalate:
- Phase 1: query-side LoRA (adapter) fine-tune.
- Phase 2: 5-fold OOF training + blind candidate.

## Training data

Reuse R84 Phase 0A pair manifest (committed at
`cache/r84/phase0a/pair_manifest.parquet`, 153K pairs, sha256
`a6ecba53...`). For fold-0 training: 127,992 pairs (121K train_split + 6.4K
dev folds 1-4). Pool exclusion as for R84.

## Phase 0 architecture details

```python
class TrackFusionModel(nn.Module):
    def __init__(self, dim_text=1024, dim_img=768, dim_lyrics=1024,
                 dim_attrs=1024, dim_audio=512, proj_dim=256, out_dim=1024):
        super().__init__()
        # Per-modality projections
        self.proj_text = nn.Sequential(
            nn.Linear(dim_text, proj_dim), nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.proj_img = nn.Sequential(
            nn.Linear(dim_img, proj_dim), nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        # ... same for lyrics, attrs, audio
        # Learnable gates per modality
        self.gates = nn.Parameter(torch.zeros(5))  # logits → sigmoid(g)
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 5, 1024), nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, out_dim),
        )

    def forward(self, embs):
        # embs is dict of modality → tensor (B, dim_mod)
        # Modality may be missing (zero tensor); gate suppresses
        gates = torch.sigmoid(self.gates)  # (5,)
        projs = [
            gates[0] * self.proj_text(embs["text"]),
            gates[1] * self.proj_img(embs["img"]),
            gates[2] * self.proj_lyrics(embs["lyrics"]),
            gates[3] * self.proj_attrs(embs["attrs"]),
            gates[4] * self.proj_audio(embs["audio"]),
        ]
        x = torch.cat(projs, dim=-1)  # (B, 1280)
        out = self.fusion(x)
        return F.normalize(out, dim=-1)
```

Trainable parameter budget:
- Per-modality projection (5×): (dim_mod × 256 + 256 LN params) ≈ ~250K each
- Gates: 5
- Fusion MLP: 1280×1024 + LN + 1024×1024 = ~2.4M
- **Total: ~3.5M trainable.**

Vastly smaller than R84's 335M BGE-large fine-tune.

## Phase 0 training budget

- Batch size: 64 (small model, can fit much larger than R84)
- Negatives: 64 random catalog per step (cached pre-projection catalog → just look up)
- Epochs: 3 (small model, faster convergence)
- Steps/epoch: 127,992 / 64 = ~2000
- Total steps: ~6000
- Per-step cost: 1 forward + 1 backward on ~3.5M params + 64 catalog lookups + softmax over 128 candidates ≈ ~5ms on A100
- Wall: ~30 sec/epoch × 3 = **~2 minutes** training
- Catalog re-projection at eval: 47K tracks × ~3M params forward ≈ ~10 sec
- Eval (1600 fold-0 dev queries × 47K catalog cosine): ~5 sec
- **Total wall: ~3 min A100, ~$0.15.**

Then encoder forward isn't required (we use frozen R84 query embeddings).
If we choose to encode queries with BGE-large fresh, add ~5 min.

## Phase 0 gate (matches R84c PROCEED rule)

- h7 nDCG@20 Δ ≥ +0.005 OR
- ≥10 unique h7 top-30 recoveries with rec ≥ lost OR
- h7 Δ ≥ −0.003 AND ≥5 unique h7 top-30 recoveries with canaries clean

Same-artist canary: Δ ≥ −0.005 (mandatory).
Diff-artist canary: Δ ≥ −0.005.
Top-20 overlap ≥ 8/20.

## Phase 0 outputs

- `cache/r89/phase0_fold0/model.pt` (TrackFusionModel weights, ~14 MB)
- `cache/r89/phase0_fold0/catalog_track_embs.npy` (47K × 1024 fp16, 96 MB)
- `cache/r89/phase0_fold0/oof_r89_lists.json` (1600 dev cases × 300 candidates with scores)
- `exp/eval/expR89_phase0.json` (gate report)
- `docs/r89_phase0_result.md`

## Escalation path

- **Phase 0 fails**: archive learned multimodal. Means even encoder-level fusion can't extract signal.
- **Phase 0 passes ambiguously** (one of the OR conditions): Phase 0b adds query-side LoRA.
- **Phase 0 passes cleanly** (h7 +0.005+): proceed directly to Phase 1 5-fold + blind candidate.

## Risks

- **Query embeddings frozen**: forces the model to find a fused track representation that already matches R84 query space. If R84 query space doesn't have the right structure for image/lyrics signals, projections can't compensate. Phase 1 LoRA addresses this.
- **Modality coverage**: audio CLAP has ~99% coverage; tracks missing audio get zero vector through gate (effectively excluded). Image: 99% coverage. Lyrics: 99%. Attrs: same as text metadata, near-100%.
- **Same-artist canary**: R85/R88 showed multimodal heavily biases to same-artist. Learned fusion may amplify this. Phase 0 gate will catch.
- **Overfitting on 3.5M params with 128K pairs**: should be fine but worth monitoring.
- **Cost overrun**: $0.15 budget is tight; if model needs to scale up (more proj_dim, deeper fusion), budget doubles to $0.30. Still trivially cheap vs R84's $7.50.

## What this design notably excludes

- Audio CLAP first run unless coverage check passes (~99% should be fine)
- Hard-negative anything (R79 + R85 lessons)
- Query-side LoRA in Phase 0 (variable isolation — defer to Phase 1)
- Multi-task losses (just InfoNCE)
- Pre-trained multimodal model adaptation (CLIP/BLIP/etc.) — out of scope; this is "fuse pre-extracted embeddings", not "retrain a foundation model"

## Files (to be created)

- `scripts/expR89_phase0_build_modality_cache.py` — one-time: build per-catalog-track fused
  embedding matrices (text, image, lyrics, attrs, audio), save as fp16 npy
- `scripts/expR89_phase0_train_fusion.py` — Phase 0 training script (A100)
- `scripts/expR89_phase0_eval.py` — Phase 0 eval against R84c OOF
- `cache/r89/` (gitignored bulk; pair_manifest already shipped)

## Estimated total Phase 0 work

| step | wall | $ |
|---|---|---:|
| Build modality cache (one-time, ~1.5 GB) | ~10 min Mac | 0 |
| Train fold-0 fusion model | ~5 min A100 | 0.15 |
| Eval fold-0 + compare to R84c OOF | ~5 min Mac | 0 |
| **Total Phase 0** | ~20 min | **0.15** |

If Phase 0 passes, Phase 1 (5-fold + blind candidate) adds ~$5-10 + Opus response regen.

## Decision needed before launch

1. Build modality cache + design freeze: Y/N?
2. Audio CLAP included Phase 0 or held for Phase 0b? (recommend include — coverage 99%)
3. Phase 0 fold-0 only or fold-0 + fold-1 sanity check? (recommend fold-0 only for cost)

Pending user approval before any GPU work.

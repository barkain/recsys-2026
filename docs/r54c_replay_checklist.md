# R54c Replay Checklist

**Purpose:** Reproduce the R54c production submission (composite 0.6106, #5 Blind-A) from scratch.
Covers environment → data → retriever → ranking → polish → zip → verification.

**Last validated:** 2026-05-16 (original build). This doc describes the replay path only.

---

## 1. Environment Setup

```bash
# Clone & enter
git clone <origin-url> recsys-2026
cd recsys-2026
git checkout main          # R54c merged; or use r54c-response-polish for history

# Python 3.13+ via uv
uv sync                    # installs all deps from pyproject.toml

# Key env vars
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME=~/.cache/huggingface       # or project-local .hf_cache/
export ANTHROPIC_API_KEY=<your-key>       # needed for response generation (Haiku)
```

**GPU requirements:**
- R54 5-fold training: GPU recommended (T4+ / ~40 min per fold on CUDA; hours on CPU).
  Script: `expR54_phase3_full5fold_train.py --device cuda`
- R54 production model training: GPU required (Colab T4, ~40 min).
  Script: `expR54_phase3_production_blind.py --device cuda`
- All other steps (LR, response gen, polish): CPU-only, local machine.

**HF dataset auth:**
- `talkpl-ai/TalkPlayData-Challenge-Dataset` — requires HF token with dataset access.
- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` — same.
- `talkpl-ai/TalkPlayData-Challenge-Blind-A` — same.
- Run `huggingface-cli login` or set `HF_TOKEN` env var.

---

## 2. Data Ingestion

```bash
# Pre-download datasets (will cache under HF_HOME)
uv run python -c "
from datasets import load_dataset
load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset')
load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata')
load_dataset('talkpl-ai/TalkPlayData-Challenge-Blind-A')
"
```

**Expected datasets:**
| Dataset | Split | Rows | Purpose |
|---|---|---:|---|
| TalkPlayData-Challenge-Dataset | dev | 1000 sessions (8000 cases) | Training/eval |
| TalkPlayData-Challenge-Dataset | train | 15199 sessions | Train-split augmentation |
| TalkPlayData-Challenge-Track-Metadata | all_tracks | ~39705 tracks | Catalog features |
| TalkPlayData-Challenge-Blind-A | test | 80 sessions | Blind submission target |

---

## 3. R54 Retriever Training

The R54 retriever is a 5-fold BGE-base-en-v1.5 ensemble, each fold trained on 80% dev + 20K train-split pairs.

### 3a. 5-fold OOF training (for LR dev features)

```bash
# All 5 folds (CUDA recommended, ~40 min/fold)
uv run python scripts/expR54_phase3_full5fold_train.py --device cuda

# Or single-fold:
uv run python scripts/expR54_phase3_full5fold_train.py --fold 0 --device cuda
```

**Script:** `scripts/expR54_phase3_full5fold_train.py`
**Hyperparams:** BAAI/bge-base-en-v1.5, 1 epoch, batch=32, LR=2e-5, tau=0.05, max_seq=256, top-300.
**Train augmentation:** 20K pairs, session-balanced (max 2/session), seed=0.
**Output:** `cache/r54/phase3_full/oof_r54_lists.json` (aggregated OOF lists with cosine scores).
**Fold models:** `cache/r54/phase3_full/fold_{1,2,3,4}/model/` + `track_embs.npy`
  - Fold 0 reuses smoke checkpoint: `cache/r54/phase3_smoke/fold_0/`

### 3b. 5-fold ensemble blind retrieval (for blind-A features)

```bash
uv run python scripts/expR54_phase3_ensemble_blind.py
```

**Script:** `scripts/expR54_phase3_ensemble_blind.py`
**Requires:** All 5 fold model dirs + `track_embs.npy` per fold.
**Method:** Average cosines across 5 fold models → top-300 per blind query (played excluded).
**Output:** `cache/r54_production/blind_r54_lists.json`
  - Format: `{ "lists": { sid: [(tid, score), ...] }, "manifest": {...} }`

**Alternative (single production model, NOT used in R54c):**
`scripts/expR54_phase3_production_blind.py` — trains single all-data model. Used by R55 only.

---

## 4. R21 / Source Cache Build

The LR uses 8 retrieval sources: A, B, C, D, F, ALS, R21, R54. Sources A/B/C/D/F/ALS are produced by earlier retrieval modules; R21 is the first-gen sentence-transformer retriever.

**R21 artifacts (must already exist):**
| Path | Content |
|---|---|
| `cache/r21_production/model/` | R21 SentenceTransformer |
| `cache/r21_production/track_embeddings.npy` | R21 catalog embeddings |
| `cache/r21_production/track_ids.json` | Track ID ordering |
| `cache/r21_production/dev_r21_oof_lists.json` | R21 OOF retrieval for 8000 dev cases |

**Dev payload (all sources pre-computed):**
- `exp/eval/_R12_all_turns_payload.pkl` — contains `cases`, `src_a`, `src_b`, `src_c`, `src_d`, `src_f`, track metadata maps.

**RRF fusion at pool@300:**
- Source weights: `{"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}`
- RRF k=20, pool top-300.
- Implementation: `scripts/expF1_cfbpr_retrieval.weighted_rrf()`

---

## 5. LR Training (37-Feature LambdaRank)

```bash
uv run python scripts/expR54_phase3_blind_submission.py --phase train
```

**Script:** `scripts/expR54_phase3_blind_submission.py`, function `phase_train()`
**Training data:** All 8000 dev cases. GT label = 1 if candidate == ground-truth track.
**Feature set (37):**

| Group | Features | Count |
|---|---|---:|
| Base rank/presence | rrf_rank_inv, src_{a,b,c,d,f,als}_{rank_inv,pres}, r21_{rank_inv,presence}, n_sources, source_count_v2 | 16 |
| Query/context | last_artist_match, last_tag_jaccard, query_{artist,title,meta}_tok_overlap, is_played, recency_score, n_history | 8 |
| ALS/popularity | als_dot, popularity, pool_artist_frac, pool_artist_count | 4 |
| Album (R39) | same_album_{last1,last3,any}, album_history_count, pool_same_album_count | 5 |
| R54 | r54_rank_inv, r54_presence, r54_cosine | 3 |
| **Total** | | **37** |

**LightGBM params:** lambdarank, ndcg@20, num_leaves=31, LR=0.05, min_data_in_leaf=10, 300 rounds, seed=0.
**Output:**
- `cache/r54_phase3_lr_model.txt` — the production LR model
- `cache/r54_phase3_als.npz` — ALS factors for blind retrieval
- `cache/r54_phase3_track_pop.json` — popularity stats
- `cache/r54_phase3_payload_maps.pkl` — track metadata maps

**Expected dev pool_hit@300:** 0.6220 (same as CV5 production baseline).
**Runtime:** ~15-20 min on local CPU (featurization is the bottleneck).

---

## 6. Blind Retrieval + LR Scoring

### 6a. Feature extraction

```bash
uv run python scripts/expR54_phase3_blind_submission.py --phase blind
```

**Script:** `phase_blind()` in `expR54_phase3_blind_submission.py`
**Requires:** R54 blind lists (`cache/r54_production/blind_r54_lists.json`), R21 model, ALS/pop/maps caches, HF Blind-A dataset, BM25 index, track-similarity index, CFBPR index.
**Retrieves per query:** A (track-sim), B (BM25 last_music_meta), C (BM25 full), D (track neighbor), F (CFBPR), ALS, R21, R54.
**Output:** `cache/r54_phase3_blind_features.pkl` — 80 rows with pool + feature matrices.

### 6b. Scoring + response assembly

```bash
uv run python scripts/expR54_phase3_blind_submission.py --phase score
```

**Script:** `phase_score()` in `expR54_phase3_blind_submission.py`
**Steps:**
1. Load LR model, predict on 80 feature matrices.
2. Top-20 per session by LR score.
3. Response reuse: if prior submission (R39/R27b/R25/R21) has a top-1 that's in the new top-20, reuse its response text.
4. Response generation: for changed top-1s, regenerate with Haiku.
5. Validation: 20 tracks/row, unique, non-empty response, valid catalog IDs.

**Output:** `exp/inference/blind_a/r54_phase3_exploratory_submission.{json,zip}`
**Expected churn vs R39:** ~15-20 top-1 changes.

---

## 7. Response Polish (R54c Step)

### 7a. R54b response alignment

```bash
uv run python scripts/expR54b_response_aligned.py
```

**Script:** `scripts/expR54b_response_aligned.py`
**Purpose:** Take R54 exploratory track IDs (bitwise), regenerate responses that align to the actual top-1 recommendation (not inherited from R39). Fixes the LLM judge score from 4.65→4.70.
**Output:** `exp/inference/blind_a/r54b_aligned_submission.{json,zip}`

### 7b. R54c polish (audit + targeted regen)

```bash
uv run python scripts/expR54c_response_polish.py --phase audit
uv run python scripts/expR54c_response_polish.py --phase polish
```

**Script:** `scripts/expR54c_response_polish.py`
**Phase audit:** Scores all 80 R54b responses against 7 weakness criteria (trailing_question, too_short/long, boilerplate, no_user_query_overlap, no_track_or_artist_mention, descriptor_heavy). Plus the "you're drawn to" repeated-opener cluster.
**Phase polish:**
1. Regenerate only flagged rows (~21) with enriched prompt + forbidden openers list.
2. Apply `strip_tag_prefix()` universally — removes metadata leaks like "country\n\n...".
3. Hard gates: bitwise identical track IDs, LexDiv >= 0.83, no trailing questions, 40-150 words.

**Output:**
- `exp/eval/expR54c_audit.json` — audit flags per row
- `exp/inference/blind_a/r54c_polish_submission.{json,zip}` — **PRODUCTION**
- `exp/inference/blind_a/r54c_polish_metadata.json`

**Verification that polish didn't regress:**
- Track IDs must be bitwise identical to R54b.
- LexDiv should be ≥0.83 (actual: 0.8381).
- LLM judge should hold at 4.70.
- Composite should match R54b (both: 0.6106).

---

## 8. Submission Zip Build

The final zip is produced by `expR54c_response_polish.py --phase polish`.

**Format:**
```
r54c_polish_submission.zip
└── prediction.json
```

**prediction.json structure:**
```json
[
  {
    "session_id": "<uuid>",
    "turn_number": <int>,
    "predicted_track_ids": ["<track_id_1>", ..., "<track_id_20>"],
    "predicted_response": "<response text>"
  },
  ... // 80 rows
]
```

**Validation commands:**
```bash
# Verify row count
uv run python -c "
import json, zipfile
with zipfile.ZipFile('exp/inference/blind_a/r54c_polish_submission.zip') as zf:
    data = json.loads(zf.read('prediction.json'))
    assert len(data) == 80, f'Expected 80, got {len(data)}'
    for r in data:
        assert len(r['predicted_track_ids']) == 20
        assert len(set(r['predicted_track_ids'])) == 20
        assert r['predicted_response'].strip()
    print(f'PASS: 80 rows × 20 tracks = {80*20} total predictions')
"
```

**Expected:** 80 sessions × 20 tracks = 1600 track predictions.

---

## 9. Verification Table

After rebuild, verify artifact matches the submitted version:

| Check | Expected | How to verify |
|---|---|---|
| Row count | 80 | `len(data) == 80` |
| Tracks per row | 20, unique | `len(set(r['predicted_track_ids'])) == 20` |
| Total predictions | 1600 | 80 × 20 |
| Track IDs vs R54b | Bitwise identical | `r54c[i]['predicted_track_ids'] == r54b[i]['predicted_track_ids']` for all i |
| nDCG@20 | 0.4925 | Leaderboard (determined by track IDs, not responses) |
| LexDiv (Distinct-2) | 0.8381 | Local estimate or leaderboard |
| LLM judge | 4.70 | Leaderboard |
| Composite | **0.6106** | Leaderboard |
| Leaderboard rank | #5 (as of 2026-05-16) | Public scoreboard |
| No prefix leaks | All rows pass `strip_tag_prefix` | Audit phase output |
| No trailing questions | 0 flagged | Audit phase output |
| No forbidden openers | 0 instances of "you're drawn to" etc. | Grep responses |

**Hash verification (if zip was preserved):**
```bash
shasum -a 256 exp/inference/blind_a/r54c_polish_submission.zip
# Compare against known hash from original build
```

---

## 10. Recovery Scenarios

| Problem | Recovery |
|---|---|
| `cache/r54_phase3_lr_model.txt` missing | Re-run `--phase train`. Deterministic (seed=0, same data). |
| `cache/r54_production/blind_r54_lists.json` missing | Re-run `expR54_phase3_ensemble_blind.py`. Requires all 5 fold models present. |
| Fold model(s) missing | Re-run `expR54_phase3_full5fold_train.py --fold <N>`. GPU required. Fold 0 can reuse smoke if available. |
| `cache/r21_production/` missing | Must retrain R21 from scratch (separate pipeline, `scripts/expR21_*`). Large dependency. |
| `exp/eval/_R12_all_turns_payload.pkl` missing | Rebuild via R12 preprocessing pipeline. Contains all dev cases + source lists. Critical. |
| HF dataset version changes | Pin to the version used during competition. If `local_files_only=True` fails, re-download same version. Arrow cache at `.hf_cache/datasets/talkpl-ai___*`. |
| BM25 / track-sim / CFBPR index missing | These are built by earlier scripts (`offline_retrieval_sweep.py`, `run_inference_blind_f1.py`). Rebuild from catalog. |
| Haiku API unavailable | Response generation fails. R54b responses are the fallback (equivalent composite). |
| ALS cache missing | Re-run `--phase train` which rebuilds `cache/r54_phase3_als.npz`. |
| R54 OOF (`phase3_full/oof_r54_lists.json`) missing | Re-run 5-fold training. Phase 2 OOF (`cache/r54/phase2_full/oof_r54_lists.json`) is an acceptable proxy for quick checks but NOT for production LR training. |
| Disk space issues | Safe to delete: `cache/r54_production/{model, track_embeddings.npy}` (R55 single-model, ~580MB, archived). Keep everything in the preserved list from `blind_a_final_state.md §6`. |

---

## Quick Reference: Full Replay Sequence

```bash
# 1. Environment
uv sync && huggingface-cli login

# 2. Data (skip if cached)
uv run python -c "from datasets import load_dataset; ..."

# 3. R54 5-fold training (GPU, ~3.5h total for 5 folds)
uv run python scripts/expR54_phase3_full5fold_train.py --device cuda

# 4. R54 ensemble blind retrieval (CPU, ~10 min)
uv run python scripts/expR54_phase3_ensemble_blind.py

# 5. LR training (CPU, ~20 min)
uv run python scripts/expR54_phase3_blind_submission.py --phase train

# 6. Blind feature extraction (CPU, ~15 min)
uv run python scripts/expR54_phase3_blind_submission.py --phase blind

# 7. LR scoring + initial responses (CPU, ~5 min + API calls)
uv run python scripts/expR54_phase3_blind_submission.py --phase score

# 8. R54b response alignment (CPU + API calls)
uv run python scripts/expR54b_response_aligned.py

# 9. R54c polish (CPU + API calls)
uv run python scripts/expR54c_response_polish.py --phase audit
uv run python scripts/expR54c_response_polish.py --phase polish

# 10. Verify
# Check zip, diff vs prior, leaderboard score
```

**Total wall time:** ~5-6 hours (dominated by GPU retriever training).
**Total cost:** Haiku API calls for ~20-30 response regenerations (~$0.05).

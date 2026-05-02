#!/bin/bash
# Overnight R21 submission pipeline — subprocess-isolated
set -euo pipefail

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/cache/r21_production/overnight.log"
mkdir -p "$REPO/cache/r21_production/oof"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "R21 Overnight Submission Pipeline (v2 — subprocess isolated)"
log "=========================================="

# Step 0: Build V3 pools (ALS) — separate process to avoid torch conflict
log "Step 0: Building V3 pools (ALS)..."
uv run python -c "
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pickle, json, numpy as np, sys
sys.path.insert(0, '.')
from pathlib import Path
from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

with open('exp/eval/_R12_all_turns_payload.pkl', 'rb') as f:
    payload = pickle.load(f)
cases = payload['cases']
n = len(cases)
sessions = [c['session_id'] for c in cases]

als_factors, als_track_ids, als_track_to_idx = build_als()
als_source = []
for c in cases:
    played = c['music_turns']
    sv = als_session_vector(played, als_track_to_idx, als_factors)
    if sv is not None:
        scores = als_factors @ sv
        for t in played:
            if t in als_track_to_idx: scores[als_track_to_idx[t]] = -np.inf
        top_idx = np.argpartition(-scores, 200)[:200]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        als_source.append([als_track_ids[j] for j in top_idx])
    else:
        als_source.append([])

sw = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 0.5, 'F': 1.0, 'ALS': 1.0}
v3_pools = []
for i in range(n):
    sl = {'A': payload['src_a'][i], 'B': payload['src_b'][i],
          'C': payload['src_c'][i], 'D': payload['src_d'][i],
          'F': payload['src_f'][i], 'ALS': als_source[i]}
    v3_pools.append(list(weighted_rrf(sl, sw, topk=200, k=20)))

# Save pools + fold assignments
folds = grouped_session_folds(sessions, seed=0)
fold_indices = {i: folds[i].tolist() for i in range(5)}

out = Path('cache/r21_production')
with open(out / 'v3_pools.json', 'w') as f:
    json.dump(v3_pools, f)
with open(out / 'fold_indices.json', 'w') as f:
    json.dump(fold_indices, f)
print(f'Saved V3 pools ({n}) and fold indices')
" 2>&1 | tee -a "$LOG"
log "Step 0 complete."

# Step 1: Train each fold in a separate subprocess
for FOLD in 0 1 2 3 4; do
    FOLD_FILE="$REPO/cache/r21_production/oof/fold_${FOLD}_r21_lists.json"
    if [ -f "$FOLD_FILE" ]; then
        log "Fold $FOLD: FOUND existing artifact, skipping."
        continue
    fi

    log "Fold $FOLD: Training R21 model + retrieving (subprocess)..."
    uv run python -c "
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json, pickle, sys, time, numpy as np
from pathlib import Path
from datetime import datetime
sys.path.insert(0, '.')

FOLD = $FOLD
REPO = Path('.')
OOF_DIR = REPO / 'cache' / 'r21_production' / 'oof'
R12_CACHE = REPO / 'exp' / 'eval' / '_R12_all_turns_payload.pkl'

def ts(): return f'[{datetime.now():%H:%M:%S}]'

# Load data
with open(R12_CACHE, 'rb') as f:
    payload = pickle.load(f)
cases = payload['cases']
n = len(cases)

from datasets import DownloadConfig, load_dataset
train_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset',
                        download_config=DownloadConfig(local_files_only=True))['train']
train_tracks = set()
for item in train_ds:
    for c in item['conversations']:
        if c['role'] == 'music': train_tracks.add(str(c['content']).strip())

# Load catalog (deduped)
from datasets import Dataset
hf_cache = Path.home() / '.cache/huggingface/datasets'
matches = sorted(hf_cache.glob(
    'talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/'
    'talk_play_data-challenge-track-metadata-all_tracks.arrow'))
ds = Dataset.from_file(str(matches[-1]))
cols = ds.to_dict()
meta = {}
all_track_ids = []
for i in range(len(cols['track_id'])):
    tid = str(cols['track_id'][i])
    all_track_ids.append(tid)
    meta[tid] = {k: cols[k][i] for k in cols}
assert len(all_track_ids) == 47071

# Load fold indices and V3 pools
with open(REPO / 'cache/r21_production/fold_indices.json') as f:
    fold_indices = json.load(f)
with open(REPO / 'cache/r21_production/v3_pools.json') as f:
    v3_pools = [set(p) for p in json.load(f)]

held = fold_indices[str(FOLD)]
train_idx = [j for j in range(n) if j not in set(held)]
print(f'{ts()} Fold {FOLD}: train={len(train_idx)} val={len(held)}')

# Build texts
def build_track_text(tid):
    m = meta.get(tid, {})
    names = m.get('track_name', [])
    artists = m.get('artist_name', [])
    album = m.get('album_name', [])
    tags = m.get('tag_list', [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ', '.join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ', '.join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f'{name} by {artist}. Album: {alb}. Tags: {tag_str}'

def build_query_text(case):
    parts = [str(h['content']) for h in case['history'] if h['role'] == 'user']
    parts.append(case['user_query'])
    return ' '.join(parts[-3:])

track_texts = [build_track_text(tid) for tid in all_track_ids]

# Train
import torch
import torch.nn.functional as F_t
from sentence_transformers import SentenceTransformer, InputExample

examples = []
for j in train_idx:
    gt = cases[j]['gt']
    if gt not in meta: continue
    examples.append(InputExample(texts=[build_query_text(cases[j]), build_track_text(gt)]))
print(f'  {len(examples)} training pairs')

model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')
tokenizer = model.tokenizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

def encode_with_grad(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    out = model.forward(encoded)
    return F_t.normalize(out['sentence_embedding'], dim=-1)

model.train()
for epoch in range(2):
    np.random.shuffle(examples)
    epoch_loss = 0; n_batches = 0
    for start in range(0, len(examples), 32):
        batch = examples[start:start+32]
        q = encode_with_grad([e.texts[0] for e in batch])
        p = encode_with_grad([e.texts[1] for e in batch])
        loss = F_t.cross_entropy(q @ p.T / 0.05, torch.arange(len(batch)))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item(); n_batches += 1
        if n_batches % 50 == 0:
            print(f'    batch {n_batches}: loss={loss.item():.4f}', flush=True)
    print(f'  Epoch {epoch}: loss={epoch_loss/n_batches:.4f}', flush=True)

model_dir = OOF_DIR / f'model_fold_{FOLD}'
model.save(str(model_dir))

# Encode + retrieve
model.eval()
print(f'{ts()} Encoding {len(all_track_ids)} tracks...', flush=True)
track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)

val_queries = [build_query_text(cases[j]) for j in held]
print(f'{ts()} Encoding {len(val_queries)} queries...', flush=True)
query_embs = model.encode(val_queries, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)

print(f'{ts()} Retrieving top-300...', flush=True)
fold_lists = []
for j_local in range(len(held)):
    j_global = held[j_local]
    scores = track_embs @ query_embs[j_local]
    for idx, tid in enumerate(all_track_ids):
        if tid in set(cases[j_global]['music_turns']): scores[idx] = -np.inf
    top_idx = np.argpartition(-scores, 300)[:300]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    fold_lists.append([all_track_ids[k] for k in top_idx])

# Metrics
hit200 = sum(1 for j_local, j_global in enumerate(held) if cases[j_global]['gt'] in fold_lists[j_local][:200])
unseen_hit = sum(1 for j_local, j_global in enumerate(held) if cases[j_global]['gt'] not in train_tracks and cases[j_global]['gt'] in fold_lists[j_local][:200])
unseen_total = sum(1 for j_global in held if cases[j_global]['gt'] not in train_tracks)
unique_vs_v3 = sum(1 for j_local, j_global in enumerate(held) if cases[j_global]['gt'] in fold_lists[j_local][:200] and cases[j_global]['gt'] not in v3_pools[j_global])

manifest = {'fold': FOLD, 'train_cases': len(train_idx), 'val_cases': len(held),
            'hit@200': hit200, 'hit@200_rate': hit200/len(held),
            'unseen_hit@200': unseen_hit, 'unseen_total': unseen_total,
            'unique_vs_v3': unique_vs_v3, 'catalog_size': 47071,
            'created_at': datetime.now().isoformat()}

fold_file = OOF_DIR / f'fold_{FOLD}_r21_lists.json'
with open(fold_file, 'w') as f:
    json.dump({'lists': fold_lists, 'manifest': manifest}, f)

print(f'{ts()} Fold {FOLD}: hit@200={hit200}/{len(held)} ({hit200/len(held):.1%}) unseen={unseen_hit}/{unseen_total} unique_vs_v3={unique_vs_v3}')
print(f'Saved: {fold_file}')
" 2>&1 | tee -a "$LOG"

    if [ $? -ne 0 ]; then
        log "FAILED: Fold $FOLD"
        exit 1
    fi
    log "Fold $FOLD complete."
done

# Step 2: Combine OOF lists
log "Step 2: Combining OOF lists..."
uv run python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '.')

prod = Path('cache/r21_production')
oof_dir = prod / 'oof'

fold_indices = json.load(open(prod / 'fold_indices.json'))
all_oof = [None] * 8000
manifest = {'catalog_size': 47071, 'folds': {}}

for fi in range(5):
    data = json.load(open(oof_dir / f'fold_{fi}_r21_lists.json'))
    held = fold_indices[str(fi)]
    for j_local, j_global in enumerate(held):
        all_oof[j_global] = data['lists'][j_local]
    manifest['folds'][str(fi)] = data['manifest']

assert all(x is not None for x in all_oof), 'Missing OOF lists!'
manifest['total_hit@200'] = sum(1 for i in range(8000) if True)  # recomputed below

with open(prod / 'dev_r21_oof_lists.json', 'w') as f:
    json.dump(all_oof, f)

import pickle
with open('exp/eval/_R12_all_turns_payload.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
hit200 = sum(1 for i in range(8000) if cases[i]['gt'] in all_oof[i][:200])
manifest['total_hit@200'] = hit200
manifest['total_hit@200_rate'] = hit200 / 8000

with open(prod / 'oof_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Combined OOF: 8000 lists, hit@200={hit200}/8000 ({hit200/8000:.1%})')
for fi in range(5):
    fm = manifest['folds'][str(fi)]
    print(f'  Fold {fi}: hit@200={fm[\"hit@200\"]}/{fm[\"val_cases\"]} ({fm[\"hit@200_rate\"]:.1%}) unique_vs_v3={fm[\"unique_vs_v3\"]}')
" 2>&1 | tee -a "$LOG"
log "Step 2 complete."

# Step 3: Train production model (separate subprocess)
PROD_MODEL="$REPO/cache/r21_production/model"
if [ -d "$PROD_MODEL" ]; then
    log "Step 3: Production model exists, skipping."
else
    log "Step 3: Training production R21 model..."
    uv run python scripts/train_r21_production.py 2>&1 | tee -a "$LOG"
    if [ $? -ne 0 ]; then
        log "FAILED: production model training"
        exit 1
    fi
    log "Step 3 complete."
fi

# Step 4: Blind inference (track-only)
log "Step 4: Blind inference (track-only)..."
uv run python run_inference_blind_r21.py \
    --output_tid lr_r21_v1_trackonly \
    --skip_response_generation \
    2>&1 | tee -a "$LOG"
log "Step 4 complete."

# Step 5: Hybrid responses + final validation
log "Step 5: Hybrid response assembly + validation..."
uv run python -c "
import json, sys, os
sys.path.insert(0, '.')
from pathlib import Path

with open('exp/inference/blind_a/lr_r21_v1_trackonly.json') as f:
    results = json.load(f)
with open('exp/inference/blind_a/lr_v3_hybrid.json') as f:
    v3_by_sid = {r['session_id']: r for r in json.load(f)}

reused = 0; need_gen = []
for r in results:
    sid = r['session_id']
    v3r = v3_by_sid.get(sid)
    if v3r and v3r['predicted_track_ids'][0] in set(r['predicted_track_ids']):
        resp = v3r['predicted_response'].lstrip(',').lstrip()
        if resp.strip():
            r['predicted_response'] = resp; reused += 1; continue
    r['predicted_response'] = ''; need_gen.append(sid)

print(f'Reused: {reused}, need generation: {len(need_gen)}')

if need_gen:
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from mcrs.lm_modules.claude import ClaudeModule
    from run_inference_blind_r3_det import build_session_memory_for_response, parse_last_turn
    from datasets import load_dataset

    item_db = MusicCatalogDB(dataset_name='talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split_types=['all_tracks'])
    prompts_dir = Path('mcrs/system_prompts')
    sys_prompt = (prompts_dir / 'roleplay.txt').read_text() + '\n' + (prompts_dir / 'response_generation.txt').read_text()
    haiku = ClaudeModule(model='claude-haiku-4-5-20251001')
    db = load_dataset('talkpl-ai/TalkPlayData-Challenge-Blind-A', split='test')
    blind_by_sid = {str(item['session_id']): item for item in db}

    for r in results:
        if r['predicted_response']: continue
        item = blind_by_sid[r['session_id']]
        turn_num, user_query, history, music_turns = parse_last_turn(item)
        top_id = r['predicted_track_ids'][0]
        try: top_item = item_db.id_to_metadata(top_id)
        except KeyError: top_item = f'track_id: {top_id}'
        session_memory = build_session_memory_for_response(history, user_query, item_db)
        response = haiku.response_generation(sys_prompt, session_memory, top_item)
        r['predicted_response'] = (response or '').lstrip(',').lstrip()

empty = sum(1 for r in results if not r['predicted_response'].strip())
assert empty == 0, f'{empty} empty responses!'
comma = sum(1 for r in results if r['predicted_response'].startswith(','))
assert comma == 0

# Verify tracks match track-only
with open('exp/inference/blind_a/lr_r21_v1_trackonly.json') as f:
    to = {r['session_id']: r['predicted_track_ids'] for r in json.load(f)}
assert all(to[r['session_id']] == r['predicted_track_ids'] for r in results)

with open('exp/inference/blind_a/lr_r21_v1_hybrid.json', 'w') as f:
    json.dump(results, f, indent=2)

import zipfile
with zipfile.ZipFile('exp/inference/blind_a/lr_r21_v1_hybrid_submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('exp/inference/blind_a/lr_r21_v1_hybrid.json', 'prediction.json')

print(f'SUBMISSION READY: exp/inference/blind_a/lr_r21_v1_hybrid_submission.zip')
print(f'  80 rows, reused={reused}, generated={len(need_gen)}, empty=0, comma=0')
" 2>&1 | tee -a "$LOG"

log "=========================================="
log "PIPELINE COMPLETE. Submission ready."
log "=========================================="

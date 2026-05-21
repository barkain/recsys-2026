"""R68.1 Mac-side: compute R68 top-300 OOF lists for fold-0 TRAIN cases via local cosine search.

Inputs (all already on disk):
  cache/r68/phase0_fold0/query_embeddings_train.npy        (6400, 1024)
  cache/r68/phase0_fold0/track_embeddings.npy              (47071, 1024)
  cache/r68/phase0_fold0/track_ids.json
  cache/r68/phase0_fold0/query_embeddings_train_case_ids.json

Output:
  cache/r68/phase0_fold0/oof_r68_lists_train.json   (case_id -> [[track_id, score], ...300])
"""
import json, pathlib, time
import numpy as np

REPO = pathlib.Path("/Users/nadavbarkai/dev/recsys-2026")
out_dir = REPO / "cache/r68/phase0_fold0"

train_q = np.load(out_dir / "query_embeddings_train.npy")
track_embs = np.load(out_dir / "track_embeddings.npy")
track_ids = json.loads((out_dir / "track_ids.json").read_text())
train_case_ids = json.loads((out_dir / "query_embeddings_train_case_ids.json").read_text())
print(f"train_q={train_q.shape}  track_embs={track_embs.shape}  n_tracks={len(track_ids)}  n_train={len(train_case_ids)}")

# Verify norms
print(f"  q_norm mean={np.linalg.norm(train_q, axis=1).mean():.4f}  t_norm mean={np.linalg.norm(track_embs, axis=1).mean():.4f}")

TOPK = 300
N_CHUNK = 256  # 256 * 47071 * 4 bytes = ~48 MB per chunk — comfortable on Mac
all_top_idx = []
all_top_scores = []
t0 = time.time()
for s in range(0, train_q.shape[0], N_CHUNK):
    e = min(s + N_CHUNK, train_q.shape[0])
    scores = train_q[s:e] @ track_embs.T   # (chunk, 47071)
    # argpartition + sort to get top-K efficiently
    idx_part = np.argpartition(scores, -TOPK, axis=1)[:, -TOPK:]   # unsorted top-K
    rows = np.arange(scores.shape[0])[:, None]
    sc_part = scores[rows, idx_part]
    # Sort within top-K descending
    order = np.argsort(-sc_part, axis=1)
    idx_sorted = idx_part[rows, order]
    sc_sorted = sc_part[rows, order]
    all_top_idx.append(idx_sorted)
    all_top_scores.append(sc_sorted)
    if (e % 1024 == 0) or e == train_q.shape[0]:
        print(f"  {e}/{train_q.shape[0]}  elapsed={time.time()-t0:.1f}s")
all_top_idx = np.concatenate(all_top_idx, axis=0)
all_top_scores = np.concatenate(all_top_scores, axis=0)
print(f"top-300 done in {time.time()-t0:.1f}s  shape={all_top_idx.shape}")

lists_train = {}
for i, cid in enumerate(train_case_ids):
    lists_train[cid] = [[track_ids[int(j)], float(all_top_scores[i][k])] for k, j in enumerate(all_top_idx[i])]
out_path = out_dir / "oof_r68_lists_train.json"
out_path.write_text(json.dumps(lists_train))
print(f"saved {out_path}: {out_path.stat().st_size/1e6:.2f} MB  ({len(lists_train)} cases)")

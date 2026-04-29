# Retrieval Sources Inventory

| Source | Type | Embedding Column | Dim | Tracks |
|---|---|---|---|---|
| A' (max-recent-5) | qwen3 cosine sim | metadata-qwen3_embedding_0.6b | 1024 | 46,579 |
| B (last_music_meta) | BM25 text | track_name, artist_name, album_name, tag_list | — | 47,071 |
| C (full_history) | BM25 text | track_name, artist_name, album_name, tag_list | — | 47,071 |
| D (track neighbors) | qwen3 cosine sim | metadata-qwen3_embedding_0.6b | 1024 | 46,579 |
| F (CF-BPR) | collaborative | cf-bpr | 128 | 46,455 |
| **S (sequence model)** | **learned prediction** | **targets metadata-qwen3 space** | **1024** | **46,579** |

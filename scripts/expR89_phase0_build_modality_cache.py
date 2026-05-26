"""R89 Phase 0 — build aligned modality cache.

Builds aligned (47K, dim) fp16 matrices for the 5 modalities + a presence mask:
- text:    BGE-large catalog metadata projection (from R80 cache, already on disk)
- image:   image-siglip2 (768-dim)
- lyrics:  lyrics-qwen3_embedding_0.6b (1024-dim)
- attrs:   attributes-qwen3_embedding_0.6b (1024-dim, from local R12 cache)
- audio:   audio-laion_clap (512-dim)

All aligned to a common track_id ordering (the BGE-large catalog from R80).
Missing-modality tracks are zero vectors with mask=0.

Outputs: cache/r89/modality_cache/
  - text.fp16.npy            (47071, 1024)
  - image.fp16.npy           (47071, 768)
  - lyrics.fp16.npy          (47071, 1024)
  - attrs.fp16.npy           (47071, 1024)
  - audio.fp16.npy           (47071, 512)
  - mask.fp16.npy            (47071, 5)   1.0 if modality present, 0.0 if missing
  - track_ids.json           list[str]    aligned ordering
  - dims.json                {"text": 1024, "image": 768, ...}
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R80_CATALOG_NPY = REPO / "cache" / "r80" / "catalog_track_embs_fp16.npy"
R80_CATALOG_IDS = REPO / "cache" / "r80" / "catalog_track_ids.json"
META_QWEN_DIR = REPO / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"
OUT_DIR = REPO / "cache" / "r89" / "modality_cache"


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 0, norms, 1.0).astype(arr.dtype)


def load_hf_modalities() -> tuple[list, dict[str, np.ndarray]]:
    """Load image/lyrics/attrs/audio from talkpl-ai/TalkPlayData-Challenge-Track-Embeddings."""
    from datasets import DownloadConfig, load_dataset  # type: ignore
    try:
        ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            download_config=DownloadConfig(local_files_only=True),
        )["all_tracks"]
        print(f"  HF dataset: local cache ({len(ds)} tracks)")
    except Exception:
        print(f"  HF dataset: downloading...")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
        print(f"  Downloaded ({len(ds)} tracks)")

    col_map = {
        "image": "image-siglip2",
        "lyrics": "lyrics-qwen3_embedding_0.6b",
        "attrs": "attributes-qwen3_embedding_0.6b",
        "audio": "audio-laion_clap",
    }
    tids = []
    cols: dict[str, list] = {k: [] for k in col_map}
    for item in ds:
        tids.append(str(item["track_id"]))
        for key, col in col_map.items():
            cols[key].append(item.get(col))

    out = {}
    for key, vecs in cols.items():
        dim = None
        for v in vecs:
            if v is not None and len(v) > 0:
                dim = len(v)
                break
        if dim is None:
            print(f"  {key}: NO VECTORS found")
            continue
        arr = np.zeros((len(vecs), dim), dtype=np.float32)
        n_valid = 0
        for i, v in enumerate(vecs):
            if v is not None and len(v) == dim:
                arr[i] = v
                n_valid += 1
        # L2-normalize where valid; zero rows stay zero
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr_norm = np.where(norms > 0, arr / np.where(norms > 0, norms, 1.0), 0.0).astype(np.float32)
        out[key] = arr_norm
        print(f"  {key}: shape={arr.shape}  valid={n_valid}/{len(vecs)}")
    return tids, out


def main():
    t0 = time.time()
    print(f"{ts()} R89 Phase 0 — build modality cache")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load R80 BGE-large catalog (47K × 1024 fp16) as the canonical text embedding + ordering
    print(f"\n{ts()} Loading R80 BGE-large catalog (canonical alignment)...")
    text_emb = np.load(R80_CATALOG_NPY).astype(np.float32)
    text_emb = l2_normalize(text_emb).astype(np.float16)
    text_ids = json.load(open(R80_CATALOG_IDS))
    n_tracks = len(text_ids)
    text_dim = text_emb.shape[1]
    text_id_to_idx = {tid: i for i, tid in enumerate(text_ids)}
    print(f"  text: {text_emb.shape} ({text_dim}-dim), n_tracks={n_tracks}")
    np.save(OUT_DIR / "text.fp16.npy", text_emb)

    # text mask: all present
    text_mask = np.ones(n_tracks, dtype=np.float16)

    # 2. Load HF modalities (image, lyrics, attrs, audio) and align to text_ids
    print(f"\n{ts()} Loading HF modalities (image/lyrics/attrs/audio)...")
    hf_ids, hf_mods = load_hf_modalities()
    hf_id_to_row = {tid: i for i, tid in enumerate(hf_ids)}

    masks = {"text": text_mask}
    for key, src_arr in hf_mods.items():
        dim = src_arr.shape[1]
        aligned = np.zeros((n_tracks, dim), dtype=np.float32)
        mask = np.zeros(n_tracks, dtype=np.float16)
        n_found = 0
        for i, tid in enumerate(text_ids):
            src_idx = hf_id_to_row.get(tid)
            if src_idx is None:
                continue
            v = src_arr[src_idx]
            if np.any(v):  # non-zero (valid)
                aligned[i] = v
                mask[i] = 1.0
                n_found += 1
        aligned = aligned.astype(np.float16)
        print(f"  {key}: aligned shape={aligned.shape}, coverage {n_found}/{n_tracks}")
        np.save(OUT_DIR / f"{key}.fp16.npy", aligned)
        masks[key] = mask

    # 4. Stack masks: (n_tracks, 5) ordered [text, image, lyrics, attrs, audio]
    modality_order = ["text", "image", "lyrics", "attrs", "audio"]
    mask_arr = np.stack([masks[k] for k in modality_order], axis=1)
    np.save(OUT_DIR / "mask.fp16.npy", mask_arr.astype(np.float16))
    print(f"\n  mask: {mask_arr.shape} (ordered {modality_order})")
    for i, key in enumerate(modality_order):
        cov = float(mask_arr[:, i].sum() / n_tracks)
        print(f"    {key}: coverage {cov:.3f}")

    # 5. Track IDs + dims
    with open(OUT_DIR / "track_ids.json", "w") as f:
        json.dump(text_ids, f)
    dims = {
        "text": text_dim,
        "image": int(np.load(OUT_DIR / "image.fp16.npy").shape[1]),
        "lyrics": int(np.load(OUT_DIR / "lyrics.fp16.npy").shape[1]),
        "attrs": int(np.load(OUT_DIR / "attrs.fp16.npy").shape[1]),
        "audio": int(np.load(OUT_DIR / "audio.fp16.npy").shape[1]),
    }
    with open(OUT_DIR / "dims.json", "w") as f:
        json.dump(dims, f, indent=2)
    print(f"\n  dims: {dims}")

    print(f"\n{ts()} Done. Cache dir: {OUT_DIR}")
    print(f"  Total elapsed: {(time.time() - t0)/60:.1f} min")
    total_size_mb = sum(p.stat().st_size for p in OUT_DIR.iterdir()) / 1e6
    print(f"  Total cache size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""Build per-row review bundles for R27 agentic submission audit."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pyarrow as pa

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R25_ZIP = REPO / "exp" / "inference" / "blind_a" / "r25_lexdiv_v2_submission.zip"
R21_ZIP = REPO / "exp" / "inference" / "blind_a" / "lr_r21_v1_hybrid_submission.zip"
OUT_PATH = REPO / "exp" / "inference" / "blind_a" / "r27_review_bundle.json"


def load_submission(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("prediction.json") as f:
            return json.load(f)


def load_blind_sessions():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        split="test",
        download_config=DownloadConfig(local_files_only=True),
    )
    return {str(item["session_id"]): item for item in ds}


def load_catalog():
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    with pa.memory_map(str(matches[-1]), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    cols = {col: table.column(col).to_pylist() for col in table.column_names}
    meta = {}
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        m = {k: cols[k][i] for k in cols}
        names = m.get("track_name", [])
        artists = m.get("artist_name", [])
        tags = m.get("tag_list", [])
        meta[tid] = {
            "track_id": tid,
            "name": names[0] if isinstance(names, list) and names else str(names),
            "artist": ", ".join(artists) if isinstance(artists, list) else str(artists),
            "tags": [str(t) for t in tags[:8]] if isinstance(tags, list) else [],
        }
    return meta


def format_conversation(session_item):
    convs = sorted(session_item["conversations"], key=lambda x: x["turn_number"])
    lines = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "music":
            lines.append(f"[Track played: {content}]")
        elif role == "assistant":
            lines.append(f"Assistant: {content[:200]}")
    return "\n".join(lines)


def main():
    print("Loading submissions...")
    r25 = load_submission(R25_ZIP)
    r21 = load_submission(R21_ZIP)

    print("Loading blind sessions...")
    sessions = load_blind_sessions()

    print("Loading catalog metadata...")
    meta = load_catalog()

    bundle = []
    for i in range(80):
        row = r25[i]
        sid = row["session_id"]
        session = sessions.get(sid)

        conversation = format_conversation(session) if session else "(unavailable)"

        # Last user message
        user_query = ""
        if session:
            convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
            for msg in reversed(convs):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break

        top20_meta = []
        for tid in row["predicted_track_ids"]:
            m = meta.get(tid, {"track_id": tid, "name": "?", "artist": "?", "tags": []})
            top20_meta.append(m)

        bundle.append({
            "row_index": i,
            "session_id": sid,
            "turn_number": row["turn_number"],
            "user_query": user_query,
            "conversation": conversation,
            "top20_track_ids": row["predicted_track_ids"],
            "top20_metadata": top20_meta,
            "r25_response": row["predicted_response"],
            "r21_response": r21[i]["predicted_response"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved {len(bundle)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

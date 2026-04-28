import os
from pathlib import Path

from datasets import Dataset, DownloadConfig, concatenate_datasets, load_dataset


# All available metadata fields in TalkPlayData-Challenge-Track-Metadata
ALL_CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "release_date",
    "tag_list",
]

# Fields shown to LLM in reranker/generation prompts.
DISPLAY_FIELDS = ["track_name", "artist_name", "album_name", "release_date", "tag_list"]
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


def _cached_track_arrow(split_type: str) -> str | None:
    matches = sorted(
        DEFAULT_HF_CACHE.glob(
            "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split_type}.arrow"
        )
    )
    return str(matches[-1]) if matches else None


class MusicCatalogDB:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str] = DISPLAY_FIELDS,
    ):
        self.corpus_types = corpus_types
        self.metadata_dict: dict[str, dict] = {}
        self._row_cache: dict[str, dict] = {}
        self._columns = None
        self._id_to_idx: dict[str, int] = {}

        arrow_paths = [_cached_track_arrow(split) for split in split_types]
        if all(arrow_paths):
            self._load_arrow_columns([p for p in arrow_paths if p])
        else:
            self._load_hf_dataset(dataset_name, split_types)

    def _load_arrow_columns(self, arrow_paths: list[str]) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        tables = []
        keep = set(DISPLAY_FIELDS + ["track_id"])
        for path in arrow_paths:
            with pa.memory_map(path, "r") as source:
                table = ipc.open_stream(source).read_all()
            tables.append(table.select([c for c in table.column_names if c in keep]))
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self._columns = {name: table[name] for name in table.column_names}
        track_ids = self._columns["track_id"].to_pylist()
        self._id_to_idx = {str(tid): i for i, tid in enumerate(track_ids)}

    def _load_hf_dataset(self, dataset_name: str, split_types: list[str]) -> None:
        cached_splits = []
        for split in split_types:
            arrow_path = _cached_track_arrow(split)
            if arrow_path:
                cached_splits.append(Dataset.from_file(arrow_path))
        if len(cached_splits) == len(split_types):
            metadata_concat_dataset = concatenate_datasets(cached_splits)
        else:
            metadata_dataset = load_dataset(
                dataset_name,
                download_config=DownloadConfig(local_files_only=True),
            )
            metadata_concat_dataset = concatenate_datasets(
                [metadata_dataset[s] for s in split_types]
            )
        keep = set(DISPLAY_FIELDS + ["track_id"])
        columns = {
            key: value for key, value in metadata_concat_dataset.to_dict().items()
            if key in keep
        }
        track_ids = [str(tid) for tid in columns["track_id"]]
        self.metadata_dict = {
            tid: {key: values[i] for key, values in columns.items()}
            for i, tid in enumerate(track_ids)
        }

    def _metadata_for(self, track_id: str) -> dict:
        track_id = str(track_id)
        if track_id in self._row_cache:
            return self._row_cache[track_id]
        if self.metadata_dict:
            return self.metadata_dict[track_id]
        idx = self._id_to_idx[track_id]
        row = {
            key: column[idx].as_py()
            for key, column in self._columns.items()
        }
        self._row_cache[track_id] = row
        return row

    def id_to_metadata(self, track_id: str) -> str:
        """Return a compact string representation of a track for LLM prompts."""
        metadata = self._metadata_for(track_id)
        parts = [f"track_id: {track_id}"]
        for field in DISPLAY_FIELDS:
            if field not in metadata:
                continue
            val = metadata[field]
            if isinstance(val, list):
                # For tag_list: keep top 5 tags to stay concise
                val = ", ".join(str(v) for v in val[:5])
            elif field == "release_date" and val:
                # Show only the year (first 4 chars) to save tokens
                val = str(val)[:4]
            parts.append(f"{field}: {str(val).lower()}")
        return ", ".join(parts)

    def id_to_full_metadata(self, track_id: str) -> dict:
        """Return the raw metadata dict for a track."""
        return self._metadata_for(track_id)

    def stringify_for_retrieval(self, track_id: str, fields: list[str]) -> str:
        """Build a corpus string from selected fields for indexing/retrieval."""
        metadata = self._metadata_for(track_id)
        parts = []
        for field in fields:
            if field not in metadata:
                continue
            val = metadata[field]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            parts.append(f"{field}: {val}")
        return "\n".join(parts)

from datasets import load_dataset, concatenate_datasets


# All available metadata fields in TalkPlayData-Challenge-Track-Metadata
ALL_CORPUS_TYPES = [
    "track_name",
    "artist_name",
    "album_name",
    "release_date",
    "tag_list",
]

# Minimal fields for terse item representation in LLM prompts
DISPLAY_FIELDS = ["track_name", "artist_name", "album_name"]


class MusicCatalogDB:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str] = DISPLAY_FIELDS,
    ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[s] for s in split_types]
        )
        self.corpus_types = corpus_types
        self.metadata_dict: dict[str, dict] = {
            item["track_id"]: item for item in metadata_concat_dataset
        }

    def id_to_metadata(self, track_id: str) -> str:
        """Return a compact string representation of a track for LLM prompts."""
        metadata = self.metadata_dict[track_id]
        parts = [f"track_id: {track_id}"]
        for field in DISPLAY_FIELDS:
            if field in metadata:
                val = metadata[field]
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                parts.append(f"{field}: {str(val).lower()}")
        return ", ".join(parts)

    def id_to_full_metadata(self, track_id: str) -> dict:
        """Return the raw metadata dict for a track."""
        return self.metadata_dict[track_id]

    def stringify_for_retrieval(self, track_id: str, fields: list[str]) -> str:
        """Build a corpus string from selected fields for indexing/retrieval."""
        metadata = self.metadata_dict[track_id]
        parts = []
        for field in fields:
            if field not in metadata:
                continue
            val = metadata[field]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            parts.append(f"{field}: {val}")
        return "\n".join(parts)

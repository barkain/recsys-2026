from pathlib import Path

from datasets import Dataset, DownloadConfig, concatenate_datasets, load_dataset


DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


def _cached_user_arrow(split_type: str) -> str | None:
    matches = sorted(
        DEFAULT_HF_CACHE.glob(
            "talkpl-ai___talk_play_data-challenge-user-metadata/default/*/*/"
            f"talk_play_data-challenge-user-metadata-{split_type}.arrow"
        )
    )
    return str(matches[-1]) if matches else None


class UserProfileDB:
    def __init__(self, dataset_name: str, split_types: list[str]):
        cached_splits = []
        for split in split_types:
            arrow_path = _cached_user_arrow(split)
            if arrow_path:
                cached_splits.append(Dataset.from_file(arrow_path))
        if len(cached_splits) == len(split_types):
            user_concat = concatenate_datasets(cached_splits)
        else:
            user_dataset = load_dataset(
                dataset_name,
                download_config=DownloadConfig(local_files_only=True),
            )
            user_concat = concatenate_datasets(
                [user_dataset[s] for s in split_types]
            )
        columns = user_concat.to_dict()
        user_ids = [str(uid) for uid in columns["user_id"]]
        self.profile_dict: dict[str, dict] = {
            uid: {key: values[i] for key, values in columns.items()}
            for i, uid in enumerate(user_ids)
        }

    def id_to_profile_str(self, user_id: str) -> str:
        if user_id not in self.profile_dict:
            return ""
        profile = self.profile_dict[user_id]
        parts = []
        for key in ("age_group", "gender", "country"):
            if key in profile and profile[key]:
                parts.append(f"{key}: {profile[key]}")
        return ", ".join(parts)

    def id_to_profile(self, user_id: str) -> dict:
        return self.profile_dict.get(user_id, {})

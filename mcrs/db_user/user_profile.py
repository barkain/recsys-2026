from datasets import load_dataset, concatenate_datasets


class UserProfileDB:
    def __init__(self, dataset_name: str, split_types: list[str]):
        user_dataset = load_dataset(dataset_name)
        user_concat = concatenate_datasets(
            [user_dataset[s] for s in split_types]
        )
        self.profile_dict: dict[str, dict] = {
            item["user_id"]: item for item in user_concat
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

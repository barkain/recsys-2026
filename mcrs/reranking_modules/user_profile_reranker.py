"""User-profile-aware reranker using pre-computed embeddings from TalkPlayData."""
import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets


class UserProfileReranker:
    """Reranks candidate tracks by user-track embedding similarity.

    Uses pre-computed user and track embeddings from:
      - talkpl-ai/TalkPlayData-2-User-Embeddings
      - talkpl-ai/TalkPlayData-2-Track-Embeddings

    Scores are combined with retrieval rank via linear interpolation.
    """

    def __init__(
        self,
        user_emb_dataset: str = "talkpl-ai/TalkPlayData-2-User-Embeddings",
        track_emb_dataset: str = "talkpl-ai/TalkPlayData-2-Track-Embeddings",
        cache_dir: str = "./cache",
        alpha: float = 0.3,  # weight for user-profile score vs. retrieval rank
    ):
        self.alpha = alpha
        self.cache_dir = cache_dir
        self.user_embeddings, self.user_ids = self._load_user_embeddings(user_emb_dataset, cache_dir)
        self.track_embeddings, self.track_ids = self._load_track_embeddings(track_emb_dataset, cache_dir)
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

    def _load_user_embeddings(self, dataset_name: str, cache_dir: str):
        cache_file = os.path.join(cache_dir, "user_embeddings.pt")
        ids_file = os.path.join(cache_dir, "user_ids.json")
        if os.path.exists(cache_file) and os.path.exists(ids_file):
            embs = torch.load(cache_file, map_location="cpu")
            with open(ids_file) as f:
                ids = json.load(f)
            return embs, ids
        ds = load_dataset(dataset_name)
        combined = concatenate_datasets([ds[s] for s in ds.keys()])
        ids = [item["user_id"] for item in combined]
        embs = torch.tensor([item["embedding"] for item in combined], dtype=torch.float32)
        embs = F.normalize(embs, p=2, dim=1)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(embs, cache_file)
        with open(ids_file, "w") as f:
            json.dump(ids, f)
        return embs, ids

    def _load_track_embeddings(self, dataset_name: str, cache_dir: str):
        cache_file = os.path.join(cache_dir, "track_embeddings.pt")
        ids_file = os.path.join(cache_dir, "track_emb_ids.json")
        if os.path.exists(cache_file) and os.path.exists(ids_file):
            embs = torch.load(cache_file, map_location="cpu")
            with open(ids_file) as f:
                ids = json.load(f)
            return embs, ids
        ds = load_dataset(dataset_name)
        combined = concatenate_datasets([ds[s] for s in ds.keys()])
        ids = [item["track_id"] for item in combined]
        embs = torch.tensor([item["embedding"] for item in combined], dtype=torch.float32)
        embs = F.normalize(embs, p=2, dim=1)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(embs, cache_file)
        with open(ids_file, "w") as f:
            json.dump(ids, f)
        return embs, ids

    def rerank(
        self,
        candidates: list[str],
        user_id: str | None,
        topk: int = 20,
    ) -> list[str]:
        """Rerank candidates by blending retrieval rank with user-profile similarity.

        If user_id is unknown, returns candidates unchanged (truncated to topk).
        """
        if user_id is None or user_id not in self.user_id_to_idx:
            return candidates[:topk]

        user_idx = self.user_id_to_idx[user_id]
        user_emb = self.user_embeddings[user_idx]  # [D]

        # Score each candidate by cosine similarity with user embedding
        # Only score candidates that have embeddings
        scores = {}
        for rank, track_id in enumerate(candidates):
            retrieval_score = 1.0 / (rank + 1)  # reciprocal rank
            if track_id in self.track_id_to_idx:
                t_idx = self.track_id_to_idx[track_id]
                t_emb = self.track_embeddings[t_idx]
                profile_score = float(torch.dot(user_emb, t_emb))
            else:
                profile_score = 0.0
            scores[track_id] = (1 - self.alpha) * retrieval_score + self.alpha * profile_score

        return sorted(scores, key=lambda x: scores[x], reverse=True)[:topk]

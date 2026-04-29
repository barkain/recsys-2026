"""Sequence-aware target-embedding model (Source S).

Predicts the next track's metadata-qwen3 embedding from conversational history.
Small transformer (~2.7M params) trained with InfoNCE + anti-collapse loss.

Architecture:
  - Per-turn input: track_emb (1024→256) + utt_emb (384→256) + accept_emb + turn_emb
  - 4-layer causal transformer encoder (d=256, 4 heads)
  - Readout: query_token hidden → Linear(256→1024) → L2-normalize
  - Scoring: cosine similarity against catalog metadata-qwen3 embeddings
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceRecommender(nn.Module):
    """Transformer that predicts next-track embedding from conversation history."""

    def __init__(
        self,
        track_emb_dim: int = 1024,
        utt_emb_dim: int = 384,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_turns: int = 8,
        output_dim: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_turns = max_turns

        # Input projections
        self.track_proj = nn.Linear(track_emb_dim, d_model)
        self.utt_proj = nn.Linear(utt_emb_dim, d_model)

        # Learned embeddings
        self.accept_emb = nn.Embedding(2, d_model)  # 0=rejected, 1=accepted
        self.turn_emb = nn.Embedding(max_turns, d_model)
        self.query_marker = nn.Parameter(torch.randn(d_model) * 0.02)

        # Layer norms for input fusion
        self.history_ln = nn.LayerNorm(d_model)
        self.query_ln = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        track_embs: torch.Tensor,       # (B, max_T_hist, 1024) — history track embeddings (padded)
        utt_embs: torch.Tensor,         # (B, max_T, 384) — all utterance embeddings (padded)
        accept_labels: torch.Tensor,    # (B, max_T_hist) — accept/reject labels (padded)
        turn_indices: torch.Tensor,     # (B, max_T) — turn position indices (padded)
        seq_lengths: torch.Tensor,      # (B,) — actual total sequence length (hist + 1 query)
    ) -> torch.Tensor:
        """Forward pass. Returns L2-normalized target embeddings (B, output_dim).

        seq_lengths[i] = number of history turns + 1 (the query token).
        The query utterance for example i is at utt_embs[i, seq_lengths[i]-1].
        """
        B = utt_embs.shape[0]
        device = utt_embs.device
        max_T = int(seq_lengths.max().item())
        max_T_hist = max_T - 1

        # Build history tokens from the non-padded prefix
        if max_T_hist > 0:
            hist_track = self.track_proj(track_embs[:, :max_T_hist])
            hist_utt = self.utt_proj(utt_embs[:, :max_T_hist])
            hist_accept = self.accept_emb(accept_labels[:, :max_T_hist])
            hist_turn = self.turn_emb(turn_indices[:, :max_T_hist])
            hist_tokens = self.history_ln(hist_track + hist_utt + hist_accept + hist_turn)
        else:
            hist_tokens = torch.zeros(B, 0, self.d_model, device=device)

        # Build query token — gather from each example's actual query position
        query_pos_in_utt = seq_lengths - 1  # (B,) — index of query utterance
        query_utt_gathered = utt_embs[torch.arange(B, device=device), query_pos_in_utt]  # (B, 384)
        query_utt = self.utt_proj(query_utt_gathered).unsqueeze(1)  # (B, 1, d)
        query_mark = self.query_marker.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        query_turn_gathered = turn_indices[torch.arange(B, device=device), query_pos_in_utt]  # (B,)
        query_turn = self.turn_emb(query_turn_gathered).unsqueeze(1)  # (B, 1, d)
        query_token = self.query_ln(query_utt + query_mark + query_turn)

        # Concatenate: [hist_1, ..., hist_{T-1}, query_T]
        tokens = torch.cat([hist_tokens, query_token], dim=1)  # (B, max_T, d)

        # Causal attention mask
        seq_len = tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

        # Padding mask: True = ignore. Each example has seq_lengths[i] valid tokens.
        pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= seq_lengths.unsqueeze(1)

        # Transformer
        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Readout: query token is always at position seq_lengths[i]-1 in the token sequence
        # (it's appended right after the history tokens, and history has seq_lengths[i]-1 tokens)
        query_pos = seq_lengths - 1  # (B,)
        query_hidden = hidden[torch.arange(B, device=device), query_pos]  # (B, d)

        # Project to output space and normalize
        target = self.output_proj(query_hidden)  # (B, output_dim)
        target = F.normalize(target, dim=-1)

        return target


def info_nce_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Args:
        query_emb: (B, D), L2-normalized
        positive_emb: (B, D), L2-normalized
        negative_embs: (B, K, D), L2-normalized
        tau: temperature
    """
    pos_sim = (query_emb * positive_emb).sum(-1, keepdim=True) / tau  # (B, 1)
    neg_sim = torch.einsum("bd,bkd->bk", query_emb, negative_embs) / tau  # (B, K)
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+K)
    labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)


def anti_collapse_loss(
    query_emb: torch.Tensor,
    last_track_emb: torch.Tensor,
    lam: float = 0.05,
) -> torch.Tensor:
    """Penalize cosine similarity to the last-played track embedding."""
    return lam * (query_emb * last_track_emb).sum(-1).mean()


class SequenceModelRetriever:
    """Source S — sequence-aware target-embedding retriever for inference."""

    def __init__(
        self,
        model_path: str,
        catalog_track_ids: list[str],
        catalog_emb_matrix: np.ndarray,
        utt_embeddings: np.ndarray,
        utt_index: dict[str, int],
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.catalog_track_ids = catalog_track_ids
        self.catalog_matrix = torch.from_numpy(catalog_emb_matrix).float()
        self.utt_embeddings = utt_embeddings
        self.utt_index = utt_index

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        config = checkpoint.get("config", {})
        self.model = SequenceRecommender(
            track_emb_dim=config.get("track_emb_dim", 1024),
            utt_emb_dim=config.get("utt_emb_dim", 384),
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 4),
            output_dim=config.get("output_dim", 1024),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Track embedding lookup (metadata-qwen3)
        self.track_emb_lookup = {
            tid: catalog_emb_matrix[i]
            for i, tid in enumerate(catalog_track_ids)
        }

    def _get_utt_emb(self, session_id: str, turn_number: int) -> np.ndarray:
        key = f"{session_id}:{turn_number}"
        idx = self.utt_index.get(key)
        if idx is not None:
            return self.utt_embeddings[idx]
        return np.zeros(384, dtype=np.float32)

    @torch.no_grad()
    def topn(
        self,
        session_id: str,
        history: list[dict],
        user_query: str,
        current_turn: int,
        topn: int = 200,
    ) -> list[str]:
        """Return top-N track_ids by predicted target embedding similarity.

        Returns [] for hist_0 (no prior music). The model requires at least one
        prior track embedding to condition on. This is by design — hist_0 cases
        are served by BM25 sources (B/C) which work well without music history.
        """
        music_turns = []
        user_turns = []
        for msg in history:
            if msg["role"] == "music":
                music_turns.append((int(msg["turn_number"]), str(msg["content"]).strip()))
            elif msg["role"] == "user":
                user_turns.append((int(msg["turn_number"]), str(msg["content"])))

        if not music_turns:
            return []

        # Build input tensors
        T_hist = len(music_turns)
        T = T_hist + 1  # +1 for current query

        track_embs = np.zeros((T_hist, 1024), dtype=np.float32)
        utt_embs = np.zeros((T, 384), dtype=np.float32)
        accept_labels = np.ones(T_hist, dtype=np.int64)
        turn_indices = np.zeros(T, dtype=np.int64)

        for i, (turn_num, tid) in enumerate(music_turns):
            emb = self.track_emb_lookup.get(tid)
            if emb is not None:
                track_embs[i] = emb
            utt_embs[i] = self._get_utt_emb(session_id, turn_num)
            turn_indices[i] = min(turn_num - 1, 7)

        # Current query
        utt_embs[T_hist] = self._get_utt_emb(session_id, current_turn)
        turn_indices[T_hist] = min(current_turn - 1, 7)

        # To tensors (batch=1)
        track_t = torch.from_numpy(track_embs).unsqueeze(0).to(self.device)
        utt_t = torch.from_numpy(utt_embs).unsqueeze(0).to(self.device)
        accept_t = torch.from_numpy(accept_labels).unsqueeze(0).to(self.device)
        turn_t = torch.from_numpy(turn_indices).unsqueeze(0).to(self.device)
        lengths = torch.tensor([T], dtype=torch.long, device=self.device)

        # Forward
        target_emb = self.model(track_t, utt_t, accept_t, turn_t, lengths)  # (1, 1024)

        # Score against catalog
        scores = (target_emb @ self.catalog_matrix.to(self.device).T).squeeze(0)  # (N_catalog,)

        # Exclude played tracks
        played_set = {tid for _, tid in music_turns}
        played_indices = [
            i for i, tid in enumerate(self.catalog_track_ids) if tid in played_set
        ]
        if played_indices:
            scores[played_indices] = -float("inf")

        # Top-N
        topk_indices = torch.topk(scores, min(topn, len(scores))).indices.cpu().numpy()
        return [self.catalog_track_ids[i] for i in topk_indices]

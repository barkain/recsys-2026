# Competition Strategy — RecSys Challenge 2026

## Baseline Analysis

The baseline system:
- **Retrieval**: BM25 or BERT over {track_name, artist_name, album_name}
- **LLM**: Llama-3.2-1B-Instruct for response generation
- **Corpus**: Only 3 text fields per track (very sparse signal)

## Improvement Roadmap

### Phase 1: Enriched Item Representation (High impact, low effort)
- Add `tag_list` to corpus_types (genre, mood, style tags)  
- Use pre-computed track embeddings from `talkpl-ai/TalkPlayData-2-Track-Embeddings`
- Enrich item strings with more descriptive metadata

### Phase 2: Better Retrieval Model
- Replace bert-base-uncased with a music-aware or instruction-tuned embedding model
  - E5-large-v2 or BGE-M3 for better semantic understanding
  - Optionally: CLAP for audio-text cross-modal retrieval
- Hybrid retrieval: BM25 (sparse) + dense = reciprocal rank fusion

### Phase 3: Reranking Stage
- Post-retrieval reranking using:
  - User profile embeddings from `TalkPlayData-2-User-Embeddings`
  - Conversation-aware scoring: weight recent turns more heavily
  - LLM-as-reranker: prompt an LLM to score top-50 candidates

### Phase 4: Better Response Generation
- Upgrade from Llama-1B to a better model (Claude via API)
- Improve system prompts for diversity of responses
- Include more track context in generation prompt

### Phase 5: Fine-tuning (if time allows)
- Semantic ID approach: fine-tune LLM to directly generate track IDs
- Collaborative filtering signals from user interaction data

## Implementation Order

1. [x] Set up repo structure, clone baseline
2. [x] Explore HuggingFace datasets to understand data schema
3. [x] Implement enriched BM25 with tag_list (quick win)
4. [x] Replace BERT with E5-base-v2 dense retriever
5. [x] Add hybrid retrieval (BM25 + dense, RRF fusion)
6. [x] Add reranking stage: user-profile reranker + LLM listwise reranker
7. [x] Claude API for generation (Haiku) and query reformulation
8. [x] Upgrade dense model to E5-large-v2 + add release_date to corpus
9. [x] Improve LLM reranker prompt with music-aware scoring criteria
10. [x] Richer id_to_metadata: includes tag_list + year for LLM context
11. [ ] Evaluate on devset, iterate
12. [ ] Try BGE-M3 or music-specific embedding model
13. [ ] Explore pre-computed TalkPlayData-2-Track-Embeddings for dense retrieval
14. [ ] Sliding-window LLM reranker (rerank by smaller windows, merge)

## Key Datasets
- Conversations: `talkpl-ai/TalkPlayData-Challenge-Dataset`
- Track metadata: `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`
- User profiles: `talkpl-ai/TalkPlayData-Challenge-User-Metadata`
- Track embeddings: `talkpl-ai/TalkPlayData-2-Track-Embeddings`
- User embeddings: `talkpl-ai/TalkPlayData-2-User-Embeddings`

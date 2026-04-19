# RecSys Challenge 2026 — Echo's Music CRS System

Conversational music recommendation system for the RecSys Challenge 2026.
Primary metric: nDCG@20.

## Strategy

Two-stage pipeline with enhancements:
1. **Retrieval**: BM25 + dense retrieval hybrid with enriched item representations (tags, mood)
2. **Reranking**: LLM-based reranker with user profile personalization  
3. **Generation**: Claude-powered response generation

See `STRATEGY.md` for full plan.

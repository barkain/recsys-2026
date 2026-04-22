"""Test the parallel batch_rerank pattern used in batch_chat.

Tests the core logic in isolation — no model imports, no dataset loading.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# ── Track concurrency ─────────────────────────────────────────────────────────
_active = 0
_peak = 0
_lock = threading.Lock()

def mock_llm_rerank(candidates, session_memory, item_db, topk=20):
    """Simulates an LLM reranker API call (~50ms latency)."""
    global _active, _peak
    with _lock:
        _active += 1
        if _active > _peak:
            _peak = _active
    time.sleep(0.05)
    with _lock:
        _active -= 1
    return candidates[:topk]

def mock_user_profile_rerank(candidates, user_id, topk=None):
    return candidates[:topk] if topk else candidates

def mock_rerank_one(i, candidates, user_ids, session_memories_full,
                    reranker=None, llm_reranker=None, item_db=None):
    """Mirrors _rerank() logic from crs_system.py."""
    if reranker:
        candidates = mock_user_profile_rerank(candidates, user_ids[i], topk=len(candidates))
    if llm_reranker:
        candidates = mock_llm_rerank(candidates, session_memories_full[i], item_db, topk=20)
    return candidates[:20]


# ── Test 1: Parallel execution is faster than sequential ─────────────────────
def test_parallel_speedup():
    global _active, _peak
    _active = 0; _peak = 0

    BATCH = 8
    candidates_batch = [[f"t{j}" for j in range(50)] for _ in range(BATCH)]
    user_ids = [f"u{i}" for i in range(BATCH)]
    memories = [[{"role": "user", "content": f"q{i}"}] for i in range(BATCH)]

    def rerank_one(args):
        i, cands = args
        return mock_rerank_one(i, cands, user_ids, memories, llm_reranker=True, item_db=None)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(rerank_one, enumerate(candidates_batch)))
    elapsed = time.time() - t0

    sequential_expected = BATCH * 0.05
    print(f"[test_parallel_speedup] elapsed={elapsed:.3f}s, peak_concurrent={_peak}")
    assert elapsed < sequential_expected * 0.6, (
        f"Expected parallel speedup: {elapsed:.3f}s should be < {sequential_expected*0.6:.3f}s"
    )
    assert _peak > 1, f"Expected >1 concurrent workers, got peak={_peak}"
    assert len(results) == BATCH
    print(f"  ✓ {BATCH}x parallel in {elapsed:.3f}s vs ~{sequential_expected:.2f}s sequential")


# ── Test 2: Result ordering is preserved ─────────────────────────────────────
def test_ordering():
    BATCH = 6
    candidates_batch = [[f"t{i}_{j}" for j in range(10)] for i in range(BATCH)]
    user_ids = [f"u{i}" for i in range(BATCH)]
    memories = [[] for _ in range(BATCH)]

    def rerank_one(args):
        i, cands = args
        # Return a sentinel so we can check ordering
        return [f"result_{i}"] + cands[1:]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(rerank_one, enumerate(candidates_batch)))

    for i, r in enumerate(results):
        assert r[0] == f"result_{i}", f"Order mismatch at {i}: {r[0]}"
    print(f"  ✓ Ordering preserved for {BATCH}-item batch")


# ── Test 3: No reranker (graceful passthrough) ───────────────────────────────
def test_no_reranker():
    BATCH = 3
    candidates_batch = [[f"t{j}" for j in range(50)] for _ in range(BATCH)]
    user_ids = [f"u{i}" for i in range(BATCH)]
    memories = [[] for _ in range(BATCH)]

    def rerank_one(args):
        i, cands = args
        return mock_rerank_one(i, cands, user_ids, memories,
                               reranker=None, llm_reranker=None, item_db=None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(rerank_one, enumerate(candidates_batch)))

    assert len(results) == BATCH
    assert all(len(r) == 20 for r in results)
    print(f"  ✓ No-reranker passthrough: {BATCH} results, all len=20")


if __name__ == "__main__":
    print("Testing parallel batch_rerank pattern...\n")
    test_parallel_speedup()
    test_ordering()
    test_no_reranker()
    print("\nAll tests passed ✓")

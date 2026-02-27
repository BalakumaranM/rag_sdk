# Vector Search Scaling — How Query Time Grows with Corpus Size

## The Two Algorithms in This SDK

### InMemoryVectorStore — Brute Force

Source: `rag_sdk/vectorstore/memory.py`

```python
dot_product = np.dot(self._vector_matrix, query_vec)          # every vector
cosine_similarities = dot_product / (norm_matrix * norm_query) # every vector
```

Every query computes cosine similarity against **all N vectors**. No indexing, no
shortcut. Time complexity: **O(N × d)** where N = number of vectors, d = embedding
dimension (typically 1536).

If N doubles, query time doubles. Strictly linear.

### ChromaDB / Qdrant / FAISS (HNSW) — Approximate Nearest Neighbour

Source: `rag_sdk/vectorstore/chroma_store.py`

```python
metadata={"hnsw:space": self.config.distance_function}
```

Chroma builds an HNSW (Hierarchical Navigable Small World) graph internally.
HNSW is a multi-layer graph: the top layer is sparse (long-range connections),
the bottom layer is dense (local connections). A query navigates greedily from
the top layer downward, visiting only O(log N) nodes total.

Time complexity: **O(log N × d)** — effectively sub-linear.

If N doubles, query time increases by log(2) ≈ a fraction of a millisecond.
This is why it is described as "nearly constant" for practical purposes.

---

## Actual Query Time by Corpus Size

Embedding dimension = 1536 (OpenAI text-embedding-3-small). Numbers are approximate,
assuming a modern CPU (50B float32 FLOPS/s).

| Corpus size | Brute force (InMemory) | HNSW (Chroma/Qdrant) | Typical LLM call |
|------------|----------------------|---------------------|------------------|
| 20 chunks | < 0.01 ms | ~0.1 ms | 1,000–5,000 ms |
| 2,000 chunks | ~0.1 ms | ~0.5 ms | 1,000–5,000 ms |
| 100,000 chunks | ~5 ms | ~1 ms | 1,000–5,000 ms |
| 1,000,000 chunks | ~50 ms | ~2 ms | 1,000–5,000 ms |
| 10,000,000 chunks | ~500 ms | ~5 ms | 1,000–5,000 ms |
| 100,000,000 chunks | ~5,000 ms (5 s!) | ~8 ms | 1,000–5,000 ms |

**Key observation:** The LLM generation call dominates total latency for any corpus
below ~1M chunks with brute force, or essentially indefinitely with HNSW. Vector
search is not the bottleneck in any realistic RAG scenario.

---

## Why "Same Query Time" for Our Research

Our experiment uses:
- Per-question mode: ~20 chunks
- Shared corpus mode: ~2,000 chunks

```
20 chunks   → 20 × 1,536 multiply-adds  =    30,720 ops  →  ~0.01 ms
2,000 chunks → 2,000 × 1,536 multiply-adds = 3,072,000 ops → ~0.10 ms
```

Against a 1–5 second LLM call, the 0.09 ms difference is invisible. That is what
"same query time" means in our context — the search overhead does not affect the
latency metric we report in experiment results.

---

## At What Scale Does It Actually Matter?

### When to switch from InMemoryVectorStore to HNSW-based store

| Trigger | Reason |
|---------|--------|
| > 100,000 chunks | Brute-force starts adding >5ms per query; also RAM becomes a concern |
| Need persistence | InMemory loses all data on restart |
| Need filtering at scale | InMemory's metadata filter is a Python loop; HNSW stores support native filtering |

### 1 Lakh (100,000) Times Our Current Corpus

```
2,000 chunks × 100,000 = 200,000,000 chunks

Brute force: ~10,000 ms (10 seconds) per query  →  completely unusable
HNSW:        ~8–15 ms per query                 →  completely fine
```

HNSW wins decisively. The difference between the two algorithms is about 1,000× at
this scale.

---

## Where HNSW Itself Breaks Down

HNSW is not infinitely scalable. Two failure modes:

### 1. Recall Degrades (Quality Issue)
HNSW is *approximate* — it does not guarantee finding the exact top-k vectors, only
the very likely ones. Below ~100M vectors, recall stays above 99% with default
settings. Above that, you need to increase the `ef_search` parameter, which trades
accuracy for speed.

### 2. Memory Becomes the Bottleneck (Hardware Issue)
```
1 vector at 1536 dims × 4 bytes (float32) = 6.1 KB
1,000,000 vectors                          = 6.1 GB (just vectors, before HNSW graph)
HNSW graph overhead                        ≈ +25%
```

| Scale | RAM needed | Approach |
|-------|-----------|---------|
| < 10M vectors | 64 GB | Single machine, HNSW in RAM |
| 10M–100M vectors | 512 GB+ | Single machine or SSD-backed (Qdrant on_disk) |
| 100M–1B vectors | Multiple machines | Sharded vector search |
| > 1B vectors | Distributed cluster | Pinecone, Weaviate distributed, Qdrant distributed |

---

## Recommendation for This Research Project

| Phase | Corpus | Use |
|-------|--------|-----|
| Phase 1–2 (baseline, chunking) | ~20 chunks per question | InMemoryVectorStore — fine |
| Phase 3–5 (retrieval, reranking, generation) | ~2,000 chunks (shared corpus) | InMemoryVectorStore — still fine |
| Phase 7 (PDF corpus) | ~5,000–50,000 chunks | Chroma ephemeral (HNSW) if InMemory gets slow |
| Production deployment | Millions of chunks | Qdrant or Pinecone with HNSW |

The rule of thumb: **switch to an HNSW-backed store when your corpus exceeds ~100,000
chunks**, not because of query speed (that is fine until tens of millions), but because
InMemoryVectorStore holds everything in RAM with no persistence and no native filtering.

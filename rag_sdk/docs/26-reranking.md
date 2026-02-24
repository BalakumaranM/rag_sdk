# Reranking

Reranking is an optional post-retrieval step that re-scores documents for better relevance. Disabled by default.

```yaml
retrieval:
  reranking:
    enabled: true
    provider: "cohere"  # "cohere" | "cross-encoder"
```

All rerankers implement `BaseReranker`:

```python
class BaseReranker(ABC):
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]: ...
```

## Over-Fetch Behavior

When reranking is enabled, `RAG.query()` automatically over-fetches `3 * top_k` candidates from the retriever, then passes them to the reranker which returns the best `top_k`:

```
query(top_k=5) → retriever fetches 15 → reranker returns top 5
```

This ensures the reranker has enough candidates to choose from.

## Cohere Reranker

Uses the Cohere Rerank API.

```yaml
retrieval:
  reranking:
    enabled: true
    provider: "cohere"
    cohere:
      model: "rerank-v3.5"
      top_n: 5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `COHERE_API_KEY` |
| `model` | `str` | `"rerank-v3.5"` | Rerank model name |
| `top_n` | `int` | `5` | Number of results to return from the API |

**When to use:** Fast, high-quality reranking without local compute. Requires a Cohere API key.

## CrossEncoder Reranker

Uses a cross-encoder model from `sentence-transformers` that jointly encodes query and document for more accurate relevance scoring.

```bash
pip install rag_sdk[cross-encoder]
```

```yaml
retrieval:
  reranking:
    enabled: true
    provider: "cross-encoder"
    cross_encoder:
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
      batch_size: 32
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model |
| `batch_size` | `int` | `32` | Prediction batch size |

**When to use:** Local reranking without API calls. Slower than Cohere for large batches but no API cost.

## Comparison

| Reranker | Latency | Cost | Accuracy | Local |
|----------|---------|------|----------|-------|
| Cohere | Low | Per-query | High | No |
| CrossEncoder | Medium | Free | High | Yes |

## Pipeline Integration

Reranking fits between retrieval and generation:

```
Retriever (3x over-fetch) → Reranker (score & trim) → GenerationStrategy
```

The `RAG.query()` method handles this automatically. You don't need to call the reranker directly.

## See Also

- [Retrieval Strategies](25-retrieval-strategies.md) — retrieval stage
- [Generation Strategies](27-generation-strategies.md) — generation stage
- [API: Providers](33-api-providers.md) — `BaseReranker` ABC

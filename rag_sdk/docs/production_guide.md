# Production Guide {#production-guide}

## Performance & Scalability

### Embedding Rate Limits
The SDK handles API rate limits (e.g., OpenAI) using intelligent queuing and backoff strategies.
*   **Batching:** Dynamic batch sizing to maximize throughput.
*   **Key Rotation:** Support for multiple API keys to distribute load.

```yaml
embedding_rate_limiting:
  queue:
    enabled: true
    backend: "redis"
  key_rotation:
    enabled: true
    keys:
      openai:
        - api_key: "${KEY_1}"
        - api_key: "${KEY_2}"
```

### Vector Store Indexing
For large datasets (1M+ documents), use async background indexing to keep the application responsive.
*   **Progressive Availability:** Query new data as it becomes indexed.
*   **Background Workers:** Use Celery/Redis for offloading indexing tasks.

## Caching Strategy

Multi-level caching to reduce latency and costs:
1.  **Embedding Cache:** Cache vector embeddings to avoid re-computing.
2.  **Retrieval Cache:** Cache search results for common queries.
3.  **LLM Cache:** Cache generated answers for identical queries.

## Observability

The SDK integrates with OpenTelemetry for comprehensive tracing and metrics.

### Metrics (Prometheus)
*   Query latency (p95, p99)
*   Token usage / Cost
*   Retrieval quality scores
*   Error rates

### Cost Tracking
Built-in integration with LiteLLM for detailed cost attribution per tenant or user.

```python
from rag_sdk.monitoring import CostTracker

tracker = CostTracker()
print(f"Monthly Cost: ${tracker.get_monthly_cost()}")
```

## Cold Start Optimization

To reduce latency on the first query:
*   **Pre-warming:** Load embedding models and establish database connections on startup.
*   **Keep-Alive:** Periodic pings to keep connections active.

## Disaster Recovery

*   **Replicas:** Configure vector store replicas.
*   **Backups:** Automated backups of metadata and vector indices.
*   **Fallbacks:** Graceful degradation if the primary LLM or Vector Store is unavailable. 

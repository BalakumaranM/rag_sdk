# Debugging & Monitoring {#debugging-monitoring}

## Error Handling

The SDK provides robust error handling for common RAG issues:

*   **Corrupted Files:** Automatic repair attempts (using tools like `qpdf`) or moving to a Dead Letter Queue.
*   **Password-Protected PDFs:** Mechanisms to request passwords or skip.
*   **Rate Limits:** Automatic backoff and retry for API calls.

```yaml
error_handling:
  pdf_errors:
    corrupted:
      strategy: "repair_and_retry"
    password_protected:
      strategy: "attempt_common_then_skip"
```

## Observability

### Health Checks
Monitor the health of your RAG system components:
*   Vector Store connectivity
*   LLM API availability
*   OCR model loading status

```yaml
monitoring:
  health_probes:
    readiness: { enabled: true, endpoint: "/health/ready" }
    liveness: { enabled: true, endpoint: "/health/live" }
```

### Alerting
Set up alerts for critical thresholds:
*   Query latency > 5s
*   Embedding queue backing up
*   High error rates on ingestion

### Debugging Tools (Future)
We are developing tools to visualize retrieval paths and inspect intermediate steps in the query pipeline.

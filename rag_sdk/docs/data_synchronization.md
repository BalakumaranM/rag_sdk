# Data Synchronization {#data-synchronization}

## Real-Time Updates

Keep your knowledge base in sync with your data sources using our hybrid approach:
*   **Webhooks:** Immediate updates for supported providers (Notion, Google Drive, Slack, GitHub).
*   **Polling:** Fallback scheduled polling for other sources.

```yaml
sync:
  strategy: "hybrid"
  webhooks:
    enabled: true
    server:
      port: 8080
  polling:
    intervals:
      google_drive: 3600  # 1 hour
```

## Incremental Indexing

We track changes efficiently to avoid re-indexing the entire corpus.
*   **Change Detection:** Using ETags or content hashes to identify modified files.
*   **Differential Sync:** Only updating changed chunks, protecting bandwidth and API costs.

```python
syncer = IncrementalSyncer(config)
syncer.sync_google_drive(folder_id="...")
```

## Deletion Propagation

When a document is deleted from a source (e.g., a file removed from Google Drive):
1.  **Cascade Deletion:** The SDK automatically removes the document, its chunks, and embeddings from the vector store.
2.  **Verification:** A verification step ensures no orphaned data remains.
3.  **Tombstones:** Optional soft-delete for audit trails.

```yaml
deletion_propagation:
  enabled: true
  cascade:
    delete_embeddings: true
    delete_chunks: true
```

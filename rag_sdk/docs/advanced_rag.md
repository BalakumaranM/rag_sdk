# Advanced RAG Techniques {#advanced-rag}

## Complex Retrieval Strategies

### Multi-Hop Reasoning
The SDK supports multi-hop reasoning to answer complex questions that require synthesizing information from multiple documents.
*   **Graph Traversal:** Using a knowledge graph (e.g., Neo4j) to find relationships between entities.
*   **Iterative Retrieval:** Breaking down a complex query into sub-questions and retrieving answers sequentially.

```python
from rag_sdk.reasoning import MultiHopReasoner

reasoner = MultiHopReasoner(config)
response = await reasoner.query("Who led the project that reduced costs by 20%?")
```

### Query Understanding
We employ advanced query preprocessing to improve retrieval:
*   **Acronym Expansion:** Automatically expanding industry-specific terms.
*   **Entity Resolution:** Resolving pronouns ("we", "it") and relative dates ("last week").
*   **Intent Classification:** Classifying queries (e.g., "factual", "how-to") to adjust retrieval strategy.

```yaml
query_understanding:
  preprocessing:
    enabled: true
    steps:
      - spell_correction
      - acronym_expansion
      - entity_recognition
```

## Long Document Handling

For very large documents (e.g., 100+ page PDFs), the SDK uses a hierarchical approach:
1.  **Hierarchical Chunking:** Breaking documents into chapters -> sections -> paragraphs.
2.  **Streaming Ingestion:** Processing large files in streams to keep memory usage low.

```python
ingestor.ingest_large_document("huge_manual.pdf")
```

## Conflict Resolution

When dealing with duplicate or conflicting information from multiple sources (e.g., Google Drive vs. Notion), the SDK uses a **Smart Deduplication** engine.

It resolves conflicts based on:
*   **Freshness:** Newer content wins.
*   **Source Authority:** Official docs > Slack messages.
*   **Completeness:** More detailed content > summaries.

```yaml
conflict_resolution:
  resolution:
    strategy: "smart_merge"
    smart_merge:
      enabled: true
      rules:
        content: "longest"
        metadata: "union"
```

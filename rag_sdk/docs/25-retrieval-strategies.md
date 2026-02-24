# Retrieval Strategies

The SDK provides 7 base retrieval strategies and 2 composable wrappers. Select the base strategy via config:

```yaml
retrieval:
  strategy: "dense"  # "dense" | "graph_rag" | "advanced_graph_rag" | "raptor" | "multi_query" | "hybrid" | "self_rag"
  top_k: 5
```

All retrievers implement `BaseRetriever`:

```python
class BaseRetriever(ABC):
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]: ...
```

## Base Strategies

### Dense (Default)

Standard vector similarity search. Embeds the query and searches the vector store.

```yaml
retrieval:
  strategy: "dense"
  top_k: 5
```

Simple, fast, and effective for most use cases.

### Multi-Query

Generates multiple query variations using an LLM, retrieves for each, and deduplicates results.

```yaml
retrieval:
  strategy: "multi_query"
  multi_query:
    num_queries: 3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_queries` | `int` | `3` | Number of alternative queries to generate |

Wraps a dense retriever internally. Returns the union of results from all query variations, deduplicated by document ID.

**When to use:** When a single query phrasing may miss relevant documents.

### Hybrid

Combines dense vector retrieval with BM25 sparse retrieval using Reciprocal Rank Fusion (RRF).

```yaml
retrieval:
  strategy: "hybrid"
  hybrid:
    bm25_weight: 0.5
    rrf_k: 60
    bm25_k1: 1.5
    bm25_b: 0.75
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bm25_weight` | `float` | `0.5` | Weight for BM25 in RRF (0.0–1.0); dense weight = 1.0 - bm25_weight |
| `rrf_k` | `int` | `60` | RRF constant to prevent high-ranked items from dominating |
| `bm25_k1` | `float` | `1.5` | BM25 term frequency saturation |
| `bm25_b` | `float` | `0.75` | BM25 length normalization |

The BM25 index is built automatically during `ingest_documents()`. Dense retrieval captures semantic meaning while BM25 captures exact keyword matches.

**When to use:** When queries contain specific terms or names that semantic search alone may miss.

### Graph RAG

Combines dense retrieval with knowledge graph traversal. Builds an in-memory graph of entities and relationships extracted from documents, then boosts results connected to query entities.

```yaml
retrieval:
  strategy: "graph_rag"
  graph_rag:
    max_entities_per_chunk: 10
    max_relationships_per_chunk: 15
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entities_per_chunk` | `int` | `10` | Max entities to extract per chunk |
| `max_relationships_per_chunk` | `int` | `15` | Max relationships to extract per chunk |

The knowledge graph is built during `ingest_documents()` via LLM-based entity/relationship extraction. At query time, graph-connected documents get a 1.2x score boost.

**When to use:** Entity-rich domains (legal, medical, knowledge bases) where relationship traversal finds relevant context that vector search alone misses.

### Advanced Graph RAG

Microsoft-style GraphRAG with hierarchical community detection, entity/relationship enrichment, and three search modes. Requires `networkx` (`pip install rag_sdk[advanced-graph-rag]`).

**Ingestion pipeline** (runs once during `ingest_documents()`):
1. Extract entities and relationships with full descriptions and relationship weights via few-shot LLM prompting
2. Merge duplicate entities and relationships across chunks (accumulate descriptions, LLM-merge)
3. Detect hierarchical communities (Leiden or Louvain algorithm at multiple resolutions)
4. Generate structured community reports: title, summary, findings, rank score
5. Embed community summaries for semantic search

**Three search modes** at query time:

| Mode | Description |
|------|-------------|
| `local` | Traverses entity neighborhood from query-matched entities; prepends graph context to dense chunks |
| `global` | Map-reduce over top community summaries; synthesizes a comprehensive answer |
| `drift` | HyDE entry point → initial retrieval → iterative follow-up questions → refined answer |

```yaml
retrieval:
  strategy: "advanced_graph_rag"
  advanced_graph_rag:
    search_mode: "local"                     # "local" | "global" | "drift"
    max_entities_per_chunk: 10               # entities to extract per chunk
    max_relationships_per_chunk: 15          # relationships to extract per chunk
    top_communities: 3                       # communities to use in global/drift search
    max_graph_hops: 2                        # neighborhood hops for local search
    drift_max_rounds: 2                      # follow-up rounds for drift search
    drift_follow_up_questions: 3             # questions generated per round
    community_detection_algorithm: "louvain" # "leiden" | "louvain"
    community_levels: 2                      # hierarchy depth (2 = coarse + fine)
    relationship_weight_in_graph: true       # use weight on graph edges
    entity_types:                            # domain-specific entity types
      - "person"
      - "organization"
      - "location"
      - "event"
      - "concept"
      - "technology"
      - "product"
      - "process"
      - "document"
      - "system"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `str` | `"local"` | `"local"`, `"global"`, or `"drift"` |
| `max_entities_per_chunk` | `int` | `10` | Max entities extracted per chunk |
| `max_relationships_per_chunk` | `int` | `15` | Max relationships extracted per chunk |
| `top_communities` | `int` | `3` | Communities used in global/drift modes |
| `max_graph_hops` | `int` | `2` | Neighborhood traversal depth for local mode |
| `drift_max_rounds` | `int` | `2` | Iterative follow-up rounds for drift mode |
| `drift_follow_up_questions` | `int` | `3` | Follow-up questions per drift round |
| `community_detection_algorithm` | `str` | `"louvain"` | `"leiden"` (requires `leidenalg`) or `"louvain"` |
| `community_levels` | `int` | `2` | Number of hierarchy levels (coarse→fine) |
| `node2vec_enabled` | `bool` | `False` | Structural entity embeddings (requires `node2vec`) |
| `relationship_weight_in_graph` | `bool` | `True` | Apply relationship weight to graph edges |
| `entity_types` | `List[str]` | 10 generic types | Override with domain-specific types for better extraction precision |

**Using `GraphIndexer` directly:**

For advanced workflows, access the ingestion pipeline directly:

```python
from rag_sdk.graph import GraphIndexer
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
indexer = GraphIndexer(embedding_provider, llm_provider, config.retrieval)
indexer.build_graph(documents)

# Inspect extracted data
print(indexer.entities)       # Dict[str, Entity]
print(indexer.relationships)  # List[Relationship]
print(indexer.communities)    # Dict[str, Community]
```

**When to use:** Complex multi-hop queries across large document collections; when you need both specific entity-level context (local) and broad dataset-spanning synthesis (global).

### RAPTOR

Builds a hierarchical tree of clustered document summaries. Clusters documents at multiple levels, generates summaries per cluster, and stores them alongside originals for multi-level retrieval.

```yaml
retrieval:
  strategy: "raptor"
  raptor:
    num_levels: 3
    clustering_method: "kmeans"
    max_clusters_per_level: 10
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_levels` | `int` | `3` | Number of tree levels |
| `clustering_method` | `str` | `"kmeans"` | Clustering algorithm |
| `max_clusters_per_level` | `int` | `10` | Max clusters per level |

The tree is built during `ingest_documents()`. Retrieval returns a mix of leaf documents and higher-level summaries, prioritizing leaf documents.

**When to use:** Large document collections where both specific details and high-level summaries are valuable.

### Self-RAG

Implements self-reflective retrieval: decides whether retrieval is needed, evaluates document relevance, and filters to only supported documents.

```yaml
retrieval:
  strategy: "self_rag"
  self_rag:
    check_support: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_support` | `bool` | `True` | Verify each document provides sufficient support |

Steps:
1. LLM decides if retrieval is needed (may return empty if not)
2. Retrieves 2x candidates from the base dense retriever
3. LLM evaluates relevance of each document
4. Optionally checks if each document supports answering the query

**When to use:** When you want the system to self-assess retrieval quality. High LLM cost per query.

## Composable Wrappers

These wrap any base strategy and are enabled via boolean flags:

```yaml
retrieval:
  strategy: "dense"  # any base strategy
  corrective_rag_enabled: true
  contextual_compression_enabled: true
```

### Corrective RAG

Evaluates relevance of retrieved results and refines the query if too few are relevant.

```yaml
retrieval:
  corrective_rag_enabled: true
  corrective_rag:
    relevance_threshold: 0.7
    max_refinement_attempts: 2
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relevance_threshold` | `float` | `0.7` | Min fraction of top_k that must be relevant |
| `max_refinement_attempts` | `int` | `2` | Max query refinement retries |

The LLM evaluates each document's relevance. If fewer than `threshold * top_k` are relevant, it rewrites the query and retries.

### Contextual Compression

Uses an LLM to extract only query-relevant content from each retrieved document.

```yaml
retrieval:
  contextual_compression_enabled: true
```

Over-fetches 2x documents, then compresses each with an LLM prompt asking for only the relevant extract. Documents with no relevant content are filtered out.

### Wrapper Composition

Wrappers are applied in order: base strategy → Corrective RAG → Contextual Compression. You can enable both:

```yaml
retrieval:
  strategy: "hybrid"
  corrective_rag_enabled: true
  contextual_compression_enabled: true
```

This produces: `ContextualCompression(CorrectiveRAG(HybridRetriever(...)))`.

## Over-Fetch Behavior with Reranking

When reranking is enabled, the retriever fetches `3 * top_k` documents to give the reranker more candidates:

```yaml
retrieval:
  top_k: 5
  reranking:
    enabled: true
```

This retrieves 15 candidates, reranks them, and returns the top 5.

## See Also

- [Reranking](26-reranking.md) — post-retrieval reranking
- [Generation Strategies](27-generation-strategies.md) — answer generation
- [API: RAG](30-api-rag.md) — `RAG.query()` reference

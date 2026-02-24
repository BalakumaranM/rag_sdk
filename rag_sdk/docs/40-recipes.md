# Recipes

Production-ready configurations for common use cases.

---

## 1. Minimal Local Setup (No API Keys)

Run entirely locally with no external API calls.

```yaml
project_name: "local-rag"

embeddings:
  provider: "local"
  local:
    model: "BAAI/bge-small-en-v1.5"

vectorstore:
  provider: "memory"

llm:
  provider: "openai"
  openai:
    base_url: "http://localhost:11434/v1"  # Ollama or local server
    model: "llama3"

retrieval:
  strategy: "dense"

generation:
  strategy: "standard"
```

```bash
pip install rag_sdk[local-embeddings]
```

## 2. Production with Pinecone

Cloud-hosted vector store with OpenAI for embeddings and generation.

```yaml
project_name: "production-rag"
environment: "production"

embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-small"
    dimensions: 1536

vectorstore:
  provider: "pinecone"
  pinecone:
    index_name: "prod-index"
    namespace: "default"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"
    temperature: 0.3

retrieval:
  strategy: "dense"
  top_k: 5
  reranking:
    enabled: true
    provider: "cohere"

generation:
  strategy: "standard"
```

## 3. High-Accuracy Pipeline

Maximize answer quality with semantic chunking, hybrid retrieval, reranking, and Chain of Verification.

```yaml
project_name: "high-accuracy"

document_processing:
  chunking:
    strategy: "semantic"
  semantic_chunking:
    breakpoint_percentile: 25.0
    min_chunk_size: 100

embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-large"
    dimensions: 3072

vectorstore:
  provider: "chroma"
  chroma:
    mode: "persistent"
    persist_path: "./chroma_db"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"
    temperature: 0.2
    max_tokens: 2000

retrieval:
  strategy: "hybrid"
  top_k: 10
  hybrid:
    bm25_weight: 0.4
  reranking:
    enabled: true
    provider: "cohere"
    cohere:
      model: "rerank-v3.5"

generation:
  strategy: "cove"
cove:
  max_verification_questions: 3
```

## 4. PDF-Heavy Pipeline

Optimized for ingesting and querying PDFs with Docling for complex layouts.

```yaml
project_name: "pdf-pipeline"

document_processing:
  chunk_size: 800
  chunk_overlap: 150
  chunking:
    strategy: "recursive"
  pdf_parser:
    backend: "docling"
    docling_do_ocr: true
    docling_do_table_structure: true
    docling_table_mode: "accurate"

embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-small"

vectorstore:
  provider: "faiss"
  faiss:
    index_type: "Flat"
    persist_path: "./faiss_index"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"

retrieval:
  strategy: "dense"

generation:
  strategy: "attributed"
attributed_generation:
  citation_style: "numeric"
```

```bash
pip install rag_sdk[docling,faiss]
```

## 5. Cited Answers

Generate answers with inline `[N]` citations for traceability.

```yaml
project_name: "cited-answers"

embeddings:
  provider: "openai"

vectorstore:
  provider: "memory"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"

retrieval:
  strategy: "dense"
  top_k: 5

generation:
  strategy: "attributed"
attributed_generation:
  citation_style: "numeric"
```

```python
result = rag.query("What are the key findings?")
print(result["answer"])
# "The study found significant improvements [1] with a 30% reduction [2]."

for cite in result["citations"]:
    print(f"  [{cite['citation_number']}] {cite['source']}: {cite['content_preview'][:80]}...")
```

## 6. Knowledge Graph Enhanced

Use Graph RAG for entity-rich domains.

```yaml
project_name: "graph-rag"

embeddings:
  provider: "openai"

vectorstore:
  provider: "chroma"
  chroma:
    mode: "persistent"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"

retrieval:
  strategy: "graph_rag"
  top_k: 5
  graph_rag:
    max_entities_per_chunk: 10
    max_relationships_per_chunk: 15

generation:
  strategy: "standard"
```

## 7. Self-Correcting Pipeline

Combine corrective RAG with contextual compression for self-improving retrieval.

```yaml
project_name: "self-correcting"

embeddings:
  provider: "openai"

vectorstore:
  provider: "memory"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"

retrieval:
  strategy: "dense"
  top_k: 5
  corrective_rag_enabled: true
  contextual_compression_enabled: true
  corrective_rag:
    relevance_threshold: 0.7
    max_refinement_attempts: 2

generation:
  strategy: "standard"
```

## 8. Multi-Provider (Gemini + Cohere)

Mix providers — Gemini for LLM, Cohere for embeddings, CrossEncoder for reranking.

```yaml
project_name: "multi-provider"

embeddings:
  provider: "cohere"
  cohere:
    model: "embed-english-v3.0"

vectorstore:
  provider: "qdrant"
  qdrant:
    url: "http://localhost:6333"

llm:
  provider: "gemini"
  gemini:
    model: "gemini-2.5-flash"
    temperature: 0.5

retrieval:
  strategy: "multi_query"
  top_k: 5
  multi_query:
    num_queries: 3
  reranking:
    enabled: true
    provider: "cross-encoder"
    cross_encoder:
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

generation:
  strategy: "standard"
```

```bash
pip install rag_sdk[qdrant,cross-encoder]
```

## 9. RAPTOR for Large Collections

Hierarchical summarization for large document sets.

```yaml
project_name: "raptor-pipeline"

document_processing:
  chunk_size: 500
  chunk_overlap: 100

embeddings:
  provider: "openai"

vectorstore:
  provider: "faiss"
  faiss:
    index_type: "HNSW"
    persist_path: "./raptor_index"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"

retrieval:
  strategy: "raptor"
  top_k: 10
  raptor:
    num_levels: 3
    max_clusters_per_level: 10

generation:
  strategy: "standard"
```

```bash
pip install rag_sdk[faiss]
```

## 10. Advanced GraphRAG (Microsoft-style)

Full Microsoft GraphRAG pipeline with community detection and multi-modal search. Use `local` mode for specific entity queries, `global` for broad cross-dataset synthesis, and `drift` for exploratory research questions.

```yaml
project_name: "advanced-graph-rag"

embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-small"

vectorstore:
  provider: "chroma"
  chroma:
    mode: "persistent"
    persist_path: "./graph_rag_db"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"
    temperature: 0.2

retrieval:
  strategy: "advanced_graph_rag"
  top_k: 5
  advanced_graph_rag:
    search_mode: "local"           # switch to "global" for broad queries
    community_levels: 2
    community_detection_algorithm: "louvain"
    top_communities: 5
    max_graph_hops: 2

generation:
  strategy: "standard"
```

```bash
pip install rag_sdk[advanced-graph-rag]
```

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)

# Ingest — builds vector store + knowledge graph + communities
rag.ingest_documents(documents)

# Local search: entity neighborhood context
config.retrieval.advanced_graph_rag.search_mode = "local"
result = rag.query("How does RLHF relate to Constitutional AI?")

# Global search: cross-dataset synthesis
config.retrieval.advanced_graph_rag.search_mode = "global"
result = rag.query("What are the main themes across all documents?")

# Domain-specific entity types for better extraction precision
config.retrieval.advanced_graph_rag.entity_types = [
    "drug", "disease", "gene", "protein", "clinical_trial", "institution"
]
```

## See Also

- [Configuration](10-configuration.md) — config loading
- [YAML Reference](11-yaml-reference.md) — all config fields
- [Extending](41-extending.md) — custom providers

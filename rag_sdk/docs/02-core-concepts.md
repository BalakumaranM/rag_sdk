# Core Concepts

## Architecture

The RAG SDK is organized as a pipeline with pluggable components at each stage:

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Orchestrator                     │
│                      (rag_sdk.core.RAG)                  │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Ingestion   │  Retrieval   │  Reranking   │ Generation │
│  Pipeline    │  Pipeline    │  (optional)  │ Pipeline   │
└──────┬───────┴──────┬───────┴──────┬───────┴─────┬──────┘
       │              │              │             │
  ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐  ┌────▼─────┐
  │Document │   │ Retrieval │  │Reranker │  │Generation│
  │Loader   │   │ Strategy  │  │         │  │ Strategy │
  │Splitter │   │           │  │         │  │          │
  │Embedder │   │           │  │         │  │          │
  │VecStore │   │           │  │         │  │          │
  └─────────┘   └───────────┘  └─────────┘  └──────────┘
```

## Data Flow

### Ingestion Pipeline

```
PDF/Text Files
    │
    ▼
DocumentLoader.load_file()     → Document(content, metadata)
    │
    ▼
TextSplitter.split_documents() → List[Document] (chunked)
    │
    ▼
EmbeddingProvider.embed_documents() → List[List[float]]
    │
    ▼
VectorStoreProvider.add_documents(docs, embeddings)
    │
    ▼ (if applicable)
GraphRAGRetriever.build_graph() / GraphIndexer.build_graph() / RAPTORRetriever.build_tree() / HybridRetriever.index_documents()
```

### Query Pipeline

```
User Query (str)
    │
    ▼
Retriever.retrieve(query, top_k, filters) → List[Document]
    │
    ▼ (optional)
Reranker.rerank(query, docs, top_k) → List[(Document, float)]
    │
    ▼
GenerationStrategy.generate(query, docs) → Dict[str, Any]
    │                                        ├─ "answer": str
    │                                        ├─ "sources": List[Document]
    │                                        └─ "latency": float
    ▼
Result dict returned to caller
```

## The Document Model

`Document` is the universal data unit throughout the pipeline:

```python
from rag_sdk.document import Document

doc = Document(
    id="auto-generated-uuid",  # auto-assigned if not provided
    content="The actual text content...",
    metadata={
        "source": "report.pdf",
        "page_number": 3,
        "chunk_index": 0,
        "parent_id": "original-doc-uuid",
    },
)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID, auto-generated |
| `content` | `str` | Text content |
| `metadata` | `Dict[str, Any]` | Arbitrary key-value metadata |

After splitting, each chunk Document gets `chunk_index` and `parent_id` metadata linking it back to its source.

## Component Lifecycle

All components are initialized by the `RAG` constructor based on the `Config` object:

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)
# All components now initialized:
#   rag.embedding_provider  → EmbeddingProvider
#   rag.vector_store        → VectorStoreProvider
#   rag.llm_provider        → LLMProvider
#   rag.text_splitter       → BaseTextSplitter
#   rag.retriever           → BaseRetriever
#   rag.reranker            → BaseReranker | None
#   rag.generation_strategy → GenerationStrategy
#   rag.pdf_parser          → BasePDFParser
```

You typically interact with the `RAG` class directly rather than individual components:

```python
# Ingest
stats = rag.ingest_documents(documents)
stats = rag.ingest_pdf("report.pdf")

# Query
result = rag.query("What is the main finding?")
print(result["answer"])
```

## Pluggable Providers

Each pipeline stage supports multiple providers, selected via config:

| Stage | Providers |
|-------|-----------|
| **Embeddings** | `openai`, `cohere`, `gemini`, `voyage`, `local` |
| **Vector Store** | `memory`, `faiss`, `chroma`, `pinecone`, `weaviate`, `qdrant` |
| **LLM** | `openai`, `gemini`, `anthropic`, `cohere` |
| **Chunking** | `recursive`, `agentic`, `proposition`, `semantic`, `late` |
| **Retrieval** | `dense`, `graph_rag`, `advanced_graph_rag`, `raptor`, `multi_query`, `hybrid`, `self_rag` |
| **Reranking** | `cohere`, `cross-encoder` (optional, disabled by default) |
| **Generation** | `standard`, `cove`, `attributed` |

## Composable Retrieval Wrappers

Some retrieval features are **wrappers** that can layer on top of any base strategy:

- **Corrective RAG** — evaluates relevance and refines the query if results are poor
- **Contextual Compression** — uses an LLM to extract only query-relevant content from retrieved docs

These are enabled via boolean flags in config and apply regardless of which base strategy you choose:

```yaml
retrieval:
  strategy: "dense"
  corrective_rag_enabled: true
  contextual_compression_enabled: true
```

## Next Steps

- [Configuration](10-configuration.md) — configure all providers
- [Quickstart](01-quickstart.md) — build a pipeline end-to-end
- [Module Guides](20-document-loading.md) — deep dive into each stage

# API Reference: RAG

```python
from rag_sdk import RAG
```

## RAG

Main orchestrator that initializes all components from config and provides the ingestion/query interface.

### `__init__`

```python
RAG(config: Config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Config` | Validated configuration object |

Initializes all components based on config:
- `self.embedding_provider` — `EmbeddingProvider`
- `self.vector_store` — `VectorStoreProvider`
- `self.llm_provider` — `LLMProvider`
- `self.text_splitter` — `BaseTextSplitter`
- `self.retriever` — `BaseRetriever`
- `self.reranker` — `Optional[BaseReranker]` (None if reranking disabled)
- `self.generation_strategy` — `GenerationStrategy`
- `self.pdf_parser` — `BasePDFParser`

**Raises:** `ValueError` if any configured provider is unsupported or its config is missing.

### `ingest_documents`

```python
def ingest_documents(self, documents: List[Document]) -> Dict[str, int]
```

Splits documents into chunks, embeds them, stores in the vector store, and builds any required indexes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `documents` | `List[Document]` | Documents to ingest |

**Returns:** `Dict[str, int]` with keys:
- `"source_documents"` — number of input documents
- `"chunks"` — number of chunks after splitting

**Side effects:**
- Calls `GraphRAGRetriever.build_graph()` if using graph_rag strategy
- Calls `RAPTORRetriever.build_tree()` if using raptor strategy
- Calls `HybridRetriever.index_documents()` if using hybrid strategy

These are detected by unwrapping any Corrective RAG / Contextual Compression wrappers to find the innermost retriever.

### `ingest_pdf`

```python
def ingest_pdf(self, file_path: str) -> Dict[str, int]
```

Parses a PDF file and ingests the resulting documents.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the PDF file |

**Returns:** Same as `ingest_documents`.

Uses the configured `pdf_parser` backend and `one_document_per_page` setting. Internally calls `DocumentLoader.load_file()` then `ingest_documents()`.

### `query`

```python
def query(
    self,
    query: str,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]
```

Retrieves relevant documents, optionally reranks, and generates an answer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | The user's question |
| `top_k` | `Optional[int]` | `None` | Number of results (defaults to 5) |
| `filters` | `Optional[Dict[str, Any]]` | `None` | Metadata filters for retrieval |

**Returns:** `Dict[str, Any]` with keys:
- `"answer"` — `str`, the generated answer
- `"sources"` — `List[Document]`, retrieved (and optionally reranked) documents
- `"latency"` — `float`, total query time in seconds
- Strategy-specific keys (see [Generation Strategies](27-generation-strategies.md)):
  - CoVe: `"initial_answer"`, `"verification_qa"`
  - Attributed: `"citations"`

**Reranking behavior:** When reranking is enabled, the retriever fetches `3 * top_k` candidates. The reranker then returns the best `top_k`.

### `_unwrap_retriever` (static)

```python
@staticmethod
def _unwrap_retriever(retriever: BaseRetriever) -> BaseRetriever
```

Walks the `base_retriever` chain to find the innermost retriever. Used internally to detect Graph RAG, RAPTOR, and Hybrid retrievers through Corrective RAG / Contextual Compression wrappers.

## See Also

- [API: Config](31-api-config.md) — Config classes
- [Quickstart](01-quickstart.md) — usage examples

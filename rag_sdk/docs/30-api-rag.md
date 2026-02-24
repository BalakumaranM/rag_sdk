# API Reference: RAG

```python
from rag_sdk import RAG, Settings
```

---

## RAG

Main orchestrator that initialises all components from config and provides the
ingestion/query interface.

---

### `__init__`

```python
RAG(
    config: Config,
    *,
    embedding_provider: Optional[EmbeddingProvider] = None,
    llm_provider: Optional[LLMProvider] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `Config` | required | Validated configuration object |
| `embedding_provider` | `EmbeddingProvider` | `None` | Override the config-driven embedding provider |
| `llm_provider` | `LLMProvider` | `None` | Override the config-driven LLM provider |

Provider resolution priority (highest → lowest):
1. Explicit kwarg to `RAG()` / `RAG.from_providers()`
2. `Settings.embedding_provider` / `Settings.llm_provider`
3. `config.embeddings.provider` / `config.llm.provider`

Initialises all components:
- `self.embedding_provider` — `EmbeddingProvider`
- `self.vector_store` — `VectorStoreProvider`
- `self.llm_provider` — `LLMProvider`
- `self.text_splitter` — `BaseTextSplitter`
- `self.retriever` — `BaseRetriever`
- `self.reranker` — `Optional[BaseReranker]` (None if reranking disabled)
- `self.generation_strategy` — `GenerationStrategy`
- `self.pdf_parser` — `BasePDFParser`

**Raises:** `ValueError` if any configured provider is unsupported or its config is missing.

---

### `from_providers` (classmethod)

```python
@classmethod
RAG.from_providers(
    config: Config,
    *,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
) -> RAG
```

Alternative constructor for supplying live provider objects directly.

| Parameter | Type | Description |
|---|---|---|
| `config` | `Config` | Config for all non-provider settings (vector store, retrieval strategy, etc.) |
| `embedding_provider` | `EmbeddingProvider` | Embedding provider to use |
| `llm_provider` | `LLMProvider` | LLM provider to use |

```python
rag = RAG.from_providers(
    config,
    embedding_provider=MyLocalEmbedding("http://localhost:8080"),
    llm_provider=MyLocalLLM("http://localhost:11434"),
)
```

---

### `embedding_provider` (property)

```python
@property
def embedding_provider(self) -> EmbeddingProvider

@embedding_provider.setter
def embedding_provider(self, value: EmbeddingProvider) -> None
```

Reading returns the current embedding provider.

**Writing triggers a cascade:** rebuilds `text_splitter` (if semantic chunking) and
`retriever` with the new provider.

```python
rag.embedding_provider = MyNewEmbedding()
```

> **Warning:** If documents have already been ingested, swapping the embedding model
> causes vector space mismatch — the stored vectors were built with the old model.
> Call `clear_index()` and re-ingest after swapping. See [`clear_index()`](#clear_index).

---

### `llm_provider` (property)

```python
@property
def llm_provider(self) -> LLMProvider

@llm_provider.setter
def llm_provider(self, value: LLMProvider) -> None
```

Reading returns the current LLM provider.

**Writing triggers a cascade:** rebuilds `text_splitter` (if agentic/proposition
chunking), `retriever`, and `generation_strategy` with the new provider.

```python
rag.llm_provider = MyNewLLM()
```

---

### `ingest_documents`

```python
def ingest_documents(self, documents: List[Document]) -> Dict[str, int]
```

Splits documents into chunks, embeds them, stores in the vector store, and builds
any required indexes.

| Parameter | Type | Description |
|---|---|---|
| `documents` | `List[Document]` | Documents to ingest |

**Returns:** `Dict[str, int]` with keys:
- `"source_documents"` — number of input documents
- `"chunks"` — number of chunks after splitting

**Side effects:**
- Calls `GraphRAGRetriever.build_graph()` if using `graph_rag` or `advanced_graph_rag` strategy
- Calls `RAPTORRetriever.build_tree()` if using `raptor` strategy
- Calls `HybridRetriever.index_documents()` if using `hybrid` strategy
- Records the embedding model fingerprint for mismatch detection on later provider swaps

---

### `ingest_pdf`

```python
def ingest_pdf(self, file_path: str) -> Dict[str, int]
```

Parses a PDF file and ingests the resulting documents.

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Path to the PDF file |

**Returns:** Same as `ingest_documents`.

Uses the configured `pdf_parser` backend and `one_document_per_page` setting.
Internally calls `DocumentLoader.load_file()` then `ingest_documents()`.

---

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
|---|---|---|---|
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

**Reranking behaviour:** When reranking is enabled, the retriever fetches `3 * top_k`
candidates. The reranker then returns the best `top_k`.

---

### `clear_index`

```python
def clear_index(self) -> None
```

Wipes the vector store and resets all retriever state.

Use this when swapping to a different embedding model after documents have already been
ingested. The old vector store contains embeddings from the previous model — they are
incompatible with the new model's embedding space.

```python
rag.embedding_provider = MyNewEmbedding()
rag.clear_index()            # wipes the vector store, resets graph/RAPTOR state
rag.ingest_documents(docs)   # re-embeds everything with the new model
```

**What it resets:**
- Reinitialises the vector store (brand-new empty instance)
- Reinitialises the retriever (gets the fresh empty store; graph and RAPTOR tree state is lost)
- Clears the embedding fingerprint so the next `ingest_documents` call starts clean

---

### `_unwrap_retriever` (static)

```python
@staticmethod
def _unwrap_retriever(retriever: BaseRetriever) -> BaseRetriever
```

Walks the `base_retriever` chain to find the innermost retriever. Used internally to
detect Graph RAG, RAPTOR, and Hybrid retrievers through Corrective RAG / Contextual
Compression wrapper layers.

---

## Settings

```python
from rag_sdk import Settings
```

Module-level singleton for global provider defaults. Components resolve providers
lazily against `Settings` at call time, so assigning here is instantly reflected
everywhere — no re-initialisation needed.

```python
Settings.embedding_provider: Optional[EmbeddingProvider]
Settings.llm_provider:       Optional[LLMProvider]
```

### Methods

#### `Settings.reset()`

```python
Settings.reset() -> None
```

Clears both `embedding_provider` and `llm_provider` back to `None`. Useful in tests
to prevent state leaking between test cases.

### Example

```python
from rag_sdk import RAG, Settings

Settings.embedding_provider = MyEmbedding("http://localhost:8080")
Settings.llm_provider       = MyLLM("http://localhost:11434")

rag = RAG(config)   # picks up both providers automatically
```

---

## See Also

- [Extending](41-extending.md) — implementing and wiring custom providers
- [API: Config](31-api-config.md) — Config classes
- [Quickstart](01-quickstart.md) — usage examples

# Vector Stores

The SDK supports 6 vector store backends. Select one via config:

```yaml
vectorstore:
  provider: "memory"  # "memory" | "faiss" | "chroma" | "pinecone" | "weaviate" | "qdrant"
```

All stores implement the `VectorStoreProvider` interface:

```python
class VectorStoreProvider(ABC):
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None: ...
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]: ...
    def delete(self, document_ids: List[str]) -> None: ...
```

## Memory (Default)

In-memory store using numpy cosine similarity. No persistence — data is lost when the process exits.

```yaml
vectorstore:
  provider: "memory"
```

No additional configuration. Good for testing, prototyping, and small datasets.

## FAISS

Facebook AI Similarity Search. Fast approximate nearest neighbor search with optional disk persistence.

```bash
pip install rag_sdk[faiss]
```

```yaml
vectorstore:
  provider: "faiss"
  faiss:
    index_type: "Flat"
    metric: "cosine"
    persist_path: "./faiss_index"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_type` | `str` | `"Flat"` | `"Flat"` (exact), `"IVFFlat"` (approximate), `"HNSW"` (graph-based) |
| `metric` | `str` | `"cosine"` | `"cosine"`, `"l2"`, `"ip"` (inner product) |
| `persist_path` | `Optional[str]` | `None` | Path to save/load index |

## Chroma

Open-source embedding database with ephemeral, persistent, and client/server modes.

```bash
pip install rag_sdk[chroma]
```

```yaml
vectorstore:
  provider: "chroma"
  chroma:
    mode: "persistent"
    persist_path: "./chroma_db"
    collection_name: "rag-collection"
    distance_function: "cosine"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"ephemeral"` | `"ephemeral"`, `"persistent"`, `"http"` |
| `persist_path` | `str` | `"./chroma_db"` | Directory for persistent mode |
| `host` | `str` | `"localhost"` | Server host (http mode) |
| `port` | `int` | `8000` | Server port (http mode) |
| `collection_name` | `str` | `"rag-collection"` | Collection name |
| `distance_function` | `str` | `"cosine"` | `"cosine"`, `"l2"`, `"ip"` |

## Pinecone

Managed cloud vector database.

```yaml
vectorstore:
  provider: "pinecone"
  pinecone:
    index_name: "rag-index"
    namespace: "default"
    environment: "us-east-1-aws"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `PINECONE_API_KEY` |
| `index_host` | `str` | `""` | Index host URL (auto-resolved if empty) |
| `index_name` | `str` | `"rag-index"` | Index name |
| `namespace` | `str` | `"default"` | Namespace for multi-tenant isolation |
| `environment` | `str` | `"us-east-1-aws"` | Pinecone environment |

## Weaviate

Open-source vector database with built-in hybrid search.

```bash
pip install rag_sdk[weaviate]
```

```yaml
vectorstore:
  provider: "weaviate"
  weaviate:
    url: "http://localhost:8080"
    class_name: "Document"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"http://localhost:8080"` | Weaviate server URL |
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `WEAVIATE_API_KEY` |
| `class_name` | `str` | `"Document"` | Weaviate class name |

## Qdrant

Open-source vector database with advanced filtering.

```bash
pip install rag_sdk[qdrant]
```

```yaml
vectorstore:
  provider: "qdrant"
  qdrant:
    url: "http://localhost:6333"
    collection_name: "rag-collection"
    on_disk: false
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"http://localhost:6333"` | Qdrant server URL |
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `QDRANT_API_KEY` |
| `collection_name` | `str` | `"rag-collection"` | Collection name |
| `on_disk` | `bool` | `False` | Store vectors on disk instead of RAM |

## Comparison

| Store | Persistence | Managed | Filtering | Extra Install |
|-------|-------------|---------|-----------|---------------|
| Memory | No | No | Basic (exact match) | None |
| FAISS | Optional | No | No | `faiss-cpu` |
| Chroma | Optional | No | Yes | `chromadb` |
| Pinecone | Yes | Yes | Yes | `pinecone` |
| Weaviate | Yes | Optional | Yes | `weaviate-client` |
| Qdrant | Yes | Optional | Yes | `qdrant-client` |

## See Also

- [Embeddings](22-embeddings.md) — embedding providers
- [Retrieval Strategies](25-retrieval-strategies.md) — how vectors are searched
- [API: Providers](33-api-providers.md) — `VectorStoreProvider` ABC

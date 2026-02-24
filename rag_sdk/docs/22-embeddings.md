# Embeddings

The SDK supports 5 embedding providers. Select one via config:

```yaml
embeddings:
  provider: "openai"  # "openai" | "cohere" | "gemini" | "voyage" | "local"
```

All providers implement the `EmbeddingProvider` interface:

```python
class EmbeddingProvider(ABC):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
```

## OpenAI

```yaml
embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-small"
    dimensions: 1536
    batch_size: 100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `OPENAI_API_KEY` |
| `model` | `str` | `"text-embedding-3-small"` | Model name |
| `dimensions` | `Optional[int]` | `1536` | Output dimensions (`None` = model default) |
| `batch_size` | `int` | `100` | Documents per API call |

Newlines are replaced with spaces before embedding.

## Cohere

```yaml
embeddings:
  provider: "cohere"
  cohere:
    model: "embed-english-v3.0"
    input_type: "search_document"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `COHERE_API_KEY` |
| `model` | `str` | `"embed-english-v3.0"` | Model name |
| `input_type` | `str` | `"search_document"` | Used for documents; queries use `"search_query"` |

The `input_type` is automatically switched to `"search_query"` for `embed_query()`.

## Gemini

```yaml
embeddings:
  provider: "gemini"
  gemini:
    model: "gemini-embedding-001"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `GOOGLE_API_KEY` |
| `model` | `str` | `"gemini-embedding-001"` | Model name |

Uses `RETRIEVAL_DOCUMENT` task type for documents and `RETRIEVAL_QUERY` for queries.

## Voyage

```yaml
embeddings:
  provider: "voyage"
  voyage:
    model: "voyage-large-2"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `VOYAGE_API_KEY` |
| `model` | `str` | `"voyage-large-2"` | Model name |

Uses `"document"` input type for documents and `"query"` for queries.

## Local

Runs embedding models locally using `sentence-transformers`. No API calls or keys required.

```bash
pip install rag_sdk[local-embeddings]
```

```yaml
embeddings:
  provider: "local"
  local:
    model: "BAAI/bge-small-en-v1.5"
    query_prefix: ""
    document_prefix: ""
    batch_size: 32
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"BAAI/bge-small-en-v1.5"` | Any sentence-transformers model |
| `query_prefix` | `str` | `""` | Prefix added to queries (e.g., `"query: "` for E5) |
| `document_prefix` | `str` | `""` | Prefix added to documents |
| `batch_size` | `int` | `32` | Encoding batch size |

Compatible models include:
- `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`
- `intfloat/e5-small-v2`, `intfloat/e5-base-v2`, `intfloat/e5-large-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

Embeddings are L2-normalized by default.

## Comparison

| Provider | API Key | Speed | Cost | Dimensions |
|----------|---------|-------|------|------------|
| OpenAI | Yes | Fast | Per-token | 1536 (configurable) |
| Cohere | Yes | Fast | Per-token | 1024 |
| Gemini | Yes | Fast | Free tier available | 768 |
| Voyage | Yes | Fast | Per-token | 1024–1536 |
| Local | No | Depends on hardware | Free | Model-dependent |

## See Also

- [Vector Stores](23-vector-stores.md) — where embeddings are stored
- [API: Providers](33-api-providers.md) — `EmbeddingProvider` ABC

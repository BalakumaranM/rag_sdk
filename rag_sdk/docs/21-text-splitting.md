# Text Splitting

The SDK provides 5 chunking strategies, selected via config:

```yaml
document_processing:
  chunking:
    strategy: "recursive"  # "recursive" | "agentic" | "proposition" | "semantic" | "late"
```

All splitters implement `BaseTextSplitter` with two methods:
- `split_text(text: str) -> List[str]` — split raw text
- `split_documents(documents: List[Document]) -> List[Document]` — split documents, preserving metadata

## Recursive (Default)

Splits text hierarchically using a list of separators, falling back to finer-grained separators when chunks exceed `chunk_size`.

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", ".", "!", "?", ",", " "]
  chunking:
    strategy: "recursive"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | `1000` | Max characters per chunk |
| `chunk_overlap` | `int` | `200` | Overlap between consecutive chunks |
| `separators` | `List[str]` | `["\n\n", "\n", ".", ...]` | Split hierarchy |

**When to use:** General-purpose default. Fast, deterministic, no external dependencies.

## Agentic

Uses an LLM to identify semantic boundaries in text. The LLM analyzes numbered sentences and returns indices where topic shifts occur.

```yaml
document_processing:
  chunking:
    strategy: "agentic"
  agentic_chunking:
    max_chunk_size: 1000
    similarity_threshold: 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_chunk_size` | `int` | `1000` | Max chunk size; oversized chunks are sub-split |
| `similarity_threshold` | `float` | `0.5` | Reserved for future use |

Falls back to fixed-size splitting if the LLM fails. Adds `"chunking_strategy": "agentic"` to chunk metadata.

**When to use:** Documents with clear topic shifts (articles, reports). Requires an LLM call per document.

## Proposition

Uses an LLM to decompose text into atomic, self-contained propositions, then groups them into chunks.

```yaml
document_processing:
  chunking:
    strategy: "proposition"
  proposition_chunking:
    max_propositions_per_chunk: 5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_propositions_per_chunk` | `int` | `5` | Number of propositions per chunk |

Each proposition is a single fact or claim with resolved entity references (no dangling pronouns). Falls back to sentence splitting if the LLM fails.

**When to use:** Knowledge-base content where each chunk should be a self-contained fact. Highest retrieval precision at the cost of LLM calls.

## Semantic

Embeds each sentence, computes cosine similarity between consecutive sentences, and splits at points where similarity drops below a percentile threshold.

```yaml
document_processing:
  chunking:
    strategy: "semantic"
  semantic_chunking:
    breakpoint_percentile: 25.0
    min_chunk_size: 100
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `breakpoint_percentile` | `float` | `25.0` | Nth percentile of similarity scores used as threshold |
| `min_chunk_size` | `int` | `100` | Chunks below this size are merged with their neighbor |

**When to use:** When you want semantically coherent chunks without LLM calls. Requires embedding API calls per sentence.

## Late Chunking

Embeds the full document through a transformer to get token-level contextual embeddings, then splits text into chunks and mean-pools token embeddings per chunk. This preserves cross-chunk context that is lost with embed-after-split approaches.

```bash
pip install rag_sdk[late-chunking]
```

```yaml
document_processing:
  chunking:
    strategy: "late"
  late_chunking:
    model: "jinaai/jina-embeddings-v2-base-en"
    chunk_size: 512
    max_tokens: 8192
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"jinaai/jina-embeddings-v2-base-en"` | HuggingFace transformer model |
| `chunk_size` | `int` | `512` | Target chunk size in characters |
| `max_tokens` | `int` | `8192` | Max tokens for model input |

Each chunk gets a `late_embedding` key in metadata containing the pooled embedding. Falls back to simple chunking if the transformer fails.

**When to use:** When cross-chunk context matters. Requires local GPU or CPU compute.

## Comparison

| Strategy | Speed | Quality | LLM Required | Extra Deps |
|----------|-------|---------|--------------|------------|
| Recursive | Fast | Good | No | None |
| Agentic | Slow | High | Yes | None |
| Proposition | Slow | Highest | Yes | None |
| Semantic | Medium | High | No | Embedding API |
| Late | Medium | High | No | `transformers`, `torch` |

## See Also

- [Document Loading](20-document-loading.md) — loading files
- [Embeddings](22-embeddings.md) — embedding providers
- [API: Document](32-api-document.md) — full splitter API

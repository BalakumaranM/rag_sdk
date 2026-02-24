# API Reference: Config

All config classes are Pydantic `BaseModel` subclasses defined in `rag_sdk.config`.

```python
from rag_sdk.config import Config, ConfigLoader
```

## ConfigLoader

```python
class ConfigLoader:
    @staticmethod
    def from_yaml(file_path: str) -> Config
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Config
    @staticmethod
    def from_env() -> Config
```

## Config (Root)

```python
class Config(BaseModel):
    project_name: str = "rag-application"
    environment: str = "development"
    logging: LoggingConfig
    document_processing: DocumentProcessingConfig
    embeddings: EmbeddingConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    cove: CoVeConfig
    attributed_generation: AttributedGenerationConfig
```

---

## Logging

### LoggingConfig

| Field | Type | Default |
|-------|------|---------|
| `level` | `str` | `"INFO"` |
| `format` | `str` | `"json"` |
| `output` | `str` | `"stdout"` |
| `file_path` | `Optional[str]` | `None` |
| `rotation` | `str` | `"1 day"` |
| `retention` | `str` | `"30 days"` |

---

## Document Processing

### DocumentProcessingConfig

| Field | Type | Default |
|-------|------|---------|
| `chunk_size` | `int` | `1000` |
| `chunk_overlap` | `int` | `200` |
| `separators` | `List[str]` | `["\n\n", "\n", ".", "!", "?", ",", " "]` |
| `chunking` | `ChunkingConfig` | default |
| `agentic_chunking` | `AgenticChunkingConfig` | default |
| `proposition_chunking` | `PropositionChunkingConfig` | default |
| `semantic_chunking` | `SemanticChunkingConfig` | default |
| `late_chunking` | `LateChunkingConfig` | default |
| `pdf_parser` | `PDFParserConfig` | default |

### ChunkingConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `strategy` | `str` | `"recursive"` | `"recursive"`, `"agentic"`, `"proposition"`, `"semantic"`, `"late"` |

### AgenticChunkingConfig

| Field | Type | Default |
|-------|------|---------|
| `max_chunk_size` | `int` | `1000` |
| `similarity_threshold` | `float` | `0.5` |

### PropositionChunkingConfig

| Field | Type | Default |
|-------|------|---------|
| `max_propositions_per_chunk` | `int` | `5` |

### SemanticChunkingConfig

| Field | Type | Default |
|-------|------|---------|
| `breakpoint_percentile` | `float` | `25.0` |
| `min_chunk_size` | `int` | `100` |

### LateChunkingConfig

| Field | Type | Default |
|-------|------|---------|
| `model` | `str` | `"jinaai/jina-embeddings-v2-base-en"` |
| `chunk_size` | `int` | `512` |
| `max_tokens` | `int` | `8192` |

### PDFParserConfig

| Field | Type | Default |
|-------|------|---------|
| `backend` | `str` | `"pymupdf"` |
| `line_y_tolerance` | `float` | `2.0` |
| `word_x_gap_threshold` | `float` | `5.0` |
| `min_segment_length` | `float` | `10.0` |
| `grid_snap_tolerance` | `float` | `3.0` |
| `min_table_rows` | `int` | `2` |
| `min_table_cols` | `int` | `2` |
| `segment_merge_gap` | `float` | `2.0` |
| `checkbox_min_size` | `float` | `6.0` |
| `checkbox_max_size` | `float` | `24.0` |
| `checkbox_aspect_ratio_tolerance` | `float` | `0.3` |
| `one_document_per_page` | `bool` | `True` |
| `include_tables_in_text` | `bool` | `True` |
| `docling_do_ocr` | `bool` | `True` |
| `docling_do_table_structure` | `bool` | `True` |
| `docling_table_mode` | `str` | `"accurate"` |
| `docling_timeout` | `Optional[float]` | `None` |

---

## Embeddings

### EmbeddingConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `provider` | `str` | `"openai"` | `"openai"`, `"cohere"`, `"gemini"`, `"voyage"`, `"local"` |
| `openai` | `Optional[OpenAIEmbeddingConfig]` | default | |
| `cohere` | `Optional[CohereEmbeddingConfig]` | default | |
| `gemini` | `Optional[GeminiEmbeddingConfig]` | default | |
| `voyage` | `Optional[VoyageEmbeddingConfig]` | default | |
| `local` | `Optional[LocalEmbeddingConfig]` | default | |

### OpenAIEmbeddingConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"text-embedding-3-small"` |
| `dimensions` | `Optional[int]` | `1536` |
| `batch_size` | `int` | `100` |

Method: `get_api_key() -> str` — returns config key or `OPENAI_API_KEY` env var.

### CohereEmbeddingConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"embed-english-v3.0"` |
| `input_type` | `str` | `"search_document"` |

Method: `get_api_key() -> str` — returns config key or `COHERE_API_KEY` env var.

### GeminiEmbeddingConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"gemini-embedding-001"` |

Method: `get_api_key() -> str` — returns config key or `GOOGLE_API_KEY` env var.

### VoyageEmbeddingConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"voyage-large-2"` |

Method: `get_api_key() -> str` — returns config key or `VOYAGE_API_KEY` env var.

### LocalEmbeddingConfig

| Field | Type | Default |
|-------|------|---------|
| `model` | `str` | `"BAAI/bge-small-en-v1.5"` |
| `query_prefix` | `str` | `""` |
| `document_prefix` | `str` | `""` |
| `batch_size` | `int` | `32` |

---

## Vector Store

### VectorStoreConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `provider` | `str` | `"memory"` | `"memory"`, `"faiss"`, `"chroma"`, `"pinecone"`, `"weaviate"`, `"qdrant"` |
| `faiss` | `Optional[FAISSConfig]` | default | |
| `chroma` | `Optional[ChromaConfig]` | default | |
| `pinecone` | `Optional[PineconeConfig]` | default | |
| `weaviate` | `Optional[WeaviateConfig]` | default | |
| `qdrant` | `Optional[QdrantConfig]` | default | |

### FAISSConfig

| Field | Type | Default |
|-------|------|---------|
| `index_type` | `str` | `"Flat"` |
| `metric` | `str` | `"cosine"` |
| `persist_path` | `Optional[str]` | `None` |

### ChromaConfig

| Field | Type | Default |
|-------|------|---------|
| `mode` | `str` | `"ephemeral"` |
| `persist_path` | `str` | `"./chroma_db"` |
| `host` | `str` | `"localhost"` |
| `port` | `int` | `8000` |
| `collection_name` | `str` | `"rag-collection"` |
| `distance_function` | `str` | `"cosine"` |

### PineconeConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `index_host` | `str` | `""` |
| `index_name` | `str` | `"rag-index"` |
| `namespace` | `str` | `"default"` |
| `environment` | `str` | `"us-east-1-aws"` |

Method: `get_api_key() -> str`

### WeaviateConfig

| Field | Type | Default |
|-------|------|---------|
| `url` | `str` | `"http://localhost:8080"` |
| `api_key` | `Optional[SecretStr]` | `None` |
| `class_name` | `str` | `"Document"` |

Method: `get_api_key() -> str`

### QdrantConfig

| Field | Type | Default |
|-------|------|---------|
| `url` | `str` | `"http://localhost:6333"` |
| `api_key` | `Optional[SecretStr]` | `None` |
| `collection_name` | `str` | `"rag-collection"` |
| `on_disk` | `bool` | `False` |

Method: `get_api_key() -> str`

---

## LLM

### LLMConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `provider` | `str` | `"openai"` | `"openai"`, `"gemini"`, `"anthropic"`, `"cohere"` |
| `openai` | `Optional[OpenAIConfig]` | default | |
| `gemini` | `Optional[GeminiConfig]` | default | |
| `anthropic` | `Optional[AnthropicConfig]` | default | |
| `cohere` | `Optional[CohereConfig]` | default | |

### OpenAIConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `base_url` | `Optional[str]` | `None` |
| `model` | `str` | `"gpt-4-turbo-preview"` |
| `temperature` | `float` | `0.7` |
| `max_tokens` | `int` | `1000` |

Method: `get_api_key() -> str`

### GeminiConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"gemini-2.5-flash"` |
| `temperature` | `float` | `0.7` |
| `max_output_tokens` | `int` | `1000` |

Method: `get_api_key() -> str`

### AnthropicConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"claude-3-5-sonnet-20240620"` |
| `temperature` | `float` | `0.7` |
| `max_tokens` | `int` | `1024` |

Method: `get_api_key() -> str`

### CohereConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"command-r-plus"` |
| `temperature` | `float` | `0.7` |
| `max_tokens` | `int` | `1000` |

Method: `get_api_key() -> str`

---

## Retrieval

### RetrievalConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `strategy` | `str` | `"dense"` | `"dense"`, `"graph_rag"`, `"advanced_graph_rag"`, `"raptor"`, `"multi_query"`, `"hybrid"`, `"self_rag"` |
| `top_k` | `int` | `5` | |
| `corrective_rag_enabled` | `bool` | `False` | |
| `contextual_compression_enabled` | `bool` | `False` | |
| `reranking` | `RerankingConfig` | default | |
| `multi_query` | `MultiQueryConfig` | default | |
| `hybrid` | `HybridRetrievalConfig` | default | |
| `self_rag` | `SelfRAGConfig` | default | |
| `contextual_compression` | `ContextualCompressionConfig` | default | |
| `graph_rag` | `GraphRAGConfig` | default | |
| `advanced_graph_rag` | `AdvancedGraphRAGConfig` | default | |
| `raptor` | `RAPTORConfig` | default | |
| `corrective_rag` | `CorrectiveRAGConfig` | default | |

### RerankingConfig

| Field | Type | Default |
|-------|------|---------|
| `enabled` | `bool` | `False` |
| `provider` | `str` | `"cohere"` |
| `cohere` | `CohereRerankConfig` | default |
| `cross_encoder` | `CrossEncoderRerankConfig` | default |

### CohereRerankConfig

| Field | Type | Default |
|-------|------|---------|
| `api_key` | `Optional[SecretStr]` | `None` |
| `model` | `str` | `"rerank-v3.5"` |
| `top_n` | `int` | `5` |

Method: `get_api_key() -> str`

### CrossEncoderRerankConfig

| Field | Type | Default |
|-------|------|---------|
| `model` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| `batch_size` | `int` | `32` |

### MultiQueryConfig

| Field | Type | Default |
|-------|------|---------|
| `num_queries` | `int` | `3` |

### HybridRetrievalConfig

| Field | Type | Default |
|-------|------|---------|
| `bm25_weight` | `float` | `0.5` |
| `rrf_k` | `int` | `60` |
| `bm25_k1` | `float` | `1.5` |
| `bm25_b` | `float` | `0.75` |

### SelfRAGConfig

| Field | Type | Default |
|-------|------|---------|
| `check_support` | `bool` | `True` |

### ContextualCompressionConfig

| Field | Type | Default |
|-------|------|---------|
| `enabled` | `bool` | `False` |

### GraphRAGConfig

| Field | Type | Default |
|-------|------|---------|
| `max_entities_per_chunk` | `int` | `10` |
| `max_relationships_per_chunk` | `int` | `15` |

### AdvancedGraphRAGConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `search_mode` | `str` | `"local"` | `"local"`, `"global"`, or `"drift"` |
| `max_entities_per_chunk` | `int` | `10` | Max entities to extract per chunk |
| `max_relationships_per_chunk` | `int` | `15` | Max relationships to extract per chunk |
| `top_communities` | `int` | `3` | Top communities used in global/drift modes |
| `max_graph_hops` | `int` | `2` | Neighborhood hops for local search |
| `drift_max_rounds` | `int` | `2` | Iterative follow-up rounds for drift mode |
| `drift_follow_up_questions` | `int` | `3` | Follow-up questions per drift round |
| `community_detection_algorithm` | `str` | `"louvain"` | `"leiden"` (requires `leidenalg`) or `"louvain"` |
| `community_levels` | `int` | `2` | Hierarchy depth for community detection |
| `node2vec_enabled` | `bool` | `False` | Structural entity embeddings (requires `node2vec`) |
| `relationship_weight_in_graph` | `bool` | `True` | Apply relationship weight to graph edges |
| `entity_types` | `List[str]` | 10 generic types | Entity types to extract; override with domain-specific types |

### RAPTORConfig

| Field | Type | Default |
|-------|------|---------|
| `num_levels` | `int` | `3` |
| `clustering_method` | `str` | `"kmeans"` |
| `max_clusters_per_level` | `int` | `10` |

### CorrectiveRAGConfig

| Field | Type | Default |
|-------|------|---------|
| `relevance_threshold` | `float` | `0.7` |
| `max_refinement_attempts` | `int` | `2` |

---

## Generation

### GenerationConfig

| Field | Type | Default | Options |
|-------|------|---------|---------|
| `strategy` | `str` | `"standard"` | `"standard"`, `"cove"`, `"attributed"` |

### CoVeConfig

| Field | Type | Default |
|-------|------|---------|
| `max_verification_questions` | `int` | `3` |

### AttributedGenerationConfig

| Field | Type | Default |
|-------|------|---------|
| `citation_style` | `str` | `"numeric"` |

## See Also

- [Configuration](10-configuration.md) — loading and key resolution
- [YAML Reference](11-yaml-reference.md) — annotated YAML

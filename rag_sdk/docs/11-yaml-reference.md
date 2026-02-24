# YAML Configuration Reference

Complete annotated YAML with every field, its type, default value, and options.

```yaml
# ─── Top-level ────────────────────────────────────────────
project_name: "rag-application"    # str — project identifier
environment: "development"         # str — "development" | "production" | custom

# ─── Logging ──────────────────────────────────────────────
logging:
  level: "INFO"              # str — "DEBUG" | "INFO" | "WARNING" | "ERROR"
  format: "json"             # str — log format
  output: "stdout"           # str — "stdout" | "file"
  file_path: null            # Optional[str] — log file path (when output=file)
  rotation: "1 day"          # str — log rotation interval
  retention: "30 days"       # str — log retention period

# ─── Document Processing ─────────────────────────────────
document_processing:
  chunk_size: 1000           # int — max characters per chunk (recursive splitter)
  chunk_overlap: 200         # int — overlap between consecutive chunks
  separators:                # List[str] — split hierarchy (tried in order)
    - "\n\n"
    - "\n"
    - "."
    - "!"
    - "?"
    - ","
    - " "

  chunking:
    strategy: "recursive"    # str — "recursive" | "agentic" | "proposition" | "semantic" | "late"

  agentic_chunking:
    max_chunk_size: 1000           # int — max chunk size for agentic splitter
    similarity_threshold: 0.5     # float — unused currently (reserved)

  proposition_chunking:
    max_propositions_per_chunk: 5  # int — propositions grouped per chunk

  semantic_chunking:
    breakpoint_percentile: 25.0   # float — percentile threshold for split points
    min_chunk_size: 100           # int — merge chunks below this size

  late_chunking:
    model: "jinaai/jina-embeddings-v2-base-en"  # str — HuggingFace model
    chunk_size: 512                              # int — target chunk size (chars)
    max_tokens: 8192                             # int — max tokens for model input

  pdf_parser:
    backend: "pymupdf"                    # str — "pymupdf" | "docling"
    # PyMuPDF-specific:
    line_y_tolerance: 2.0                 # float — Y-axis tolerance for line grouping
    word_x_gap_threshold: 5.0             # float — X-axis gap for word separation
    min_segment_length: 10.0              # float — min line segment length
    grid_snap_tolerance: 3.0              # float — tolerance for grid alignment
    min_table_rows: 2                     # int — min rows to detect as table
    min_table_cols: 2                     # int — min columns to detect as table
    segment_merge_gap: 2.0                # float — gap threshold for merging segments
    checkbox_min_size: 6.0                # float — min checkbox size (pts)
    checkbox_max_size: 24.0               # float — max checkbox size (pts)
    checkbox_aspect_ratio_tolerance: 0.3  # float — checkbox aspect ratio tolerance
    one_document_per_page: true           # bool — one Document per page vs. whole PDF
    include_tables_in_text: true          # bool — include table text in content
    # Docling-specific:
    docling_do_ocr: true                  # bool — enable OCR
    docling_do_table_structure: true       # bool — enable table structure extraction
    docling_table_mode: "accurate"        # str — "accurate" | "fast"
    docling_timeout: null                 # Optional[float] — processing timeout (seconds)

# ─── Embeddings ───────────────────────────────────────────
embeddings:
  provider: "openai"         # str — "openai" | "cohere" | "gemini" | "voyage" | "local"

  openai:
    api_key: null            # Optional[SecretStr] — falls back to OPENAI_API_KEY
    model: "text-embedding-3-small"  # str
    dimensions: 1536         # Optional[int] — output dimensions (null = model default)
    batch_size: 100          # int — documents per API call

  cohere:
    api_key: null            # Optional[SecretStr] — falls back to COHERE_API_KEY
    model: "embed-english-v3.0"  # str
    input_type: "search_document"  # str — "search_document" | "search_query"

  gemini:
    api_key: null            # Optional[SecretStr] — falls back to GOOGLE_API_KEY
    model: "gemini-embedding-001"  # str

  voyage:
    api_key: null            # Optional[SecretStr] — falls back to VOYAGE_API_KEY
    model: "voyage-large-2"  # str

  local:
    model: "BAAI/bge-small-en-v1.5"  # str — any sentence-transformers model
    query_prefix: ""         # str — prefix added to queries
    document_prefix: ""      # str — prefix added to documents
    batch_size: 32           # int — encoding batch size

# ─── Vector Store ─────────────────────────────────────────
vectorstore:
  provider: "memory"         # str — "memory" | "faiss" | "chroma" | "pinecone" | "weaviate" | "qdrant"

  faiss:
    index_type: "Flat"       # str — "Flat" | "IVFFlat" | "HNSW"
    metric: "cosine"         # str — "cosine" | "l2" | "ip"
    persist_path: null       # Optional[str] — save/load index to disk

  chroma:
    mode: "ephemeral"        # str — "ephemeral" | "persistent" | "http"
    persist_path: "./chroma_db"  # str — directory for persistent mode
    host: "localhost"        # str — Chroma server host (http mode)
    port: 8000               # int — Chroma server port (http mode)
    collection_name: "rag-collection"  # str
    distance_function: "cosine"  # str — "cosine" | "l2" | "ip"

  pinecone:
    api_key: null            # Optional[SecretStr] — falls back to PINECONE_API_KEY
    index_host: ""           # str — auto-resolved if empty
    index_name: "rag-index"  # str
    namespace: "default"     # str
    environment: "us-east-1-aws"  # str

  weaviate:
    url: "http://localhost:8080"  # str — Weaviate server URL
    api_key: null            # Optional[SecretStr] — falls back to WEAVIATE_API_KEY
    class_name: "Document"   # str — Weaviate class name

  qdrant:
    url: "http://localhost:6333"  # str — Qdrant server URL
    api_key: null            # Optional[SecretStr] — falls back to QDRANT_API_KEY
    collection_name: "rag-collection"  # str
    on_disk: false           # bool — store vectors on disk

# ─── LLM ─────────────────────────────────────────────────
llm:
  provider: "openai"         # str — "openai" | "gemini" | "anthropic" | "cohere"

  openai:
    api_key: null            # Optional[SecretStr] — falls back to OPENAI_API_KEY
    base_url: null           # Optional[str] — custom API base URL
    model: "gpt-4-turbo-preview"  # str
    temperature: 0.7         # float — 0.0 to 2.0
    max_tokens: 1000         # int

  gemini:
    api_key: null            # Optional[SecretStr] — falls back to GOOGLE_API_KEY
    model: "gemini-2.5-flash"  # str
    temperature: 0.7         # float
    max_output_tokens: 1000  # int

  anthropic:
    api_key: null            # Optional[SecretStr] — falls back to ANTHROPIC_API_KEY
    model: "claude-3-5-sonnet-20240620"  # str
    temperature: 0.7         # float
    max_tokens: 1024         # int

  cohere:
    api_key: null            # Optional[SecretStr] — falls back to COHERE_API_KEY
    model: "command-r-plus"  # str
    temperature: 0.7         # float
    max_tokens: 1000         # int

# ─── Retrieval ────────────────────────────────────────────
retrieval:
  strategy: "dense"          # str — "dense" | "graph_rag" | "advanced_graph_rag" | "raptor" | "multi_query" | "hybrid" | "self_rag"
  top_k: 5                   # int — default number of results
  corrective_rag_enabled: false       # bool — wrap retriever with corrective RAG
  contextual_compression_enabled: false  # bool — wrap retriever with compression

  reranking:
    enabled: false           # bool — enable reranking after retrieval
    provider: "cohere"       # str — "cohere" | "cross-encoder"
    cohere:
      api_key: null          # Optional[SecretStr] — falls back to COHERE_API_KEY
      model: "rerank-v3.5"   # str
      top_n: 5               # int
    cross_encoder:
      model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # str
      batch_size: 32         # int

  multi_query:
    num_queries: 3           # int — number of query variations to generate

  hybrid:
    bm25_weight: 0.5         # float — weight for BM25 in RRF (0.0–1.0)
    rrf_k: 60                # int — RRF constant
    bm25_k1: 1.5             # float — BM25 term frequency saturation
    bm25_b: 0.75             # float — BM25 length normalization

  self_rag:
    check_support: true      # bool — verify document support for query

  contextual_compression:
    enabled: false           # bool

  graph_rag:
    max_entities_per_chunk: 10       # int
    max_relationships_per_chunk: 15  # int

  advanced_graph_rag:
    search_mode: "local"                     # str — "local" | "global" | "drift"
    max_entities_per_chunk: 10               # int — entities per chunk
    max_relationships_per_chunk: 15          # int — relationships per chunk
    top_communities: 3                       # int — communities used in global/drift
    max_graph_hops: 2                        # int — neighborhood hops for local search
    drift_max_rounds: 2                      # int — follow-up rounds for drift
    drift_follow_up_questions: 3             # int — questions per drift round
    community_detection_algorithm: "louvain" # str — "leiden" | "louvain"
    community_levels: 2                      # int — hierarchy depth
    node2vec_enabled: false                  # bool — structural entity embeddings
    relationship_weight_in_graph: true       # bool — use weight on graph edges
    entity_types:                            # List[str] — types to extract
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

  raptor:
    num_levels: 3                   # int — tree depth
    clustering_method: "kmeans"     # str
    max_clusters_per_level: 10      # int

  corrective_rag:
    relevance_threshold: 0.7        # float — min fraction of relevant docs
    max_refinement_attempts: 2      # int — query refinement retries

# ─── Generation ───────────────────────────────────────────
generation:
  strategy: "standard"       # str — "standard" | "cove" | "attributed"

cove:
  max_verification_questions: 3  # int — max questions for Chain of Verification

attributed_generation:
  citation_style: "numeric"  # str — "numeric"
```

## See Also

- [Configuration](10-configuration.md) — loading methods and API key resolution
- [Quickstart](01-quickstart.md) — minimal working config

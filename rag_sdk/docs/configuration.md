# Configuration System {#configuration}

## Configuration File Structure

The SDK uses a YAML configuration file to manage settings for all components.

```yaml
# config.yaml

# SDK Metadata
sdk_version: "1.0.0"
project_name: "my-rag-application"
environment: "production"  # development, staging, production

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"  # json, text
  output: "stdout"  # stdout, file
  file_path: "/var/log/rag-sdk.log"
  rotation: "1 day"
  retention: "30 days"

# Document Processing
document_processing:
  loader:
    type: "auto"  # auto-detect or specify: pdf, docx, txt, html, markdown
    encoding: "utf-8"
    extract_metadata: true
    
  splitter:
    type: "recursive"  # recursive, semantic, fixed, token
    chunk_size: 1000
    chunk_overlap: 200
    separators: ["\n\n", "\n", ".", "!", "?", ",", " "]
    
  # Semantic splitter config (if type: semantic)
  semantic_splitter:
    embedding_model: "text-embedding-3-small"
    breakpoint_threshold: 0.7
    
  preprocessing:
    remove_html_tags: true
    normalize_whitespace: true
    lowercase: false
    remove_special_chars: false
    deduplicate: true
    min_chunk_length: 50

# Embedding Configuration
embeddings:
  provider: "openai"  # openai, cohere, huggingface, anthropic, custom
  
  # OpenAI Configuration
  openai:
    api_key: "${OPENAI_API_KEY}"  # Environment variable
    model: "text-embedding-3-small"
    dimensions: 1536  # Optional: reduce dimensions
    batch_size: 100
    encoding_format: "float"
    
  # Cohere Configuration
  cohere:
    api_key: "${COHERE_API_KEY}"
    model: "embed-english-v3.0"
    input_type: "search_document"
    
  # HuggingFace Configuration
  huggingface:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cuda"  # cuda, cpu, mps

# Vector Store Configuration
vectorstore:
  provider: "pinecone"  # pinecone, weaviate, qdrant, milvus, chroma, faiss, elasticsearch
  
  # Pinecone Configuration
  pinecone:
    api_key: "${PINECONE_API_KEY}"
    environment: "us-east-1-aws"
    index_name: "rag-index"
    namespace: "default"
    metric: "cosine"
    dimension: 1536
    
  # Weaviate Configuration
  weaviate:
    url: "${WEAVIATE_URL}"
    api_key: "${WEAVIATE_API_KEY}"
    class_name: "Document"
    
  # Qdrant Configuration
  qdrant:
    url: "${QDRANT_URL}"
    api_key: "${QDRANT_API_KEY}"
    collection_name: "documents"
    
  # Local FAISS Configuration
  faiss:
    index_type: "IndexFlatIP"
    persist_directory: "./faiss_index"

# Retrieval Configuration
retrieval:
  strategy: "hybrid"  # dense, sparse, hybrid
  
  # Dense Retrieval
  dense:
    top_k: 5
    score_threshold: 0.7
    
  # Sparse Retrieval (BM25)
  sparse:
    top_k: 5
    b: 0.75
    k1: 1.2
    
  # Hybrid Search
  hybrid:
    alpha: 0.5  # 0 = full sparse, 1 = full dense
    top_k: 5
    
  # Re-ranking
  reranker:
    enabled: true
    provider: "cohere"
    model: "rerank-english-v2.0"
    top_n: 3

# LLM Configuration
llm:
  provider: "anthropic"  # openai, anthropic, cohere, google, azure, bedrock, litellm
  
  # OpenAI Configuration
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 1000
    streaming: true
    
  # Anthropic Configuration
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-sonnet-4-20250514"
    max_tokens: 4096
    temperature: 0.7
    
  # LiteLLM Integration
  litellm:
    enabled: true
    router_settings:
      fallback_models: ["gpt-3.5-turbo", "claude-haiku-3"]
    cost_tracking:
      enabled: true
      budget_manager:
        monthly_limit: 1000.00
        alert_webhook: "${SLACK_WEBHOOK}"

# Monitoring & Observability
monitoring:
  enabled: true
  metrics:
    provider: "prometheus"
    port: 9090
  tracing:
    enabled: true
    provider: "opentelemetry"
    endpoint: "${OTEL_ENDPOINT}"
```

## Configuration Management in Code

```python
from rag_sdk.config import Config, ConfigLoader

# Method 1: Load from YAML
config = ConfigLoader.from_yaml("config.yaml")

# Method 2: Load from environment variables
config = ConfigLoader.from_env(prefix="RAG_")

# Method 3: Load from dictionary
config = Config.from_dict({
    "embeddings": {
        "provider": "openai",
        "openai": {"model": "text-embedding-3-small"}
    }
})

# Method 4: Builder pattern
config = (Config.builder()
    .set_embedding_provider("openai")
    .set_llm_provider("anthropic")
    .set_vectorstore_provider("pinecone")
    .build())

# Access configuration
embedding_model = config.embeddings.openai.model
top_k = config.retrieval.dense.top_k
```

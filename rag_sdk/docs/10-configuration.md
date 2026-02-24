# Configuration

## Overview

The SDK uses a single `Config` Pydantic model that controls every component. There are three ways to create a config:

```python
from rag_sdk.config import Config, ConfigLoader
```

## Loading Methods

### From YAML

```python
config = ConfigLoader.from_yaml("config.yaml")
```

Reads a YAML file and validates it against the `Config` schema. Any missing fields use their defaults.

### From Dictionary

```python
config = ConfigLoader.from_dict({
    "project_name": "my-app",
    "embeddings": {"provider": "cohere"},
    "llm": {"provider": "gemini"},
})
```

Useful for programmatic configuration or when loading from other sources.

### From Defaults (Environment)

```python
config = ConfigLoader.from_env()
```

Returns a `Config()` with all defaults. API keys are resolved from environment variables via each provider's `get_api_key()` method.

### Direct Construction

```python
config = Config(
    project_name="my-app",
    embeddings=EmbeddingConfig(provider="openai"),
)
```

## Config Tree

```
Config
├── project_name: str = "rag-application"
├── environment: str = "development"
├── logging: LoggingConfig
├── document_processing: DocumentProcessingConfig
│   ├── chunk_size, chunk_overlap, separators
│   ├── chunking: ChunkingConfig (strategy selector)
│   ├── agentic_chunking: AgenticChunkingConfig
│   ├── proposition_chunking: PropositionChunkingConfig
│   ├── semantic_chunking: SemanticChunkingConfig
│   ├── late_chunking: LateChunkingConfig
│   └── pdf_parser: PDFParserConfig
├── embeddings: EmbeddingConfig
│   ├── provider selector
│   ├── openai: OpenAIEmbeddingConfig
│   ├── cohere: CohereEmbeddingConfig
│   ├── gemini: GeminiEmbeddingConfig
│   ├── voyage: VoyageEmbeddingConfig
│   └── local: LocalEmbeddingConfig
├── vectorstore: VectorStoreConfig
│   ├── provider selector
│   ├── faiss: FAISSConfig
│   ├── chroma: ChromaConfig
│   ├── pinecone: PineconeConfig
│   ├── weaviate: WeaviateConfig
│   └── qdrant: QdrantConfig
├── llm: LLMConfig
│   ├── provider selector
│   ├── openai: OpenAIConfig
│   ├── gemini: GeminiConfig
│   ├── anthropic: AnthropicConfig
│   └── cohere: CohereConfig
├── retrieval: RetrievalConfig
│   ├── strategy, top_k
│   ├── corrective_rag_enabled, contextual_compression_enabled
│   ├── reranking: RerankingConfig
│   ├── multi_query: MultiQueryConfig
│   ├── hybrid: HybridRetrievalConfig
│   ├── self_rag: SelfRAGConfig
│   ├── contextual_compression: ContextualCompressionConfig
│   ├── graph_rag: GraphRAGConfig
│   ├── advanced_graph_rag: AdvancedGraphRAGConfig
│   ├── raptor: RAPTORConfig
│   └── corrective_rag: CorrectiveRAGConfig
├── generation: GenerationConfig
├── cove: CoVeConfig
└── attributed_generation: AttributedGenerationConfig
```

## API Key Resolution

Each provider config with an `api_key` field follows the same resolution order:

1. **Config value** — if `api_key` is set in YAML or code, use it
2. **Environment variable** — fallback to the provider-specific env var

```python
class OpenAIConfig(BaseModel):
    api_key: Optional[SecretStr] = None

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY", "")
```

| Provider | Env Variable |
|----------|-------------|
| OpenAI (LLM & Embeddings) | `OPENAI_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Cohere (LLM, Embeddings, Reranking) | `COHERE_API_KEY` |
| Voyage | `VOYAGE_API_KEY` |
| Pinecone | `PINECONE_API_KEY` |
| Weaviate | `WEAVIATE_API_KEY` |
| Qdrant | `QDRANT_API_KEY` |

API keys are stored as `SecretStr` in Pydantic, so they won't appear in logs or serialized output.

## Provider Selection Pattern

Each module uses a `provider` string to select the active implementation. Only the selected provider's config needs to be populated:

```yaml
embeddings:
  provider: "cohere"  # Only cohere config matters
  cohere:
    model: "embed-english-v3.0"
```

## Next Steps

- [YAML Reference](11-yaml-reference.md) — every YAML field documented
- [Core Concepts](02-core-concepts.md) — architecture overview

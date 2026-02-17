import os
import yaml  # type: ignore
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv

load_dotenv()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"
    file_path: Optional[str] = None
    rotation: str = "1 day"
    retention: str = "30 days"


class ChunkingConfig(BaseModel):
    strategy: str = (
        "recursive"  # "recursive", "agentic", "proposition", "semantic", "late"
    )


class AgenticChunkingConfig(BaseModel):
    max_chunk_size: int = 1000
    similarity_threshold: float = 0.5


class PropositionChunkingConfig(BaseModel):
    max_propositions_per_chunk: int = 5


class SemanticChunkingConfig(BaseModel):
    breakpoint_percentile: float = 25.0
    min_chunk_size: int = 100


class LateChunkingConfig(BaseModel):
    model: str = "jinaai/jina-embeddings-v2-base-en"
    chunk_size: int = 512
    max_tokens: int = 8192


class PDFParserConfig(BaseModel):
    backend: str = "pymupdf"  # "pymupdf" or "docling"
    # PyMuPDF-specific fields:
    line_y_tolerance: float = 2.0
    word_x_gap_threshold: float = 5.0
    min_segment_length: float = 10.0
    grid_snap_tolerance: float = 3.0
    min_table_rows: int = 2
    min_table_cols: int = 2
    segment_merge_gap: float = 2.0
    checkbox_min_size: float = 6.0
    checkbox_max_size: float = 24.0
    checkbox_aspect_ratio_tolerance: float = 0.3
    one_document_per_page: bool = True
    include_tables_in_text: bool = True
    # Docling-specific fields:
    docling_do_ocr: bool = True
    docling_do_table_structure: bool = True
    docling_table_mode: str = "accurate"  # "accurate" or "fast"
    docling_timeout: Optional[float] = None


class DocumentProcessingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", ".", "!", "?", ",", " "]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agentic_chunking: AgenticChunkingConfig = Field(
        default_factory=AgenticChunkingConfig
    )
    proposition_chunking: PropositionChunkingConfig = Field(
        default_factory=PropositionChunkingConfig
    )
    semantic_chunking: SemanticChunkingConfig = Field(
        default_factory=SemanticChunkingConfig
    )
    late_chunking: LateChunkingConfig = Field(default_factory=LateChunkingConfig)
    pdf_parser: PDFParserConfig = Field(default_factory=PDFParserConfig)


class OpenAIEmbeddingConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = 1536
    batch_size: int = 100

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY", "")


class CohereEmbeddingConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "embed-english-v3.0"
    input_type: str = "search_document"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("COHERE_API_KEY", "")


class GeminiEmbeddingConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "gemini-embedding-001"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("GOOGLE_API_KEY", "")


class VoyageEmbeddingConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "voyage-large-2"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("VOYAGE_API_KEY", "")


class LocalEmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32


class EmbeddingConfig(BaseModel):
    provider: str = "openai"  # "openai", "cohere", "gemini", "voyage", "local"
    openai: Optional[OpenAIEmbeddingConfig] = Field(
        default_factory=OpenAIEmbeddingConfig
    )
    cohere: Optional[CohereEmbeddingConfig] = Field(
        default_factory=CohereEmbeddingConfig
    )
    gemini: Optional[GeminiEmbeddingConfig] = Field(
        default_factory=GeminiEmbeddingConfig
    )
    voyage: Optional[VoyageEmbeddingConfig] = Field(
        default_factory=VoyageEmbeddingConfig
    )
    local: Optional[LocalEmbeddingConfig] = Field(default_factory=LocalEmbeddingConfig)


class FAISSConfig(BaseModel):
    index_type: str = "Flat"  # "Flat" | "IVFFlat" | "HNSW"
    metric: str = "cosine"  # "cosine" | "l2" | "ip"
    persist_path: Optional[str] = None


class ChromaConfig(BaseModel):
    mode: str = "ephemeral"  # "ephemeral" | "persistent" | "http"
    persist_path: str = "./chroma_db"
    host: str = "localhost"
    port: int = 8000
    collection_name: str = "rag-collection"
    distance_function: str = "cosine"  # "cosine" | "l2" | "ip"


class PineconeConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    index_host: str = ""
    index_name: str = "rag-index"
    namespace: str = "default"
    environment: str = "us-east-1-aws"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("PINECONE_API_KEY", "")


class WeaviateConfig(BaseModel):
    url: str = "http://localhost:8080"
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    class_name: str = "Document"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("WEAVIATE_API_KEY", "")


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    collection_name: str = "rag-collection"
    on_disk: bool = False

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("QDRANT_API_KEY", "")


class VectorStoreConfig(BaseModel):
    provider: str = (
        "memory"  # "memory" | "faiss" | "chroma" | "pinecone" | "weaviate" | "qdrant"
    )
    faiss: Optional[FAISSConfig] = Field(default_factory=FAISSConfig)
    chroma: Optional[ChromaConfig] = Field(default_factory=ChromaConfig)
    pinecone: Optional[PineconeConfig] = Field(default_factory=PineconeConfig)
    weaviate: Optional[WeaviateConfig] = Field(default_factory=WeaviateConfig)
    qdrant: Optional[QdrantConfig] = Field(default_factory=QdrantConfig)


class OpenAIConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    base_url: Optional[str] = None
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1000

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("OPENAI_API_KEY", "")


class GeminiConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 1000

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("GOOGLE_API_KEY", "")


class AnthropicConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "claude-3-5-sonnet-20240620"
    temperature: float = 0.7
    max_tokens: int = 1024

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("ANTHROPIC_API_KEY", "")


class CohereConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "command-r-plus"
    temperature: float = 0.7
    max_tokens: int = 1000

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("COHERE_API_KEY", "")


class LLMConfig(BaseModel):
    provider: str = "openai"
    openai: Optional[OpenAIConfig] = Field(default_factory=OpenAIConfig)
    gemini: Optional[GeminiConfig] = Field(default_factory=GeminiConfig)
    anthropic: Optional[AnthropicConfig] = Field(default_factory=AnthropicConfig)
    cohere: Optional[CohereConfig] = Field(default_factory=CohereConfig)


class GraphRAGConfig(BaseModel):
    max_entities_per_chunk: int = 10
    max_relationships_per_chunk: int = 15


class RAPTORConfig(BaseModel):
    num_levels: int = 3
    clustering_method: str = "kmeans"
    max_clusters_per_level: int = 10


class CorrectiveRAGConfig(BaseModel):
    relevance_threshold: float = 0.7
    max_refinement_attempts: int = 2


class CohereRerankConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    model: str = "rerank-v3.5"
    top_n: int = 5

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("COHERE_API_KEY", "")


class CrossEncoderRerankConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32


class RerankingConfig(BaseModel):
    enabled: bool = False
    provider: str = "cohere"  # "cohere", "cross-encoder"
    cohere: CohereRerankConfig = Field(default_factory=CohereRerankConfig)
    cross_encoder: CrossEncoderRerankConfig = Field(
        default_factory=CrossEncoderRerankConfig
    )


class MultiQueryConfig(BaseModel):
    num_queries: int = 3


class HybridRetrievalConfig(BaseModel):
    bm25_weight: float = 0.5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class SelfRAGConfig(BaseModel):
    check_support: bool = True


class ContextualCompressionConfig(BaseModel):
    enabled: bool = False


class RetrievalConfig(BaseModel):
    strategy: str = (
        "dense"  # "dense", "graph_rag", "raptor", "multi_query", "hybrid", "self_rag"
    )
    top_k: int = 5
    corrective_rag_enabled: bool = False
    contextual_compression_enabled: bool = False
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    multi_query: MultiQueryConfig = Field(default_factory=MultiQueryConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    self_rag: SelfRAGConfig = Field(default_factory=SelfRAGConfig)
    contextual_compression: ContextualCompressionConfig = Field(
        default_factory=ContextualCompressionConfig
    )
    graph_rag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    raptor: RAPTORConfig = Field(default_factory=RAPTORConfig)
    corrective_rag: CorrectiveRAGConfig = Field(default_factory=CorrectiveRAGConfig)


class GenerationConfig(BaseModel):
    strategy: str = "standard"  # "standard", "cove", "attributed"


class CoVeConfig(BaseModel):
    max_verification_questions: int = 3


class AttributedGenerationConfig(BaseModel):
    citation_style: str = "numeric"  # "numeric"


class Config(BaseModel):
    project_name: str = "rag-application"
    environment: str = "development"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig
    )
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    cove: CoVeConfig = Field(default_factory=CoVeConfig)
    attributed_generation: AttributedGenerationConfig = Field(
        default_factory=AttributedGenerationConfig
    )


class ConfigLoader:
    @staticmethod
    def from_yaml(file_path: str) -> Config:
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)

    @staticmethod
    def from_env() -> Config:
        # Minimal implementation for env var loading override could be added here
        # For now, relying on pydantic defaults + os.getenv in sub-models
        return Config()

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Config:
        return Config(**config_dict)

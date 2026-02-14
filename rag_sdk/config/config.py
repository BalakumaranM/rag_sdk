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
    strategy: str = "recursive"  # "recursive", "agentic", "proposition"


class AgenticChunkingConfig(BaseModel):
    max_chunk_size: int = 1000
    similarity_threshold: float = 0.5


class PropositionChunkingConfig(BaseModel):
    max_propositions_per_chunk: int = 5


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
    model: str = "models/text-embedding-004"

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


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
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


class PineconeConfig(BaseModel):
    api_key: Optional[SecretStr] = Field(default=None, validate_default=True)
    environment: str = "us-east-1-aws"
    index_name: str = "rag-index"
    namespace: str = "default"

    def get_api_key(self) -> str:
        if self.api_key:
            return self.api_key.get_secret_value()
        return os.getenv("PINECONE_API_KEY", "")


class VectorStoreConfig(BaseModel):
    provider: str = "memory"  # Default to memory for ease of use
    pinecone: Optional[PineconeConfig] = Field(default_factory=PineconeConfig)


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
    model: str = "gemini-1.5-pro"
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


class RetrievalConfig(BaseModel):
    strategy: str = "dense"  # "dense", "graph_rag", "raptor"
    top_k: int = 5
    corrective_rag_enabled: bool = False
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

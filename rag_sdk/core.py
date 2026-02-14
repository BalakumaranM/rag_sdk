import time
import logging
from typing import List, Optional, Dict, Any
from .config import Config
from .document import Document, TextSplitter, AgenticSplitter, PropositionSplitter
from .document.base import BaseTextSplitter
from .embeddings import OpenAIEmbedding, EmbeddingProvider
from .vectorstore import VectorStoreProvider, InMemoryVectorStore, PineconeVectorStore
from .llm import LLMProvider, OpenAILLM
from .retrieval import (
    Retriever,
    GraphRAGRetriever,
    RAPTORRetriever,
    CorrectiveRAGRetriever,
)
from .retrieval.base import BaseRetriever
from .generation import (
    GenerationStrategy,
    StandardGeneration,
    ChainOfVerificationGeneration,
    AttributedGeneration,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAG:
    """
    Main RAG orchestrator.
    """

    def __init__(self, config: Config):
        self.config = config
        self._init_components()

    def _init_components(self) -> None:
        # 1. Embeddings
        self._init_embeddings()

        # 2. Vector Store
        self._init_vectorstore()

        # 3. LLM (must come before splitter/retriever since they may need it)
        self._init_llm()

        # 4. Text Splitter
        self._init_splitter()

        # 5. Retriever
        self._init_retriever()

        # 6. Generation Strategy
        self._init_generation()

    def _init_embeddings(self) -> None:
        if self.config.embeddings.provider == "openai":
            if not self.config.embeddings.openai:
                raise ValueError("OpenAI embedding config is missing")
            self.embedding_provider: EmbeddingProvider = OpenAIEmbedding(
                self.config.embeddings.openai
            )
        elif self.config.embeddings.provider == "cohere":
            if not self.config.embeddings.cohere:
                raise ValueError("Cohere embedding config is missing")
            from .embeddings import CohereEmbedding

            self.embedding_provider = CohereEmbedding(self.config.embeddings.cohere)
        elif self.config.embeddings.provider == "gemini":
            if not self.config.embeddings.gemini:
                raise ValueError("Gemini embedding config is missing")
            from .embeddings import GeminiEmbedding

            self.embedding_provider = GeminiEmbedding(self.config.embeddings.gemini)
        elif self.config.embeddings.provider == "voyage":
            if not self.config.embeddings.voyage:
                raise ValueError("Voyage embedding config is missing")
            from .embeddings import VoyageEmbedding

            self.embedding_provider = VoyageEmbedding(self.config.embeddings.voyage)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.config.embeddings.provider}"
            )

    def _init_vectorstore(self) -> None:
        if self.config.vectorstore.provider == "memory":
            self.vector_store: VectorStoreProvider = InMemoryVectorStore()
        elif self.config.vectorstore.provider == "pinecone":
            if not self.config.vectorstore.pinecone:
                raise ValueError("Pinecone vector store config is missing")
            self.vector_store = PineconeVectorStore(self.config.vectorstore.pinecone)
        else:
            raise ValueError(
                f"Unsupported vector store provider: {self.config.vectorstore.provider}"
            )

    def _init_llm(self) -> None:
        if self.config.llm.provider == "openai":
            if not self.config.llm.openai:
                raise ValueError("OpenAI LLM config is missing")
            self.llm_provider: LLMProvider = OpenAILLM(self.config.llm.openai)
        elif self.config.llm.provider == "gemini":
            if not self.config.llm.gemini:
                raise ValueError("Gemini LLM config is missing")
            from .llm import GeminiLLM

            self.llm_provider = GeminiLLM(self.config.llm.gemini)
        elif self.config.llm.provider == "anthropic":
            if not self.config.llm.anthropic:
                raise ValueError("Anthropic LLM config is missing")
            from .llm import AnthropicLLM

            self.llm_provider = AnthropicLLM(self.config.llm.anthropic)
        elif self.config.llm.provider == "cohere":
            if not self.config.llm.cohere:
                raise ValueError("Cohere LLM config is missing")
            from .llm import CohereLLM

            self.llm_provider = CohereLLM(self.config.llm.cohere)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _init_splitter(self) -> None:
        strategy = self.config.document_processing.chunking.strategy

        if strategy == "recursive":
            self.text_splitter: BaseTextSplitter = TextSplitter(
                chunk_size=self.config.document_processing.chunk_size,
                chunk_overlap=self.config.document_processing.chunk_overlap,
                separators=self.config.document_processing.separators,
            )
        elif strategy == "agentic":
            agentic_config = self.config.document_processing.agentic_chunking
            self.text_splitter = AgenticSplitter(
                llm_provider=self.llm_provider,
                max_chunk_size=agentic_config.max_chunk_size,
                similarity_threshold=agentic_config.similarity_threshold,
            )
        elif strategy == "proposition":
            prop_config = self.config.document_processing.proposition_chunking
            self.text_splitter = PropositionSplitter(
                llm_provider=self.llm_provider,
                max_propositions_per_chunk=prop_config.max_propositions_per_chunk,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _init_retriever(self) -> None:
        strategy = self.config.retrieval.strategy

        if strategy == "dense":
            base_retriever: BaseRetriever = Retriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                config=self.config.retrieval,
            )
        elif strategy == "graph_rag":
            base_retriever = GraphRAGRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                llm_provider=self.llm_provider,
                config=self.config.retrieval,
            )
        elif strategy == "raptor":
            base_retriever = RAPTORRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                llm_provider=self.llm_provider,
                config=self.config.retrieval,
            )
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        # Optionally wrap with Corrective RAG
        if self.config.retrieval.corrective_rag_enabled:
            self.retriever: BaseRetriever = CorrectiveRAGRetriever(
                base_retriever=base_retriever,
                llm_provider=self.llm_provider,
                config=self.config.retrieval.corrective_rag,
            )
        else:
            self.retriever = base_retriever

    def _init_generation(self) -> None:
        strategy = self.config.generation.strategy

        if strategy == "standard":
            self.generation_strategy: GenerationStrategy = StandardGeneration(
                llm_provider=self.llm_provider,
            )
        elif strategy == "cove":
            self.generation_strategy = ChainOfVerificationGeneration(
                llm_provider=self.llm_provider,
                max_verification_questions=self.config.cove.max_verification_questions,
            )
        elif strategy == "attributed":
            self.generation_strategy = AttributedGeneration(
                llm_provider=self.llm_provider,
                citation_style=self.config.attributed_generation.citation_style,
            )
        else:
            raise ValueError(f"Unsupported generation strategy: {strategy}")

    def ingest_documents(self, documents: List[Document]) -> Dict[str, int]:
        """
        Ingest documents into the vector store.
        """
        # 1. Split
        splitted_docs = self.text_splitter.split_documents(documents)
        logger.info(
            f"Split {len(documents)} documents into {len(splitted_docs)} chunks."
        )

        # 2. Embed
        texts = [doc.content for doc in splitted_docs]
        embeddings = self.embedding_provider.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        # 3. Store
        self.vector_store.add_documents(splitted_docs, embeddings)
        logger.info(f"Stored documents in {self.config.vectorstore.provider}.")

        # 4. Post-ingestion hooks for advanced retrievers
        if isinstance(self.retriever, CorrectiveRAGRetriever):
            inner = self.retriever.base_retriever
        else:
            inner = self.retriever

        if isinstance(inner, GraphRAGRetriever):
            logger.info("Building knowledge graph for Graph RAG...")
            inner.build_graph(splitted_docs)

        if isinstance(inner, RAPTORRetriever):
            logger.info("Building RAPTOR tree...")
            inner.build_tree(splitted_docs)

        return {"source_documents": len(documents), "chunks": len(splitted_docs)}

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        """
        start_time = time.time()

        # 1. Retrieve
        retrieved_docs = self.retriever.retrieve(
            query, top_k=top_k or 5, filters=filters
        )

        # 2. Generate
        result = self.generation_strategy.generate(query, retrieved_docs)

        latency = time.time() - start_time
        result["sources"] = retrieved_docs
        result["latency"] = latency

        return result

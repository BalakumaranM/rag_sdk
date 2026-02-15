import time
import logging
from typing import List, Optional, Dict, Any
from .config import Config
from .document import Document, TextSplitter, AgenticSplitter, PropositionSplitter
from .document.base import BaseTextSplitter, BasePDFParser
from .document.loader import DocumentLoader
from .document.pdf_parser import PyMuPDFParser
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
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_splitter()
        self._init_retriever()
        self._init_generation()
        self._init_pdf_parser()

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

    def _init_pdf_parser(self) -> None:
        pdf_config = self.config.document_processing.pdf_parser
        if pdf_config.backend == "pymupdf":
            self.pdf_parser: BasePDFParser = PyMuPDFParser(
                line_y_tolerance=pdf_config.line_y_tolerance,
                min_segment_length=pdf_config.min_segment_length,
                grid_snap_tolerance=pdf_config.grid_snap_tolerance,
                min_table_rows=pdf_config.min_table_rows,
                min_table_cols=pdf_config.min_table_cols,
                segment_merge_gap=pdf_config.segment_merge_gap,
                checkbox_min_size=pdf_config.checkbox_min_size,
                checkbox_max_size=pdf_config.checkbox_max_size,
                checkbox_aspect_ratio_tolerance=pdf_config.checkbox_aspect_ratio_tolerance,
                include_tables_in_text=pdf_config.include_tables_in_text,
            )
        else:
            raise ValueError(f"Unsupported PDF parser backend: {pdf_config.backend}")

    def ingest_pdf(self, file_path: str) -> Dict[str, int]:
        """Parse a PDF file and ingest it into the RAG pipeline.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict with source_documents and chunks counts.
        """
        pdf_config = self.config.document_processing.pdf_parser
        documents = DocumentLoader.load_file(
            file_path,
            pdf_parser=self.pdf_parser,
            one_doc_per_page=pdf_config.one_document_per_page,
        )
        if isinstance(documents, Document):
            documents = [documents]
        logger.info(f"Loaded {len(documents)} documents from PDF: {file_path}")
        return self.ingest_documents(documents)

    def ingest_documents(self, documents: List[Document]) -> Dict[str, int]:
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")

        texts = [doc.content for doc in split_docs]
        embeddings = self.embedding_provider.embed_documents(texts)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        self.vector_store.add_documents(split_docs, embeddings)
        logger.info(f"Stored documents in {self.config.vectorstore.provider}.")

        inner = (
            self.retriever.base_retriever
            if isinstance(self.retriever, CorrectiveRAGRetriever)
            else self.retriever
        )

        if isinstance(inner, GraphRAGRetriever):
            logger.info("Building knowledge graph for Graph RAG...")
            inner.build_graph(split_docs)

        if isinstance(inner, RAPTORRetriever):
            logger.info("Building RAPTOR tree...")
            inner.build_tree(split_docs)

        return {"source_documents": len(documents), "chunks": len(split_docs)}

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        retrieved_docs = self.retriever.retrieve(
            query, top_k=top_k or 5, filters=filters
        )
        result = self.generation_strategy.generate(query, retrieved_docs)

        result["sources"] = retrieved_docs
        result["latency"] = time.time() - start_time
        return result

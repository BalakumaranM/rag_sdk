import time
import logging
from typing import List, Optional, Dict, Any
from .config import Config
from .document import (
    Document,
    TextSplitter,
    AgenticSplitter,
    PropositionSplitter,
    SemanticSplitter,
    LateSplitter,
)
from .document.base import BaseTextSplitter, BasePDFParser
from .document.loader import DocumentLoader
from .document.pdf_parser import PyMuPDFParser
from .embeddings import OpenAIEmbedding, EmbeddingProvider
from .vectorstore import VectorStoreProvider, InMemoryVectorStore
from .llm import LLMProvider, OpenAILLM
from .retrieval import (
    Retriever,
    GraphRAGRetriever,
    RAPTORRetriever,
    CorrectiveRAGRetriever,
)
from .retrieval.base import BaseRetriever
from .reranking.base import BaseReranker
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
        self._init_reranker()
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
        elif self.config.embeddings.provider == "local":
            if not self.config.embeddings.local:
                raise ValueError("Local embedding config is missing")
            from .embeddings import LocalEmbedding

            self.embedding_provider = LocalEmbedding(self.config.embeddings.local)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.config.embeddings.provider}"
            )

    def _init_vectorstore(self) -> None:
        provider = self.config.vectorstore.provider
        if provider == "memory":
            self.vector_store: VectorStoreProvider = InMemoryVectorStore()
        elif provider == "faiss":
            if not self.config.vectorstore.faiss:
                raise ValueError("FAISS vector store config is missing")
            from .vectorstore.faiss_store import FAISSVectorStore

            self.vector_store = FAISSVectorStore(self.config.vectorstore.faiss)
        elif provider == "chroma":
            if not self.config.vectorstore.chroma:
                raise ValueError("Chroma vector store config is missing")
            from .vectorstore.chroma_store import ChromaVectorStore

            self.vector_store = ChromaVectorStore(self.config.vectorstore.chroma)
        elif provider == "pinecone":
            if not self.config.vectorstore.pinecone:
                raise ValueError("Pinecone vector store config is missing")
            from .vectorstore.pinecone import PineconeVectorStore

            self.vector_store = PineconeVectorStore(self.config.vectorstore.pinecone)
        elif provider == "weaviate":
            if not self.config.vectorstore.weaviate:
                raise ValueError("Weaviate vector store config is missing")
            from .vectorstore.weaviate_store import WeaviateVectorStore

            self.vector_store = WeaviateVectorStore(self.config.vectorstore.weaviate)
        elif provider == "qdrant":
            if not self.config.vectorstore.qdrant:
                raise ValueError("Qdrant vector store config is missing")
            from .vectorstore.qdrant_store import QdrantVectorStore

            self.vector_store = QdrantVectorStore(self.config.vectorstore.qdrant)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

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
        elif strategy == "semantic":
            self.text_splitter = SemanticSplitter(
                embedding_provider=self.embedding_provider,
                config=self.config.document_processing.semantic_chunking,
            )
        elif strategy == "late":
            self.text_splitter = LateSplitter(
                config=self.config.document_processing.late_chunking,
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
        elif strategy == "multi_query":
            from .retrieval import MultiQueryRetriever

            dense_retriever = Retriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                config=self.config.retrieval,
            )
            base_retriever = MultiQueryRetriever(
                base_retriever=dense_retriever,
                llm_provider=self.llm_provider,
                config=self.config.retrieval.multi_query,
            )
        elif strategy == "hybrid":
            from .retrieval import HybridRetriever

            base_retriever = HybridRetriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                config=self.config.retrieval,
            )
        elif strategy == "self_rag":
            from .retrieval import SelfRAGRetriever

            dense_retriever = Retriever(
                embedding_provider=self.embedding_provider,
                vector_store=self.vector_store,
                config=self.config.retrieval,
            )
            base_retriever = SelfRAGRetriever(
                base_retriever=dense_retriever,
                llm_provider=self.llm_provider,
                config=self.config.retrieval.self_rag,
            )
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")

        # Optionally wrap with Corrective RAG
        if self.config.retrieval.corrective_rag_enabled:
            base_retriever = CorrectiveRAGRetriever(
                base_retriever=base_retriever,
                llm_provider=self.llm_provider,
                config=self.config.retrieval.corrective_rag,
            )

        # Optionally wrap with Contextual Compression
        if self.config.retrieval.contextual_compression_enabled:
            from .retrieval import ContextualCompressionRetriever

            base_retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                llm_provider=self.llm_provider,
                config=self.config.retrieval.contextual_compression,
            )

        self.retriever: BaseRetriever = base_retriever

    def _init_reranker(self) -> None:
        reranking_config = self.config.retrieval.reranking
        if not reranking_config.enabled:
            self.reranker: Optional[BaseReranker] = None
            return

        if reranking_config.provider == "cohere":
            from .reranking import CohereReranker

            self.reranker = CohereReranker(reranking_config.cohere)
        elif reranking_config.provider == "cross-encoder":
            from .reranking import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(reranking_config.cross_encoder)
        else:
            raise ValueError(
                f"Unsupported reranking provider: {reranking_config.provider}"
            )

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
        elif pdf_config.backend == "docling":
            from .document.docling_parser import DoclingParser

            self.pdf_parser = DoclingParser(
                do_ocr=pdf_config.docling_do_ocr,
                do_table_structure=pdf_config.docling_do_table_structure,
                table_mode=pdf_config.docling_table_mode,
                timeout=pdf_config.docling_timeout,
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

        # Build specialized indexes for retriever wrappers
        inner = self._unwrap_retriever(self.retriever)

        if isinstance(inner, GraphRAGRetriever):
            logger.info("Building knowledge graph for Graph RAG...")
            inner.build_graph(split_docs)

        if isinstance(inner, RAPTORRetriever):
            logger.info("Building RAPTOR tree...")
            inner.build_tree(split_docs)

        from .retrieval import HybridRetriever

        if isinstance(inner, HybridRetriever):
            logger.info("Building BM25 index for hybrid retrieval...")
            inner.index_documents(split_docs)

        return {"source_documents": len(documents), "chunks": len(split_docs)}

    @staticmethod
    def _unwrap_retriever(retriever: BaseRetriever) -> BaseRetriever:
        """Unwrap chained retrievers to find the innermost base retriever."""
        inner = retriever
        while hasattr(inner, "base_retriever"):
            inner = inner.base_retriever  # type: ignore[attr-defined]
        return inner

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        k = top_k or 5
        # Over-fetch when reranking to give the reranker more candidates
        fetch_k = k * 3 if self.reranker else k
        retrieved_docs = self.retriever.retrieve(query, top_k=fetch_k, filters=filters)

        # Apply reranking if configured
        if self.reranker and retrieved_docs:
            reranked = self.reranker.rerank(query, retrieved_docs, top_k=k)
            retrieved_docs = [doc for doc, _score in reranked]

        result = self.generation_strategy.generate(query, retrieved_docs)

        result["sources"] = retrieved_docs
        result["latency"] = time.time() - start_time
        return result

import time
from typing import List, Optional, Dict, Any
from .config import Config
from .document import Document, TextSplitter
from .embeddings import OpenAIEmbedding, EmbeddingProvider
from .vectorstore import VectorStoreProvider, InMemoryVectorStore, PineconeVectorStore
from .llm import LLMProvider, OpenAILLM
from .retrieval import Retriever


import logging

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

        # 2. Vector Store
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

        # 3. LLM
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

        # 4. Processing
        self.text_splitter = TextSplitter(
            chunk_size=self.config.document_processing.chunk_size,
            chunk_overlap=self.config.document_processing.chunk_overlap,
            separators=self.config.document_processing.separators,
        )

        # 5. Retriever
        self.retriever = Retriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            config=self.config.retrieval,
        )

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
        retrieved_docs = self.retriever.retrieve(query, top_k, filters)

        # 2. Generate
        context_text = "\n\n".join([doc.content for doc in retrieved_docs])

        system_prompt = (
            "You are a helpful assistant. Use the following pieces of context to answer the user's question.\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context_text}"
        )

        answer = self.llm_provider.generate(prompt=query, system_prompt=system_prompt)

        latency = time.time() - start_time

        return {"answer": answer, "sources": retrieved_docs, "latency": latency}

# API Reference: Provider ABCs

All provider base classes are abstract (ABC). Each defines the interface that concrete
implementations must follow.

> **Lazy provider resolution:** All built-in components (`Retriever`,
> `BasicGraphRAGRetriever`, `AdvancedGraphRAGRetriever`, `RAPTORRetriever`,
> `HybridRetriever`, `GraphIndexer`, all generation strategies, and all splitters)
> accept `None` for `embedding_provider` and `llm_provider` in their constructors.
> When `None`, the component reads from `Settings.embedding_provider` /
> `Settings.llm_provider` lazily at call time.  Pass an explicit value to override
> `Settings` for a specific instance.

---

## EmbeddingProvider

```python
from rag_sdk.embeddings import EmbeddingProvider
```

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: List of document strings.

        Returns:
            List of embedding vectors (one per text).
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query string.

        Returns:
            Embedding vector.
        """
```

**Implementations:** `OpenAIEmbedding`, `CohereEmbedding`, `GeminiEmbedding`, `VoyageEmbedding`, `LocalEmbedding`

---

## LLMProvider

```python
from rag_sdk.llm import LLMProvider
```

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response for the prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated text.
        """

    @abstractmethod
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """Stream the response for the prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Yields:
            Text chunks as they are generated.
        """
```

**Implementations:** `OpenAILLM`, `GeminiLLM`, `AnthropicLLM`, `CohereLLM`

---

## VectorStoreProvider

```python
from rag_sdk.vectorstore import VectorStoreProvider
```

```python
class VectorStoreProvider(ABC):
    @abstractmethod
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        """Store documents and their embeddings.

        Args:
            documents: Documents to store.
            embeddings: Corresponding embedding vectors.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Max results.
            filters: Optional metadata filters.

        Returns:
            List of (Document, score) tuples, sorted by similarity descending.
        """

    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """Delete documents by ID.

        Args:
            document_ids: IDs to delete.
        """
```

**Implementations:** `InMemoryVectorStore`, `FAISSVectorStore`, `ChromaVectorStore`, `PineconeVectorStore`, `WeaviateVectorStore`, `QdrantVectorStore`

---

## BaseRetriever

```python
from rag_sdk.retrieval import BaseRetriever
```

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query.
            top_k: Max documents to return.
            filters: Optional metadata filters.

        Returns:
            List of relevant documents, ordered by relevance.
        """
```

**Implementations:** `Retriever` (dense), `BasicGraphRAGRetriever`, `AdvancedGraphRAGRetriever`, `RAPTORRetriever`, `MultiQueryRetriever`, `HybridRetriever`, `SelfRAGRetriever`, `CorrectiveRAGRetriever`, `ContextualCompressionRetriever`

---

## BaseReranker

```python
from rag_sdk.reranking import BaseReranker
```

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Max documents to return.

        Returns:
            List of (Document, score) tuples, ordered by relevance descending.
        """
```

**Implementations:** `CohereReranker`, `CrossEncoderReranker`

---

## GenerationStrategy

```python
from rag_sdk.generation import GenerationStrategy
```

```python
class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate an answer from a query and retrieved documents.

        Args:
            query: The user's question.
            documents: Retrieved context documents.

        Returns:
            Dict with at least an 'answer' key. Strategies may include
            additional keys (e.g., 'citations', 'verification_qa').
        """
```

**Implementations:** `StandardGeneration`, `ChainOfVerificationGeneration`, `AttributedGeneration`

---

## BaseTextSplitter

```python
from rag_sdk.document import BaseTextSplitter
```

```python
class BaseTextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]: ...
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]: ...
```

**Implementations:** `TextSplitter`, `AgenticSplitter`, `PropositionSplitter`, `SemanticSplitter`, `LateSplitter`

---

## BasePDFParser

```python
from rag_sdk.document import BasePDFParser
```

```python
class BasePDFParser(ABC):
    @abstractmethod
    def parse_file(self, file_path: str, pages: Optional[List[int]] = None) -> ParsedDocument: ...
    @abstractmethod
    def parse_page(self, page: Any, page_number: int) -> ParsedPage: ...
    def to_documents(self, parsed: ParsedDocument, one_doc_per_page: bool = True) -> List[Document]: ...
```

**Implementations:** `PyMuPDFParser`, `DoclingParser`

---

## GraphIndexer

```python
from rag_sdk.graph import GraphIndexer
```

Manages the Advanced GraphRAG ingestion pipeline. Handles entity/relationship extraction, cross-chunk merging, community detection, and community report generation. Used internally by `AdvancedGraphRAGRetriever` and available directly for custom workflows.

```python
class GraphIndexer:
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None: ...

    def build_graph(self, documents: List[Document]) -> None:
        """Run the full ingestion pipeline:
        1. Extract entities/relationships with descriptions and weights
        2. Merge duplicates across chunks (LLM-merged descriptions)
        3. Detect hierarchical communities (Leiden or Louvain)
        4. Generate structured community reports (title, summary, findings, rank)
        5. Embed community summaries
        """

    @property
    def entities(self) -> Dict[str, Entity]: ...

    @property
    def relationships(self) -> List[Relationship]: ...

    @property
    def communities(self) -> Dict[str, Community]: ...

    @property
    def graph(self) -> nx.Graph: ...
```

**Graph data models** (importable from `rag_sdk.graph`):

```python
from rag_sdk.graph import Entity, Relationship, Community

@dataclass
class Entity:
    name: str
    entity_type: str = ""
    description: str = ""          # LLM-generated description
    document_ids: List[str] = ...  # chunks this entity appears in

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    description: str = ""          # LLM-generated description
    weight: float = 1.0            # strength 1–10
    document_ids: List[str] = ...

@dataclass
class Community:
    id: str
    level: int                               # 0 = coarsest community level
    entities: List[str]
    title: str = ""                          # short topic label
    summary: str = ""                        # 1-2 sentence summary
    full_content: str = ""                   # markdown report
    findings: List[Dict[str, str]] = ...     # structured finding bullets
    rank: float = 0.0                        # importance score (1–10)
    embedding: Optional[List[float]] = None  # community summary embedding
```

## See Also

- [Extending](41-extending.md) — implementing custom providers
- [API: Document](32-api-document.md) — concrete classes

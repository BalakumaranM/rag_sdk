# API Reference {#api-reference}

## Core RAG Class

```python
from rag_sdk import RAG

class RAG:
    """Main RAG orchestrator"""
    
    def __init__(self, config: Config):
        """
        Initialize RAG system with configuration
        Args:
            config: Configuration object
        """
        
    def ingest_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> IngestResult:
        """
        Ingest documents into the vector store
        Args:
            documents: List of Document objects
            batch_size: Number of documents to process in each batch
            show_progress: Show progress bar
        Returns:
            IngestResult with statistics
        """
        
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
        return_sources: bool = True,
        stream: bool = False
    ) -> Union[RAGResponse, Iterator[RAGStreamResponse]]:
        """
        Query the RAG system
        Args:
            query: User query string
            top_k: Override default top_k for retrieval
            filters: Metadata filters for retrieval
            return_sources: Include source documents in response
            stream: Stream the response
        Returns:
            RAGResponse or iterator of RAGStreamResponse
        """
        
    def async_query(self, query: str, **kwargs) -> Awaitable[RAGResponse]:
        """Async version of query"""
        
    def batch_query(self, queries: List[str], max_concurrent: int = 5) -> List[RAGResponse]:
        """Process multiple queries in batch"""

    def delete_documents(self, document_ids: List[str]) -> DeleteResult:
        """Delete documents from vector store"""
        
    def get_statistics(self) -> RAGStatistics:
        """Get system statistics"""
```

## Document Class

```python
from rag_sdk.document import Document

class Document:
    """Represents a document"""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ):
        """
        Args:
            content: Document text content
            metadata: Optional metadata dict
            doc_id: Optional document ID (auto-generated if not provided)
        """
        
    @classmethod
    def from_file(cls, file_path: str) -> "Document":
        """Load document from file"""
        
    @classmethod
    def from_url(cls, url: str) -> "Document":
        """Load document from URL"""
```

## Response Classes

```python
from rag_sdk.response import RAGResponse, Source

class RAGResponse:
    """Response from RAG query"""
    answer: str  # Generated answer
    sources: List[Source]  # Retrieved sources
    metadata: Dict  # Additional metadata
    latency: float  # Response time in seconds
    
class Source:
    """Retrieved source document"""
    content: str  # Source text
    score: float  # Relevance score
    metadata: Dict  # Document metadata
    document_id: str  # Source document ID
```

## Advanced Retrieval

```python
from rag_sdk.retrieval import AdvancedRetriever

retriever = AdvancedRetriever(config)

# Hybrid search
results = retriever.hybrid_search(
    query="What is machine learning?",
    dense_weight=0.7,
    sparse_weight=0.3,
    top_k=5
)

# MMR (Maximal Marginal Relevance)
results = retriever.mmr_search(
    query="AI safety",
    fetch_k=20,
    lambda_mult=0.5  # diversity parameter
)

# Filtered search
results = retriever.search(
    query="latest research",
    filters={
        "category": "AI",
        "year": {"$gte": 2023},
        "author": {"$in": ["John Doe", "Jane Smith"]}
    }
)
```

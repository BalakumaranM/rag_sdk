# Extending the SDK

Add custom providers, strategies, and components by implementing the base ABCs.

## Custom Embedding Provider

```python
from typing import List
from rag_sdk.embeddings import EmbeddingProvider


class MyEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str):
        # Initialize your embedding model
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return list of embedding vectors
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        # Return single embedding vector
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        # Your embedding logic here
        ...
```

## Custom LLM Provider

```python
from typing import Optional, Iterator
from rag_sdk.llm import LLMProvider


class MyLLM(LLMProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Call your LLM and return the response
        ...

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        # Yield response chunks
        ...
```

## Custom Vector Store

```python
from typing import List, Optional, Tuple, Dict, Any
from rag_sdk.vectorstore import VectorStoreProvider
from rag_sdk.document import Document


class MyVectorStore(VectorStoreProvider):
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        # Store documents and embeddings
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        # Return (document, score) tuples sorted by similarity
        ...

    def delete(self, document_ids: List[str]) -> None:
        # Delete by ID
        ...
```

## Custom Retrieval Strategy

```python
from typing import List, Optional, Dict, Any
from rag_sdk.retrieval import BaseRetriever
from rag_sdk.document import Document


class MyRetriever(BaseRetriever):
    def __init__(self, embedding_provider, vector_store, **kwargs):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        # Your retrieval logic
        ...
```

## Custom Reranker

```python
from typing import List, Tuple
from rag_sdk.reranking import BaseReranker
from rag_sdk.document import Document


class MyReranker(BaseReranker):
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        # Return (document, score) tuples sorted by relevance
        ...
```

## Custom Generation Strategy

```python
from typing import List, Dict, Any
from rag_sdk.generation import GenerationStrategy
from rag_sdk.document import Document


class MyGeneration(GenerationStrategy):
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        # Must return dict with at least "answer" key
        ...
        return {"answer": answer}
```

## Custom Text Splitter

```python
from typing import List
from rag_sdk.document import BaseTextSplitter, Document


class MySplitter(BaseTextSplitter):
    def split_text(self, text: str) -> List[str]:
        # Return list of text chunks
        ...

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                content=chunk,
                metadata={**doc.metadata, "chunk_index": i, "parent_id": doc.id},
            )
            for doc in documents
            for i, chunk in enumerate(self.split_text(doc.content))
        ]
```

## Custom PDF Parser

```python
from typing import Any, List, Optional
from rag_sdk.document import BasePDFParser, ParsedDocument, ParsedPage


class MyPDFParser(BasePDFParser):
    def parse_file(
        self, file_path: str, pages: Optional[List[int]] = None
    ) -> ParsedDocument:
        # Parse the PDF and return structured data
        ...

    def parse_page(self, page: Any, page_number: int) -> ParsedPage:
        # Parse a single page
        ...
    # to_documents() is inherited from BasePDFParser
```

## Swapping Components on RAG

After constructing a `RAG` instance, you can swap any component:

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)

# Swap embedding provider
rag.embedding_provider = MyEmbedding(model_name="custom-model")

# Swap LLM
rag.llm_provider = MyLLM(endpoint="http://localhost:8080")

# Swap vector store
rag.vector_store = MyVectorStore()

# Swap retriever
rag.retriever = MyRetriever(
    embedding_provider=rag.embedding_provider,
    vector_store=rag.vector_store,
)

# Swap reranker
rag.reranker = MyReranker()

# Swap generation strategy
rag.generation_strategy = MyGeneration(llm_provider=rag.llm_provider)

# Swap splitter
rag.text_splitter = MySplitter()

# Swap PDF parser
rag.pdf_parser = MyPDFParser()
```

Components are plain instance attributes, so swapping is straightforward. Ensure type compatibility — the `RAG` methods expect the standard ABC interfaces.

## See Also

- [API: Providers](33-api-providers.md) — all ABC interfaces
- [Recipes](40-recipes.md) — production configurations

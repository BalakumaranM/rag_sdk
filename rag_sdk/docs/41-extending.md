# Extending the SDK

Add custom providers, strategies, and components by implementing the base ABCs.

---

## Using Custom Providers

There are three ways to plug a custom provider into the SDK, in order of preference.

### Option 1 — `Settings` singleton (global default)

Set a provider once before creating any `RAG` instance. Every component in the SDK
resolves providers lazily at call time, so they all pick up the override automatically.

```python
from rag_sdk import RAG, Settings
from rag_sdk.config import ConfigLoader

# Define your custom providers (see sections below for how to build them)
Settings.embedding_provider = MyEmbedding("http://localhost:8080")
Settings.llm_provider       = MyLLM("http://localhost:11434")

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)  # uses your providers — no config changes needed
```

`Settings` is a module-level singleton backed by Python's module cache. Every import
of `Settings` in every component resolves to the same object, so setting it once is
instantly visible everywhere.

Reset during tests:

```python
Settings.reset()   # clears both embedding_provider and llm_provider back to None
```

### Option 2 — `from_providers()` classmethod (explicit, per-instance)

Pass live provider objects directly. Most explicit option — nothing implicit about it.

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG.from_providers(
    config,
    embedding_provider=MyEmbedding("http://localhost:8080"),
    llm_provider=MyLLM("http://localhost:11434"),
)
```

All other settings (vector store, retrieval strategy, chunking, etc.) still come from
`config`. Only `embedding_provider` and `llm_provider` are overridden.

### Option 3 — constructor kwargs (inline override)

Same as `from_providers()` but inline with the normal constructor:

```python
rag = RAG(
    config,
    embedding_provider=MyEmbedding("http://localhost:8080"),
    llm_provider=MyLLM("http://localhost:11434"),
)
```

### Priority chain

When `RAG` initialises, providers are resolved in this order:

```
1. Explicit kwarg to RAG() / RAG.from_providers()   ← highest priority
2. Settings.embedding_provider / Settings.llm_provider
3. config.embeddings.provider / config.llm.provider  ← lowest priority (default)
```

---

## Custom Embedding Provider

Subclass `EmbeddingProvider` and implement both abstract methods.

```python
from typing import List
from rag_sdk.embeddings import EmbeddingProvider


class MyEmbedding(EmbeddingProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.model = "my-model-v1"   # expose model name for fingerprinting

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return a list of vectors, one per text
        return [self._call_api(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        # Return a single vector
        return self._call_api(text)

    def _call_api(self, text: str) -> List[float]:
        # Call your local server or custom model
        ...
```

> **Tip:** Expose a `model` or `model_name` attribute on your class. The SDK uses it
> to build an embedding fingerprint and will warn you if you swap to a different model
> after ingesting documents.

---

## Custom LLM Provider

Subclass `LLMProvider` and implement `generate` and `stream`.

```python
from typing import Optional, Iterator
from rag_sdk.llm import LLMProvider


class MyLLM(LLMProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Call your LLM endpoint and return the full response string
        ...

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        # Yield response tokens/chunks as they arrive
        ...
```

---

## Custom Vector Store

Subclass `VectorStoreProvider` and implement the three abstract methods.

```python
from typing import List, Optional, Tuple, Dict, Any
from rag_sdk.vectorstore import VectorStoreProvider
from rag_sdk.document import Document


class MyVectorStore(VectorStoreProvider):
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        # Store documents and their corresponding embedding vectors
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        # Return (document, similarity_score) tuples, highest score first
        ...

    def delete(self, document_ids: List[str]) -> None:
        ...
```

---

## Custom Retrieval Strategy

Subclass `BaseRetriever`.

```python
from typing import List, Optional, Dict, Any
from rag_sdk.retrieval import BaseRetriever
from rag_sdk.document import Document
from rag_sdk.settings import Settings


class MyRetriever(BaseRetriever):
    def __init__(self, embedding_provider=None, vector_store=None):
        self._embedding_provider = embedding_provider
        self.vector_store = vector_store

    @property
    def embedding_provider(self):
        return self._embedding_provider or Settings.embedding_provider

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        emb = self.embedding_provider.embed_query(query)
        results = self.vector_store.search(emb, top_k, filters)
        return [doc for doc, _ in results]
```

> **Best practice:** Follow the lazy property pattern shown above. Store providers
> with a leading underscore and expose them via a property that falls back to `Settings`.
> This makes your retriever work whether providers are passed explicitly or set globally.

---

## Custom Reranker

```python
from typing import List, Tuple
from rag_sdk.reranking import BaseReranker
from rag_sdk.document import Document


class MyReranker(BaseReranker):
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        # Return (document, score) tuples sorted by relevance descending
        ...
```

---

## Custom Generation Strategy

```python
from typing import List, Dict, Any, Optional
from rag_sdk.generation import GenerationStrategy
from rag_sdk.document import Document
from rag_sdk.llm import LLMProvider
from rag_sdk.settings import Settings


class MyGeneration(GenerationStrategy):
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self._llm_provider = llm_provider

    @property
    def llm_provider(self) -> LLMProvider:
        return self._llm_provider or Settings.llm_provider

    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        context = "\n\n".join(d.content for d in documents)
        answer = self.llm_provider.generate(prompt=query, system_prompt=context)
        return {"answer": answer}   # must include "answer" key
```

---

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

---

## Custom PDF Parser

```python
from typing import Any, List, Optional
from rag_sdk.document import BasePDFParser, ParsedDocument, ParsedPage


class MyPDFParser(BasePDFParser):
    def parse_file(
        self, file_path: str, pages: Optional[List[int]] = None
    ) -> ParsedDocument:
        ...

    def parse_page(self, page: Any, page_number: int) -> ParsedPage:
        ...
    # to_documents() is inherited from BasePDFParser
```

---

## Swapping Providers at Runtime

### Embedding and LLM providers — cascading setters

Assigning to `rag.embedding_provider` or `rag.llm_provider` triggers a cascade that
rebuilds all dependent components (retriever, splitter, generation strategy) with the
new provider:

```python
rag.embedding_provider = MyNewEmbedding()   # rebuilds splitter + retriever
rag.llm_provider       = MyNewLLM()         # rebuilds splitter + retriever + generation
```

### Embedding model change — critical warning

The vector store holds raw float vectors with no knowledge of which model generated
them. Swapping to a different embedding model after ingestion causes one of two failures:

- **Dimension mismatch** — if models output different vector sizes, the similarity search
  throws a shape error immediately.
- **Space mismatch** — if models output the same size but different spaces (e.g.,
  `text-embedding-3-small` → `text-embedding-ada-002`), there is no error but results
  are silently wrong: cosine similarity between vectors from different spaces is
  geometrically meaningless.

The SDK detects this and logs a warning when the fingerprint changes after ingestion.
The correct recovery path is:

```python
rag.embedding_provider = MyNewEmbedding()
rag.clear_index()            # wipes the vector store and resets retriever state
rag.ingest_documents(docs)   # re-embeds everything with the new model
```

### Other components — direct attribute assignment

All other components are plain attributes. Swap them directly:

```python
rag.vector_store        = MyVectorStore()
rag.retriever           = MyRetriever(rag.embedding_provider, rag.vector_store)
rag.reranker            = MyReranker()
rag.generation_strategy = MyGeneration(llm_provider=rag.llm_provider)
rag.text_splitter       = MySplitter()
rag.pdf_parser          = MyPDFParser()
```

Ensure type compatibility — `RAG` methods expect the standard ABC interfaces.

---

## See Also

- [API: RAG](30-api-rag.md) — full `RAG` constructor, `from_providers()`, and `clear_index()` reference
- [API: Providers](33-api-providers.md) — all ABC interfaces
- [Recipes](40-recipes.md) — production configurations

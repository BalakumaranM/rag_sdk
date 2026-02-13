# Core Components {#core-components}

## 1. Document Processing Module

```python
from rag_sdk.document import (
    DocumentLoader,
    TextSplitter,
    DocumentProcessor
)
```

**Capabilities:**
*   **Loaders:** PDF, DOCX, TXT, HTML, Markdown, CSV, JSON
*   **Splitters:** Recursive, semantic, fixed-size, token-based
*   **Metadata extraction:** Auto-extract titles, authors, timestamps
*   **Preprocessing:** Cleaning, normalization, deduplication

## 2. Embedding Module

```python
from rag_sdk.embeddings import EmbeddingProvider
```

**Supported Providers:**
*   OpenAI (`text-embedding-3-small`, `text-embedding-3-large`)
*   Cohere (`embed-english-v3.0`, `embed-multilingual-v3.0`)
*   Anthropic Voyage
*   HuggingFace models
*   Custom embedding endpoints
*   Local models (SentenceTransformers)

## 3. Vector Store Module

```python
from rag_sdk.vectorstore import VectorStoreProvider
```

**Supported Stores:**
*   Pinecone
*   Weaviate
*   Qdrant
*   Milvus
*   ChromaDB
*   FAISS (local)
*   Elasticsearch
*   PostgreSQL with pgvector

## 4. LLM Generation Module

```python
from rag_sdk.llm import LLMProvider
# Or use the unified LiteLLM provider
from rag_sdk.llm import LiteLLMProvider
```

**Supported Providers:**
*   OpenAI (GPT-4, GPT-3.5)
*   Anthropic (Claude 3/4)
*   Cohere (Command)
*   Google (Gemini)
*   Azure OpenAI
*   AWS Bedrock
*   Custom API endpoints
*   Local models (via llama.cpp, vLLM)

## 5. Retrieval Module

```python
from rag_sdk.retrieval import (
    Retriever,
    HybridSearch,
    Reranker
)
```

**Features:**
*   Dense retrieval (vector similarity)
*   Sparse retrieval (BM25, TF-IDF)
*   Hybrid search (combining dense + sparse)
*   MMR (Maximal Marginal Relevance)
*   Contextual compression
*   Re-ranking (Cohere, Cross-encoders)

# RAG SDK Development Rules - Part 1: Foundation & Architecture

## 🎯 Project Mission
Build a **world-class, production-ready RAG (Retrieval Augmented Generation) SDK** that rivals industry leaders like LangChain and LlamaIndex. This SDK must be provider-agnostic, highly modular, well-documented, and enterprise-ready.

---

## 📋 Core Principles

### 1. Teaching & Learning Philosophy
- **NEVER just provide code without explanation**
- Always explain the "why" behind architectural decisions
- Break down complex concepts into digestible parts
- Encourage the developer to understand each component before moving forward
- Ask clarifying questions when requirements are ambiguous
- Suggest best practices and alternatives, explaining trade-offs

### 2. Code Quality Standards
- **Production-grade only** - no shortcuts or "this will work for now" solutions
- Follow SOLID principles and clean code practices
- Type hints for ALL Python functions (using `typing` module)
- Comprehensive docstrings (Google or NumPy style)
- Error handling at every layer with meaningful error messages
- Input validation for all public APIs
- Unit test coverage minimum: 80%
- Integration tests for critical paths

### 3. Architecture Requirements
The SDK follows a layered architecture:

```
Application Layer
    ↓
RAG SDK Core Engine
    ↓
Provider Adapters Layer (OpenAI, Anthropic, Cohere, Pinecone, etc.)
```

**Key Components:**
1. **Document Processing Module** - Loaders, splitters, preprocessors
2. **Embedding Module** - Provider-agnostic embedding interface
3. **Vector Store Module** - Support for Pinecone, Weaviate, Qdrant, FAISS, ChromaDB, etc.
4. **LLM Generation Module** - Support for OpenAI, Anthropic, Cohere, Google Gemini, etc.
5. **Retrieval Module** - Dense, sparse, hybrid search with re-ranking
6. **Configuration System** - YAML-based with environment variable support
7. **Observability** - Logging, monitoring, tracing, metrics

---

## 🛠️ Technical Stack & Standards

### Required Technologies
- **Language**: Python 3.9+ (use modern Python features)
- **Async**: Use `asyncio` for all I/O operations
- **Config**: YAML with Pydantic validation
- **Testing**: pytest with fixtures and parametrization
- **Documentation**: Sphinx with autodoc
- **Type Checking**: mypy strict mode
- **Linting**: ruff (replaces flake8, black, isort)
- **Dependency Management**: Poetry or pip-tools

### Code Organization
```
rag-sdk/
├── rag_sdk/
│   ├── __init__.py
│   ├── core/              # Core abstractions and interfaces
│   ├── document/          # Document processing
│   ├── embeddings/        # Embedding providers
│   ├── vectorstore/       # Vector database adapters
│   ├── llm/              # LLM providers
│   ├── retrieval/        # Retrieval strategies
│   ├── config/           # Configuration system
│   ├── observability/    # Logging, monitoring, tracing
│   ├── security/         # Auth, encryption, PII handling
│   └── utils/            # Utilities
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

---

## 📐 Design Patterns to Follow

### 1. Strategy Pattern
Use for interchangeable components (embeddings, vector stores, LLMs):
```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass
```

### 2. Factory Pattern
For creating provider instances:
```python
class EmbeddingFactory:
    @staticmethod
    def create(provider: str, config: Dict) -> EmbeddingProvider:
        providers = {
            "openai": OpenAIEmbeddings,
            "cohere": CohereEmbeddings,
            "huggingface": HuggingFaceEmbeddings
        }
        return providers[provider](config)
```

### 3. Builder Pattern
For complex object construction (RAG pipeline):
```python
rag = (RAGBuilder()
    .with_embeddings("openai", model="text-embedding-3-small")
    .with_vectorstore("pinecone", index="my-index")
    .with_llm("anthropic", model="claude-sonnet-4")
    .with_retriever(top_k=5, rerank=True)
    .build())
```

### 4. Adapter Pattern
For third-party service integration - wrap external APIs consistently

### 5. Observer Pattern
For event-driven features (callbacks, hooks, monitoring)

---

## 🔧 Configuration System Requirements

### YAML Configuration
```yaml
# All configs must support:
# 1. Environment variables: ${ENV_VAR}
# 2. Default values
# 3. Validation
# 4. Clear error messages

embeddings:
  provider: "openai"  # Required
  openai:
    api_key: "${OPENAI_API_KEY}"  # Environment variable
    model: "text-embedding-3-small"  # Default
    timeout: 60  # Optional with default
```

### Pydantic Validation
```python
from pydantic import BaseModel, Field, validator

class OpenAIConfig(BaseModel):
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(
        "text-embedding-3-small",
        description="Embedding model name"
    )
    timeout: int = Field(60, ge=1, le=300)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
```

---

## 🎯 Priority Features (Build in This Order)

### Phase 1: Core Foundation
1. Project structure and tooling setup
2. Configuration system with Pydantic
3. Abstract interfaces for all components
4. Logging and error handling framework

### Phase 2: Basic RAG Pipeline
5. Document loader (start with TXT, PDF)
6. Text splitter (recursive)
7. Embedding provider (OpenAI first)
8. Vector store (FAISS for local, Pinecone for cloud)
9. Basic retrieval (dense vector search)
10. LLM integration (OpenAI first)
11. Simple RAG query pipeline

### Phase 3: Multi-Provider Support
12. Add Cohere embeddings
13. Add Anthropic Claude
14. Add Qdrant/Weaviate vector stores
15. Provider factory pattern

### Phase 4: Advanced Retrieval
16. Hybrid search (dense + sparse)
17. Re-ranking (Cohere, cross-encoders)
18. MMR (Maximal Marginal Relevance)
19. Query understanding/expansion

### Phase 5: Production Features
20. Async support throughout
21. Caching layer
22. Rate limiting
23. Retry logic with exponential backoff
24. Comprehensive error handling

### Phase 6: Observability
25. Structured logging
26. Metrics collection
27. Distributed tracing
28. Health checks

### Phase 7: Security & Compliance
29. PII detection and redaction
30. Multi-tenancy support
31. Encryption at rest
32. Audit logging

### Phase 8: Advanced Features
33. Real-time document sync
34. Incremental indexing
35. Multi-hop reasoning
36. Cost optimization features

---

## 📚 Required Reading

Before coding any component, understand these concepts:

### RAG Fundamentals
- Vector embeddings and semantic search
- Chunking strategies and their trade-offs
- Retrieval-augmented generation pattern
- Context window management

### Design Patterns
- Strategy, Factory, Builder, Adapter, Observer patterns
- Dependency injection
- Interface segregation

### Async Python
- asyncio event loop
- async/await syntax
- Concurrency vs parallelism
- Async context managers

### Testing
- Unit vs integration vs E2E tests
- Test doubles (mocks, stubs, fakes)
- Test-driven development (TDD)
- Property-based testing

---

**Continue to Part 2 for Implementation & Quality guidelines →**

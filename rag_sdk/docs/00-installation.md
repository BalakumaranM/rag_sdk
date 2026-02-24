# Installation

## Requirements

- Python 3.9+
- pip or any PEP 517 compatible installer

## Basic Install

```bash
pip install rag_sdk
```

This installs the core SDK with these dependencies:

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0.0` | Config validation |
| `pyyaml>=6.0` | YAML config loading |
| `python-dotenv>=1.0.0` | `.env` file support |
| `numpy>=1.24.0` | Vector math |
| `openai>=1.0.0` | OpenAI embeddings & LLM |
| `tiktoken>=0.5.0` | Token counting |
| `google-genai>=1.0.0` | Gemini embeddings & LLM |
| `anthropic>=0.25.0` | Anthropic LLM |
| `cohere>=5.0.0` | Cohere embeddings, LLM & reranking |
| `voyageai>=0.2.0` | Voyage embeddings |
| `PyMuPDF>=1.24.0` | PDF parsing (default backend) |

## Optional Extras

Install additional providers and features with pip extras:

```bash
# Individual extras
pip install rag_sdk[faiss]
pip install rag_sdk[chroma]
pip install rag_sdk[pinecone]
pip install rag_sdk[weaviate]
pip install rag_sdk[qdrant]
pip install rag_sdk[docling]
pip install rag_sdk[cross-encoder]
pip install rag_sdk[late-chunking]
pip install rag_sdk[local-embeddings]

# Multiple extras
pip install rag_sdk[faiss,chroma,cross-encoder]

# Development
pip install rag_sdk[dev]
```

| Extra | Packages | Purpose |
|-------|----------|---------|
| `faiss` | `faiss-cpu>=1.7.0` | FAISS vector store |
| `chroma` | `chromadb>=0.5.0` | Chroma vector store |
| `pinecone` | `pinecone>=5.0.0` | Pinecone vector store |
| `weaviate` | `weaviate-client>=4.0.0` | Weaviate vector store |
| `qdrant` | `qdrant-client>=1.7.0` | Qdrant vector store |
| `docling` | `docling>=2.0.0` | Docling PDF parser (ML-powered) |
| `cross-encoder` | `sentence-transformers>=2.2.0` | CrossEncoder reranking |
| `late-chunking` | `transformers>=4.30.0`, `torch>=2.0.0` | Late chunking strategy |
| `local-embeddings` | `sentence-transformers>=2.2.0` | Local embedding models |
| `dev` | `pytest`, `ruff`, `mypy`, type stubs | Development & testing |

## Environment Variables

API keys can be set in a `.env` file or as environment variables. The SDK uses `python-dotenv` to load `.env` automatically.

| Variable | Used By |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI embeddings & LLM |
| `GOOGLE_API_KEY` | Gemini embeddings & LLM |
| `ANTHROPIC_API_KEY` | Anthropic LLM |
| `COHERE_API_KEY` | Cohere embeddings, LLM & reranking |
| `VOYAGE_API_KEY` | Voyage embeddings |
| `PINECONE_API_KEY` | Pinecone vector store |
| `WEAVIATE_API_KEY` | Weaviate vector store |
| `QDRANT_API_KEY` | Qdrant vector store |

Example `.env` file:

```bash
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

> API keys can also be passed directly in YAML config or Python code. Environment variables are the fallback when no key is provided in config.

## Verification

```python
from rag_sdk import RAG
from rag_sdk.config import Config

config = Config()
print(config.project_name)  # "rag-application"
```

If this runs without error, the SDK is installed correctly.

## Next Steps

- [Quickstart](01-quickstart.md) — build your first RAG pipeline in 5 minutes
- [Configuration](10-configuration.md) — learn how to configure the SDK

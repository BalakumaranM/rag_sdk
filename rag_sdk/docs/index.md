# RAG SDK Documentation

A modular, configurable Python SDK for building Retrieval-Augmented Generation pipelines.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        RAG Orchestrator                       │
│                         (rag_sdk.RAG)                         │
├────────────┬────────────┬────────────┬───────────┬───────────┤
│  Document  │ Embedding  │  Vector    │ Retrieval │Generation │
│ Processing │ Providers  │  Stores    │ Strategies│ Strategies│
├────────────┼────────────┼────────────┼───────────┼───────────┤
│ Recursive  │ OpenAI     │ Memory     │ Dense     │ Standard  │
│ Agentic    │ Cohere     │ FAISS      │ Multi-Q   │ CoVe      │
│ Proposition│ Gemini     │ Chroma     │ Hybrid    │ Attributed│
│ Semantic   │ Voyage     │ Pinecone   │ Graph RAG │           │
│ Late       │ Local      │ Weaviate   │ Adv. Graph│           │
│            │            │ Qdrant     │ RAPTOR    │           │
│ PyMuPDF    │            │            │ Self-RAG  │           │
│ Docling    │            │            │ Corrective│           │
│            │            │            │ Compress. │           │
├────────────┤            │            ├───────────┤           │
│ 5 splitters│ 5 providers│ 6 stores   │9 strategies│3 strategies│
└────────────┴────────────┴────────────┴───────────┴───────────┘
                              │
                        4 LLM Providers
                    (OpenAI, Gemini, Anthropic, Cohere)
                              │
                     2 Rerankers (optional)
                    (Cohere, CrossEncoder)
```

## Quick Navigation

### Getting Started

| Doc | Description |
|-----|-------------|
| [Installation](00-installation.md) | pip install, extras, env vars |
| [Quickstart](01-quickstart.md) | 5-minute tutorial |
| [Core Concepts](02-core-concepts.md) | Architecture and data flow |

### Configuration

| Doc | Description |
|-----|-------------|
| [Configuration](10-configuration.md) | Config loading, API key resolution |
| [YAML Reference](11-yaml-reference.md) | Every config field documented |

### Module Guides

| Doc | Description |
|-----|-------------|
| [Document Loading](20-document-loading.md) | Document model, file loading, PDF parsing |
| [Text Splitting](21-text-splitting.md) | 5 chunking strategies |
| [Embeddings](22-embeddings.md) | 5 embedding providers |
| [Vector Stores](23-vector-stores.md) | 6 vector store backends |
| [LLM Providers](24-llm-providers.md) | 4 LLM providers + JSON utility |
| [Retrieval Strategies](25-retrieval-strategies.md) | 7 base + 2 composable strategies |
| [Reranking](26-reranking.md) | Cohere + CrossEncoder reranking |
| [Generation Strategies](27-generation-strategies.md) | Standard, CoVe, Attributed |

### API Reference

| Doc | Description |
|-----|-------------|
| [API: RAG](30-api-rag.md) | RAG class methods and return types |
| [API: Config](31-api-config.md) | All 30+ config classes, incl. AdvancedGraphRAGConfig |
| [API: Document](32-api-document.md) | Document, splitter, PDF classes |
| [API: Providers](33-api-providers.md) | All base ABC interfaces |

### Recipes & Extending

| Doc | Description |
|-----|-------------|
| [Recipes](40-recipes.md) | 10 production-ready configurations |
| [Extending](41-extending.md) | Custom providers and component swapping |

## Minimal Example

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader
from rag_sdk.document import Document

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)

rag.ingest_documents([
    Document(content="RAG combines retrieval with generation."),
])

result = rag.query("What is RAG?")
print(result["answer"])
```

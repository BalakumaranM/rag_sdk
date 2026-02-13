# RAG SDK - Complete Documentation

Welcome to the RAG SDK documentation. This SDK provides a comprehensive, production-ready foundation for building RAG applications with maximum flexibility and minimal boilerplate.

## Architecture Overview {#architecture}

### Core Philosophy

The SDK follows a modular, provider-agnostic architecture with the following layers:

```mermaid
graph TD
    App[Application Layer (Your App)] --> Core[RAG SDK Core Engine]
    Core --> Pipeline[Query Processing Pipeline]
    Core --> Embedder[Embedder Module]
    Core --> Retriever[Retriever Module]
    Core --> Generator[LLM Gen Module]
    
    Embedder --> Adapters[Provider Adapters Layer]
    Retriever --> Adapters
    Generator --> Adapters
    
    Adapters --> OpenAI[OpenAI]
    Adapters --> Anthropic[Anthropic]
    Adapters --> Pinecone[Pinecone]
    Adapters --> Weaviate[Weaviate]
    Adapters --> Cohere[Cohere]
```

The system is designed to be highly extensible, allowing you to swap out components (LLMs, Vector Stores, Embedders) via configuration without changing your application code.

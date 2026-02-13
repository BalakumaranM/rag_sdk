# Usage Examples {#examples}

## Quick Start

```python
from rag_sdk import RAG, Document
from rag_sdk.config import ConfigLoader

# Load configuration
config = ConfigLoader.from_yaml("config.yaml")

# Initialize RAG system
rag = RAG(config)

# Ingest documents
documents = [
    Document(
        content="Machine learning is a subset of AI...",
        metadata={"source": "textbook", "chapter": 1}
    ),
    Document(
        content="Neural networks are inspired by the brain...",
        metadata={"source": "textbook", "chapter": 2}
    )
]

result = rag.ingest_documents(documents)
print(f"Ingested {result.successful} documents")

# Query
response = rag.query("What is machine learning?")
print(response.answer)
print(f"\nSources ({len(response.sources)}):")
for source in response.sources:
    print(f"- {source.content[:100]}... (score: {source.score:.3f})")
```

## Advanced Example with Custom Pipeline

```python
from rag_sdk import RAG
from rag_sdk.pipeline import QueryPipeline, Stage
from rag_sdk.retrieval import HybridRetriever, Reranker
from rag_sdk.generation import AnswerGenerator

# Custom pipeline
pipeline = QueryPipeline()

# Add preprocessing stage
pipeline.add_stage(
    Stage.PREPROCESS,
    lambda query: query.strip().lower()
)

# Add query expansion
pipeline.add_stage(
    Stage.TRANSFORM,
    lambda query: generate_multi_queries(query, num=3)
)

# Custom retrieval
retriever = HybridRetriever(config)
pipeline.add_stage(
    Stage.RETRIEVE,
    retriever.search
)

# Add reranking
reranker = Reranker(provider="cohere", model="rerank-english-v2.0")
pipeline.add_stage(
    Stage.RERANK,
    reranker.rerank
)

# Custom generation
generator = AnswerGenerator(config)
pipeline.add_stage(
    Stage.GENERATE,
    generator.generate
)

# Initialize RAG with custom pipeline
rag = RAG(config, pipeline=pipeline)
```

## Streaming Responses

```python
# Streaming for real-time response
for chunk in rag.query("Explain quantum computing", stream=True):
    print(chunk.delta, end="", flush=True)
```

## Async Usage

```python
import asyncio

async def main():
    rag = RAG(config)
    
    # Concurrent queries
    queries = [
        "What is AI?",
        "How does ML work?",
        "Explain neural networks"
    ]
    
    tasks = [rag.async_query(q) for q in queries]
    responses = await asyncio.gather(*tasks)
    
    for response in responses:
        print(response.answer)

asyncio.run(main())
```

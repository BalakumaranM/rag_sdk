# Quickstart

Build a working RAG pipeline in 5 minutes.

## 1. Create a Config File

Save as `config.yaml`:

```yaml
project_name: "quickstart"

embeddings:
  provider: "openai"
  openai:
    model: "text-embedding-3-small"

vectorstore:
  provider: "memory"

llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"
    temperature: 0.7

retrieval:
  strategy: "dense"
  top_k: 5

generation:
  strategy: "standard"
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

## 2. Initialize the Pipeline

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader

config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)
```

## 3. Ingest Documents

### From text files

```python
from rag_sdk.document import Document

documents = [
    Document(content="Python is a programming language created by Guido van Rossum."),
    Document(content="RAG combines retrieval with generation for grounded answers."),
    Document(content="Vector databases store embeddings for similarity search."),
]

stats = rag.ingest_documents(documents)
print(stats)
# {"source_documents": 3, "chunks": 3}
```

### From a PDF

```python
stats = rag.ingest_pdf("report.pdf")
print(stats)
# {"source_documents": 12, "chunks": 47}
```

### From a directory

```python
from rag_sdk.document import DocumentLoader

docs = DocumentLoader.load_directory("./data", extensions=[".txt", ".md"])
stats = rag.ingest_documents(docs)
```

## 4. Query

```python
result = rag.query("What is RAG?")

print(result["answer"])
# "RAG (Retrieval-Augmented Generation) combines retrieval with generation..."

print(f"Sources: {len(result['sources'])}")
print(f"Latency: {result['latency']:.2f}s")
```

## 5. Full Script

```python
from rag_sdk import RAG
from rag_sdk.config import ConfigLoader
from rag_sdk.document import Document

# Load config
config = ConfigLoader.from_yaml("config.yaml")
rag = RAG(config)

# Ingest
documents = [
    Document(content="Python is a programming language created by Guido van Rossum."),
    Document(content="RAG combines retrieval with generation for grounded answers."),
    Document(content="Vector databases store embeddings for similarity search."),
]
rag.ingest_documents(documents)

# Query
result = rag.query("What is RAG?")
print(result["answer"])
```

## What Happened

1. `ConfigLoader.from_yaml()` parsed the YAML into a validated `Config` object
2. `RAG(config)` initialized all components: OpenAI embedder, in-memory vector store, OpenAI LLM, recursive text splitter, dense retriever, standard generation
3. `ingest_documents()` split docs into chunks, embedded them, and stored in the vector store
4. `query()` embedded the query, searched for similar chunks, and passed them to the LLM for generation

## Next Steps

- [Core Concepts](02-core-concepts.md) — understand the full architecture
- [Configuration](10-configuration.md) — customize every component
- [Recipes](40-recipes.md) — production-ready configurations

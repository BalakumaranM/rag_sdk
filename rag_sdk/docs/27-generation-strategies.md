# Generation Strategies

The SDK provides 3 generation strategies that produce answers from retrieved documents. Select one via config:

```yaml
generation:
  strategy: "standard"  # "standard" | "cove" | "attributed"
```

All strategies implement `GenerationStrategy`:

```python
class GenerationStrategy(ABC):
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]: ...
```

The `RAG.query()` method adds `sources` and `latency` keys to the result dict after generation.

## Standard

Concatenates retrieved document content as context and prompts the LLM to answer.

```yaml
generation:
  strategy: "standard"
```

**Output:**

```python
{
    "answer": "RAG combines retrieval with generation...",
    "sources": [...],   # added by RAG.query()
    "latency": 1.23,    # added by RAG.query()
}
```

1 LLM call per query. The system prompt instructs the LLM to use only the provided context and admit when it doesn't know.

## Chain of Verification (CoVe)

Generates an initial answer, creates verification questions about its claims, answers each question against the context, then produces a refined answer.

```yaml
generation:
  strategy: "cove"
cove:
  max_verification_questions: 3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_verification_questions` | `int` | `3` | Max verification questions to generate |

**Output:**

```python
{
    "answer": "Refined, more accurate answer...",
    "initial_answer": "First draft answer...",
    "verification_qa": [
        {"question": "Is X true?", "answer": "Yes, according to..."},
        {"question": "Does Y hold?", "answer": "The context states..."},
    ],
    "sources": [...],
    "latency": 4.56,
}
```

**Steps:**
1. Generate initial answer (1 LLM call)
2. Generate verification questions from the answer (1 LLM call)
3. Answer each verification question against context (N LLM calls)
4. Generate refined answer using initial answer + verification results (1 LLM call)

Total: 3 + N LLM calls. Use when answer accuracy is critical.

## Attributed

Generates answers with inline `[N]` citations referencing numbered source documents.

```yaml
generation:
  strategy: "attributed"
attributed_generation:
  citation_style: "numeric"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `citation_style` | `str` | `"numeric"` | Citation format |

**Output:**

```python
{
    "answer": "RAG combines retrieval with generation [1]. It uses vector databases [2].",
    "citations": [
        {
            "citation_number": 1,
            "document_id": "abc-123",
            "source": "intro.txt",
            "content_preview": "RAG combines retrieval...",
        },
        {
            "citation_number": 2,
            "document_id": "def-456",
            "source": "overview.md",
            "content_preview": "Vector databases store...",
        },
    ],
    "sources": [...],
    "latency": 1.89,
}
```

The system prompt presents documents as numbered sources and instructs the LLM to use `[N]` notation. Citations are parsed from the response and matched back to source documents.

1 LLM call per query. Use when traceability to source documents is important.

## Comparison

| Strategy | LLM Calls | Accuracy | Citations | Latency |
|----------|-----------|----------|-----------|---------|
| Standard | 1 | Good | No | Low |
| CoVe | 3 + N | Highest | No | High |
| Attributed | 1 | Good | Yes | Low |

## See Also

- [Retrieval Strategies](25-retrieval-strategies.md) — what feeds into generation
- [Reranking](26-reranking.md) — improving retrieval quality
- [API: RAG](30-api-rag.md) — `RAG.query()` return format

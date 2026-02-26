# Phase 1 — Baseline

## Purpose

Establish a reference point. Every subsequent phase compares its numbers against
Phase 1. The configuration is intentionally the simplest possible RAG pipeline:
no fancy chunking, no multi-hop retrieval, no self-verification.

---

## Configuration

| Component | Choice | Reason for default |
|-----------|--------|-------------------|
| Chunking | TextSplitter (chunk_size=512, overlap=50) | Most widely used, well-understood |
| Retrieval | Dense (cosine similarity) | The standard starting point for all RAG systems |
| Reranking | None | Adds complexity; measure its value in Phase 4 |
| Generation | Standard | Simple context concatenation, no self-reflection |
| VectorStore | InMemoryVectorStore | Zero overhead, fresh per question |
| Embedding | Local REST API | Project's default embedding server |
| LLM | Local Ollama (gpt-oss:20b) | Project's default generation model |

---

## Dataset

**HotpotQA distractor dev set** — 100 questions.

File downloaded to: `research/data/hotpotqa_dev_distractor.json` (gitignored, ~54 MB).

### What one sample looks like

```
question: "Were Scott Derrickson and Ed Wood of the same nationality?"
answer:   "yes"
type:     "comparison"
level:    "easy"

context (10 Wikipedia paragraphs):
  [GOLD] "Scott Derrickson"  →  "Scott Derrickson (born July 16, 1966) is an American director..."
  [GOLD] "Ed Wood"           →  "Ed Wood (born October 10, 1924) was an American filmmaker..."
  [DIS]  "Marvel Cinematic Universe"
  [DIS]  "Horror film"
  [DIS]  "American film"
  [DIS]  ... 5 more distractors ...

supporting_facts:
  ("Scott Derrickson", 0)   ← sentence 0 of that paragraph contains the evidence
  ("Ed Wood", 0)
```

**Bridge questions** need two articles that chain (A → B → answer).
**Comparison questions** need two articles to compare an attribute across them.

---

## Ingestion Trace (Per Question)

```
Step 1: rag.clear_index()
    → InMemoryVectorStore wiped. Fresh start.

Step 2: sample.to_documents()  produces 10 Document objects

    Document(
        content  = "Scott Derrickson\n\nScott Derrickson (born July 16, 1966) ..."
        metadata = {
            "source":        "Scott Derrickson",  ← Wikipedia article title
            "question_id":   "5a8b57f25...",
            "is_supporting": True                 ← gold label (not used by retriever)
        }
    )
    ... × 10

Step 3: rag.ingest_documents(10 docs)

    a) TextSplitter(512, 50) splits each Document into chunks
       - A typical HotpotQA paragraph is 150-400 words → usually 1 chunk
       - Longer articles may produce 2-3 chunks
       - Each chunk inherits all metadata from its parent Document
       Result: 10 documents → approximately 15-25 chunks

    b) embedding_provider.embed_documents(all_chunk_texts)
       → one vector per chunk

    c) vector_store.add_documents(chunks, vectors)
       → stored in InMemoryVectorStore

Final state: ~20 vectors in memory, each tagged with which Wikipedia article it came from.
```

---

## Query Trace (Per Question)

```
Step 4: rag.query("Were Scott Derrickson and Ed Wood of the same nationality?")

    a) embedding_provider.embed_query(question)  →  query_vector

    b) vector_store.search(query_vector, top_k=5)
       Returns top-5 chunks by cosine similarity

    c) StandardGeneration.generate(question, top5_chunks)
       Prompt: "Answer based on the context: [chunk1][chunk2]...[chunk5]
                Question: Were Scott Derrickson and Ed Wood of the same nationality?"
       LLM response: "yes"

    d) result = {
           "answer":  "yes",
           "sources": [chunk1, chunk2, chunk3, chunk4, chunk5],
           "latency": 1.23
       }
```

---

## Evaluation Trace (Per Question)

```
retrieved_sources = [chunk.metadata["source"] for chunk in result["sources"]]
                  = ["Scott Derrickson", "Ed Wood", "Marvel Cinematic Universe",
                     "Horror film", "American film"]

gold_titles = {"Scott Derrickson", "Ed Wood"}

context_recall    = |{"Scott Derrickson","Ed Wood"} ∩ retrieved| / 2
                  = 2/2 = 1.0

context_precision = |retrieved ∩ gold| / |retrieved|
                  = 2/5 = 0.4

exact_match = normalize("yes") == normalize("yes") = 1.0

f1 = token_F1("yes", "yes") = 1.0
```

**Normalize** strips articles (a/an/the), punctuation, lowercases, collapses whitespace.
This prevents "the United States" vs "United States" from being counted as wrong.

---

## What the Metrics Tell Us

| Metric | Good score means | Bad score means |
|--------|-----------------|-----------------|
| `context_recall` | We retrieved both gold articles | We missed evidence; LLM won't have what it needs |
| `context_precision` | Retrieved docs are mostly useful | We retrieved noisy distractors; LLM gets confused |
| `exact_match` | Answer matches gold exactly | LLM answered wrong or in different phrasing |
| `f1` | Substantial token overlap with gold | Partial credit; answer is near-correct |

For HotpotQA, **context_recall is the most critical metric for retrieval evaluation**.
If we don't retrieve both gold articles, the LLM literally cannot answer multi-hop
questions correctly — it lacks the evidence. Context precision tells us how much noise
we're feeding the LLM along with the gold evidence.

---

## Expected Baseline Behaviour

Dense retrieval on HotpotQA distractor is known to be challenging because:

1. **Distractors are topically related** — not random. "Marvel Cinematic Universe"
   is a plausible distractor for a Scott Derrickson question. A keyword-based system
   would likely retrieve it over one of the gold articles.

2. **Multi-hop questions require both articles** — dense retrieval finds the most
   similar document, which is often just the one more directly mentioned in the
   question. The second, connected article is harder to retrieve from the question
   alone.

3. **Chunk granularity matters** — if a chunk boundary splits a supporting sentence,
   the evidence is split across two chunks and may not land in the top-5.

Typical literature numbers for dense retrieval on HotpotQA:
- Context Recall @ 5: ~0.60–0.75
- Context Precision @ 5: ~0.30–0.50
- F1: ~0.45–0.60

These are the numbers Phase 1 should produce. If our numbers are significantly lower,
it points to an embedding quality issue. If they are significantly higher, we may be
running on an easier subset.

---

## How to Run

```bash
cd /path/to/rag_sdk
.venv/bin/python research/phase1_baseline/run.py
```

Results saved to: `research/results/phase1_baseline.json`

---

## What Phase 2 Will Change

Phase 1 uses TextSplitter with fixed 512-token chunks. Phase 2 will test whether
smarter chunking strategies improve the numbers. The hypothesis: PropositionSplitter
(atomic facts) or SemanticSplitter (topic boundaries) should improve context_precision
because they cut at more meaningful boundaries than a fixed character count.

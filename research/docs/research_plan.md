# RAG SDK Research Evaluation Plan

## Goal

Systematically evaluate every component of the RAG SDK using controlled ablation
studies. One variable changes per phase; everything else is held constant. This lets
us measure the isolated contribution of each component — chunking, retrieval,
reranking, generation — and ultimately identify the best configuration.

---

## Dataset: HotpotQA Distractor Dev Set

**Why HotpotQA?**

HotpotQA is a multi-hop QA dataset built from Wikipedia. Each question requires
connecting information from two different Wikipedia articles to answer. This is
important because:
- Simple factoid questions are too easy — almost any retriever handles them
- Multi-hop questions stress-test retrieval in ways that reveal real differences
- It is the standard benchmark used in the original RAG, Self-RAG, and DPR papers

**Which split?**

We use the *distractor* dev split. Each question ships with exactly 10 pre-selected
Wikipedia paragraphs: 2 gold (the actual evidence) + 8 topically-related distractors.
This makes it self-contained (~54 MB) — no need to download all of Wikipedia.

**Raw data structure (one record):**

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison",
  "level": "easy",
  "context": [
    ["Scott Derrickson", ["Scott Derrickson (born July 16, 1966) is an American director.", "..."]],
    ["Ed Wood",          ["Ed Wood (born October 10, 1924) was an American filmmaker.", "..."]],
    ["Marvel Cinematic Universe", ["...(distractor)..."]],
    ... 7 more distractor paragraphs ...
  ],
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ]
}
```

**Question types:**
- `bridge` — answer requires chaining two articles (e.g., "Who directed the film starring X?")
- `comparison` — answer requires comparing an attribute across two articles

**Difficulty levels:** easy, medium, hard

**Do the 10 documents repeat across questions?**

Yes — popular Wikipedia articles (e.g. "United States") may appear as distractors in
multiple questions. This does not affect evaluation because we `clear_index()` between
every question and each question runs in complete isolation.

---

## Ingestion Pipeline (Per Question)

For every HotpotQA sample, the experiment harness does this:

```
1. rag.clear_index()
   → Wipes the InMemoryVectorStore. Zero state from previous question.

2. sample.to_documents()  →  10 Document objects
   Each paragraph becomes one Document:

   Document(
       content  = "Scott Derrickson\n\nScott Derrickson (born July 16, 1966) ..."
       metadata = {
           "source":       "Scott Derrickson",   ← Wikipedia article title
           "question_id":  "5a8b57f25...",        ← links chunk to its question
           "is_supporting": True                  ← gold evidence label (eval only)
       }
   )

3. rag.ingest_documents(docs)
   → TextSplitter splits each Document into chunks (512 tokens, 50 overlap)
   → Each chunk INHERITS metadata from its parent Document
   → All chunks are embedded and stored in InMemoryVectorStore

   10 documents  →  ~15-25 chunks  →  ~15-25 vectors
```

**Important:** `is_supporting` in the metadata is a label we set for evaluation
purposes only. The RAG pipeline (retriever, generator) is completely blind to it.
The retriever just finds the most semantically similar chunks. We read this metadata
only after the fact to compute metrics.

---

## Evaluation (Per Question)

```
1. rag.query(question)
   → Embeds the question
   → Cosine similarity search: returns top-5 CHUNKS
   → StandardGeneration: builds context from chunks, calls LLM, returns answer

2. Extract retrieved_sources = [chunk.metadata["source"] for chunk in top5]
   → e.g. ["Scott Derrickson", "Ed Wood", "Marvel Cinematic Universe", ...]
   → Note: retrieval is at CHUNK level, evaluation is at ARTICLE (title) level

3. Compute metrics against HotpotQA ground truth:
```

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| `context_recall` | \|gold titles ∩ retrieved titles\| / \|gold titles\| | Did we retrieve ALL the evidence we needed? |
| `context_precision` | \|retrieved titles that are gold\| / \|total retrieved\| | Are retrieved docs actually useful, or noisy? |
| `exact_match` | normalize(predicted) == normalize(gold) | Binary correctness |
| `f1` | token-level F1(predicted, gold) | Partial credit for near-correct answers |
| `avg_latency` | mean wall-clock seconds per query | System speed |

**Example calculation for the Scott Derrickson question:**

```
gold titles:       {"Scott Derrickson", "Ed Wood"}
retrieved titles:  ["Scott Derrickson", "Ed Wood", "Marvel Cinematic Universe", "Horror film", "American film"]

context_recall    = 2/2 = 1.0   (retrieved both gold docs — perfect)
context_precision = 2/5 = 0.4   (3 of 5 retrieved were noise)
exact_match       = 1.0         ("yes" == "yes")
f1                = 1.0
```

---

## Phase Plan

| Phase | Variable changed | Everything else | Key research question |
|-------|-----------------|-----------------|----------------------|
| **1** | — (baseline) | TextSplitter + dense + standard | What is the reference point? |
| **2** | Chunking strategy | Dense retrieval + standard gen | Does chunking boundary affect retrieval quality? |
| **3** | Retrieval strategy | Best chunker + standard gen | Which retriever finds the right docs on multi-hop questions? |
| **4** | Reranking | Best chunker + best retriever | Does reranking recover precision without hurting recall? |
| **5** | Generation strategy | Best chunker + best retriever | Does CoVe reduce hallucination? Does attribution add verifiability? |
| **6** | Best-of-breed | All winners combined | What is the optimal full pipeline? |
| **7** | Corpus (PDF) | Best combo | Does everything hold on real domain documents? |

---

## Phase 2: Chunking Ablation (Built)

**What changes:** The splitter. Everything else (dense retrieval, standard generation) stays as Phase 1.

**Why chunking matters:** The chunk is the unit of retrieval. If a chunk boundary cuts
a key fact in half, retrieval will fail even with a perfect retriever. Different
chunking strategies make very different decisions about where to cut.

| Variant | Splitter | Key hypothesis |
|---------|----------|---------------|
| 2a | TextSplitter(512, 50) | Baseline — same as Phase 1 |
| 2b | TextSplitter(256, 25) | Smaller chunks → higher precision, lower recall |
| 2c | TextSplitter(1024, 100) | Larger chunks → higher recall, lower precision |
| 2d | SemanticSplitter | Cuts at topic boundaries → better precision |
| 2e | AgenticSplitter | LLM detects semantic sections → most nuanced cuts |
| 2f | PropositionSplitter | Decomposes to atomic facts → high precision for factoids |
| 2g | LateSplitter | Context-aware token embeddings → better ranking without boundary changes |

**Expected learning:** PropositionSplitter likely wins on precision (factoid answers),
SemanticSplitter likely wins on recall (topic-coherent chunks). LateSplitter is the
wildcard — it changes embeddings, not chunk content.

---

## Phase 3: Retrieval Strategy Ablation (Built)

**What changes:** The retriever. Uses best chunker from Phase 2.

| Variant | Retriever | What it tests |
|---------|-----------|--------------|
| 3a | Dense | Baseline semantic search |
| 3b | BM25 | Can keywords alone beat semantics on Wikipedia? |
| 3c | Hybrid (dense + BM25) | Does combining always help? |
| 3d | MultiQuery | Does query expansion improve recall? |
| 3e | SelfRAG | Does adaptive retrieval save tokens at no quality cost? |
| 3f | ContextualCompression | Does compressing retrieved context improve faithfulness? |
| 3g | CorrectiveRAG | Does iterative query refinement improve precision? |
| 3h | BasicGraphRAG | Does structural knowledge help on multi-hop? |
| 3i | AdvancedGraphRAG (local) | Microsoft-style entity-neighbourhood search |
| 3j | AdvancedGraphRAG (global) | Community summary search |
| 3k | AdvancedGraphRAG (drift) | Exploratory HyDE-based search |
| 3l | RAPTOR | Does hierarchy help for questions needing an overview? |

**Expected learning:** Graph-based and multi-query retrievers should outperform dense
on HotpotQA specifically because HotpotQA is designed for multi-hop reasoning. If they
don't, that is itself an important finding — it means the overhead isn't justified.

---

## Phase 4: Reranking Ablation (Built)

**What changes:** Whether and how retrieved chunks are reranked before generation.
Runs on top of the best retriever from Phase 3 (over-fetch 3× then rerank to top-5).

| Variant | Reranker |
|---------|---------|
| 4a | None (ordering from retriever) |
| 4b | CrossEncoderReranker (local, joint query-doc encoding) |
| 4c | CohereReranker (API-based) |

**Expected learning:** Reranking should improve precision (noisy docs get demoted)
but may not help recall (already-retrieved gold docs stay in the pool). The question
is whether precision improvement is worth the added latency.

---

## Phase 5: Generation Strategy Ablation (Planned)

**What changes:** How the answer is generated from the retrieved context.
Uses best chunker + best retriever from prior phases.

| Variant | Strategy | What it tests |
|---------|----------|--------------|
| 5a | Standard | Baseline: context → LLM → answer |
| 5b | ChainOfVerification (CoVe) | Generate → verify claims → refine. Tests hallucination reduction |
| 5c | Attributed | Answer with inline citations [N]. Tests verifiability |

**Expected learning:** CoVe should improve faithfulness at the cost of latency (3
extra LLM calls). Attributed generation trades answer fluency for traceability. On
HotpotQA exact_match, Standard may still win because it is less conservative.

---

## Phase 6: Best-of-Breed Combination (Planned)

Take the top performer from each phase and combine:

```
Chunking:   winner from Phase 2
Retrieval:  winner from Phase 3
Reranking:  winner from Phase 4 (if it helps)
Generation: winner from Phase 5
```

Compare against Phase 1 baseline. The delta is the total measurable improvement
from systematic RAG optimization.

---

## Phase 7: PDF Corpus — Synthetic QA (Planned)

**Purpose:** Validate that the Phase 6 best-of-breed configuration works on real
domain documents, not just Wikipedia.

**Approach:**
1. Pick a domain-specific PDF corpus (e.g., the research papers in `research_papers/`)
2. Generate synthetic Q&A pairs from those documents using the LLM
3. Run the Phase 6 best configuration against those Q&A pairs
4. Compare quality against HotpotQA numbers

**Why this matters:** Wikipedia paragraphs are clean, encyclopaedic, and well-structured.
Real documents (PDFs, reports, papers) have noise, tables, references, and complex
layouts. This phase tests whether our gains generalise.

---

## Build Status

| Phase | Status | Script | Result file |
|-------|--------|--------|-------------|
| 1 | ✅ Built + pushed | `research/phase1_baseline/run.py` | `research/results/phase1_baseline.json` |
| 2 | ✅ Built + pushed | `research/phase2_chunking/run.py` | `research/results/phase2_chunking_<id>.json` |
| 3 | ✅ Built + pushed | `research/phase3_retrieval/run.py` | `research/results/phase3_retrieval_<id>.json` |
| 4 | ✅ Built + pushed | `research/phase4_reranking/run.py` | `research/results/phase4_reranking_<id>.json` |
| 5 | ⬜ Planned | `research/phase5_generation/run.py` | — |
| 6 | ⬜ Planned | `research/phase6_best_combo/run.py` | — |
| 7 | ⬜ Planned | `research/phase7_pdf_corpus/run.py` | — |

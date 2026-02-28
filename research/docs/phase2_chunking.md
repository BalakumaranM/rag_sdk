# Phase 2 — Chunking Strategy Ablation

## Purpose

Find out how much the choice of text splitter affects retrieval quality. Everything
else is frozen: dense retrieval, standard generation, same embedding model, same LLM.
Only the splitter changes.

The core question: **do smarter chunk boundaries produce better retrieval?**

---

## Why Chunking Matters

The chunk is the unit of retrieval. The vector store stores one embedding per chunk.
When a query comes in, the retriever finds the top-k most similar *chunks* — not documents.

This means two problems can arise:

1. **Boundary problem**: A key fact is split across two chunks. Each half is
   semantically weaker, so neither chunk ranks high enough to be retrieved.

2. **Noise problem**: A chunk is too large and contains several topics. Its embedding
   is a semantic average — similar to many queries but not highly similar to any one.
   It ranks in the middle everywhere and wins nowhere.

Different splitters make very different boundary decisions. This phase measures
which boundary strategy works best on HotpotQA.

---

## Variants

### 2a — TextSplitter(512, 50) · Baseline

Splits by character count. Tries separators in order `[\n\n, \n, ., !, ?, ,, " "]`
and falls back to hard splitting at 512 characters with 50-character overlap.

- **Overlap** means the last 50 characters of chunk N are repeated at the start of
  chunk N+1. This ensures facts near boundaries appear in two chunks.
- Same configuration as Phase 1 — this is the reference point for Phase 2.

Expected: moderate recall, moderate precision.

---

### 2b — TextSplitter(256, 25) · Smaller chunks

Same algorithm, half the chunk size.

- More chunks → finer granularity → each chunk is more topically focused
- The embedding for a small chunk represents a narrow idea → higher precision
- But the gold fact might be spread across multiple small chunks → lower recall

Expected: **higher precision, lower recall** vs 2a.

---

### 2c — TextSplitter(1024, 100) · Larger chunks

Same algorithm, double the chunk size.

- Fewer chunks → broader coverage per chunk → less likely to miss a gold fact
- But each chunk contains more topics → its embedding is diluted → lower precision
- Also: top-k=5 chunks retrieved, but 5 large chunks may cover most of the corpus

Expected: **higher recall, lower precision** vs 2a.

---

### 2d — SemanticSplitter · Topic-aware boundaries

Source: `rag_sdk/document/semantic_splitter.py`

Algorithm (Greg Kamradt approach):
1. Split text into sentences
2. Embed each sentence using our LocalAPIEmbedding
3. Compute cosine similarity between every pair of consecutive sentences
4. Identify "breakpoints" — positions where similarity drops below the
   25th percentile (configured via `breakpoint_percentile`)
5. Cut at breakpoints

**Why it's better in theory**: Boundaries fall where topics actually change, not
where a character counter runs out. A chunk about "Scott Derrickson's nationality"
won't be mixed with a chunk about "his directing style."

**Cost vs 2a**: One embedding call per sentence during ingestion (slow for large docs,
fine for HotpotQA's short paragraphs).

Expected: **higher precision** than fixed-size chunking (cleaner topic boundaries).
Recall depends on whether the semantic boundary captures the exact gold sentence.

---

### 2e — AgenticSplitter · LLM-detected boundaries  `[SLOW]`

Source: `rag_sdk/document/agentic_splitter.py`

Algorithm:
1. Feed the full document text to the LLM
2. Ask it: "identify the sentence indices where major topic shifts occur"
3. Split at those LLM-identified positions
4. Falls back to fixed-size splitting if LLM returns unparseable output

**Why it's better in theory**: An LLM understands *meaning*, not just embedding
similarity. It can detect a topic shift even when consecutive sentences are
lexically similar.

**Cost**: One LLM call per document during ingestion.

For 100 HotpotQA questions × 10 docs = **1,000 LLM calls just for chunking**,
plus 100 more for generation. At 2s per call → ~35 minutes total.

Run with `--all` or `--variants 2e`.

Expected: **highest precision** of all variants if LLM boundary detection is accurate.
But may suffer from LLM inconsistency on short Wikipedia paragraphs (the LLM may find
no clear topic shift in a 200-word passage).

---

### 2f — PropositionSplitter · Atomic facts  `[SLOW]`

Source: `rag_sdk/document/proposition_splitter.py`

Algorithm:
1. Feed the document to the LLM
2. Ask it to decompose the text into atomic, self-contained propositions:
   `["Scott Derrickson is American.", "Scott Derrickson was born in 1966.", ...]`
3. Group propositions into chunks of at most N propositions each (default: 5)

**Why it's better in theory**: Each chunk is a small set of complete, standalone facts.
No dangling context, no ambiguous pronouns, no topic mixing. Perfect for factoid questions.

**Cost**: Same as AgenticSplitter — one LLM call per document. Same time estimate.

Run with `--all` or `--variants 2f`.

Expected: **highest F1** on factoid questions. HotpotQA answers are usually 1–3 word
facts, and proposition chunks are precisely structured for this. However, the propositions
may lose the connecting narrative that helps the retriever understand relationships.

---

### 2g — LateSplitter · Contextual token embeddings  `[needs: transformers, torch]`

Source: `rag_sdk/document/late_splitter.py`

Algorithm (Jina AI's "Late Chunking"):
1. Feed the full document to a local Jina transformer (jinaai/jina-embeddings-v2-base-en)
2. Extract **token-level** embeddings — every token gets a contextual representation
   that reflects the whole document (because transformers are bidirectional)
3. Split text into sentence-based chunks
4. Each chunk's embedding = mean-pool of its token embeddings

**Why it's better in theory**: Traditional chunking embeds each chunk in isolation.
A chunk mentioning "he" near the beginning has no idea "he" refers to Scott Derrickson
(mentioned earlier in the document). Late chunking gives every token a contextual
embedding, so "he" is represented knowing the full document context.

**Important caveat for our harness:**

Our `rag.ingest_documents()` pipeline:
```
LateSplitter.split_documents()  →  chunks with late_embedding in metadata
                                         ↓
embedding_provider.embed_documents(chunk_texts)   ← LocalAPIEmbedding
                                         ↓
vector_store.add_documents(chunks, embeddings)    ← uses LocalAPIEmbedding vectors
```

The `late_embedding` stored in metadata is **ignored** — the vector store uses the
LocalAPIEmbedding vectors instead. In our harness, LateSplitter is tested only for
its **chunking boundary decisions** (sentence-based splits), not for its contextual
embeddings, which are the actual innovation.

To truly test late chunking, the ingest pipeline would need to read the `late_embedding`
from metadata and use those as the stored vectors. This is a future SDK enhancement.

Run with `--variants 2g` if `transformers` and `torch` are installed.

---

## How to Read the Results

```
  Variant               Recall  Prec    EM      F1    Latency
  ──────────────────────────────────────────────────────────
  2a TextSplitter(512)  0.712   0.423   0.312   0.487  2.34s  ← baseline
  2b TextSplitter(256)  0.698   0.451   0.298   0.471  2.21s  ← ↑prec ↓recall
  2c TextSplitter(1024) 0.731   0.389   0.321   0.498  2.45s  ← ↑recall ↓prec
  2d SemanticSplitter   0.743   0.467   0.334   0.512  2.89s  ← both ↑?
  2e AgenticSplitter    0.751   0.489   0.341   0.523  8.21s  ← slow
  2f PropositionSplitter 0.769  0.512   0.356   0.541  12.4s  ← slow
```

(Numbers above are illustrative hypotheses, not actual results.)

**What to look for:**

| Observation | Interpretation |
|------------|---------------|
| 2c (large) recall > 2b (small) recall | Confirms boundary problem — facts weren't being captured in small chunks |
| 2d (semantic) precision > 2c (large) | Topic-aware cuts reduce noise better than just making chunks bigger |
| 2f (proposition) F1 > all others | Atomic facts align well with HotpotQA's short-answer format |
| 2e or 2f not much better than 2d | LLM chunking overhead not justified for this corpus type |
| LateSplitter (2g) similar to TextSplitter | Expected — our harness only tests its boundaries, not its contextual embeddings |

**The winner from Phase 2 becomes the fixed chunking strategy for Phase 3.**

---

## How to Run

```bash
cd /path/to/rag_sdk

# Fast variants only (2a–2d), ~30–60 minutes depending on LLM speed:
.venv/bin/python research/phase2_chunking/run.py

# All variants including slow (add 2e, 2f — each adds ~30–60 more minutes):
.venv/bin/python research/phase2_chunking/run.py --all

# Specific variants:
.venv/bin/python research/phase2_chunking/run.py --variants 2a,2d,2f

# Re-run a specific variant (ignores cached result):
.venv/bin/python research/phase2_chunking/run.py --variants 2d --force
```

Results saved to: `research/results/phase2_chunking_<id>.json`

Partial runs are safe — already-completed variants are loaded from cache. You can
run `--variants 2a,2b,2c,2d` today and add `--variants 2e,2f` tomorrow.

---

## What Phase 3 Will Change

Phase 3 fixes the best chunking strategy from Phase 2 and varies the retrieval
strategy across dense, BM25, hybrid, multi-query, self-RAG, corrective RAG, basic
graph RAG, advanced graph RAG (local/global/drift), and RAPTOR. Phase 3 will use
**shared_corpus mode** for more realistic retrieval difficulty.

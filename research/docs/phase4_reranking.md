# Phase 4 — Reranking Ablation

## Purpose

Find out whether reranking the retriever's output improves answer quality. Everything
else is frozen: TextSplitter(512, 50) chunking, the Phase 3 winning retrieval strategy,
standard generation. Only the reranker changes.

The core question: **does reranking the retrieved chunks before generation help?**

---

## How Reranking Works in This SDK

The retrieval pipeline has two stages when a reranker is configured:

```
Query
  │
  ▼
Retriever.retrieve(query, top_k = 5 × 3 = 15)   ← over-fetch 3×
  │
  ▼
Reranker.rerank(query, 15 docs, top_k=5)          ← trim to 5
  │
  ▼
GenerationStrategy.generate(query, 5 docs)
```

`rag.query()` automatically sets `fetch_k = top_k * 3` whenever a reranker is
configured. This gives the reranker more candidates to choose from — without
over-fetching, the reranker can only reorder the same 5 docs the retriever
would have returned anyway, which provides no benefit.

**Why over-fetch matters**: The retriever may rank a highly relevant document 6th
or 7th because its embedding is not quite as close as some noise documents. The
reranker (which uses a more powerful scoring model) can surface it from position 7
back to position 1 — but only if it was fetched in the first place.

---

## Retrieval vs Reranking — Two Different Ranking Signals

| | Retriever | Reranker |
|--|----------|---------|
| Signal | Embedding cosine similarity (bi-encoder) | Cross-encoder joint query+doc score |
| Speed | Fast — one embedding per doc, precomputed | Slow — one inference pass per (query, doc) pair |
| Quality | Approximate — embedding is a bag of meaning | Precise — attends to exact word overlap and position |
| Scales to | Millions of docs (index once) | Dozens of docs (score at query time) |

The two-stage pattern (fast retriever → precise reranker) is the standard in
production retrieval systems (e.g. ColBERT, MS MARCO pipelines).

---

## Variants

### 4a — No Reranking · Baseline

The retriever's ordering is used directly. The top-5 chunks by embedding similarity
go straight to generation.

This is the same as Phase 3's winning retrieval strategy but evaluated fresh here
as the Phase 4 reference point.

---

### 4b — CrossEncoderReranker · Local Model

Source: `rag_sdk/reranking/cross_encoder.py`

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)

A cross-encoder takes a (query, document) pair as a single input and produces a
single relevance score. Unlike bi-encoders (used by the retriever), which encode
query and document independently and compare their embeddings, a cross-encoder
attends to both sequences simultaneously.

**Why it's better**: The cross-encoder can detect that "American" in the document
directly answers "nationality" in the question — a signal that embedding similarity
misses because "American" is common and doesn't shift the embedding much. The
cross-encoder sees the exact token-level relationship.

**Cost**: One model inference per (query, doc) pair. With 15 fetched docs per
query and 100 questions: 15 × 100 = 1,500 inference passes. On CPU, each pass
takes ~10–50ms → 15–75 seconds total. Fast.

**Dependency**: `sentence-transformers` (`pip install sentence-transformers`). The
model (~50MB) is downloaded automatically on first use.

Expected: **higher context_precision** (irrelevant chunks ranked down); **recall
unchanged or slightly lower** (same pool of 15 fetched, just reordered). EM and F1
should improve if the reranker surfaces the right chunks above the noise.

---

### 4c — CohereReranker · API-based

Source: `rag_sdk/reranking/cohere_reranker.py`

Model: `rerank-v3.5` (default)

Cohere's Rerank API is a managed cross-encoder service. You send the query and
documents; Cohere returns relevance scores without you needing to host the model.

**Why it might outperform CrossEncoder**: Cohere's `rerank-v3.5` is a much larger
model than `ms-marco-MiniLM-L-6-v2`. It has been trained on a broader multilingual
corpus and incorporates semantic understanding beyond simple MS MARCO patterns.

**Cost**: 1 API call per query × 100 queries. Each call includes 15 documents.
Latency: ~100–300ms per query (network + inference). Requires `COHERE_API_KEY`.

**Dependency**: `cohere` package + `COHERE_API_KEY` environment variable.

Expected: **equal or better precision than CrossEncoder**, especially on nuanced
queries where the smaller local model may score incorrectly. Latency is higher
than CrossEncoder if Cohere servers are slow.

---

## How to Read the Results

```
  Variant                Recall  Prec    EM      F1    Latency
  ─────────────────────────────────────────────────────────────
  No Reranking    (4a)   0.643   0.389   0.281   0.451  2.34s  ← baseline
  CrossEncoder    (4b)   0.643   0.451   0.312   0.478  3.21s  ← ↑prec ↑F1
  CohereReranker  (4c)   0.643   0.467   0.321   0.487  4.12s  ← ↑↑prec ↑↑F1
```

(Numbers above are illustrative hypotheses, not actual results.)

**What to look for:**

| Observation | Interpretation |
|------------|----------------|
| Recall unchanged across 4a/4b/4c | Expected — reranking doesn't change which 15 docs were fetched, only how they're ordered |
| Precision 4b > 4a | CrossEncoder correctly identifies and promotes relevant docs |
| Precision 4c > 4b | Larger Cohere model has better relevance signal than tiny cross-encoder |
| F1 increase matches precision increase | Higher precision = cleaner context = more focused generation |
| Latency increase is small (< 2s) | Reranking overhead is modest; worth it if F1 improves |
| Precision flat (4b ≈ 4a) | The retriever's ordering is already good; reranking adds no signal |

**The winner from Phase 4 becomes the fixed reranking choice for Phase 5.**

---

## How to Run

```bash
cd /path/to/rag_sdk

# All 3 variants (all fast, no --all flag needed):
.venv/bin/python research/phase4_reranking/run.py

# Specific variants:
.venv/bin/python research/phase4_reranking/run.py --variants 4a,4b

# Re-run a specific variant:
.venv/bin/python research/phase4_reranking/run.py --variants 4b --force
```

Results saved to: `research/results/phase4_reranking_<id>.json`

**Before running:**

1. Update `RETRIEVAL_STRATEGY` in `run.py` to the Phase 3 winner (e.g. `"hybrid"`).
2. For 4b: `pip install sentence-transformers`
3. For 4c: `export COHERE_API_KEY=your_key_here`

---

## What Phase 5 Will Change

Phase 5 fixes the best chunking + retrieval + reranking from Phases 2–4 and varies
the generation strategy: standard vs ChainOfVerification (CoVe) vs attributed.

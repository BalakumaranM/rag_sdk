# Phase 3 — Retrieval Strategy Ablation

## Purpose

Find out which retrieval strategy works best on multi-hop questions. Everything
else is frozen: TextSplitter(512, 50) chunking, standard generation, same embedding
model, same LLM. Only the retriever changes.

The core question: **does a smarter retrieval strategy find the right documents?**

---

## Why Retrieval Is the Central Challenge

In Phase 1 and 2, every question had exactly 10 documents to search through (~20
chunks). Almost any strategy can find 2 relevant docs among 20 candidates.

Phase 3 uses **shared_corpus mode** — all 100 questions' Wikipedia articles are
pooled, deduplicated by title, and ingested once. The retriever now faces ~200
unique Wikipedia articles (~600–1000 chunks). It must find the 2 gold articles
among 200 candidates. This is much closer to a production retrieval problem.

HotpotQA is designed for **multi-hop reasoning**: answering "Were Scott Derrickson
and Ed Wood of the same nationality?" requires retrieving articles about *both*
people and connecting the information. Simple dense retrieval may miss this
because a query embedding for the full question is a semantic average — it may not
be closest to either individual article.

---

## Why Evaluation Mode Changes Here

| Phase | Mode | Corpus size | Difficulty |
|-------|------|------------|------------|
| 1–2 | per_question | ~20 chunks per question | Easy — isolation testing |
| 3+ | shared_corpus | ~600–1000 chunks total | Hard — realistic retrieval |

The shared corpus mode is also faster than per_question for Phase 3: ingestion
runs once, then all 100 queries run against the same index. For strategies like
GraphRAG that spend 5–60 minutes building a graph during ingestion, this is
critical — you would not want to rebuild the graph 100 times.

---

## Variants

### 3a — Dense · Baseline

Embeds the query and returns the top-5 chunks by cosine similarity.

This is the same retriever as Phase 1 but now operating on the shared corpus.
The expected numbers will be **lower than Phase 1** because the search space is
20× larger (200 articles vs 10).

Expected: moderate recall, moderate precision. This is the Phase 3 reference point.

---

### 3b — BM25 · Keyword Search

Uses the HybridRetriever with `bm25_weight=1.0` (dense weight = 0.0), making it
purely keyword-based. BM25 scores documents based on term frequency in the query
relative to the corpus.

**Why it might help**: HotpotQA questions often name specific people and places.
"Scott Derrickson" is a rare n-gram — keyword search will rank the article about
Scott Derrickson very highly simply because it contains those exact tokens, while
dense search might retrieve semantically similar articles about other directors.

**Why it might fail**: Multi-hop questions are phrased at a higher abstraction
level ("Were X and Y of the same nationality?"). BM25 cannot retrieve "Ed Wood"
from a question that doesn't mention Ed Wood — but dense search might, via semantic
similarity to "American filmmaker."

Expected: **higher precision** on bridge questions (where the query names the
target); **lower recall** on comparison questions (where one entity must be inferred).

---

### 3c — Hybrid (Dense + BM25) · Best of Both

Combines dense and BM25 rankings using Reciprocal Rank Fusion (RRF) with equal
weights (0.5/0.5).

RRF merges ranked lists without score normalisation:
```
rrf_score(doc) = Σ  weight_i / (k + rank_i)
```
A document ranked high by both retrievers gets a very high RRF score. A document
ranked high by only one still gets a moderate score.

**Why it's better in theory**: Dense retrieval captures semantics, BM25 captures
exact entity names. Hybrid should win on both question types.

Expected: **recall ≥ Dense, precision ≥ BM25** — the best of both worlds.

---

### 3d — MultiQuery · Query Expansion

Uses the LLM to rewrite the original query into 3 different phrasings, runs dense
retrieval for each, and returns the deduplicated union of results.

Example rewrites for "Were Scott Derrickson and Ed Wood of the same nationality?":
1. "What is the nationality of Scott Derrickson?"
2. "What country is Ed Wood from?"
3. "Are Scott Derrickson and Ed Wood both American?"

Each sub-query is more targeted than the original compound question. The union
covers what neither sub-query would have found alone.

**Cost**: 1 extra LLM call (to generate variants) + 3 dense retrieval calls per
query. Modest latency increase.

Expected: **higher recall** than Dense on multi-hop questions — the sub-queries
directly name each entity, making dense retrieval more precise for each one.

---

### 3e — SelfRAG · Adaptive Retrieval

The SelfRAG strategy uses the LLM to decide, at query time, whether retrieval is
even necessary. If the LLM determines the answer is already known (unlikely for
HotpotQA), it skips retrieval entirely. Otherwise, it retrieves and optionally
filters retrieved documents by asking the LLM which are actually relevant.

**Why it might help**: Removes irrelevant documents before generation. On HotpotQA,
the 8 distractor articles are genuinely distracting. If SelfRAG can identify and
discard them, generation gets a cleaner context.

**Cost**: 1 LLM call to decide (always) + 1 LLM call per retrieved doc to assess
relevance (5 docs = 5 calls). Significant latency increase.

Expected: **higher precision** (irrelevant docs filtered out); recall may drop if
the relevance filter is too aggressive.

---

### 3f — ContextualCompression · In-context Distillation

Wraps dense retrieval. After retrieving the top-5 chunks, passes each chunk to the
LLM and asks it to extract only the sentences relevant to the query. The LLM may
compress a 200-word chunk to a single relevant sentence.

**Why it might help**: Dense retrieval finds the right chunk, but the chunk contains
noise (other facts from the same Wikipedia paragraph). Generation's context window
is dominated by noise. Compression removes it.

**Cost**: 1 LLM call per chunk → 5 LLM calls per query for compression, plus the
generation call.

Expected: **higher context_precision** (compressed context is cleaner); **recall
unchanged** (same documents retrieved, just shorter). F1 and EM may improve because
generation reads less noise.

---

### 3g — CorrectiveRAG · Query Refinement

Wraps dense retrieval. After retrieving the top-5 chunks, the LLM scores each for
relevance. If fewer than `relevance_threshold` (70%) of the retrieved docs are
relevant, the LLM rewrites the query and retrieves again (up to 2 attempts).

**Why it might help**: On ambiguous multi-hop questions, the first dense retrieval
may return mostly distractors. CorrectiveRAG detects this automatically and refines
the query.

**Cost**: 1 LLM call for relevance scoring + potentially 1 rewrite + 1 extra
retrieval per failure.

Expected: **higher precision** when first retrieval fails; **latency varies** (cheap
when first retrieval succeeds, expensive when it doesn't).

---

### 3h — BasicGraphRAG · Entity Graph Retrieval  `[SLOW]`

Source: `rag_sdk/retrieval/graph_rag.py`

**Ingestion**: For each chunk, the LLM extracts entities and relationships:
```
chunk: "Scott Derrickson (born 1966) is an American director."
→ entities:      [Scott Derrickson, American]
→ relationships: [Scott Derrickson --nationality→ American]
```
All entities and relationships are stored in an in-memory graph.

**Query time**:
1. Extract query entities via LLM
2. Traverse the graph 1–2 hops from those entities
3. Retrieve the chunks that are connected to those graph nodes
4. Boost relevance scores based on graph connectivity

**Cost**: 1 LLM call per chunk during ingestion (~600–1000 calls for the shared
corpus). Plus 1 LLM call per query for entity extraction.

Expected: **higher recall on multi-hop questions** — if the graph correctly links
both gold articles via shared entities (e.g. "American"), traversal surfaces both.

**Caveat**: HotpotQA's Wikipedia paragraphs are short and clean, so entity extraction
is relatively reliable. But a 200-word paragraph about Scott Derrickson may yield
10+ entities — graph quality depends on extraction precision.

---

### 3i — AdvancedGraphRAG (local) · Entity Neighbourhood Search  `[SLOW, needs: networkx]`

Source: `rag_sdk/retrieval/advanced_graph_rag.py`

Microsoft-style GraphRAG with community detection. The ingestion pipeline:
1. Extract entities and relationships (same as BasicGraphRAG)
2. Run **Louvain community detection** to cluster related entities
3. Generate a **summary for each community** via LLM

At query time (local mode):
1. Embed the query, find most similar entities
2. Traverse entity neighbourhood up to `max_graph_hops` hops
3. Return: neighbourhood context + top-k dense chunks

Expected: **similar to BasicGraphRAG** on short Wikipedia paragraphs. The community
detection benefit appears more clearly on longer, interconnected document sets.

---

### 3j — AdvancedGraphRAG (global) · Community Summary Search  `[SLOW, needs: networkx]`

Same graph as 3i. At query time (global mode):
1. Score each community against the query by embedding similarity
2. For each top community, generate a partial answer from the community summary
3. Synthesise partial answers into one final answer

**This bypasses dense chunk retrieval entirely** — the answer is built from community
summaries, not from individual chunks.

Expected: **better on broad synthesis questions** ("What themes does this corpus
cover?"); potentially **worse on factoid questions** where the exact fact may not
survive summary compression. HotpotQA is mostly factoid → global mode may underperform.

---

### 3k — AdvancedGraphRAG (drift) · HyDE Iterative Search  `[SLOW, needs: networkx]`

Same graph as 3i. At query time (drift mode):
1. Use HyDE — generate a hypothetical answer document from the query
2. Use the hypothetical answer as the initial search vector
3. Retrieve documents, generate follow-up questions from gaps
4. Repeat for `drift_max_rounds` rounds

Expected: **highest recall** of the AdvancedGraphRAG modes — iterative refinement
finds information missed in the first pass. But also highest latency.

---

### 3l — RAPTOR · Hierarchical Clustering  `[SLOW]`

Source: `rag_sdk/retrieval/raptor.py`

**Ingestion**:
1. Take all leaf chunks
2. K-means cluster them at level 1 → generate an LLM summary per cluster
3. K-means cluster the summaries at level 2 → summarise again
4. Repeat for `num_levels` levels (default: 3)

This produces a **tree of summaries** — leaf nodes are original chunks, internal
nodes are cluster summaries.

**Query time**: Retrieve from all levels. If a query asks for a broad topic, the
cluster summary (which aggregates many chunks) is more relevant than any single
chunk.

**Why it helps for HotpotQA**: Multi-hop questions require connecting two entities.
A cluster summary covering "American directors" may directly mention both Scott
Derrickson and Ed Wood, even though no single leaf chunk mentions both.

Expected: **higher recall on bridge questions** where the answer requires connecting
information that spans multiple chunks; moderate precision.

---

## How to Read the Results

```
  Variant                    Recall  Prec    EM      F1    Latency
  ────────────────────────────────────────────────────────────────────
  Dense               (3a)   0.643   0.389   0.281   0.451  2.34s  ← baseline (harder corpus)
  BM25                (3b)   0.612   0.421   0.271   0.438  1.98s  ← ↑prec ↓recall
  Hybrid(dense+BM25)  (3c)   0.671   0.412   0.295   0.469  2.15s  ← ↑recall ↑prec?
  MultiQuery          (3d)   0.698   0.401   0.311   0.479  4.21s  ← ↑recall at latency cost
  SelfRAG             (3e)   0.631   0.445   0.289   0.453  6.89s  ← ↑prec ↓recall
  ContextualCompr.    (3f)   0.643   0.461   0.301   0.471  9.12s  ← ↑prec, same recall
  CorrectiveRAG       (3g)   0.658   0.432   0.294   0.462  5.43s  ← ↑prec on failures
  BasicGraphRAG       (3h)   0.712   0.412   0.318   0.489 18.4s   ← ↑recall on multi-hop
  AdvGraphRAG(local)  (3i)   0.724   0.418   0.321   0.495 24.1s   ← ...
  AdvGraphRAG(global) (3j)   0.681   0.374   0.289   0.452 41.2s   ← ↓ for factoid
  AdvGraphRAG(drift)  (3k)   0.738   0.402   0.328   0.501 52.3s   ← highest recall
  RAPTOR              (3l)   0.721   0.396   0.314   0.486 31.2s   ← ↑recall on bridge
```

(Numbers above are illustrative hypotheses, not actual results.)

**What to look for:**

| Observation | Interpretation |
|------------|----------------|
| Dense (3a) recall < Phase 1 recall | Expected — 20× harder corpus. If recall barely drops, corpus is too easy |
| BM25 (3b) recall < Dense (3a) recall | Dense beats BM25 on multi-hop — semantic similarity matters more than keywords |
| Hybrid (3c) recall > Dense (3a) | Combining both worlds helps — use Hybrid as the Phase 4 base |
| MultiQuery (3d) recall > Dense (3a) | Query expansion catches both entities in multi-hop questions |
| SelfRAG (3e) precision > Dense (3a) | Relevance filtering removes distractors — at a latency cost |
| GraphRAG (3h) recall > Dense (3a) | Entity graph helps multi-hop retrieval — structural knowledge pays off |
| AdvGraphRAG global (3j) F1 < local (3i) | Global mode hurt by summary compression of factoid answers |
| RAPTOR (3l) recall > Dense (3a) | Cluster summaries surface cross-document connections |
| No slow variant beats Hybrid | LLM overhead not justified; use Hybrid for Phase 4 |

**The winner from Phase 3 becomes the fixed retrieval strategy for Phase 4.**

---

## How to Run

```bash
cd /path/to/rag_sdk

# Fast variants only (3a–3g), ~30–90 minutes:
.venv/bin/python research/phase3_retrieval/run.py

# All variants including slow (adds 3h–3l):
.venv/bin/python research/phase3_retrieval/run.py --all

# Specific variants:
.venv/bin/python research/phase3_retrieval/run.py --variants 3a,3d,3h

# Re-run a specific variant:
.venv/bin/python research/phase3_retrieval/run.py --variants 3d --force
```

Results saved to: `research/results/phase3_retrieval_<id>.json`

Partial runs are safe — already-completed variants are loaded from cache.

---

## A Note on Chunking

This phase uses `TextSplitter(512, 50)` — the Phase 2 baseline — as the fixed
chunking strategy. Once Phase 2 results are available, you can update `_make_config`
in `run.py` to use the Phase 2 winner and re-run Phase 3 for the final numbers.

---

## What Phase 4 Will Change

Phase 4 fixes the best retrieval strategy from Phase 3 and varies the reranking
approach: no reranker vs CrossEncoderReranker (local) vs CohereReranker (API).
The retriever over-fetches 3× top-k, the reranker trims to top-k.

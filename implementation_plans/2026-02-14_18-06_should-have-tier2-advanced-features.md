# Implementation Plan: should_have.yaml Tier 2 Advanced Features

**Date:** 2026-02-14 18:06
**Status:** COMPLETED
**Branch:** main

---

## Objective

Add 7 advanced features to the RAG SDK: 2 chunking strategies, 3 retrieval strategies, and 2 generation strategies. All features follow the existing pattern of ABC base classes, Pydantic configs, and strategy selection in the orchestrator.

---

## Phase 1: Foundation Refactoring [COMPLETED]

**Goal:** Create base ABCs and extract generation into its own module.

**New files created:**
- `rag_sdk/document/base.py` — `BaseTextSplitter` ABC with `split_text` and `split_documents` abstract methods
- `rag_sdk/retrieval/base.py` — `BaseRetriever` ABC with `retrieve` abstract method
- `rag_sdk/generation/__init__.py` — New generation module package
- `rag_sdk/generation/base.py` — `GenerationStrategy` ABC with `generate(query, docs) -> dict`
- `rag_sdk/generation/standard.py` — Extracted existing inline generation logic from `core.py` into `StandardGeneration` class

**Modified files:**
- `rag_sdk/document/splitter.py` — `TextSplitter` now inherits from `BaseTextSplitter`
- `rag_sdk/retrieval/retriever.py` — `Retriever` now inherits from `BaseRetriever`
- `rag_sdk/document/__init__.py` — Added export for `BaseTextSplitter`
- `rag_sdk/retrieval/__init__.py` — Added export for `BaseRetriever`

**Design decisions:**
- ABCs use `@abstractmethod` decorators to enforce implementation
- `GenerationStrategy.generate()` returns `Dict[str, Any]` with at least an `answer` key, allowing strategies to include extra metadata (citations, verification QA, etc.)

---

## Phase 2: Config Schema Additions [COMPLETED]

**Goal:** Add all Pydantic config models for the new strategies.

**New config classes added to `rag_sdk/config/config.py`:**

| Config Class | Fields | Defaults |
|---|---|---|
| `ChunkingConfig` | `strategy` | `"recursive"` |
| `AgenticChunkingConfig` | `max_chunk_size`, `similarity_threshold` | `1000`, `0.5` |
| `PropositionChunkingConfig` | `max_propositions_per_chunk` | `5` |
| `GraphRAGConfig` | `max_entities_per_chunk`, `max_relationships_per_chunk` | `10`, `15` |
| `RAPTORConfig` | `num_levels`, `clustering_method`, `max_clusters_per_level` | `3`, `"kmeans"`, `10` |
| `CorrectiveRAGConfig` | `relevance_threshold`, `max_refinement_attempts` | `0.7`, `2` |
| `GenerationConfig` | `strategy` | `"standard"` |
| `CoVeConfig` | `max_verification_questions` | `3` |
| `AttributedGenerationConfig` | `citation_style` | `"numeric"` |

**Existing configs modified:**
- `DocumentProcessingConfig` — added `chunking`, `agentic_chunking`, `proposition_chunking` fields
- `RetrievalConfig` — added `corrective_rag_enabled`, `graph_rag`, `raptor`, `corrective_rag` fields
- `Config` — added `generation`, `cove`, `attributed_generation` fields

**Backward compatibility:** All new fields have defaults matching existing behavior (`recursive` / `dense` / `standard`).

---

## Phase 3: Chunking Features [COMPLETED]

### 3a. Agentic Splitter — `rag_sdk/document/agentic_splitter.py`

**Class:** `AgenticSplitter(BaseTextSplitter)`

**How it works:**
1. Pre-splits text into sentences using regex `(?<=[.!?])\s+`
2. Sends numbered sentences to LLM asking it to identify semantic boundary indices
3. Groups sentences between boundaries into chunks
4. Sub-splits any chunks that exceed `max_chunk_size * 1.5`
5. Falls back to simple character-based splitting if LLM fails

**Constructor params:** `llm_provider`, `max_chunk_size`, `similarity_threshold`

**Robustness:** All LLM calls wrapped in try/except. JSON parsing uses regex extraction before `json.loads`.

### 3b. Proposition Splitter — `rag_sdk/document/proposition_splitter.py`

**Class:** `PropositionSplitter(BaseTextSplitter)`

**How it works:**
1. Sends text to LLM asking it to decompose into atomic, self-contained propositions
2. Parses JSON array of proposition strings from response
3. Groups propositions into chunks of `max_propositions_per_chunk` size
4. Falls back to sentence splitting if LLM fails

**Constructor params:** `llm_provider`, `max_propositions_per_chunk`

---

## Phase 4: Retrieval Features [COMPLETED]

### 4a. Graph RAG Retriever — `rag_sdk/retrieval/graph_rag.py`

**Class:** `GraphRAGRetriever(BaseRetriever)`

**Data structures:**
- `Entity` dataclass: `name`, `entity_type`, `document_ids`
- `Relationship` dataclass: `source`, `target`, `relation`, `document_ids`
- In-memory adjacency dict for graph traversal

**How it works:**
- `build_graph(documents)` — Called during ingestion. Uses LLM to extract entities/relationships from each document chunk. Stores in in-memory graph.
- `retrieve(query)` — Extracts query entities via LLM, traverses graph (2 hops) to find relevant document IDs, runs dense retrieval, boosts scores for graph-matched documents by 1.2x.

**Fuzzy matching:** Query entities are matched against graph entities using substring containment.

### 4b. RAPTOR Retriever — `rag_sdk/retrieval/raptor.py`

**Class:** `RAPTORRetriever(BaseRetriever)`

**Helper:** `_kmeans(vectors, k, max_iter=50)` — Pure numpy k-means implementation. Deterministic with `rng seed 42`.

**How it works:**
- `build_tree(documents)` — Called during ingestion. For each level: embeds documents, clusters with k-means, generates LLM summaries per cluster, stores summaries in vector store with `raptor_level` metadata. Repeats for `num_levels`.
- `retrieve(query)` — Searches vector store with `top_k * 2`, separates leaf docs from summary docs, prioritizes leaf docs and fills remaining slots with summaries.

**No new dependencies:** k-means uses only numpy.

### 4c. Corrective RAG Retriever — `rag_sdk/retrieval/corrective_rag.py`

**Class:** `CorrectiveRAGRetriever(BaseRetriever)` — **Decorator/wrapper pattern**

**How it works:**
1. Retrieves documents using wrapped `base_retriever`
2. Evaluates relevance of each document via LLM (returns JSON array of `{index, relevant}`)
3. If fewer than `min_relevant` docs pass threshold, refines query via LLM and re-retrieves
4. Repeats up to `max_refinement_attempts` times
5. Returns relevant docs, or falls back to all docs on last attempt

**Composability:** Wraps any `BaseRetriever` — works with dense, Graph RAG, or RAPTOR. Enabled via `corrective_rag_enabled: true` in config.

---

## Phase 5: Generation Features [COMPLETED]

### 5a. Chain-of-Verification — `rag_sdk/generation/cove.py`

**Class:** `ChainOfVerificationGeneration(GenerationStrategy)`

**4-step process:**
1. **Generate initial answer** using context
2. **Generate verification questions** — LLM produces up to N questions about claims in the answer
3. **Answer each verification question independently** against the original context
4. **Generate refined answer** incorporating verification results

**Returns:** `{answer, initial_answer, verification_qa: [{question, answer}]}`

**Graceful degradation:** If no verification questions are generated (LLM failure or simple answer), returns the initial answer directly.

### 5b. Attributed Generation — `rag_sdk/generation/attributed.py`

**Class:** `AttributedGeneration(GenerationStrategy)`

**How it works:**
1. Builds numbered context: `[1] (Source: filename)\ncontent...`
2. Prompts LLM to include `[N]` inline citations in its response
3. Parses citation numbers from response using regex `\[(\d+)\]`
4. Maps citation numbers back to source documents

**Returns:** `{answer, citations: [{citation_number, document_id, source, content_preview}]}`

---

## Phase 6: Orchestrator Integration [COMPLETED]

**File:** `rag_sdk/core.py` — Major refactor

**Changes:**
1. **Reordered initialization:** embeddings → vectorstore → **LLM** → splitter → retriever → generation (LLM must come before splitter now since agentic/proposition splitters need it)
2. **Broke `_init_components` into sub-methods:**
   - `_init_embeddings()` — Same as before
   - `_init_vectorstore()` — Same as before
   - `_init_llm()` — Same as before
   - `_init_splitter()` — Selects `TextSplitter`, `AgenticSplitter`, or `PropositionSplitter` based on `chunking.strategy`
   - `_init_retriever()` — Selects `Retriever`, `GraphRAGRetriever`, or `RAPTORRetriever` based on `retrieval.strategy`, then optionally wraps with `CorrectiveRAGRetriever`
   - `_init_generation()` — Selects `StandardGeneration`, `ChainOfVerificationGeneration`, or `AttributedGeneration`
3. **`ingest_documents()` post-ingestion hooks:** After storing documents, calls `build_graph()` for GraphRAG or `build_tree()` for RAPTOR. Correctly unwraps CorrectiveRAG wrapper to find inner retriever.
4. **`query()` delegates to generation strategy:** Calls `self.generation_strategy.generate(query, retrieved_docs)` instead of inline code. Merges `sources` and `latency` into result dict.

---

## Phase 7: Config, Tests, and Verification [COMPLETED]

**config.yaml updated** with commented examples for all new options.

**pyproject.toml fixed:** Added `[tool.setuptools.packages.find]` with `include = ["rag_sdk*"]` to prevent `project_plans/` directory from breaking the build.

**Virtual environment:** Created with `uv venv .venv` and installed with `uv pip install -e ".[dev]"`.

**33 unit tests — ALL PASSING:**

| Test Category | Count | Tests |
|---|---|---|
| Existing tests | 2 | `test_dummy`, `test_rag_init` |
| Config defaults & fields | 4 | Defaults, new fields, dict construction, YAML loading |
| Base ABC inheritance | 2 | TextSplitter, Retriever inherit correctly |
| TextSplitter | 2 | Basic splitting, document splitting with metadata |
| Agentic Splitter | 4 | Mock LLM, fallback on failure, short text bypass, document splitting |
| Proposition Splitter | 3 | Mock LLM, fallback on failure, document splitting |
| Graph RAG | 2 | Build graph, retrieve with graph boost |
| RAPTOR | 3 | k-means clustering, build tree, retrieve with level priority |
| Corrective RAG | 2 | Pass-through relevant docs, query refinement loop |
| Standard Generation | 1 | Basic generation |
| CoVe Generation | 2 | Full 4-step flow, graceful degradation |
| Attributed Generation | 2 | Citation parsing, no-citation handling |
| Strategy selection | 4 | Defaults, invalid chunking/retrieval/generation errors |

---

## Files Summary

### New Files (12)

| File | Lines | Purpose |
|---|---|---|
| `rag_sdk/document/base.py` | 24 | BaseTextSplitter ABC |
| `rag_sdk/document/agentic_splitter.py` | 107 | LLM-based semantic boundary chunking |
| `rag_sdk/document/proposition_splitter.py` | 85 | LLM-based atomic proposition chunking |
| `rag_sdk/retrieval/base.py` | 17 | BaseRetriever ABC |
| `rag_sdk/retrieval/graph_rag.py` | 175 | In-memory knowledge graph + dense retrieval |
| `rag_sdk/retrieval/raptor.py` | 155 | Hierarchical clustering + LLM summaries |
| `rag_sdk/retrieval/corrective_rag.py` | 115 | Composable relevance-checking wrapper |
| `rag_sdk/generation/__init__.py` | 12 | Generation module exports |
| `rag_sdk/generation/base.py` | 18 | GenerationStrategy ABC |
| `rag_sdk/generation/standard.py` | 27 | Extracted standard generation |
| `rag_sdk/generation/cove.py` | 107 | Chain-of-Verification (4-step) |
| `rag_sdk/generation/attributed.py` | 72 | Inline citation generation |

### Modified Files (8)

| File | Changes |
|---|---|
| `rag_sdk/config/config.py` | +9 config classes, +fields on 3 existing |
| `rag_sdk/config/__init__.py` | +9 exports |
| `rag_sdk/document/splitter.py` | Inherits BaseTextSplitter |
| `rag_sdk/document/__init__.py` | +3 exports |
| `rag_sdk/retrieval/retriever.py` | Inherits BaseRetriever |
| `rag_sdk/retrieval/__init__.py` | +4 exports |
| `rag_sdk/core.py` | Full refactor into sub-methods, wired all 7 strategies |
| `config.yaml` | Added commented config sections |
| `pyproject.toml` | Fixed setuptools package discovery |

---

## Key Design Decisions

1. **Corrective RAG is a composable wrapper**, not a standalone strategy — can layer on top of dense, Graph RAG, or RAPTOR
2. **All defaults match existing behavior** — `recursive`, `dense`, `standard` — full backward compatibility
3. **No new external dependencies** — k-means implemented with numpy, graph stored in-memory with dicts
4. **LLM JSON parsing wrapped in try/except** with fallbacks for robustness
5. **Generation strategies return dicts** with at least `answer` key — allows flexible extra metadata per strategy

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

## Dataset: QASPER (added for chunking research)

**Why HotpotQA is insufficient for chunking ablation**

HotpotQA documents are pre-extracted Wikipedia paragraphs — typically 300–800 characters
each. With a 512-character chunk size, most documents fit in a single chunk.
Phase 2 (chunking ablation) produces near-identical numbers across all splitter variants
because the splitter barely fires: the "documents" are already paragraph-sized.

**Why QASPER**

QASPER (Dasigi et al. 2021, Allen AI) is a QA dataset built from NLP research papers.
Each document is a full paper — 6,000–12,000 tokens across 20–50 paragraphs.
At chunk_size=512, one paper produces 40–100 chunks. Now chunking decisions genuinely
matter: SemanticSplitter can align cuts to section boundaries, PropositionSplitter
decomposes dense Methods sections into self-contained claims, and chunk_size differences
produce measurably different retrieval pools.

| Property | HotpotQA | QASPER |
|---|---|---|
| Document type | Wikipedia paragraphs | NLP research papers |
| Avg document length | ~300–800 chars | ~8,000–12,000 tokens |
| Chunks per doc (512 chars) | 1–2 | 40–100 |
| QA ground truth | Yes (multi-hop) | Yes (expert-annotated) |
| Evidence type | Article titles | Verbatim paragraph text |
| Question types | bridge / comparison | extractive / abstractive / boolean |
| Tests chunking boundaries | Barely | Strongly |

**Which split**

QASPER dev set: 281 papers, ~1,000 expert-annotated QA pairs.
We use the first 100 non-unanswerable QA pairs for consistency with HotpotQA (100 questions).

**Data format**

```json
{
  "<paper_id>": {
    "title": "...",
    "abstract": "...",
    "full_text": [
      {"section_name": "Introduction", "paragraphs": ["para1", "para2"]},
      {"section_name": "Methods",      "paragraphs": ["para3"]}
    ],
    "qas": [
      {
        "question": "What dataset is used?",
        "question_id": "...",
        "answers": [
          {
            "type": "extractive",
            "free_form_answer": "SQuAD",
            "evidence": ["para3 verbatim text"],
            "highlighted_evidence": ["SQuAD"]
          }
        ]
      }
    ]
  }
}
```

**Answer types**

- `extractive` — answer is a span copied from the paper; `highlighted_evidence` is the span
- `abstractive` — free-form answer paraphrasing the paper; `free_form_answer` is the gold
- `boolean` — yes/no question; `free_form_answer` is "Yes" or "No"
- `unanswerable` — skipped entirely during loading

**Getting the dataset**

Download the dev JSON (inside a .tgz archive) from Allen AI's S3 bucket and place it
at `research/data/qasper_dev_v0.3.json`:

```bash
# Option A — download and extract (internet required)
curl -LO https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar -xzf qasper-train-dev-v0.3.tgz qasper-dev-v0.3.json
mv qasper-dev-v0.3.json research/data/

# Option B — auto-download on first run (if S3 is accessible)
.venv/bin/python research/phase1_baseline/run_qasper.py
# The loader downloads and extracts automatically if research/data/qasper_dev_v0.3.json
# does not exist.
```

`load_qasper()` always checks for the file first and skips the download if it exists —
so manual placement works with no code changes needed.

**Evaluation differences from HotpotQA**

| Aspect | HotpotQA | QASPER |
|---|---|---|
| Gold context identifier | Article title (string) | Verbatim paragraph text |
| context_recall | Retrieved titles ∩ gold titles | Chunk's source paragraph ∈ evidence set |
| context_precision | Retrieved titles that are gold | Retrieved chunks whose paragraph is in evidence |
| MRR | Rank of first gold-title chunk | Rank of first evidence-paragraph chunk |
| EM / F1 | vs gold answer string | vs free_form_answer (or highlighted_evidence) |
| Strata breakdowns | by_type (bridge/comparison), by_level (easy/medium/hard) | by_answer_type (extractive/abstractive/boolean) |

Each chunk stores `paragraph_text` in its metadata (the original un-split paragraph).
Evidence matching is exact: `chunk.metadata["paragraph_text"] in evidence_set`.
This works because QASPER evidence is verbatim paragraph text from `full_text`.

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
| `context_recall` | \|gold titles ∩ retrieved titles\| / \|gold titles\| | Did we retrieve ALL the evidence we needed? (article level) |
| `context_precision` | \|retrieved titles that are gold\| / \|total retrieved\| | Are retrieved docs actually useful, or noisy? (article level) |
| `sentence_recall` | gold sentences found in retrieved chunks / total gold sentences | Did the *specific supporting sentences* land inside a retrieved chunk? Finer than title-level recall. |
| `mrr` | 1 / rank of first gold doc | Ranking quality: gold at rank 1 → 1.0; gold at rank 5 → 0.2. Two runs with identical context_recall can differ drastically here. |
| `hit_rate` | 1.0 if any retrieved doc is gold, else 0.0 | Binary floor: did we retrieve *anything* useful? Especially relevant for bridge questions where even one hop found is meaningful. |
| `exact_match` | normalize(predicted) == normalize(gold) | Binary correctness |
| `f1` | token-level F1(predicted, gold) | Partial credit for near-correct answers |
| `faithfulness` | LLM-as-judge: fraction of answer claims grounded in context | Hallucination detection. **Opt-in** (`eval_faithfulness=True`), costs one extra LLM call per question. Off by default. |
| `avg_latency` | mean wall-clock seconds per query | System speed |

All metrics except `faithfulness` are computed with zero extra LLM calls and no external
dependencies. `faithfulness` uses the same `llm_provider` already passed to `run_experiment`
— no separate judge model or RAGAS dependency required (see *Why not RAGAS?* below).

**Aggregate breakdowns (always computed, no extra cost):**

Every result JSON also includes `metrics["by_type"]` and `metrics["by_level"]` containing
the full metric set stratified by question type (bridge / comparison) and difficulty
(easy / medium / hard). This prevents a retriever that excels on comparison questions but
fails bridge questions from hiding behind an averaged score.

**Why not RAGAS?**

RAGAS was designed for evaluation *without* ground truth labels. We have HotpotQA gold
labels, which are categorically superior:

| RAGAS metric | Our equivalent | Why ours is better |
|---|---|---|
| Context Precision | `context_precision` | Gold labels beat LLM-as-judge |
| Context Recall | `context_recall` | Gold labels beat LLM-as-judge |
| Answer Relevancy | `exact_match` + `f1` | Ground truth beats embedding-similarity proxy |
| Faithfulness | `faithfulness` (opt-in) | Implemented directly via our existing `llm_provider` |

RAGAS also requires a separate judge LLM (typically the OpenAI API), adds a heavy
dependency with its own versioning, and makes results harder to reproduce. The one metric
RAGAS offers that our original harness lacked — faithfulness — is now implemented in the
harness itself.

**Example calculation for the Scott Derrickson question:**

```
gold titles:       {"Scott Derrickson", "Ed Wood"}
supporting_facts:  [("Scott Derrickson", 0), ("Ed Wood", 0)]
  → gold sentences: ["Scott Derrickson (born July 16, 1966) is an American director.",
                     "Ed Wood (born October 10, 1924) was an American filmmaker."]

retrieved (in order): ["Scott Derrickson", "Ed Wood", "Marvel Cinematic Universe",
                       "Horror film", "American film"]

context_recall    = 2/2 = 1.0   (retrieved both gold titles — perfect)
context_precision = 2/5 = 0.4   (3 of 5 retrieved were noise)
sentence_recall   = 2/2 = 1.0   (both supporting sentences appear in retrieved chunks)
mrr               = 1/1 = 1.0   (first gold doc is at rank 1)
hit_rate          = 1.0         (at least one gold doc retrieved)
exact_match       = 1.0         ("yes" == "yes")
f1                = 1.0

# MRR contrast — same recall, very different ranking:
retrieved (buried): ["Horror film", "American film", "Marvel", "Ed Wood", "Scott Derrickson"]
mrr               = 1/4 = 0.25  (first gold "Ed Wood" at rank 4)
context_recall    = 1.0         (still retrieved both — recall is blind to order)
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

**Key metrics for Phase 4:** `mrr` is the most sensitive signal here. Rerankers
don't change *which* docs are in the candidate pool (recall is fixed by the upstream
retriever), but they change the *order*. An improvement in `mrr` with no change in
`context_recall` is exactly what a good reranker produces — gold docs are promoted
to earlier ranks, giving the LLM cleaner leading context.

---

## Phase 5: Generation Strategy Ablation (Built)

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

**Key metrics for Phase 5:** This is the one phase where `faithfulness` (opt-in) is
worth enabling. Run Phase 5 with `eval_faithfulness=True` to directly measure whether
CoVe reduces hallucination relative to Standard. Without `faithfulness`, you can only
infer hallucination from EM/F1 misses — ambiguous, since a miss might be a retrieval
failure rather than generation fabrication. The `faithfulness` metric resolves that
ambiguity.

**Note on Attributed (5c) EM/F1:** The LLM embeds citation markers like `[1]` and
`[2]` in the answer. After metric normalisation, brackets are stripped but the digits
remain (`"yes [1]"` → `"yes 1"`), adding extra tokens. Lower EM/F1 for 5c does not
mean lower quality — it reflects the citation format, not accuracy.

---

## Phase 6: Best-of-Breed Combination (Built)

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

## Phase 7: PDF Corpus — Synthetic QA (Built)

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

### HotpotQA

| Phase | Status | Script | Result file |
|-------|--------|--------|-------------|
| 1 | ✅ Built + pushed | `research/phase1_baseline/run.py` | `research/results/phase1_baseline.json` |
| 2 | ✅ Built + pushed | `research/phase2_chunking/run.py` | `research/results/phase2_chunking_<id>.json` |
| 3 | ✅ Built + pushed | `research/phase3_retrieval/run.py` | `research/results/phase3_retrieval_<id>.json` |
| 4 | ✅ Built + pushed | `research/phase4_reranking/run.py` | `research/results/phase4_reranking_<id>.json` |
| 5 | ✅ Built + pushed | `research/phase5_generation/run.py` | `research/results/phase5_generation_<id>.json` |
| 6 | ✅ Built + pushed | `research/phase6_best_combo/run.py` | `research/results/phase6_best_combo_<id>.json` |
| 7 | ✅ Built + pushed | `research/phase7_pdf_corpus/run.py` | `research/results/phase7_pdf_corpus.json` |

### QASPER

| Phase | Status | Script | Result file |
|-------|--------|--------|-------------|
| 1 | ✅ Built + pushed | `research/phase1_baseline/run_qasper.py` | `research/results/phase1_baseline_qasper.json` |
| 2 | ✅ Built + pushed | `research/phase2_chunking/run_qasper.py` | `research/results/phase2_chunking_<id>_qasper.json` |
| 3 | ✅ Built + pushed | `research/phase3_retrieval/run_qasper.py` | `research/results/phase3_retrieval_<id>_qasper.json` |
| 4 | ✅ Built + pushed | `research/phase4_reranking/run_qasper.py` | `research/results/phase4_reranking_<id>_qasper.json` |
| 5 | ✅ Built + pushed | `research/phase5_generation/run_qasper.py` | `research/results/phase5_generation_<id>_qasper.json` |
| 6 | ✅ Built + pushed | `research/phase6_best_combo/run_qasper.py` | `research/results/phase6_best_combo_qasper.json` |
| 7 | ✅ Built + pushed | `research/phase7_pdf_corpus/run_qasper.py` | `research/results/phase7_qasper.json` |

## SDK Tools

### Chunk Inspector (`rag_sdk.document.inspector`)

A standalone utility that shows exactly what a splitter produces from any set of
documents — no vector store, no embeddings, no LLM calls.

```python
from rag_sdk.document import TextSplitter, SemanticSplitter, inspect_chunks

splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
report = inspect_chunks(docs, splitter)

report.summary()          # aggregate stats + size histogram
report.table()            # every chunk as a fixed-width table row
report.detail(3)          # full content of chunk #3
report.for_source("Introduction [0]").table()  # filter to one source

# Programmatic access — plain Python list of dataclasses
tiny = [c for c in report.chunks if c.char_count < 100]

# Optional: pandas DataFrame (requires: pip install pandas)
df = report.to_dataframe()
df.groupby("source").char_count.mean()
df.sort_values("char_count", ascending=False)
```

**Why this matters for QASPER research:** calling `inspect_chunks` on a QASPER paper
before running an experiment immediately reveals the difference — 40–100 chunks from
a real paper vs 1–2 chunks from a HotpotQA paragraph. This validates the dataset
choice and shows what each splitter actually does to long, structured documents.

Phase 2 QASPER (`run_qasper.py --inspect`) prints a full chunk table for each
splitter variant on the first paper without running the full evaluation.

---

## Harness Changelog

### v2 — Evaluation harness expanded (`research/shared/harness.py`)

**New metrics added** (all phases automatically benefit on next run):

| Metric | Default | Cost | Primary diagnostic phase |
|--------|---------|------|--------------------------|
| `sentence_recall` | Always on | 0 extra calls | Phase 2 (chunking) |
| `mrr` | Always on | 0 extra calls | Phase 4 (reranking) |
| `hit_rate` | Always on | 0 extra calls | Phase 3 (retrieval) |
| `faithfulness` | Opt-in | +1 LLM call/question | Phase 5 (generation) |

**New aggregate breakdowns** (always computed):

- `metrics["by_type"]["bridge"|"comparison"]` — HotpotQA question type strata
- `metrics["by_level"]["easy"|"medium"|"hard"]` — HotpotQA difficulty strata

**API change** (backward compatible — all existing callers unchanged):

```python
# New optional parameter added to run_experiment:
run_experiment(..., eval_faithfulness=False)  # default: off

# Enable for Phase 5 to measure hallucination directly:
run_experiment(..., eval_faithfulness=True)
```

**No new dependencies.** `faithfulness` uses the same `llm_provider` already passed to
`run_experiment`. RAGAS was evaluated and rejected — see *Why not RAGAS?* above.

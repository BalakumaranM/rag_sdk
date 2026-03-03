# RAG Evaluation Module — Research Opinion & Concrete Plan

## The Problem

All current evaluation in this project is tied to HotpotQA. The harness in
`research/shared/harness.py` measures context_recall, mrr, sentence_recall etc. using
HotpotQA's gold labels (supporting titles, sentence indices). You cannot reuse any of
this for your own documents — there are no pre-existing gold labels, no `supporting_facts`
field, no distractor paragraph structure.

When you move to a custom domain (legal contracts, product manuals, internal knowledge
bases, research papers), you need:

1. **A way to create a ground truth dataset** from your own documents — without hand-labelling
   hundreds of Q&A pairs.

2. **A way to evaluate your RAG pipeline** against that dataset using metrics that don't
   require gold retrieval labels (because you won't have them).

3. **Both of these must work with any LLM** — local or API — using the providers already
   in the SDK.

---

## Framework Landscape Review

Before deciding what to build, the major evaluation frameworks were assessed:

| Framework | License | Local LLM | Synthetic Gen | Key problem |
|---|---|---|---|---|
| **RAGAS** | Apache 2.0 | Yes (LiteLLM adapter) | Yes (TestsetGenerator) | Tightly coupled to LangChain/LlamaIndex adapter layer; bridging to our own providers adds friction |
| **DeepEval** | Apache 2.0 | Yes (Ollama native) | Yes (Synthesizer) | Large surface area; commercial Confident AI cloud entanglement; 50+ metrics increase maintenance burden |
| **TruLens** | MIT | Yes (LiteLLM) | No | Observability-first platform, not a library; no synthetic data generation; heavyweight |
| **ARES** | MIT+Apache | No | Yes (with 150+ human labels) | Requires GPU fine-tuning (A100+); known multi-dataset bugs; academic prototype |
| **SelfCheckGPT** | MIT | No | No | Single-purpose hallucination detector only; requires sampling multiple generations |

### Why NOT to use any of them as a dependency

Every framework above solves the provider-coupling problem by adding its own adapter
layer (LiteLLM, LangChain, LlamaIndex). We already have `LLMProvider` and
`EmbeddingProvider` abstractions that work with local endpoints, OpenAI, Anthropic,
Cohere, Gemini, and Voyage — with no third-party coupling.

Pulling in RAGAS or DeepEval would mean:
- A new dependency ecosystem with its own versioning and breaking changes
- Adapter boilerplate to bridge our providers to their provider system
- Loss of control over evaluation prompts (RAGAS prompts are hard-coded)
- The user would now need to manage two provider configurations

The algorithms behind the best RAGAS metrics are well-understood and implementable
in ~200 lines. We adopt the algorithms, not the library.

### What we DO take from each framework

| Framework | What we adopt |
|---|---|
| RAGAS | Claim-extraction faithfulness algorithm; embedding-based answer relevancy; evolutionary QA generation strategy |
| DeepEval | G-Eval structured reasoning approach for answer correctness |
| ARES | Idea of PPI confidence intervals (future, not MVP) |
| SelfCheckGPT | Not adopted — the sampling approach requires too many LLM calls per question |

---

## Decision: SDK Module vs Research-Only

**Recommendation: Inside the SDK (`rag_sdk/evaluation/`)**

The evaluation module should live in the SDK, not in `research/`, for these reasons:

1. **Users of the SDK need evaluation in production.** The development loop is:
   build RAG → evaluate → tune → repeat. Evaluation is not a research-only concern.

2. **Synthetic QA generation is a first-class RAG feature.** It unlocks the ability to
   test any document corpus without hand-labelling. This belongs in the product.

3. **The providers are already there.** `LLMProvider.generate()` and
   `EmbeddingProvider.embed_documents()` are exactly the interfaces the evaluation
   metrics need. Putting evaluation in the SDK means zero bridging code.

4. **The research harness becomes a thin wrapper.** `research/shared/harness.py` today
   duplicates logic (faithfulness, etc.) that should live in the SDK. After this module
   is built, the harness can delegate to `rag_sdk.evaluation` and focus only on the
   HotpotQA-specific orchestration.

5. **It matches the original project_plans/rag_evaluations.yaml vision** which already
   listed RAGAS-style metrics and frameworks as planned features.

---

## Architecture

```
rag_sdk/
└── evaluation/
    ├── __init__.py              ← public API: RAGEvaluator, SyntheticDatasetGenerator, EvalSample
    ├── dataset.py               ← EvalSample, EvalDataset (data model + serialization)
    ├── evaluator.py             ← RAGEvaluator (main class, orchestrates all metrics)
    ├── synthetic.py             ← SyntheticDatasetGenerator
    └── metrics/
        ├── __init__.py
        ├── string_match.py      ← exact_match, token_f1, rouge_l  (no LLM, no deps)
        ├── faithfulness.py      ← claim extraction + verification  (2 LLM calls)
        ├── relevancy.py         ← answer_relevancy, context_relevancy  (1-2 LLM calls)
        ├── correctness.py       ← answer_correctness vs ground truth  (1 LLM call)
        └── retrieval.py         ← mrr, hit_rate, ndcg, context_recall/precision (labeled)
```

**Nothing in `rag_sdk/evaluation/` imports from `research/`.** The flow is:

```
research/shared/harness.py   →   uses   →   rag_sdk.evaluation (for faithfulness, etc.)
rag_sdk/evaluation/          →   uses   →   rag_sdk.llm.base.LLMProvider
                             →   uses   →   rag_sdk.embeddings.base.EmbeddingProvider
```

---

## Phase A — Data Model (`evaluation/dataset.py`)

### `EvalSample`

The atomic unit of evaluation — provider-agnostic, dataset-agnostic.

```python
@dataclass
class EvalSample:
    question: str
    ground_truth: Optional[str] = None       # gold answer, if available
    ground_truth_contexts: Optional[List[str]] = None  # gold source chunks, if available
    metadata: Dict[str, Any] = field(default_factory=dict)  # type, level, source file, etc.
```

### `EvalResult`

One row of evaluation output.

```python
@dataclass
class EvalResult:
    question: str
    answer: str
    retrieved_contexts: List[str]            # chunk content strings
    ground_truth: Optional[str]
    scores: Dict[str, Optional[float]]       # metric_name → score (None if not computed)
    metadata: Dict[str, Any]
```

### `EvalDataset`

```python
@dataclass
class EvalDataset:
    samples: List[EvalSample]
    metadata: Dict[str, Any]                 # source, generation_method, num_samples, etc.

    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> "EvalDataset": ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
```

Saves/loads as JSON. Portable, inspectable, version-controllable.

---

## Phase B — String-Match Metrics (`metrics/string_match.py`)

No LLM, no embedding model, no dependencies beyond stdlib.

| Function | Description |
|---|---|
| `exact_match(pred, gold)` | Normalized equality. Identical to harness `_exact_match`. |
| `token_f1(pred, gold)` | Token-level F1. Identical to harness `_token_f1`. |
| `rouge_l(pred, gold)` | ROUGE-L (longest common subsequence recall). Better than F1 for multi-sentence answers. Uses `difflib.SequenceMatcher` — no extra dep. |

These are always available regardless of whether an LLM is configured.

---

## Phase C — LLM-as-Judge Metrics

### C1 — Faithfulness (`metrics/faithfulness.py`)

**Algorithm (RAGAS-style, two LLM calls):**

```
Step 1 — Claim extraction:
  prompt: "Given this answer: {answer}\nList every factual claim as atomic statements."
  output: ["Claim 1", "Claim 2", ...]

Step 2 — Claim verification (one call with all claims batched):
  prompt: "Context: {context}\nFor each claim below, output 1 if supported, 0 if not.
           Claims: {claims_json}"
  output: [1, 0, 1, ...]

faithfulness = sum(verdicts) / len(claims)
```

Why two calls vs one: multi-call is more accurate and returns which specific claims
failed — interpretable, actionable. The harness currently uses a single-call approach;
the SDK module adopts the RAGAS algorithm since it's the production version.

```python
def faithfulness(
    answer: str,
    contexts: List[str],
    llm_provider: LLMProvider,
) -> Tuple[float, List[str], List[int]]:
    """Returns (score, claims, verdicts) for interpretability."""
```

### C2 — Answer Relevancy (`metrics/relevancy.py`)

**Algorithm (RAGAS-style, embedding-based):**

```
Step 1 — Question regeneration (1 LLM call):
  prompt: "Given this answer: {answer}\nGenerate {n} questions this answer could address."
  output: ["Q1", "Q2", "Q3"]

Step 2 — Embedding similarity:
  sim_i = cosine(embed(original_question), embed(Q_i))
  answer_relevancy = mean(sim_i for i in 1..n)
```

If no `embedding_provider` is given, fall back to a simpler single LLM-as-judge call:
```
prompt: "Rate 0.0–1.0: does this answer directly address the question? ..."
```

```python
def answer_relevancy(
    question: str,
    answer: str,
    llm_provider: LLMProvider,
    embedding_provider: Optional[EmbeddingProvider] = None,
    num_questions: int = 3,
) -> float: ...
```

**Context Relevancy** — per retrieved chunk:

```
prompt: "Rate 0.0–1.0: how relevant is this context chunk to the question?
         Question: {question}
         Context: {chunk}"
context_relevancy = mean score across all chunks
```

```python
def context_relevancy(
    question: str,
    contexts: List[str],
    llm_provider: LLMProvider,
) -> float: ...
```

### C3 — Answer Correctness (`metrics/correctness.py`)

Only computable when `ground_truth` is available.

Two-component score (configurable weights):
- **String component (30%):** ROUGE-L or token F1 against ground truth
- **Semantic component (70%):** LLM judge comparing predicted vs ground truth

```
prompt: "Ground truth: {ground_truth}
         Predicted: {answer}
         Rate semantic correctness 0.0–1.0. Are the core facts identical?
         Minor wording differences = 1.0. Missing facts = proportionally lower."
```

```python
def answer_correctness(
    answer: str,
    ground_truth: str,
    llm_provider: LLMProvider,
    string_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> float: ...
```

---

## Phase D — Retrieval Metrics (`metrics/retrieval.py`)

These are already implemented in the harness. In the SDK module they become proper
functions with the same signatures, and the harness imports them from here.

| Function | Requires gold labels | Description |
|---|---|---|
| `mrr(retrieved, gold_set)` | Yes | 1/rank of first gold item |
| `hit_rate(retrieved, gold_set)` | Yes | Binary: any gold item retrieved |
| `ndcg(retrieved, gold_set, k)` | Yes (binary OK) | NDCG@k, works with binary relevance |
| `context_recall_labeled(retrieved, gold_set)` | Yes | Fraction of gold items retrieved |
| `context_precision_labeled(retrieved, gold_set)` | Yes | Fraction of retrieved that are gold |

**NDCG formula (binary relevance):**
```
DCG@k  = Σ rel_i / log2(i+1)  for i in 1..k   (rel_i ∈ {0,1})
IDCG@k = Σ 1 / log2(i+1)      for i in 1..min(k, |gold|)
NDCG@k = DCG@k / IDCG@k
```

NDCG is more sensitive than MRR when multiple gold docs exist (HotpotQA always has 2).
MRR only cares about the first hit; NDCG rewards retrieving all gold docs early.

---

## Phase E — RAGEvaluator (`evaluation/evaluator.py`)

The main class that wires everything together.

```python
class RAGEvaluator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        metrics: Optional[List[str]] = None,   # subset to run; default: all applicable
    ): ...

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],                   # retrieved chunk text content
        ground_truth: Optional[str] = None,
        retrieved_sources: Optional[List[str]] = None,  # titles or IDs
        gold_sources: Optional[Set[str]] = None,        # for labeled retrieval metrics
    ) -> EvalResult: ...

    def evaluate_batch(
        self,
        samples: List[EvalSample],
        answers: List[str],
        contexts_per_sample: List[List[str]],
    ) -> List[EvalResult]: ...

    def evaluate_rag(
        self,
        rag: "RAG",
        dataset: EvalDataset,
        top_k: int = 5,
    ) -> List[EvalResult]:
        """Run rag.query() for each sample, then evaluate. One-shot convenience method."""
        ...

    def summary(self, results: List[EvalResult]) -> Dict[str, float]:
        """Aggregate means across all results. Skips None scores."""
        ...
```

**Metric selection logic:**
- If `ground_truth` is None → skip `answer_correctness`, `rouge_l` (exact_match, token_f1 still run but will score 0)
- If `embedding_provider` is None → answer_relevancy uses LLM-only fallback
- If `gold_sources` is None → skip labeled retrieval metrics (mrr, hit_rate, ndcg, context_recall/precision)
- `faithfulness` and `context_relevancy` always run (they are reference-free)

**Metric costs** (LLM calls per question):

| Metric | LLM calls | Embedding calls |
|---|---|---|
| `exact_match`, `token_f1`, `rouge_l` | 0 | 0 |
| `faithfulness` | 2 | 0 |
| `answer_relevancy` | 1 | 1 (or +1 LLM if no embedder) |
| `context_relevancy` | len(contexts) | 0 |
| `answer_correctness` | 1 | 0 |
| **Total (all metrics, 5 contexts)** | **~9** | **1** |

This is comparable to RAGAS's cost profile. Context_relevancy is the expensive one for
large top_k — make it optional or batch-callable.

---

## Phase F — Synthetic Dataset Generator (`evaluation/synthetic.py`)

This is the most impactful piece for custom datasets. It removes the need to hand-label
ground truth.

### Algorithm (inspired by RAGAS TestsetGenerator, but provider-agnostic)

```
For each document chunk (or chunk pair for multi-hop):

  1. Generate a question:
     prompt: "Read this context. Write a {question_type} question that:
              - Can only be answered using this specific text
              - Is not trivially answered by the title
              - Requires reading the key facts in the passage"

  2. Generate a ground truth answer:
     prompt: "Context: {chunk}\nQuestion: {question}\n
              Answer the question using ONLY information from the context above."

  3. Quality filter (LLM self-check):
     prompt: "Question: {q}\nAnswer: {a}\nContext: {chunk}
              Score 0-10: Is the answer fully answerable from the context alone,
              with no additional knowledge required?"
     → discard if score < 7

  4. Store EvalSample(question, ground_truth, ground_truth_contexts=[chunk], metadata)
```

### Question Types

| Type | How generated | What it tests |
|---|---|---|
| `simple` | From a single chunk | Basic retrieval + generation |
| `multi_hop` | From two related chunks | Multi-step retrieval |
| `abstractive` | From chunk, requiring synthesis | Summarization / reasoning |
| `null` | Unanswerable given corpus | Hallucination resistance |

### Public API

```python
class SyntheticDatasetGenerator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ): ...

    def generate(
        self,
        documents: List[Document],
        num_questions: int = 100,
        question_types: List[str] = ["simple", "multi_hop"],
        quality_threshold: float = 7.0,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> EvalDataset: ...

    def generate_from_rag(
        self,
        rag: "RAG",
        num_questions: int = 100,
        question_types: List[str] = ["simple", "multi_hop"],
    ) -> EvalDataset:
        """Convenience: use the RAG's already-ingested document store."""
        ...
```

### Why this matters for your use cases

You can now do the full evaluation loop on **any corpus** without a single hand-labelled
example:

```python
# 1. Ingest your documents
rag = RAG(config, embedding_provider=embed, llm_provider=llm)
rag.ingest_documents(my_domain_docs)

# 2. Generate a ground truth dataset from those same documents
gen = SyntheticDatasetGenerator(llm_provider=llm, embedding_provider=embed)
dataset = gen.generate_from_rag(rag, num_questions=200, question_types=["simple", "multi_hop"])
dataset.save(Path("my_domain_eval.json"))

# 3. Evaluate
evaluator = RAGEvaluator(llm_provider=llm, embedding_provider=embed)
results = evaluator.evaluate_rag(rag, dataset)
summary = evaluator.summary(results)
print(summary)
# → {"faithfulness": 0.87, "answer_relevancy": 0.82, "answer_correctness": 0.74, ...}
```

No HotpotQA, no gold labels, no hand-labelling. Swap in any domain.

---

## Phase G — Research Harness Integration

After the SDK module is built, `research/shared/harness.py` gets updated:

1. **Import and use SDK metrics** instead of local duplicates:
   - `_faithfulness()` → `from rag_sdk.evaluation.metrics.faithfulness import faithfulness`
   - `_mrr()`, `_hit_rate()` → `from rag_sdk.evaluation.metrics.retrieval import mrr, hit_rate`
   - `_exact_match()`, `_token_f1()` → `from rag_sdk.evaluation.metrics.string_match import ...`

2. **Keep HotpotQA-specific code in the harness** (not in the SDK):
   - `sentence_recall` — uses HotpotQA's `supporting_facts` sentence-index format
   - `_build_shared_corpus()` — HotpotQA distractor structure
   - `_run_per_question()` / `_run_shared_corpus()` — HotpotQA evaluation modes
   - `HotpotQASample` data model

3. **Add a generic evaluation path** alongside the HotpotQA path:
   ```python
   # For any EvalDataset (not just HotpotQA)
   def run_custom_experiment(
       config: Config,
       dataset: EvalDataset,          # from SyntheticDatasetGenerator or hand-labelled
       embedding_provider: EmbeddingProvider,
       llm_provider: LLMProvider,
       experiment_name: str,
   ) -> Dict[str, Any]: ...
   ```

---

## Implementation Order & Dependencies

```
Phase A  (dataset.py)           → no dependencies; do first
Phase B  (string_match.py)      → depends on A
Phase D  (retrieval.py)         → depends on A; mirrors existing harness functions
Phase C1 (faithfulness.py)      → depends on A; needs LLMProvider
Phase C2 (relevancy.py)         → depends on A; needs LLMProvider
Phase C3 (correctness.py)       → depends on A, B; needs LLMProvider
Phase E  (evaluator.py)         → depends on A, B, C1-C3, D
Phase F  (synthetic.py)         → depends on A; needs LLMProvider
Phase G  (harness integration)  → depends on E; last step

Approximate size:  ~800–1000 lines total, spread across 8 files.
```

All phases can be tested independently. Phase G is last and has zero risk — existing
experiments still run unchanged; harness just delegates to SDK internally.

---

## What Gets Added to `rag_sdk/__init__.py` and README

```python
# New public exports
from rag_sdk.evaluation import RAGEvaluator, SyntheticDatasetGenerator, EvalSample, EvalDataset
```

The README gets a new **Evaluation** section showing the two workflows:
1. Evaluate against a hand-labelled dataset (EvalDataset from JSON)
2. Generate + evaluate in one block (SyntheticDatasetGenerator → evaluate_rag)

---

## Summary: What This Unlocks

| Capability | Before | After |
|---|---|---|
| Evaluate RAG on HotpotQA | ✅ (research only) | ✅ (SDK + research) |
| Evaluate RAG on custom docs | ❌ | ✅ |
| Create ground truth without labelling | ❌ | ✅ |
| Faithfulness metric | ✅ (single-call, research only) | ✅ (two-call claim extraction, SDK) |
| Answer relevancy | ❌ | ✅ |
| Context relevancy | ❌ | ✅ |
| Answer correctness vs ground truth | ❌ | ✅ |
| NDCG | ❌ | ✅ |
| Works with local LLM | ✅ (existing providers) | ✅ (same providers) |
| Works with any embedding | ✅ (existing providers) | ✅ (same providers) |
| External framework dependency | ❌ | ❌ (none added) |

# Phase 5 — Generation Strategy Ablation

## Purpose

Find out whether a more sophisticated generation strategy produces better answers
given the same retrieved context. Everything else is frozen: TextSplitter(512, 50)
chunking, the Phase 3 winning retrieval strategy, the Phase 4 winning reranking
choice. Only the generation strategy changes.

The core question: **does verifying or attributing the answer improve quality?**

---

## What Generation Strategies Do

The generation strategy takes a query and the top-k retrieved chunks, then produces
an answer. The simplest strategy calls the LLM once. More sophisticated strategies
add verification loops or citation requirements.

```
Retrieved chunks (5 docs)
        │
        ▼
GenerationStrategy.generate(query, docs)
        │
        ▼
{"answer": str, ...}      ← harness reads result["answer"]
```

---

## Variants

### 5a — Standard · Baseline

Source: `rag_sdk/generation/standard.py`

Builds a context string from the retrieved chunks and calls the LLM once:

```
system_prompt = "Use the following context to answer the question.\n\n{context}"
answer = llm.generate(query, system_prompt=system_prompt)
```

One LLM call per query. This is the same strategy used in all previous phases.

Expected: moderate F1 and EM — the reference point for Phase 5.

---

### 5b — CoVe · Chain-of-Verification  `[SLOW]`

Source: `rag_sdk/generation/cove.py`

Algorithm (Dhuliawala et al., 2023):

```
Step 1  Initial answer
        llm(query, context) → initial_answer

Step 2  Verification questions
        llm(initial_answer) → ["What nationality is Scott Derrickson?",
                                "What nationality is Ed Wood?", ...]

Step 3  Verify each claim (1 LLM call per question)
        llm(question, context) → factual answer for that specific claim

Step 4  Refined answer
        llm(query, context, initial_answer, verification_qa) → refined_answer
```

Total LLM calls per query: 1 + 1 + N + 1 = **3 + N** where N = number of
verification questions (up to 3 by default) → **up to 6 calls per query**.

**Why it's better in theory**: The initial answer may hallucinate or conflate
facts. Verification questions force the LLM to check each claim independently
against the context. The refined answer incorporates what was verified.

**Why it may not help on HotpotQA**: HotpotQA answers are usually 1–3 words
("yes", "Scott Derrickson", "American"). There's little to verify in a one-word
answer. CoVe is designed for paragraph-length answers with multiple factual claims.
The verification loop may not fire at all for short answers if the LLM can't
generate meaningful verification questions.

**Cost**: Up to 6 LLM calls × 100 questions = up to 600 calls. At 2s/call,
~20 minutes of extra compute vs Standard.

Expected: **similar or slightly higher F1** on HotpotQA (CoVe's benefit is bigger
on open-ended generation tasks); **significantly higher latency**.

---

### 5c — Attributed · Answer with Citations

Source: `rag_sdk/generation/attributed.py`

Numbers the retrieved chunks `[1], [2], ..., [5]`, then instructs the LLM:

```
system_prompt = "Use the numbered sources to answer. Include inline [N] citations.
                 [1] (Source: Scott Derrickson) Scott Derrickson...
                 [2] (Source: Ed Wood) Ed Wood..."

answer = llm.generate(query, system_prompt=system_prompt)
```

The LLM produces answers like: *"Yes, both Scott Derrickson [1] and Ed Wood [2]
are American."*

One LLM call per query — same cost as Standard.

**Why it's useful**: Each claim is traceable to a specific source. The user can
verify which chunk the LLM drew from. This is important for production deployments
where hallucination must be auditable.

**Metric caveat — IMPORTANT:**

The `_normalize()` function in the evaluation harness strips `[` and `]` as
punctuation but leaves digits. So `"yes [1] [2]"` normalises to `"yes 1 2"`.
The gold answer `"yes"` normalises to `"yes"`. This means:

- Precision decreases (extra tokens "1" and "2")
- Recall is unchanged (gold token "yes" is found)
- F1 decreases

**This is a metric artefact, not a quality regression.** If you strip `[N]`
patterns before normalisation, Attributed scores identically to Standard on
factual content. The raw F1/EM numbers in the table will be lower for 5c — read
the note in the output.

Expected: **F1/EM below Standard** in the raw numbers (due to citation digits);
**qualitatively better** for production use cases requiring traceability.

---

## How to Read the Results

```
  Variant        Recall  Prec    EM      F1    Latency
  ──────────────────────────────────────────────────────
  Standard (5a)  0.643   0.389   0.281   0.451  2.34s  ← baseline
  CoVe     (5b)  0.643   0.389   0.289   0.463  9.12s  ← ↑F1 at 4× latency cost
  Attributed(5c) 0.643   0.389   0.241   0.392 2.51s   ← ↓raw F1 (metric artefact)
```

(Numbers above are illustrative hypotheses, not actual results.)

**What to look for:**

| Observation | Interpretation |
|------------|----------------|
| Recall identical across 5a/5b/5c | Expected — generation doesn't change which docs were retrieved |
| Precision identical across 5a/5b/5c | Same retrieved docs, same context |
| CoVe F1 > Standard F1 | Verification loop helped refine the answer for HotpotQA |
| CoVe F1 ≈ Standard F1 | CoVe's benefit doesn't apply to 1–3 word answers |
| Attributed F1 < Standard F1 | Expected artefact — not a quality problem |
| CoVe latency >> Standard latency | Each extra LLM call adds 1–3 seconds |

**Choosing the Phase 5 winner:**

- If CoVe F1 > Standard by a meaningful margin (>0.01): use CoVe in Phase 6
- If CoVe ≈ Standard: use Standard (same quality, lower cost)
- Attributed is a separate dimension — choose it for production deployability,
  not for benchmark scores. Its raw F1 will always be lower due to the metric
  artefact, but the factual quality is identical to Standard.

---

## How to Run

```bash
cd /path/to/rag_sdk

# Fast variants only (5a Standard + 5c Attributed):
.venv/bin/python research/phase5_generation/run.py

# All variants including slow CoVe:
.venv/bin/python research/phase5_generation/run.py --all

# Specific variants:
.venv/bin/python research/phase5_generation/run.py --variants 5a,5b

# Re-run a specific variant:
.venv/bin/python research/phase5_generation/run.py --variants 5b --force
```

Results saved to: `research/results/phase5_generation_<id>.json`

**Before running:**
1. Update `RETRIEVAL_STRATEGY` to the Phase 3 winner.
2. Update `RERANKING_PROVIDER` to the Phase 4 winner (`None` | `"cross-encoder"` | `"cohere"`).

---

## What Phase 6 Will Change

Phase 6 takes the best component from each phase and combines them into a single
best-of-breed pipeline, then compares the full combination against the Phase 1
baseline to measure total improvement.

# Phase 7 — PDF Corpus with Synthetic QA

## Purpose

Validate that the Phase 6 best-of-breed pipeline generalises beyond Wikipedia.
HotpotQA uses clean, encyclopaedic Wikipedia paragraphs — real domain documents
(research papers, reports, manuals) are messier, longer, and structured differently.

The core question: **do the optimisations from Phases 2–5 hold on real PDFs?**

---

## Why This Phase Is Different

Every previous phase used HotpotQA, which provides:
- Pre-labeled gold Q&A pairs
- Pre-selected document sets (10 per question)
- Pre-identified supporting documents

Phase 7 has none of that. We start with raw PDFs and must:
1. **Parse** them into text (with the SDK's PDF parser)
2. **Generate** synthetic Q&A pairs using the LLM
3. **Ingest** all parsed pages into the RAG pipeline
4. **Evaluate** the pipeline on those synthetic pairs

---

## Setup

Drop your PDF files into `research/data/pdfs/`:

```bash
cp /path/to/your/paper.pdf research/data/pdfs/
cp /path/to/your/report.pdf research/data/pdfs/
# ... add as many as you like
```

That directory is gitignored — PDFs are never committed to the repo.

There are no constraints on which PDFs to use, but for meaningful results:
- Use **domain-specific** documents (technical papers, product manuals, reports)
- 5–20 PDFs with 10–50 pages each is a good starting size
- Avoid very short PDFs (< 5 pages) — not enough content to generate good questions

---

## How Synthetic QA Works

For each document page, the LLM is prompted:

```
Read the following text and generate N factoid question-answer pairs.
Rules:
  - Each question must be answerable from this text alone.
  - Each answer must be a short phrase (1-5 words).
  - Questions must be specific, not vague.
Return ONLY a JSON array:
[{"question": "...", "answer": "..."}, ...]

Text: {page_content}
```

Each QA pair is tagged with its `source` (the PDF filename without extension).
This allows computing `source_recall` at evaluation time.

**QA caching:** Generated pairs are saved to `research/data/phase7_qa_cache.json`
(gitignored). Re-runs load from cache unless `--force_qa` is passed.

---

## Metrics

| Metric | Definition | HotpotQA equivalent |
|--------|-----------|---------------------|
| `source_recall` | Did the retriever return a chunk from the PDF that generated the Q? | `context_recall` |
| `exact_match` | Normalised predicted answer == normalised gold answer | `exact_match` |
| `f1` | Token-level F1 between predicted and gold | `f1` |
| `avg_latency` | Mean wall-clock seconds per query | `avg_latency` |

**Note on synthetic QA quality:** The gold answers are LLM-generated, not human-
verified. If the LLM generates a wrong gold answer or the pipeline gives a
synonymous but differently worded correct answer, EM/F1 will be artificially low.
This is an inherent limitation of automated evaluation on synthetic data.

---

## How to Run

```bash
cd /path/to/rag_sdk

# Use the default PDF directory (research/data/pdfs/):
.venv/bin/python research/phase7_pdf_corpus/run.py

# Point at a specific directory:
.venv/bin/python research/phase7_pdf_corpus/run.py --pdf_dir /path/to/pdfs

# Control QA pair count:
.venv/bin/python research/phase7_pdf_corpus/run.py --num_qa 100 --qa_per_doc 10

# Regenerate QA pairs (ignores cache):
.venv/bin/python research/phase7_pdf_corpus/run.py --force_qa

# Re-run evaluation (ignores cached result):
.venv/bin/python research/phase7_pdf_corpus/run.py --force
```

Results saved to: `research/results/phase7_pdf_corpus.json`
QA cache saved to: `research/data/phase7_qa_cache.json` (gitignored)

**Before running:**
1. Update the winner constants at the top of `run.py` to the Phase 2–5 results.
2. Add PDFs to `research/data/pdfs/`.

---

## How to Read the Output

```
==================================================================
Phase 7 — PDF Corpus Results
==================================================================
  Metric               Phase 7 (PDF)
  ────────────────────────────────────────
  Source Recall        0.612
  Exact Match          0.241
  F1                   0.389
  Avg Latency          3.12s
  Samples              50

  Comparison vs Phase 6 (HotpotQA, shared_corpus):
  Metric                   Phase 6    Phase 7    Delta
  ──────────────────────────────────────────────────────
  Context/Src Recall         0.731      0.612   -0.119
  F1                         0.523      0.389   -0.134
  Exact Match                0.341      0.241   -0.100
==================================================================
```

(Numbers above are illustrative hypotheses, not actual results.)

**Interpreting the comparison:**

| Phase 7 vs Phase 6 | Interpretation |
|-------------------|----------------|
| Phase 7 F1 ≈ Phase 6 F1 | Pipeline generalises well — optimisations are domain-agnostic |
| Phase 7 F1 < Phase 6 F1 by < 0.1 | Moderate generalisation gap — expected for domain shift |
| Phase 7 F1 < Phase 6 F1 by > 0.1 | Poor generalisation — pipeline over-fits to Wikipedia structure |
| Phase 7 source_recall << Phase 6 context_recall | Retriever struggles with PDF layout noise; consider chunking tuning |
| Phase 7 F1 > Phase 6 F1 | Surprising — synthetic QA may be easier or PDF domain is well-matched |

**Common failure modes on PDFs:**
- **Source recall drops**: PDF pages split mid-sentence, losing context at chunk boundaries.
  SemanticSplitter or larger chunk sizes may help.
- **F1 drops**: PDF text has OCR noise, headers, footers, and table artefacts that dilute
  chunk embeddings. Pre-processing (stripping headers/footers) can help.
- **Synthetic QA answer mismatch**: The gold answer is LLM-generated; the LLM may
  phrase the same fact differently as a retrieved answer. This inflates "wrong" answers
  that are factually correct.

---

## What This Tells You

This phase completes the research loop:

```
Phase 1   Baseline on HotpotQA (per_question)
  ↓
Phase 2–5 Ablation: find the best chunk / retriever / reranker / generator
  ↓
Phase 6   Best-of-breed vs baseline on HotpotQA (shared_corpus)
  ↓
Phase 7   Best-of-breed on real PDFs (generalisation test)
```

If Phase 7 scores are close to Phase 6, your pipeline is robust. If they diverge,
you have a domain adaptation problem — go back to Phase 2 and tune chunking for
PDF structure specifically.

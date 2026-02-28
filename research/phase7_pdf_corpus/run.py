"""Phase 7 — PDF Corpus with Synthetic QA
==========================================
Tests the Phase 6 best-of-breed pipeline on real domain documents (PDFs) rather
than the clean Wikipedia paragraphs of HotpotQA. Uses the LLM to generate
synthetic question-answer pairs from the PDF content, then evaluates retrieval
and answer quality on those pairs.

This answers: **do the optimisations from Phases 2–5 generalise to real documents?**

Workflow
--------
  1. Parse all PDFs in --pdf_dir (default: research/data/pdfs/)
  2. Generate synthetic Q&A pairs from the document pages (LLM, cached)
  3. Ingest all PDF pages into the RAG pipeline
  4. Query each Q&A pair and compute metrics
  5. Compare against Phase 6 results (if available)

Metrics
-------
  source_recall    Did the retriever find a chunk from the document that generated the Q?
  exact_match      Normalised predicted answer == normalised gold answer
  f1               Token-level F1 between predicted and gold answer
  avg_latency      Mean wall-clock seconds per query

Usage
-----
  # Point at your PDF directory:
  .venv/bin/python research/phase7_pdf_corpus/run.py --pdf_dir /path/to/pdfs

  # Use the default directory (research/data/pdfs/):
  .venv/bin/python research/phase7_pdf_corpus/run.py

  # Re-generate QA pairs even if cache exists:
  .venv/bin/python research/phase7_pdf_corpus/run.py --force_qa

  # Re-run evaluation even if result file exists:
  .venv/bin/python research/phase7_pdf_corpus/run.py --force

Setup
-----
  Drop your PDF files into research/data/pdfs/ (or pass --pdf_dir).
  That directory is gitignored — PDFs are never committed to the repo.

  Update the winner constants below to the Phase 2–5 results before
  running the definitive experiment.
"""

import argparse
import json
import logging
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.shared.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    RESULTS_DIR,
    TOP_K,
)
from research.shared.providers import LocalAPIEmbedding, make_llm
from rag_sdk.config import ConfigLoader
from rag_sdk.document.loader import DocumentLoader
from rag_sdk.document.models import Document
from rag_sdk.document.pdf_parser import PyMuPDFParser
from rag_sdk import RAG

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# ── Winner constants — update after running Phases 2–5 ───────────────────────

BEST_CHUNKING_STRATEGY = "recursive"
BEST_CHUNK_SIZE = 512
BEST_CHUNK_OVERLAP = 50
BEST_RETRIEVAL_STRATEGY = "dense"
BEST_RERANKING_PROVIDER: Optional[str] = None
BEST_GENERATION_STRATEGY = "standard"

# ── Paths ─────────────────────────────────────────────────────────────────────

RESEARCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PDF_DIR = RESEARCH_DIR / "data" / "pdfs"
QA_CACHE_PATH = RESEARCH_DIR / "data" / "phase7_qa_cache.json"
RESULT_PATH = RESULTS_DIR / "phase7_pdf_corpus.json"
PHASE6_RESULT_PATH = RESULTS_DIR / "phase6_best_combo_6b.json"


# ── Metric helpers ────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if not num_common:
        return 0.0
    precision = num_common / len(pred_toks) if pred_toks else 0.0
    recall = num_common / len(gold_toks) if gold_toks else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Config builder ────────────────────────────────────────────────────────────


def _make_config() -> Any:
    reranking_dict: Dict[str, Any] = {"enabled": False}
    if BEST_RERANKING_PROVIDER is not None:
        reranking_dict = {"enabled": True, "provider": BEST_RERANKING_PROVIDER}

    return ConfigLoader.from_dict(
        {
            "document_processing": {
                "chunk_size": BEST_CHUNK_SIZE,
                "chunk_overlap": BEST_CHUNK_OVERLAP,
                "chunking": {"strategy": BEST_CHUNKING_STRATEGY},
            },
            "embeddings": {"provider": "openai"},
            "vectorstore": {"provider": "memory"},
            "llm": {
                "provider": "openai",
                "openai": {
                    "model": LOCAL_LLM_MODEL,
                    "base_url": LOCAL_LLM_BASE_URL,
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
            "retrieval": {
                "strategy": BEST_RETRIEVAL_STRATEGY,
                "top_k": TOP_K,
                "reranking": reranking_dict,
            },
            "generation": {"strategy": BEST_GENERATION_STRATEGY},
        }
    )


# ── PDF loading ───────────────────────────────────────────────────────────────


def _load_pdfs(pdf_dir: Path) -> List[Document]:
    """Parse all PDFs in pdf_dir. Sets metadata['source'] to filename stem."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("  No PDF files found in %s", pdf_dir)
        logger.info("  Drop your PDF files there and re-run.")
        return []

    logger.info("  Found %d PDF file(s) in %s", len(pdf_files), pdf_dir)
    parser = PyMuPDFParser()
    all_docs: List[Document] = []

    for pdf_path in pdf_files:
        try:
            docs = DocumentLoader.load_file(
                str(pdf_path), pdf_parser=parser, one_doc_per_page=True
            )
            if isinstance(docs, Document):
                docs = [docs]
            # Override source to the clean filename stem for tracking
            stem = pdf_path.stem
            for doc in docs:
                doc.metadata["source"] = stem
            all_docs.extend(docs)
            logger.info("    %s → %d page(s)", pdf_path.name, len(docs))
        except Exception as exc:
            logger.info("    %s → ERROR: %s (skipped)", pdf_path.name, exc)

    return all_docs


# ── Synthetic QA generation ───────────────────────────────────────────────────


def _generate_qa_for_doc(doc: Document, llm: Any, n: int) -> List[Dict[str, str]]:
    """Ask the LLM to generate n factoid Q&A pairs from a single document page."""
    content = doc.content.strip()
    if len(content) < 100:
        return []

    prompt = (
        f"Read the following text and generate {n} factoid question-answer pairs.\n"
        "Rules:\n"
        "  - Each question must be answerable from this text alone.\n"
        "  - Each answer must be a short phrase (1-5 words).\n"
        "  - Questions must be specific, not vague.\n"
        "Return ONLY a JSON array (no explanation):\n"
        '[{"question": "...", "answer": "..."}, ...]\n\n'
        f"Text:\n{content[:2000]}"
    )

    try:
        response = llm.generate(prompt=prompt)
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            return []
        pairs = json.loads(match.group())
        result = []
        for pair in pairs:
            if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                q = str(pair["question"]).strip()
                a = str(pair["answer"]).strip()
                if q and a:
                    result.append({"question": q, "answer": a})
        return result[:n]
    except Exception as exc:
        logger.info(
            "    QA generation failed for doc (source=%s): %s",
            doc.metadata.get("source", "?"),
            exc,
        )
        return []


def _generate_all_qa(
    docs: List[Document],
    llm: Any,
    total: int,
    qa_per_doc: int,
    cache_path: Path,
    force: bool,
) -> List[Dict[str, str]]:
    """Generate QA pairs across documents. Loads from cache unless force=True."""
    if not force and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        logger.info("  Loaded %d QA pairs from cache (%s).", len(cached), cache_path)
        return cached

    if not docs:
        return []

    # Distribute target evenly across docs, cap at qa_per_doc
    n_per_doc = min(qa_per_doc, max(1, total // len(docs)))
    # If we have fewer docs than needed to hit total, ask for more per doc
    if len(docs) * n_per_doc < total:
        n_per_doc = min(qa_per_doc, (total + len(docs) - 1) // len(docs))

    logger.info(
        "  Generating ~%d QA pairs from %d pages (%d per page)...",
        total,
        len(docs),
        n_per_doc,
    )

    all_pairs: List[Dict[str, str]] = []
    for i, doc in enumerate(docs, 1):
        pairs = _generate_qa_for_doc(doc, llm, n_per_doc)
        for pair in pairs:
            pair["source"] = doc.metadata.get("source", f"doc_{i}")
        all_pairs.extend(pairs)
        logger.info(
            "    [%d/%d] %s → %d pairs (total so far: %d)",
            i,
            len(docs),
            doc.metadata.get("source", "?"),
            len(pairs),
            len(all_pairs),
        )
        if len(all_pairs) >= total:
            break

    all_pairs = all_pairs[:total]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(all_pairs, indent=2))
    logger.info("  Saved %d QA pairs to cache (%s).", len(all_pairs), cache_path)

    return all_pairs


# ── Evaluation ────────────────────────────────────────────────────────────────


def _eval_pair(qa: Dict[str, str], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one synthetic QA pair against a RAG result."""
    retrieved_sources = [
        doc.metadata.get("source", "") for doc in rag_result.get("sources", [])
    ]
    gold_source = qa.get("source", "")
    source_recall = float(gold_source in retrieved_sources) if gold_source else 1.0
    return {
        "question": qa["question"],
        "gold_answer": qa["answer"],
        "predicted_answer": rag_result["answer"],
        "gold_source": gold_source,
        "retrieved_sources": retrieved_sources,
        "source_recall": source_recall,
        "exact_match": _exact_match(rag_result["answer"], qa["answer"]),
        "f1": _token_f1(rag_result["answer"], qa["answer"]),
        "latency": rag_result.get("latency", 0.0),
    }


def _run_evaluation(rag: RAG, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []
    for i, qa in enumerate(qa_pairs, 1):
        logger.info("  [%d/%d] %s", i, len(qa_pairs), qa["question"][:70])
        result = rag.query(qa["question"])
        rows.append(_eval_pair(qa, result))
    return rows


# ── Comparison table ──────────────────────────────────────────────────────────


def _print_results(rows: List[Dict[str, Any]], phase6_path: Optional[Path]) -> None:
    n = len(rows)
    metrics = {
        "source_recall": sum(r["source_recall"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "avg_latency": sum(r["latency"] for r in rows) / n,
        "num_samples": n,
    }

    logger.info("")
    logger.info("=" * 66)
    logger.info("Phase 7 — PDF Corpus Results")
    logger.info("=" * 66)
    logger.info("  Metric               Phase 7 (PDF)")
    logger.info("  " + "-" * 40)
    logger.info("  Source Recall        %.3f", metrics["source_recall"])
    logger.info("  Exact Match          %.3f", metrics["exact_match"])
    logger.info("  F1                   %.3f", metrics["f1"])
    logger.info("  Avg Latency          %.2fs", metrics["avg_latency"])
    logger.info("  Samples              %d", n)

    # Compare against Phase 6 if available
    if phase6_path and phase6_path.exists():
        p6 = json.loads(phase6_path.read_text())
        pm = p6["metrics"]
        logger.info("")
        logger.info("  Comparison vs Phase 6 (HotpotQA, shared_corpus):")
        logger.info(
            "  %-22s  %10s  %10s  %10s", "Metric", "Phase 6", "Phase 7", "Delta"
        )
        logger.info("  " + "-" * 58)
        for key, label in [
            ("context_recall", "Context/Src Recall"),
            ("f1", "F1"),
            ("exact_match", "Exact Match"),
        ]:
            p6_val = pm.get(key, 0.0)
            p7_val = metrics.get(
                key.replace("context_recall", "source_recall"), metrics.get(key, 0.0)
            )
            if key == "context_recall":
                p7_val = metrics["source_recall"]
            else:
                p7_val = metrics[key]
            diff = p7_val - p6_val
            sign = "+" if diff >= 0 else ""
            logger.info(
                "  %-22s  %10.3f  %10.3f  %s%.3f",
                label,
                p6_val,
                p7_val,
                sign,
                diff,
            )
        logger.info("")
        logger.info("  Note: Phase 6 used Wikipedia (clean, encyclopaedic).")
        logger.info("  Phase 7 uses real PDFs (noisy, structured differently).")
        logger.info("  Lower Phase 7 scores = pipeline doesn't fully generalise.")
        logger.info("  Similar scores = optimisations are domain-agnostic.")
    logger.info("=" * 66)
    logger.info("")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: PDF corpus evaluation with synthetic QA"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=str(DEFAULT_PDF_DIR),
        help="Directory containing PDF files (default: research/data/pdfs/)",
    )
    parser.add_argument(
        "--num_qa",
        type=int,
        default=50,
        help="Total number of synthetic QA pairs to generate (default: 50)",
    )
    parser.add_argument(
        "--qa_per_doc",
        type=int,
        default=5,
        help="Maximum QA pairs to generate per document page (default: 5)",
    )
    parser.add_argument(
        "--force_qa",
        action="store_true",
        help="Re-generate QA pairs even if cache exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluation even if result file exists",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)

    logger.info("=" * 55)
    logger.info("Phase 7 — PDF Corpus with Synthetic QA")
    logger.info("=" * 55)
    logger.info("  PDF directory: %s", pdf_dir)
    logger.info("  Target QA pairs: %d (max %d per page)", args.num_qa, args.qa_per_doc)
    logger.info(
        "  Pipeline: %s chunking / %s retrieval / %s reranking / %s generation",
        BEST_CHUNKING_STRATEGY,
        BEST_RETRIEVAL_STRATEGY,
        BEST_RERANKING_PROVIDER or "none",
        BEST_GENERATION_STRATEGY,
    )
    logger.info("")

    # Load cached result if available
    if not args.force and RESULT_PATH.exists():
        logger.info("  Result already exists: %s", RESULT_PATH)
        logger.info("  Use --force to re-run. Loading cached result for display.")
        cached = json.loads(RESULT_PATH.read_text())
        _print_results(cached["rows"], PHASE6_RESULT_PATH)
        return

    # Load PDFs
    logger.info("Loading PDFs...")
    if not pdf_dir.exists():
        logger.info("  PDF directory not found: %s", pdf_dir)
        logger.info("  Create it and add PDFs, then re-run:")
        logger.info("    mkdir -p %s", pdf_dir)
        sys.exit(1)

    docs = _load_pdfs(pdf_dir)
    if not docs:
        sys.exit(1)

    logger.info("  Loaded %d pages from PDFs.", len(docs))
    logger.info("")

    llm = make_llm()
    embedding = LocalAPIEmbedding()

    # Generate synthetic QA pairs
    logger.info("Generating synthetic QA pairs...")
    qa_pairs = _generate_all_qa(
        docs=docs,
        llm=llm,
        total=args.num_qa,
        qa_per_doc=args.qa_per_doc,
        cache_path=QA_CACHE_PATH,
        force=args.force_qa,
    )

    if not qa_pairs:
        logger.info("  No QA pairs generated. Check LLM connectivity and PDF content.")
        sys.exit(1)

    logger.info("  Using %d QA pairs for evaluation.", len(qa_pairs))
    logger.info("")

    # Build RAG and ingest documents
    logger.info("Building RAG pipeline and ingesting %d pages...", len(docs))
    config = _make_config()
    rag = RAG(config, embedding_provider=embedding, llm_provider=llm)
    ingest_stats = rag.ingest_documents(docs)
    logger.info(
        "  Ingested %d pages → %d chunks.",
        ingest_stats["source_documents"],
        ingest_stats["chunks"],
    )
    logger.info("")

    # Evaluate
    logger.info("Evaluating %d QA pairs...", len(qa_pairs))
    t0 = time.time()
    rows = _run_evaluation(rag, qa_pairs)
    elapsed = time.time() - t0
    logger.info("  Evaluation complete in %.1fs.", elapsed)
    logger.info("")

    # Save results
    n = len(rows)
    metrics = {
        "source_recall": sum(r["source_recall"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "avg_latency": sum(r["latency"] for r in rows) / n,
        "num_samples": n,
    }
    output = {
        "experiment": "phase7_pdf_corpus",
        "pdf_dir": str(pdf_dir),
        "num_pages": len(docs),
        "pipeline": {
            "chunking": BEST_CHUNKING_STRATEGY,
            "chunk_size": BEST_CHUNK_SIZE,
            "chunk_overlap": BEST_CHUNK_OVERLAP,
            "retrieval": BEST_RETRIEVAL_STRATEGY,
            "reranking": BEST_RERANKING_PROVIDER,
            "generation": BEST_GENERATION_STRATEGY,
        },
        "metrics": metrics,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2))
    logger.info("  Results saved to %s", RESULT_PATH)

    _print_results(rows, PHASE6_RESULT_PATH)


if __name__ == "__main__":
    main()

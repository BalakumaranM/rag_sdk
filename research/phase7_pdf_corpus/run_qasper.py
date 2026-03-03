"""Phase 7 — QASPER as Expert-Annotated PDF Corpus
====================================================
Phase 7's original design used synthetically generated Q&A pairs from
domain PDFs.  QASPER is the real-data equivalent: expert-annotated Q&A
pairs from actual NLP research papers.

Relationship to the original Phase 7
--------------------------------------
  Original Phase 7  : Best combo on local PDFs + LLM-generated QA
  This script        : Best combo on QASPER  + expert-annotated QA

QASPER is strictly better for this purpose:
- QA pairs written by NLP researchers who read the paper (not an LLM)
- Diverse answer types: extractive spans, abstractive summaries, yes/no
- Ground truth evidence paragraphs — no LLM judge needed
- 100 papers × ~3 questions = rich evaluation set

This script uses the same best-combo configuration as Phase 6 QASPER and
serves as the cross-dataset validation: does the best pipeline hold up
on NLP domain text compared to HotpotQA Wikipedia text?

Update the four variables below to the Phase 6 QASPER winners before
running the definitive experiment.

Run
---
  .venv/bin/python research/phase7_pdf_corpus/run_qasper.py
  .venv/bin/python research/phase7_pdf_corpus/run_qasper.py --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_sdk.document import TextSplitter, inspect_chunks

from research.shared.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    NUM_QASPER_QUESTIONS,
    RESULTS_DIR,
    TOP_K,
)
from research.shared.providers import LocalAPIEmbedding, make_llm
from research.shared.qasper_dataset import load_qasper
from research.shared.qasper_harness import run_qasper_experiment
from rag_sdk.config import ConfigLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

# ── Set to Phase 6 QASPER winners ─────────────────────────────────────────────
CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
RETRIEVAL_STRATEGY = "dense"
RERANKING_STRATEGY: Optional[str] = None
GENERATION_STRATEGY = "standard"


def _make_config() -> Any:
    reranking_dict: dict = {"enabled": RERANKING_STRATEGY is not None}
    if RERANKING_STRATEGY:
        reranking_dict["strategy"] = RERANKING_STRATEGY
        reranking_dict["top_n"] = TOP_K

    return ConfigLoader.from_dict(
        {
            "document_processing": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunking": {"strategy": CHUNKING_STRATEGY},
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
            "retrieval": {"strategy": RETRIEVAL_STRATEGY, "top_k": TOP_K},
            "reranking": reranking_dict,
            "generation": {"strategy": GENERATION_STRATEGY},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: QASPER as expert-annotated PDF corpus"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 7 — Expert-Annotated PDF Corpus  [QASPER]")
    logger.info("=" * 60)
    logger.info("  Dataset:     QASPER dev (NLP papers, expert QA)")
    logger.info(
        "  Chunking:    %s (%d/%d)", CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP
    )
    logger.info("  Retrieval:   %s", RETRIEVAL_STRATEGY)
    logger.info("  Reranking:   %s", RERANKING_STRATEGY or "None")
    logger.info("  Generation:  %s", GENERATION_STRATEGY)
    logger.info("  Questions:   %d", NUM_QASPER_QUESTIONS)
    logger.info("  Mode:        per_question (each question on its own paper)")
    logger.info("")

    out_path = RESULTS_DIR / "phase7_qasper.json"
    if not args.force and out_path.exists():
        logger.info("Result already exists: %s", out_path)
        logger.info("Use --force to re-run.")
        return

    samples = load_qasper(num_questions=NUM_QASPER_QUESTIONS)

    # ── Chunk inspection on the first sample ─────────────────────────────────
    logger.info("── Chunk inspection (first paper) ───────────────────────────")
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    first_docs = samples[0].to_documents()
    report = inspect_chunks(first_docs, splitter)
    report.summary()
    logger.info("")

    result = run_qasper_experiment(
        config=_make_config(),
        samples=samples,
        embedding_provider=LocalAPIEmbedding(),
        llm_provider=make_llm(),
        experiment_name="phase7_qasper",
        results_dir=RESULTS_DIR,
        mode="per_question",
    )

    m = result["metrics"]
    logger.info("\n%s", "=" * 60)
    logger.info("Results  [QASPER — Expert PDF Corpus]")
    logger.info("%s", "=" * 60)
    logger.info("  Context Recall:    %.3f", m["context_recall"])
    logger.info("  Context Precision: %.3f", m["context_precision"])
    logger.info("  MRR:               %.3f", m["mrr"])
    logger.info("  Hit Rate:          %.3f", m["hit_rate"])
    logger.info("  Exact Match:       %.3f", m["exact_match"])
    logger.info("  F1:                %.3f", m["f1"])
    logger.info("  Avg Latency:       %.2fs", m["avg_latency"])

    by_type = m.get("by_answer_type", {})
    if by_type:
        logger.info("")
        logger.info("  By answer type:")
        for atype, agg in by_type.items():
            if agg.get("num_samples", 0) > 0:
                logger.info(
                    "    %-12s  Recall=%.3f  EM=%.3f  F1=%.3f  n=%d",
                    atype,
                    agg["context_recall"],
                    agg["exact_match"],
                    agg["f1"],
                    agg["num_samples"],
                )

    logger.info("\n  Full results: %s", result["output_file"])


if __name__ == "__main__":
    main()

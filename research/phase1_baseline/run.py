"""Phase 1 — Baseline
====================
The simplest possible RAG pipeline. Every subsequent phase changes exactly
one variable against this reference point.

Configuration
-------------
  Chunking:   TextSplitter  (chunk_size=512, overlap=50)
  Retrieval:  Dense (semantic search)
  Reranking:  None
  Generation: Standard
  Embedding:  Local API
  LLM:        Local Ollama
  VectorDB:   InMemoryVectorStore (fresh per question)
  Dataset:    HotpotQA dev distractor (100 questions)

Run
---
  cd /path/to/rag_sdk
  .venv/bin/python research/phase1_baseline/run.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.shared.config import NUM_EVAL_QUESTIONS, RESULTS_DIR
from research.shared.dataset import load_hotpotqa
from research.shared.harness import run_experiment
from research.shared.providers import LocalAPIEmbedding, make_config, make_llm

# Suppress SDK INFO noise; show only research output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def main() -> None:
    logger.info("=" * 55)
    logger.info("Phase 1 — Baseline")
    logger.info("=" * 55)
    logger.info("  Chunking:   TextSplitter (chunk_size=512, overlap=50)")
    logger.info("  Retrieval:  Dense")
    logger.info("  Generation: Standard")
    logger.info("  Questions:  %d", NUM_EVAL_QUESTIONS)
    logger.info("")

    samples = load_hotpotqa(num_questions=NUM_EVAL_QUESTIONS)

    config = make_config(
        chunking_strategy="recursive",
        retrieval_strategy="dense",
        generation_strategy="standard",
        chunk_size=512,
        chunk_overlap=50,
    )

    result = run_experiment(
        config=config,
        samples=samples,
        embedding_provider=LocalAPIEmbedding(),
        llm_provider=make_llm(),
        experiment_name="phase1_baseline",
        results_dir=RESULTS_DIR,
    )

    m = result["metrics"]
    logger.info("\n%s", "=" * 55)
    logger.info("Results")
    logger.info("%s", "=" * 55)
    logger.info("  Context Recall:    %.3f", m["context_recall"])
    logger.info("  Context Precision: %.3f", m["context_precision"])
    logger.info("  Exact Match:       %.3f", m["exact_match"])
    logger.info("  F1:                %.3f", m["f1"])
    logger.info("  Avg Latency:       %.2fs", m["avg_latency"])
    logger.info("\n  Full results: %s", result["output_file"])


if __name__ == "__main__":
    main()

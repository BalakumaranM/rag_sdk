"""Phase 6 — Best-of-Breed Combination (QASPER)
================================================
Combines the winning components from each QASPER phase and compares
against the QASPER Phase 1 baseline.

Update the four variables below to the actual QASPER phase winners
before running the definitive experiment.

Configuration
-------------
  CHUNKING_STRATEGY  ← Phase 2 QASPER winner
  RETRIEVAL_STRATEGY ← Phase 3 QASPER winner
  RERANKING_STRATEGY ← Phase 4 QASPER winner (or None)
  GENERATION_STRATEGY← Phase 5 QASPER winner

Run
---
  .venv/bin/python research/phase6_best_combo/run_qasper.py
  .venv/bin/python research/phase6_best_combo/run_qasper.py --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

# ── Fill in QASPER phase winners before running ───────────────────────────────
CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
RETRIEVAL_STRATEGY = "dense"
RERANKING_STRATEGY: Optional[str] = None  # e.g. "cross_encoder" or None
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
    parser = argparse.ArgumentParser(description="Phase 6: Best-of-Breed [QASPER]")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 6 — Best-of-Breed Combination  [QASPER]")
    logger.info("=" * 60)
    logger.info(
        "  Chunking:    %s (%d/%d)", CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP
    )
    logger.info("  Retrieval:   %s", RETRIEVAL_STRATEGY)
    logger.info("  Reranking:   %s", RERANKING_STRATEGY or "None")
    logger.info("  Generation:  %s", GENERATION_STRATEGY)
    logger.info("  Questions:   %d", NUM_QASPER_QUESTIONS)
    logger.info("  Mode:        shared_corpus")
    logger.info("")

    out_path = RESULTS_DIR / "phase6_best_combo_qasper.json"
    if not args.force and out_path.exists():
        logger.info("Result already exists: %s", out_path)
        logger.info("Use --force to re-run.")
        return

    samples = load_qasper(num_questions=NUM_QASPER_QUESTIONS)

    result = run_qasper_experiment(
        config=_make_config(),
        samples=samples,
        embedding_provider=LocalAPIEmbedding(),
        llm_provider=make_llm(),
        experiment_name="phase6_best_combo",
        results_dir=RESULTS_DIR,
        mode="shared_corpus",
    )

    m = result["metrics"]
    logger.info("\n%s", "=" * 60)
    logger.info("Results  [QASPER]")
    logger.info("%s", "=" * 60)
    logger.info("  Context Recall:    %.3f", m["context_recall"])
    logger.info("  Context Precision: %.3f", m["context_precision"])
    logger.info("  MRR:               %.3f", m["mrr"])
    logger.info("  Hit Rate:          %.3f", m["hit_rate"])
    logger.info("  Exact Match:       %.3f", m["exact_match"])
    logger.info("  F1:                %.3f", m["f1"])
    logger.info("  Avg Latency:       %.2fs", m["avg_latency"])
    logger.info("\n  Full results: %s", result["output_file"])


if __name__ == "__main__":
    main()

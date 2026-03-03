"""Phase 1 — Baseline (QASPER)
==============================
Identical configuration to phase1_baseline/run.py but evaluated on QASPER:
NLP research papers instead of Wikipedia paragraphs.

Why QASPER changes the baseline numbers
-----------------------------------------
HotpotQA documents are pre-extracted Wikipedia paragraphs (~300–800 chars).
Most fit in a single 512-char chunk, so chunking makes little difference.

QASPER papers are 6,000–12,000 tokens with 20–50 paragraphs each.
Even with 512-char chunks a paper produces 40–100 chunks — the retriever
must find the 1–2 evidence paragraphs among a much larger, noisier pool.
This baseline tells us the reference point on that harder task.

inspect_chunks is run on the first sample before the experiment so you can
see what the baseline splitter produces on a real QASPER paper.

Configuration
-------------
  Chunking:   TextSplitter  (chunk_size=512, overlap=50)
  Retrieval:  Dense (semantic search)
  Reranking:  None
  Generation: Standard
  Dataset:    QASPER dev (100 non-unanswerable QA pairs)
  Mode:       per_question (each question runs on its own paper)

Run
---
  .venv/bin/python research/phase1_baseline/run_qasper.py

  # Save chunk CSV for offline browsing (opens in Excel / pandas):
  .venv/bin/python research/phase1_baseline/run_qasper.py --chunks-out research/results/chunks_phase1.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_sdk.document import TextSplitter, inspect_chunks

from research.shared.config import NUM_QASPER_QUESTIONS, RESULTS_DIR
from research.shared.providers import LocalAPIEmbedding, make_config, make_llm
from research.shared.qasper_dataset import load_qasper
from research.shared.qasper_harness import run_qasper_experiment

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Baseline [QASPER]")
    parser.add_argument(
        "--chunks-out",
        metavar="PATH",
        help="Save chunk CSV for offline browsing (Excel / pandas). "
        "Example: research/results/chunks_phase1.csv",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 1 — Baseline  [QASPER]")
    logger.info("=" * 60)
    logger.info("  Chunking:   TextSplitter (chunk_size=512, overlap=50)")
    logger.info("  Retrieval:  Dense")
    logger.info("  Generation: Standard")
    logger.info("  Questions:  %d", NUM_QASPER_QUESTIONS)
    logger.info("")

    samples = load_qasper(num_questions=NUM_QASPER_QUESTIONS)

    # ── Chunk inspection on the first sample ─────────────────────────────────
    # Shows what 512-char splitting produces on a real paper vs HotpotQA.
    logger.info("── Chunk inspection (first paper, TextSplitter 512/50) ──────")
    splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
    sample_docs = samples[0].to_documents()
    report = inspect_chunks(sample_docs, splitter)
    report.summary()

    if args.chunks_out:
        csv_path = report.to_csv(args.chunks_out)
        logger.info("  Chunks saved → %s", csv_path)
    logger.info("")

    # ── Run experiment ────────────────────────────────────────────────────────
    config = make_config(
        chunking_strategy="recursive",
        retrieval_strategy="dense",
        generation_strategy="standard",
        chunk_size=512,
        chunk_overlap=50,
    )

    result = run_qasper_experiment(
        config=config,
        samples=samples,
        embedding_provider=LocalAPIEmbedding(),
        llm_provider=make_llm(),
        experiment_name="phase1_baseline",
        results_dir=RESULTS_DIR,
        mode="per_question",
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

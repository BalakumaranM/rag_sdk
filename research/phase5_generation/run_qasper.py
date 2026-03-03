"""Phase 5 — Generation Strategy Ablation (QASPER)
===================================================
Same 3 generation variants as phase5_generation/run.py, evaluated on QASPER.

QASPER answers tend to be longer and more technical than HotpotQA answers,
making generation strategy more impactful — CoVe claim verification can catch
more hallucinations, and attributed generation is especially useful for NLP
paper claims where precise source tracing matters.

Enable --faithfulness to measure hallucination rates directly (costs two
extra LLM calls per question — run on a smaller subset if needed).

Variants
--------
  5a  Standard          baseline: context → LLM → answer
  5b  ChainOfVerification (CoVe)  hallucination reduction via verification loop
  5c  Attributed        inline citations [N] for verifiability

Usage
-----
  .venv/bin/python research/phase5_generation/run_qasper.py
  .venv/bin/python research/phase5_generation/run_qasper.py --faithfulness
  .venv/bin/python research/phase5_generation/run_qasper.py --variants 5a,5b
  .venv/bin/python research/phase5_generation/run_qasper.py --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Update to the Phase 3 winner before running the definitive experiment.
RETRIEVAL_STRATEGY = "dense"
CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

VARIANTS: List[Dict[str, Any]] = [
    {"id": "5a", "label": "Standard", "strategy": "standard"},
    {"id": "5b", "label": "ChainOfVerification", "strategy": "chain_of_verification"},
    {"id": "5c", "label": "Attributed", "strategy": "attributed"},
]
VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


def _experiment_name(variant_id: str) -> str:
    return f"phase5_generation_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json"
    return json.loads(path.read_text()) if path.exists() else None


def _make_config(generation_strategy: str) -> Any:
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
            "generation": {"strategy": generation_strategy},
        }
    )


def _print_table(rows: List[Dict[str, Any]], has_faithfulness: bool) -> None:
    if not rows:
        return
    col = 22
    faith_header = "  %6s" % "Faith" if has_faithfulness else ""
    logger.info("")
    logger.info("=" * 75)
    logger.info("Phase 5 — Generation Ablation Results  [QASPER]")
    logger.info("=" * 75)
    logger.info(
        "  %-*s  %6s  %6s  %6s  %6s%s  %8s",
        col,
        "Variant",
        "Recall",
        "Prec",
        "EM",
        "F1",
        faith_header,
        "Latency",
    )
    logger.info("  " + "-" * 71)
    for row in rows:
        m = row["metrics"]
        faith_val = "  %6.3f" % m.get("faithfulness", 0.0) if has_faithfulness else ""
        logger.info(
            "  %-*s  %6.3f  %6.3f  %6.3f  %6.3f%s  %7.2fs",
            col,
            row["label"][:col],
            m["context_recall"],
            m["context_precision"],
            m["exact_match"],
            m["f1"],
            faith_val,
            m["avg_latency"],
        )
    logger.info("=" * 75)
    logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Generation Ablation [QASPER]"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variants", type=str, metavar="IDS")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="Enable per-row faithfulness scoring (costs 2 extra LLM calls per question)",
    )
    args = parser.parse_args()

    if args.variants:
        ids = [x.strip() for x in args.variants.split(",")]
        unknown = [i for i in ids if i not in VARIANT_BY_ID]
        if unknown:
            logger.info("Unknown variant IDs: %s", ", ".join(unknown))
            sys.exit(1)
        to_run = [VARIANT_BY_ID[i] for i in ids]
    else:
        to_run = VARIANTS

    logger.info("=" * 60)
    logger.info("Phase 5 — Generation Strategy Ablation  [QASPER]")
    logger.info("=" * 60)
    logger.info(
        "  Chunking:    %s (%d/%d) (fixed)",
        CHUNKING_STRATEGY,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    logger.info("  Retrieval:   %s (fixed)", RETRIEVAL_STRATEGY)
    logger.info("  Faithfulness eval: %s", "ON" if args.faithfulness else "OFF")
    logger.info("  Questions:   %d", NUM_QASPER_QUESTIONS)
    logger.info("")

    samples = load_qasper(num_questions=NUM_QASPER_QUESTIONS)
    embedding = LocalAPIEmbedding()
    llm = make_llm()
    completed: List[Dict[str, Any]] = []

    for variant in to_run:
        vid = variant["id"]

        if not args.force and _result_exists(vid):
            cached = _load_result(vid)
            if cached is not None:
                logger.info("  [%s] Loaded from cache.", vid)
                completed.append({**cached, "label": variant["label"]})
                continue

        logger.info("  [%s] Running %s …", vid, variant["label"])
        result = run_qasper_experiment(
            config=_make_config(variant["strategy"]),
            samples=samples,
            embedding_provider=embedding,
            llm_provider=llm,
            experiment_name=_experiment_name(vid),
            results_dir=RESULTS_DIR,
            mode="per_question",
            eval_faithfulness=args.faithfulness,
        )
        completed.append({**result, "label": variant["label"]})
        m = result["metrics"]
        logger.info(
            "  [%s] Done — Recall=%.3f  EM=%.3f  F1=%.3f  %.2fs",
            vid,
            m["context_recall"],
            m["exact_match"],
            m["f1"],
            m["avg_latency"],
        )

    _print_table(completed, has_faithfulness=args.faithfulness)


if __name__ == "__main__":
    main()

"""Phase 4 — Reranking Ablation (QASPER)
=========================================
Same 3 reranking variants as phase4_reranking/run.py, evaluated on QASPER.

Update RETRIEVAL_STRATEGY below to the Phase 3 QASPER winner before running
the definitive experiment.

Variants
--------
  4a  No reranking        retriever ordering only (baseline)
  4b  CrossEncoder        local cross-encoder model (sentence-transformers)
  4c  CohereReranker      Cohere Rerank API (requires COHERE_API_KEY env var)

Usage
-----
  .venv/bin/python research/phase4_reranking/run_qasper.py
  .venv/bin/python research/phase4_reranking/run_qasper.py --variants 4a,4b
  .venv/bin/python research/phase4_reranking/run_qasper.py --force
"""

import argparse
import json
import logging
import os
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

# Update to Phase 3 QASPER winner before running the definitive experiment.
RETRIEVAL_STRATEGY = "dense"

VARIANTS: List[Dict[str, Any]] = [
    {"id": "4a", "label": "No reranking", "reranking": None},
    {
        "id": "4b",
        "label": "CrossEncoder",
        "reranking": "cross_encoder",
        "needs_cross_encoder": True,
    },
    {
        "id": "4c",
        "label": "CohereReranker",
        "reranking": "cohere",
        "needs_cohere_key": True,
    },
]
VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


def _experiment_name(variant_id: str) -> str:
    return f"phase4_reranking_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json"
    return json.loads(path.read_text()) if path.exists() else None


def _make_config(reranking: Optional[str]) -> Any:
    reranking_dict: Dict[str, Any] = {"enabled": reranking is not None}
    if reranking == "cross_encoder":
        reranking_dict["strategy"] = "cross_encoder"
        reranking_dict["top_n"] = TOP_K
    elif reranking == "cohere":
        reranking_dict["strategy"] = "cohere"
        reranking_dict["top_n"] = TOP_K

    return ConfigLoader.from_dict(
        {
            "document_processing": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunking": {"strategy": "recursive"},
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
            "generation": {"strategy": "standard"},
        }
    )


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 18
    logger.info("")
    logger.info("=" * 75)
    logger.info("Phase 4 — Reranking Ablation Results  [QASPER]")
    logger.info("=" * 75)
    logger.info(
        "  %-*s  %6s  %6s  %6s  %6s  %6s  %8s",
        col,
        "Variant",
        "Recall",
        "Prec",
        "MRR",
        "Hit",
        "F1",
        "Latency",
    )
    logger.info("  " + "-" * 71)
    for row in rows:
        m = row["metrics"]
        logger.info(
            "  %-*s  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %7.2fs",
            col,
            row["label"][:col],
            m["context_recall"],
            m["context_precision"],
            m["mrr"],
            m["hit_rate"],
            m["f1"],
            m["avg_latency"],
        )
    logger.info("=" * 75)
    best_mrr = max(rows, key=lambda r: r["metrics"]["mrr"])
    logger.info("  Best MRR: %s", best_mrr["label"])
    logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Reranking Ablation [QASPER]")
    parser.add_argument("--variants", type=str, metavar="IDS")
    parser.add_argument("--force", action="store_true")
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
    logger.info("Phase 4 — Reranking Ablation  [QASPER]")
    logger.info("=" * 60)
    logger.info("  Retrieval:  %s (fixed)", RETRIEVAL_STRATEGY)
    logger.info("  Chunking:   TextSplitter(512, 50) (fixed)")
    logger.info("  Questions:  %d", NUM_QASPER_QUESTIONS)
    logger.info("  Mode:       shared_corpus")
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

        if variant.get("needs_cohere_key") and not os.getenv("COHERE_API_KEY"):
            logger.info("  [%s] Skipping — COHERE_API_KEY not set.", vid)
            continue

        logger.info("  [%s] Running %s …", vid, variant["label"])
        try:
            result = run_qasper_experiment(
                config=_make_config(variant["reranking"]),
                samples=samples,
                embedding_provider=embedding,
                llm_provider=llm,
                experiment_name=_experiment_name(vid),
                results_dir=RESULTS_DIR,
                mode="shared_corpus",
            )
        except ImportError as exc:
            logger.info("  [%s] Skipped — missing dependency: %s", vid, exc)
            continue

        completed.append({**result, "label": variant["label"]})
        m = result["metrics"]
        logger.info(
            "  [%s] Done — Recall=%.3f  MRR=%.3f  F1=%.3f  %.2fs",
            vid,
            m["context_recall"],
            m["mrr"],
            m["f1"],
            m["avg_latency"],
        )

    _print_table(completed)


if __name__ == "__main__":
    main()

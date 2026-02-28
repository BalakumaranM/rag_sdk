"""Phase 6 — Best-of-Breed Combination
========================================
Combines the winners from Phases 2–5 into a single pipeline and measures the
total improvement over the Phase 1 configuration.

Both variants run in shared_corpus mode so the comparison is apples-to-apples.
(Phase 1's original result used per_question mode — a different corpus size —
so we re-run the baseline configuration here under identical conditions.)

Variants
--------
  6a  Baseline       Phase 1 config re-run in shared_corpus mode (reference)
  6b  Best-of-breed  Phase 2+3+4+5 winners combined

Before running, update the winner constants below to the actual Phase 2–5 results:

  BEST_CHUNKING_STRATEGY  — from Phase 2 results (default: "recursive")
  BEST_CHUNK_SIZE         — from Phase 2 results (default: 512)
  BEST_CHUNK_OVERLAP      — from Phase 2 results (default: 50)
  BEST_RETRIEVAL_STRATEGY — from Phase 3 results (default: "dense")
  BEST_RERANKING_PROVIDER — from Phase 4 results (default: None)
  BEST_GENERATION_STRATEGY— from Phase 5 results (default: "standard")

Usage
-----
  .venv/bin/python research/phase6_best_combo/run.py

  # Re-run even if a result file exists:
  .venv/bin/python research/phase6_best_combo/run.py --force

  # Re-run only one variant:
  .venv/bin/python research/phase6_best_combo/run.py --variants 6b
  .venv/bin/python research/phase6_best_combo/run.py --variants 6b --force
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
    NUM_EVAL_QUESTIONS,
    RESULTS_DIR,
    TOP_K,
)
from research.shared.dataset import load_hotpotqa
from research.shared.harness import run_experiment
from research.shared.providers import LocalAPIEmbedding, make_llm
from rag_sdk.config import ConfigLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


# ── Winner constants — update after running Phases 2–5 ───────────────────────

# Phase 2 chunking winner
BEST_CHUNKING_STRATEGY = "recursive"  # e.g. "semantic", "proposition"
BEST_CHUNK_SIZE = 512  # e.g. 256, 1024
BEST_CHUNK_OVERLAP = 50  # e.g. 25, 100

# Phase 3 retrieval winner
BEST_RETRIEVAL_STRATEGY = "dense"  # e.g. "hybrid", "multi_query", "graph_rag"

# Phase 4 reranking winner (None = no reranking)
BEST_RERANKING_PROVIDER: Optional[str] = None  # e.g. "cross-encoder", "cohere"

# Phase 5 generation winner
BEST_GENERATION_STRATEGY = "standard"  # e.g. "cove", "attributed"


# ── Config builder ───────────────────────────────────────────────────────────


def _make_baseline_config() -> Any:
    """Phase 1 configuration — TextSplitter(512,50) + Dense + Standard + no reranking."""
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
            "retrieval": {
                "strategy": "dense",
                "top_k": TOP_K,
                "reranking": {"enabled": False},
            },
            "generation": {"strategy": "standard"},
        }
    )


def _make_best_config() -> Any:
    """Best-of-breed configuration — winners from Phases 2–5."""
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


# ── Variant registry ─────────────────────────────────────────────────────────

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "6a",
        "label": "Baseline (Phase 1 config)",
        "kind": "baseline",
        "note": "TextSplitter(512,50) + Dense + Standard + no reranking in shared_corpus mode",
    },
    {
        "id": "6b",
        "label": "Best-of-breed",
        "kind": "best",
        "note": "Phase 2+3+4+5 winners combined",
    },
]

VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _experiment_name(variant_id: str) -> str:
    return f"phase6_best_combo_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _delta(best: float, base: float) -> str:
    """Format a metric delta with sign and percentage."""
    diff = best - base
    pct = (diff / base * 100) if base > 0 else 0.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f} ({sign}{pct:.1f}%)"


def _print_comparison(baseline: Dict[str, Any], best: Dict[str, Any]) -> None:
    bm = baseline["metrics"]
    gm = best["metrics"]

    logger.info("")
    logger.info("=" * 72)
    logger.info("Phase 6 — Best-of-Breed vs Baseline")
    logger.info("=" * 72)
    logger.info("")
    logger.info("  Pipeline components:")
    logger.info(
        "  ┌──────────────┬────────────────────────────┬────────────────────────────┐"
    )
    logger.info(
        "  │ Component    │ Baseline (6a)              │ Best-of-breed (6b)         │"
    )
    logger.info(
        "  ├──────────────┼────────────────────────────┼────────────────────────────┤"
    )
    logger.info(
        "  │ Chunking     │ TextSplitter(512, 50)      │ %-26s │", _chunking_label()
    )
    logger.info(
        "  │ Retrieval    │ Dense                      │ %-26s │",
        BEST_RETRIEVAL_STRATEGY,
    )
    logger.info(
        "  │ Reranking    │ None                       │ %-26s │",
        BEST_RERANKING_PROVIDER or "None",
    )
    logger.info(
        "  │ Generation   │ Standard                   │ %-26s │",
        BEST_GENERATION_STRATEGY,
    )
    logger.info(
        "  └──────────────┴────────────────────────────┴────────────────────────────┘"
    )
    logger.info("")
    logger.info("  Metrics (shared_corpus mode, %d questions):", bm["num_samples"])
    logger.info("")
    logger.info("  %-20s  %8s  %8s  %12s", "Metric", "Baseline", "Best", "Delta")
    logger.info("  " + "-" * 58)
    for metric, label in [
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
        ("exact_match", "Exact Match"),
        ("f1", "F1"),
    ]:
        logger.info(
            "  %-20s  %8.3f  %8.3f  %s",
            label,
            bm[metric],
            gm[metric],
            _delta(gm[metric], bm[metric]),
        )
    logger.info("")
    logger.info(
        "  %-20s  %7.2fs  %7.2fs  %s",
        "Avg Latency",
        bm["avg_latency"],
        gm["avg_latency"],
        _delta(gm["avg_latency"], bm["avg_latency"]),
    )
    logger.info("")
    logger.info("=" * 72)
    logger.info("")

    # Verdict
    f1_gain = gm["f1"] - bm["f1"]
    recall_gain = gm["context_recall"] - bm["context_recall"]
    latency_cost = gm["avg_latency"] - bm["avg_latency"]

    if f1_gain > 0.05:
        verdict = "Strong improvement — optimisation clearly paid off."
    elif f1_gain > 0.01:
        verdict = "Moderate improvement — optimisation helped."
    elif f1_gain >= -0.01:
        verdict = (
            "Flat — optimised pipeline is no better than baseline on this dataset."
        )
    else:
        verdict = "Regression — check individual phase results to identify the culprit."

    logger.info("  Verdict: %s", verdict)
    logger.info(
        "  F1 gain: %+.3f  |  Recall gain: %+.3f  |  Latency cost: %+.2fs",
        f1_gain,
        recall_gain,
        latency_cost,
    )
    logger.info("")


def _chunking_label() -> str:
    strategy_names = {
        "recursive": f"TextSplitter({BEST_CHUNK_SIZE},{BEST_CHUNK_OVERLAP})",
        "semantic": "SemanticSplitter",
        "agentic": "AgenticSplitter",
        "proposition": "PropositionSplitter",
        "late": "LateSplitter",
    }
    return strategy_names.get(BEST_CHUNKING_STRATEGY, BEST_CHUNKING_STRATEGY)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Best-of-breed combination vs baseline"
    )
    parser.add_argument(
        "--variants",
        type=str,
        metavar="IDS",
        help="Comma-separated variant IDs to run (e.g. --variants 6b)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run variants even if a result file already exists",
    )
    args = parser.parse_args()

    # Select variants
    if args.variants:
        ids = [x.strip() for x in args.variants.split(",")]
        unknown = [i for i in ids if i not in VARIANT_BY_ID]
        if unknown:
            logger.info("Unknown variant IDs: %s", ", ".join(unknown))
            logger.info("Valid IDs: %s", ", ".join(VARIANT_BY_ID))
            sys.exit(1)
        to_run = [VARIANT_BY_ID[i] for i in ids]
    else:
        to_run = VARIANTS

    logger.info("=" * 55)
    logger.info("Phase 6 — Best-of-Breed Combination")
    logger.info("=" * 55)
    logger.info("  Mode: shared_corpus (%d questions)", NUM_EVAL_QUESTIONS)
    logger.info("")
    logger.info("  Best-of-breed pipeline:")
    logger.info("    Chunking:   %s", _chunking_label())
    logger.info("    Retrieval:  %s", BEST_RETRIEVAL_STRATEGY)
    logger.info("    Reranking:  %s", BEST_RERANKING_PROVIDER or "None")
    logger.info("    Generation: %s", BEST_GENERATION_STRATEGY)
    logger.info("")
    logger.info("  Variants scheduled:")
    for v in to_run:
        tags = []
        if not args.force and _result_exists(v["id"]):
            tags.append("cached")
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        logger.info("    %s  %s%s", v["id"], v["label"], tag_str)
    logger.info("")

    samples = load_hotpotqa(num_questions=NUM_EVAL_QUESTIONS)
    embedding = LocalAPIEmbedding()
    llm = make_llm()
    completed: Dict[str, Dict[str, Any]] = {}

    for variant in to_run:
        vid = variant["id"]

        # Use cached result if available and --force not set
        if not args.force and _result_exists(vid):
            cached = _load_result(vid)
            if cached is not None:
                logger.info("  [%s] Loaded from cache.", vid)
                completed[vid] = cached
                continue

        logger.info("  [%s] Running %s ...", vid, variant["label"])

        config = (
            _make_baseline_config()
            if variant["kind"] == "baseline"
            else _make_best_config()
        )

        result = run_experiment(
            config=config,
            samples=samples,
            embedding_provider=embedding,
            llm_provider=llm,
            experiment_name=_experiment_name(vid),
            results_dir=RESULTS_DIR,
            mode="shared_corpus",
        )

        completed[vid] = result
        m = result["metrics"]
        logger.info(
            "  [%s] Done — Recall=%.3f  Prec=%.3f  EM=%.3f  F1=%.3f  %.2fs",
            vid,
            m["context_recall"],
            m["context_precision"],
            m["exact_match"],
            m["f1"],
            m["avg_latency"],
        )

    # Print comparison only when both results are available
    if "6a" in completed and "6b" in completed:
        _print_comparison(completed["6a"], completed["6b"])
    elif completed:
        # Only one variant ran — show what we have
        for vid, result in completed.items():
            m = result["metrics"]
            logger.info(
                "  [%s] Recall=%.3f  Prec=%.3f  EM=%.3f  F1=%.3f  %.2fs",
                vid,
                m["context_recall"],
                m["context_precision"],
                m["exact_match"],
                m["f1"],
                m["avg_latency"],
            )
        logger.info("")
        logger.info("  Run both 6a and 6b to see the full comparison table.")


if __name__ == "__main__":
    main()

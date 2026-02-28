"""Phase 4 — Reranking Ablation
================================
Varies the reranking strategy across 3 variants. Everything else is held constant:
TextSplitter(512, 50) chunking, Dense retrieval (Phase 3 baseline), standard
generation, local embedding API, local LLM.

NOTE: Update ``RETRIEVAL_STRATEGY`` below to the Phase 3 winner before running
the definitive experiment.

When reranking is enabled, rag.query() automatically over-fetches top_k * 3
candidates from the retriever, then the reranker trims them back to top_k.
This gives the reranker more material to work with.

Variants
--------
  4a  No reranking        retriever ordering only (baseline)
  4b  CrossEncoder        local cross-encoder model (sentence-transformers)
  4c  CohereReranker      Cohere Rerank API (requires COHERE_API_KEY env var)

Usage
-----
  .venv/bin/python research/phase4_reranking/run.py

  # Specific variants:
  .venv/bin/python research/phase4_reranking/run.py --variants 4a,4b

  # Re-run even if a result file exists:
  .venv/bin/python research/phase4_reranking/run.py --force
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


# ── Retrieval strategy ───────────────────────────────────────────────────────
# Set this to the Phase 3 winner before running the definitive experiment.
# Default: "dense" (Phase 3 baseline).
RETRIEVAL_STRATEGY = "dense"


# ── Config builder ───────────────────────────────────────────────────────────


def _make_config(reranking_dict: Dict[str, Any]) -> Any:
    """Build a Config with the Phase 3 winner retrieval and the given reranking section."""
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
                "strategy": RETRIEVAL_STRATEGY,
                "top_k": TOP_K,
                "reranking": reranking_dict,
            },
            "generation": {"strategy": "standard"},
        }
    )


# ── Variant registry ─────────────────────────────────────────────────────────

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "4a",
        "label": "No Reranking",
        "reranking": {"enabled": False},
        "needs_sentence_transformers": False,
        "needs_cohere_api": False,
        "note": "baseline — retriever ordering only, no post-processing",
    },
    {
        "id": "4b",
        "label": "CrossEncoderReranker",
        "reranking": {"enabled": True, "provider": "cross-encoder"},
        "needs_sentence_transformers": True,
        "needs_cohere_api": False,
        "note": (
            "local cross-encoder (ms-marco-MiniLM-L-6-v2); "
            "jointly scores query+doc pairs; over-fetches 3×, reranks to top-5"
        ),
    },
    {
        "id": "4c",
        "label": "CohereReranker",
        "reranking": {"enabled": True, "provider": "cohere"},
        "needs_sentence_transformers": False,
        "needs_cohere_api": True,
        "note": "Cohere Rerank API (rerank-v3.5); requires COHERE_API_KEY env var",
    },
]

VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _has_sentence_transformers() -> bool:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cohere_api_key() -> bool:
    return bool(os.getenv("COHERE_API_KEY", ""))


def _has_cohere_package() -> bool:
    try:
        import cohere  # noqa: F401

        return True
    except ImportError:
        return False


def _experiment_name(variant_id: str) -> str:
    return f"phase4_reranking_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 22
    logger.info("")
    logger.info("=" * 72)
    logger.info("Phase 4 — Reranking Ablation Results")
    logger.info("=" * 72)
    logger.info(
        "  %-*s  %6s  %6s  %6s  %6s  %8s",
        col,
        "Variant",
        "Recall",
        "Prec",
        "EM",
        "F1",
        "Latency",
    )
    logger.info("  " + "-" * 68)
    for row in rows:
        m = row["metrics"]
        logger.info(
            "  %-*s  %6.3f  %6.3f  %6.3f  %6.3f  %7.2fs",
            col,
            row["label"][:col],
            m["context_recall"],
            m["context_precision"],
            m["exact_match"],
            m["f1"],
            m["avg_latency"],
        )
    logger.info("=" * 72)
    logger.info("")
    best_recall = max(rows, key=lambda r: r["metrics"]["context_recall"])
    best_prec = max(rows, key=lambda r: r["metrics"]["context_precision"])
    best_f1 = max(rows, key=lambda r: r["metrics"]["f1"])
    logger.info("  Best context_recall:    %s", best_recall["label"])
    logger.info("  Best context_precision: %s", best_prec["label"])
    logger.info("  Best F1:                %s", best_f1["label"])
    logger.info("")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Reranking Ablation")
    parser.add_argument(
        "--variants",
        type=str,
        metavar="IDS",
        help="Comma-separated variant IDs to run (e.g. --variants 4a,4b)",
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
    logger.info("Phase 4 — Reranking Ablation")
    logger.info("=" * 55)
    logger.info("  Chunking:   TextSplitter(512, 50) [Phase 2 baseline]")
    logger.info(
        "  Retrieval:  %s [Phase 3 winner — update RETRIEVAL_STRATEGY]",
        RETRIEVAL_STRATEGY,
    )
    logger.info("  Generation: Standard (fixed)")
    logger.info("  Mode:       shared_corpus")
    logger.info("  Questions:  %d", NUM_EVAL_QUESTIONS)
    logger.info("  Over-fetch: top_k × 3 = %d → rerank to %d", TOP_K * 3, TOP_K)
    logger.info("")
    logger.info("  Variants scheduled:")
    for v in to_run:
        tags = []
        if v["needs_sentence_transformers"] and not _has_sentence_transformers():
            tags.append("missing: sentence-transformers")
        if v["needs_cohere_api"]:
            if not _has_cohere_package():
                tags.append("missing: cohere package")
            elif not _has_cohere_api_key():
                tags.append("missing: COHERE_API_KEY")
        if not args.force and _result_exists(v["id"]):
            tags.append("cached")
        tag_str = "  [" + ", ".join(tags) + "]" if tags else ""
        logger.info("    %s  %s%s", v["id"], v["label"], tag_str)
    logger.info("")

    samples = load_hotpotqa(num_questions=NUM_EVAL_QUESTIONS)
    embedding = LocalAPIEmbedding()
    llm = make_llm()
    completed: List[Dict[str, Any]] = []

    for variant in to_run:
        vid = variant["id"]

        # Use cached result if available and --force not set
        if not args.force and _result_exists(vid):
            cached = _load_result(vid)
            if cached is not None:
                logger.info("  [%s] Loaded from cache.", vid)
                completed.append({**cached, "label": variant["label"]})
                continue

        # Dependency checks
        if variant["needs_sentence_transformers"] and not _has_sentence_transformers():
            logger.info(
                "  [%s] Skipping — sentence-transformers not installed. "
                "Install with: pip install sentence-transformers",
                vid,
            )
            continue

        if variant["needs_cohere_api"]:
            if not _has_cohere_package():
                logger.info(
                    "  [%s] Skipping — cohere package not installed. "
                    "Install with: pip install cohere",
                    vid,
                )
                continue
            if not _has_cohere_api_key():
                logger.info(
                    "  [%s] Skipping — COHERE_API_KEY not set in environment.",
                    vid,
                )
                continue

        logger.info("  [%s] Running %s ...", vid, variant["label"])

        config = _make_config(variant["reranking"])

        try:
            result = run_experiment(
                config=config,
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
            "  [%s] Done — Recall=%.3f  Prec=%.3f  EM=%.3f  F1=%.3f  %.2fs",
            vid,
            m["context_recall"],
            m["context_precision"],
            m["exact_match"],
            m["f1"],
            m["avg_latency"],
        )

    _print_table(completed)


if __name__ == "__main__":
    main()

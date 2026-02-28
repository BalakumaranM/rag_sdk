"""Phase 3 — Retrieval Strategy Ablation
==========================================
Varies the retrieval strategy across 12 variants. Everything else is held constant:
TextSplitter(512, 50) chunking, standard generation, local embedding API, local LLM.

Evaluation mode: shared_corpus — all unique Wikipedia articles are ingested once
and every sample queries the same pool (~200 unique articles, ~600–1000 chunks).
This is more realistic than Phase 2's per-question mode: the retriever must find
the 2 gold articles among ~200 candidates instead of 10.

Variants
--------
  3a  Dense                       baseline — cosine similarity
  3b  BM25                        keyword-only (hybrid w/ bm25_weight=1.0)
  3c  Hybrid                      dense + BM25 with RRF (50/50)
  3d  MultiQuery                  LLM expands query into 3 variants, unions results
  3e  SelfRAG                     adaptive — LLM decides if retrieval is needed
  3f  ContextualCompression       dense + LLM extracts relevant portions per chunk
  3g  CorrectiveRAG               dense + LLM relevance check + query rewrite on failure
  3h  BasicGraphRAG               entity graph + dense retrieval              [slow]
  3i  AdvancedGraphRAG(local)     entity neighbourhood search                 [slow, needs: networkx]
  3j  AdvancedGraphRAG(global)    community summary search                    [slow, needs: networkx]
  3k  AdvancedGraphRAG(drift)     HyDE iterative search                       [slow, needs: networkx]
  3l  RAPTOR                      hierarchical cluster summaries              [slow]

By default only fast variants (3a–3g) run. Use flags to add more.

Usage
-----
  # Fast variants only (3a–3g):
  .venv/bin/python research/phase3_retrieval/run.py

  # All variants including slow:
  .venv/bin/python research/phase3_retrieval/run.py --all

  # Specific variants:
  .venv/bin/python research/phase3_retrieval/run.py --variants 3a,3d,3h

  # Re-run even if a result file exists:
  .venv/bin/python research/phase3_retrieval/run.py --force
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


# ── Config builder ───────────────────────────────────────────────────────────


def _make_config(retrieval_dict: Dict[str, Any]) -> Any:
    """Build a Config with TextSplitter(512, 50) and the given retrieval section.

    Chunking is fixed at the Phase 2 baseline.  Once Phase 2 results are available,
    replace "recursive"/512/50 with the winning strategy before the final run.
    """
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
            "retrieval": retrieval_dict,
            "generation": {"strategy": "standard"},
        }
    )


# ── Variant registry ─────────────────────────────────────────────────────────

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "3a",
        "label": "Dense",
        "retrieval": {"strategy": "dense", "top_k": TOP_K},
        "slow": False,
        "needs_networkx": False,
        "note": "baseline — cosine similarity against all chunk embeddings",
    },
    {
        "id": "3b",
        "label": "BM25",
        "retrieval": {
            "strategy": "hybrid",
            "top_k": TOP_K,
            "hybrid": {"bm25_weight": 1.0},
        },
        "slow": False,
        "needs_networkx": False,
        "note": "keyword-only search — hybrid with bm25_weight=1.0, dense_weight=0.0",
    },
    {
        "id": "3c",
        "label": "Hybrid(dense+BM25)",
        "retrieval": {
            "strategy": "hybrid",
            "top_k": TOP_K,
            "hybrid": {"bm25_weight": 0.5},
        },
        "slow": False,
        "needs_networkx": False,
        "note": "combines dense + BM25 via RRF; equal weights",
    },
    {
        "id": "3d",
        "label": "MultiQuery",
        "retrieval": {
            "strategy": "multi_query",
            "top_k": TOP_K,
            "multi_query": {"num_queries": 3},
        },
        "slow": False,
        "needs_networkx": False,
        "note": "LLM rewrites query into 3 variants; retrieves for each, deduplicates",
    },
    {
        "id": "3e",
        "label": "SelfRAG",
        "retrieval": {"strategy": "self_rag", "top_k": TOP_K},
        "slow": False,
        "needs_networkx": False,
        "note": "adaptive — LLM decides if retrieval is needed, filters irrelevant docs",
    },
    {
        "id": "3f",
        "label": "ContextualCompression",
        "retrieval": {
            "strategy": "dense",
            "top_k": TOP_K,
            "contextual_compression_enabled": True,
        },
        "slow": False,
        "needs_networkx": False,
        "note": "dense + LLM extracts only query-relevant sentences from each chunk",
    },
    {
        "id": "3g",
        "label": "CorrectiveRAG",
        "retrieval": {
            "strategy": "dense",
            "top_k": TOP_K,
            "corrective_rag_enabled": True,
        },
        "slow": False,
        "needs_networkx": False,
        "note": "dense + LLM relevance scoring; rewrites query if too few docs pass",
    },
    {
        "id": "3h",
        "label": "BasicGraphRAG",
        "retrieval": {"strategy": "graph_rag", "top_k": TOP_K},
        "slow": True,
        "needs_networkx": False,
        "note": "LLM extracts entities per chunk at ingest; graph-boosts matching chunks",
    },
    {
        "id": "3i",
        "label": "AdvancedGraphRAG(local)",
        "retrieval": {
            "strategy": "advanced_graph_rag",
            "top_k": TOP_K,
            "advanced_graph_rag": {"search_mode": "local"},
        },
        "slow": True,
        "needs_networkx": True,
        "note": "Microsoft-style GraphRAG — entity neighbourhood (local search mode)",
    },
    {
        "id": "3j",
        "label": "AdvancedGraphRAG(global)",
        "retrieval": {
            "strategy": "advanced_graph_rag",
            "top_k": TOP_K,
            "advanced_graph_rag": {"search_mode": "global"},
        },
        "slow": True,
        "needs_networkx": True,
        "note": "Microsoft-style GraphRAG — community summaries (global search mode)",
    },
    {
        "id": "3k",
        "label": "AdvancedGraphRAG(drift)",
        "retrieval": {
            "strategy": "advanced_graph_rag",
            "top_k": TOP_K,
            "advanced_graph_rag": {"search_mode": "drift"},
        },
        "slow": True,
        "needs_networkx": True,
        "note": "Microsoft-style GraphRAG — HyDE iterative exploration (drift mode)",
    },
    {
        "id": "3l",
        "label": "RAPTOR",
        "retrieval": {"strategy": "raptor", "top_k": TOP_K},
        "slow": True,
        "needs_networkx": False,
        "note": "hierarchical k-means clustering + LLM summarisation per level",
    },
]

VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _has_networkx() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _experiment_name(variant_id: str) -> str:
    return f"phase3_retrieval_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _ingest_cost_warning(variant: Dict[str, Any]) -> None:
    """Print a cost estimate for slow (LLM-intensive) strategies."""
    strategy = variant["retrieval"]["strategy"]
    vid = variant["id"]
    if strategy in ("graph_rag", "advanced_graph_rag"):
        logger.info(
            "  [%s] Warning: %s makes 1 LLM call per chunk during ingestion.",
            vid,
            variant["label"],
        )
        logger.info(
            "  [%s]   Shared corpus: ~200 articles × ~3–5 chunks = ~600–1000 LLM calls",
            vid,
        )
        logger.info(
            "  [%s]   + %d generation calls. Estimated extra time: 20–60×",
            vid,
            NUM_EVAL_QUESTIONS,
        )
    elif strategy == "raptor":
        logger.info(
            "  [%s] Warning: %s runs k-means + LLM summarisation per cluster at ingest.",
            vid,
            variant["label"],
        )
        logger.info(
            "  [%s]   ~3 levels × ~10 clusters = ~30 LLM calls for ingestion.",
            vid,
        )
        logger.info(
            "  [%s]   + %d generation calls. Estimated extra time: 2–5×",
            vid,
            NUM_EVAL_QUESTIONS,
        )


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 26
    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 3 — Retrieval Strategy Ablation Results")
    logger.info("=" * 80)
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
    logger.info("  " + "-" * 76)
    for row in rows:
        m = row["metrics"]
        suffix = " [slow]" if row.get("slow") else ""
        logger.info(
            "  %-*s  %6.3f  %6.3f  %6.3f  %6.3f  %7.2fs%s",
            col,
            row["label"][:col],
            m["context_recall"],
            m["context_precision"],
            m["exact_match"],
            m["f1"],
            m["avg_latency"],
            suffix,
        )
    logger.info("=" * 80)
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
    parser = argparse.ArgumentParser(description="Phase 3: Retrieval Strategy Ablation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Include slow variants "
            "(3h BasicGraphRAG, 3i/3j/3k AdvancedGraphRAG, 3l RAPTOR)"
        ),
    )
    group.add_argument(
        "--variants",
        type=str,
        metavar="IDS",
        help="Comma-separated variant IDs to run (e.g. --variants 3a,3d,3h)",
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
    elif args.all:
        to_run = VARIANTS
    else:
        to_run = [v for v in VARIANTS if not v["slow"]]

    logger.info("=" * 55)
    logger.info("Phase 3 — Retrieval Strategy Ablation")
    logger.info("=" * 55)
    logger.info("  Chunking:   TextSplitter(512, 50) [Phase 2 baseline]")
    logger.info("  Generation: Standard (fixed)")
    logger.info("  Mode:       shared_corpus")
    logger.info("  Questions:  %d", NUM_EVAL_QUESTIONS)
    logger.info("")
    logger.info("  Variants scheduled:")
    for v in to_run:
        tags = []
        if v["slow"]:
            tags.append("SLOW")
        if v["needs_networkx"]:
            tags.append("needs: networkx")
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
                completed.append(
                    {**cached, "label": variant["label"], "slow": variant["slow"]}
                )
                continue

        # Check networkx for AdvancedGraphRAG variants
        if variant["needs_networkx"] and not _has_networkx():
            logger.info(
                "  [%s] Skipping — networkx not installed. "
                "Install with: pip install rag_sdk[advanced-graph-rag]",
                vid,
            )
            continue

        # Warn about ingestion cost for slow variants
        if variant["slow"]:
            _ingest_cost_warning(variant)

        logger.info("  [%s] Running %s ...", vid, variant["label"])

        config = _make_config(variant["retrieval"])

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

        completed.append({**result, "label": variant["label"], "slow": variant["slow"]})
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

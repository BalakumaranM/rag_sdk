"""Phase 3 — Retrieval Strategy Ablation (QASPER)
==================================================
Same 12 retrieval variants as phase3_retrieval/run.py, evaluated on QASPER.

QASPER shared_corpus mode is significantly harder than HotpotQA:
  HotpotQA: ~200 unique Wikipedia articles → ~600–1,000 chunks
  QASPER:   ~30–50 unique papers × ~30 paragraphs → ~1,500–3,000+ chunks

The retriever must find 1–2 evidence paragraphs among this larger, denser
pool of NLP domain text — a more realistic production scenario.

Variants
--------
  3a  Dense                       baseline
  3b  BM25                        keyword-only
  3c  Hybrid                      dense + BM25 with RRF
  3d  MultiQuery                  LLM query expansion
  3e  SelfRAG                     adaptive retrieval
  3f  ContextualCompression       dense + LLM context extraction
  3g  CorrectiveRAG               iterative query refinement
  3h  BasicGraphRAG               entity graph + dense              [slow]
  3i  AdvancedGraphRAG(local)     entity neighbourhood search       [slow, needs: networkx]
  3j  AdvancedGraphRAG(global)    community summary search          [slow, needs: networkx]
  3k  AdvancedGraphRAG(drift)     HyDE iterative search             [slow, needs: networkx]
  3l  RAPTOR                      hierarchical cluster summaries    [slow]

Usage
-----
  .venv/bin/python research/phase3_retrieval/run_qasper.py
  .venv/bin/python research/phase3_retrieval/run_qasper.py --all
  .venv/bin/python research/phase3_retrieval/run_qasper.py --variants 3a,3d,3h
  .venv/bin/python research/phase3_retrieval/run_qasper.py --force
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

CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "3a",
        "label": "Dense",
        "strategy": "dense",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3b",
        "label": "BM25",
        "strategy": "hybrid",
        "slow": False,
        "needs_networkx": False,
        "bm25_only": True,
    },
    {
        "id": "3c",
        "label": "Hybrid",
        "strategy": "hybrid",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3d",
        "label": "MultiQuery",
        "strategy": "multi_query",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3e",
        "label": "SelfRAG",
        "strategy": "self_rag",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3f",
        "label": "ContextualCompression",
        "strategy": "contextual_compression",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3g",
        "label": "CorrectiveRAG",
        "strategy": "corrective_rag",
        "slow": False,
        "needs_networkx": False,
    },
    {
        "id": "3h",
        "label": "BasicGraphRAG",
        "strategy": "graph_rag",
        "slow": True,
        "needs_networkx": False,
    },
    {
        "id": "3i",
        "label": "AdvancedGraphRAG(local)",
        "strategy": "advanced_graph_rag",
        "slow": True,
        "needs_networkx": True,
        "search_type": "local",
    },
    {
        "id": "3j",
        "label": "AdvancedGraphRAG(global)",
        "strategy": "advanced_graph_rag",
        "slow": True,
        "needs_networkx": True,
        "search_type": "global",
    },
    {
        "id": "3k",
        "label": "AdvancedGraphRAG(drift)",
        "strategy": "advanced_graph_rag",
        "slow": True,
        "needs_networkx": True,
        "search_type": "drift",
    },
    {
        "id": "3l",
        "label": "RAPTOR",
        "strategy": "raptor",
        "slow": True,
        "needs_networkx": False,
    },
]
VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


def _has_networkx() -> bool:
    try:
        import networkx  # noqa: F401

        return True
    except ImportError:
        return False


def _experiment_name(variant_id: str) -> str:
    return f"phase3_retrieval_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json"
    return json.loads(path.read_text()) if path.exists() else None


def _make_config(variant: Dict[str, Any]) -> Any:
    retrieval: Dict[str, Any] = {
        "strategy": variant["strategy"],
        "top_k": TOP_K,
    }
    if variant.get("bm25_only"):
        retrieval["bm25_weight"] = 1.0
    if "search_type" in variant:
        retrieval["search_type"] = variant["search_type"]

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
            "retrieval": retrieval,
            "generation": {"strategy": "standard"},
        }
    )


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 28
    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 3 — Retrieval Ablation Results  [QASPER]")
    logger.info("=" * 80)
    logger.info(
        "  %-*s  %6s  %6s  %6s  %6s  %8s",
        col,
        "Variant",
        "Recall",
        "Prec",
        "MRR",
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
            m["mrr"],
            m["f1"],
            m["avg_latency"],
            suffix,
        )
    logger.info("=" * 80)
    best_recall = max(rows, key=lambda r: r["metrics"]["context_recall"])
    best_mrr = max(rows, key=lambda r: r["metrics"]["mrr"])
    logger.info("  Best context_recall : %s", best_recall["label"])
    logger.info("  Best MRR            : %s", best_mrr["label"])
    logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Retrieval Ablation [QASPER]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--variants", type=str, metavar="IDS")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.variants:
        ids = [x.strip() for x in args.variants.split(",")]
        unknown = [i for i in ids if i not in VARIANT_BY_ID]
        if unknown:
            logger.info("Unknown variant IDs: %s", ", ".join(unknown))
            sys.exit(1)
        to_run = [VARIANT_BY_ID[i] for i in ids]
    elif args.all:
        to_run = VARIANTS
    else:
        to_run = [v for v in VARIANTS if not v["slow"]]

    logger.info("=" * 60)
    logger.info("Phase 3 — Retrieval Strategy Ablation  [QASPER]")
    logger.info("=" * 60)
    logger.info("  Chunking:   TextSplitter(%d, %d) (fixed)", CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info("  Generation: Standard (fixed)")
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
                completed.append(
                    {**cached, "label": variant["label"], "slow": variant["slow"]}
                )
                continue

        if variant["needs_networkx"] and not _has_networkx():
            logger.info("  [%s] Skipping — networkx not installed.", vid)
            continue

        logger.info("  [%s] Running %s …", vid, variant["label"])
        try:
            result = run_qasper_experiment(
                config=_make_config(variant),
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
            "  [%s] Done — Recall=%.3f  Prec=%.3f  MRR=%.3f  F1=%.3f  %.2fs",
            vid,
            m["context_recall"],
            m["context_precision"],
            m["mrr"],
            m["f1"],
            m["avg_latency"],
        )

    _print_table(completed)


if __name__ == "__main__":
    main()

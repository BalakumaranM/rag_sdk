"""Phase 2 — Chunking Strategy Ablation
========================================
Varies the text splitter across 7 variants. Everything else is held constant:
dense retrieval, standard generation, local embedding API, local LLM.

Variants
--------
  2a  TextSplitter(512, 50)    baseline — same as Phase 1
  2b  TextSplitter(256, 25)    smaller chunks
  2c  TextSplitter(1024, 100)  larger chunks
  2d  SemanticSplitter         embedding-based topic boundaries
  2e  AgenticSplitter          LLM-detected section boundaries     [slow]
  2f  PropositionSplitter      atomic fact extraction              [slow]
  2g  LateSplitter             contextual token embeddings         [needs: transformers torch]

By default only fast variants (2a–2d) run. Use flags to add more.

Usage
-----
  # Fast variants only (2a–2d):
  .venv/bin/python research/phase2_chunking/run.py

  # All variants including slow:
  .venv/bin/python research/phase2_chunking/run.py --all

  # Specific variants:
  .venv/bin/python research/phase2_chunking/run.py --variants 2a,2d,2f

  # Re-run even if a result file exists:
  .venv/bin/python research/phase2_chunking/run.py --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.shared.config import NUM_EVAL_QUESTIONS, RESULTS_DIR
from research.shared.dataset import load_hotpotqa
from research.shared.harness import run_experiment
from research.shared.providers import LocalAPIEmbedding, make_config, make_llm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


# ── Variant registry ────────────────────────────────────────────────────────

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "2a",
        "label": "TextSplitter(512,50)",
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "slow": False,
        "needs_transformers": False,
        "note": "baseline — identical to Phase 1",
    },
    {
        "id": "2b",
        "label": "TextSplitter(256,25)",
        "strategy": "recursive",
        "chunk_size": 256,
        "chunk_overlap": 25,
        "slow": False,
        "needs_transformers": False,
        "note": "smaller chunks — expect higher precision, lower recall",
    },
    {
        "id": "2c",
        "label": "TextSplitter(1024,100)",
        "strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "slow": False,
        "needs_transformers": False,
        "note": "larger chunks — expect higher recall, lower precision",
    },
    {
        "id": "2d",
        "label": "SemanticSplitter",
        "strategy": "semantic",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": False,
        "needs_transformers": False,
        "note": "cuts at embedding-detected topic boundaries",
    },
    {
        "id": "2e",
        "label": "AgenticSplitter",
        "strategy": "agentic",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": True,
        "needs_transformers": False,
        "note": "LLM call per document to detect semantic section breaks",
    },
    {
        "id": "2f",
        "label": "PropositionSplitter",
        "strategy": "proposition",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": True,
        "needs_transformers": False,
        "note": "LLM call per document to extract atomic self-contained facts",
    },
    {
        "id": "2g",
        "label": "LateSplitter",
        "strategy": "late",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": False,
        "needs_transformers": True,
        "note": "contextual token embeddings via local Jina model (boundaries only in our harness)",
    },
]

VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _has_transformers() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _experiment_name(variant_id: str) -> str:
    return f"phase2_chunking_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _llm_calls_warning(variant: Dict[str, Any], n: int) -> None:
    """Explain the cost of LLM-based splitters before running them."""
    # chunking: n_samples × 10 docs × 1 LLM call per doc
    chunking_calls = n * 10
    generation_calls = n
    logger.info(
        "  [%s] Warning: %s makes an LLM call per document during chunking.",
        variant["id"],
        variant["label"],
    )
    logger.info(
        "  [%s]   Chunking calls:   %d  (%d questions × 10 docs)",
        variant["id"],
        chunking_calls,
        n,
    )
    logger.info(
        "  [%s]   Generation calls: %d",
        variant["id"],
        generation_calls,
    )
    logger.info(
        "  [%s]   Estimated extra time vs fast variants: 10–30×",
        variant["id"],
    )


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 24
    logger.info("")
    logger.info("=" * 78)
    logger.info("Phase 2 — Chunking Ablation Results")
    logger.info("=" * 78)
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
    logger.info("  " + "-" * 74)
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
    logger.info("=" * 78)
    logger.info("")
    best_recall = max(rows, key=lambda r: r["metrics"]["context_recall"])
    best_prec = max(rows, key=lambda r: r["metrics"]["context_precision"])
    best_f1 = max(rows, key=lambda r: r["metrics"]["f1"])
    logger.info("  Best context_recall:    %s", best_recall["label"])
    logger.info("  Best context_precision: %s", best_prec["label"])
    logger.info("  Best F1:                %s", best_f1["label"])
    logger.info("")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Chunking Strategy Ablation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Include slow LLM-based splitters (2e AgenticSplitter, 2f PropositionSplitter)",
    )
    group.add_argument(
        "--variants",
        type=str,
        metavar="IDS",
        help="Comma-separated variant IDs to run (e.g. --variants 2a,2d,2f)",
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
    logger.info("Phase 2 — Chunking Strategy Ablation")
    logger.info("=" * 55)
    logger.info("  Retrieval:  Dense (fixed)")
    logger.info("  Generation: Standard (fixed)")
    logger.info("  Questions:  %d", NUM_EVAL_QUESTIONS)
    logger.info("  Mode:       per_question")
    logger.info("")
    logger.info("  Variants scheduled:")
    for v in to_run:
        tags = []
        if v["slow"]:
            tags.append("SLOW")
        if v["needs_transformers"]:
            tags.append("needs: transformers torch")
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

        # Check transformers for LateSplitter
        if variant["needs_transformers"] and not _has_transformers():
            logger.info(
                "  [%s] Skipping — transformers/torch not installed. "
                "Install with: pip install transformers torch",
                vid,
            )
            continue

        # Warn about LLM-based splitter cost
        if variant["slow"]:
            _llm_calls_warning(variant, NUM_EVAL_QUESTIONS)

        logger.info("  [%s] Running %s …", vid, variant["label"])

        config = make_config(
            chunking_strategy=variant["strategy"],
            retrieval_strategy="dense",
            generation_strategy="standard",
            chunk_size=variant["chunk_size"],
            chunk_overlap=variant["chunk_overlap"],
        )

        # LateSplitter raises ImportError at RAG construction if deps missing.
        # Guard here as a belt-and-suspenders check alongside _has_transformers().
        try:
            result = run_experiment(
                config=config,
                samples=samples,
                embedding_provider=embedding,
                llm_provider=llm,
                experiment_name=_experiment_name(vid),
                results_dir=RESULTS_DIR,
                mode="per_question",
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

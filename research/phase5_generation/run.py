"""Phase 5 — Generation Strategy Ablation
==========================================
Varies the generation strategy across 3 variants. Everything else is held constant:
TextSplitter(512, 50) chunking, the Phase 3 winning retrieval strategy, the Phase 4
winning reranking choice, local embedding API, local LLM.

NOTE: Update RETRIEVAL_STRATEGY and RERANKING_PROVIDER below before running the
definitive experiment. Defaults use the Phase 3/4 baselines (dense, no reranker).

Variants
--------
  5a  Standard             one LLM call — baseline generation
  5b  CoVe                 Chain-of-Verification: up to 6 LLM calls per query  [slow]
  5c  Attributed           one LLM call — answer with inline [N] source citations

NOTE on Attributed (5c) metrics:
  The LLM embeds citation markers like [1] and [2] in its answer. After metric
  normalisation, brackets are stripped but the digits remain, adding extra tokens
  ("yes [1]" → "yes 1"). This slightly lowers F1 and EM vs Standard even when
  the factual content is identical. Lower scores for 5c do NOT mean lower quality —
  they reflect the citation format, not accuracy.

Usage
-----
  # Fast variants only (5a, 5c):
  .venv/bin/python research/phase5_generation/run.py

  # All variants including slow CoVe:
  .venv/bin/python research/phase5_generation/run.py --all

  # Specific variants:
  .venv/bin/python research/phase5_generation/run.py --variants 5a,5b

  # Re-run even if a result file exists:
  .venv/bin/python research/phase5_generation/run.py --force
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


# ── Pipeline settings ────────────────────────────────────────────────────────
# Update these to the winners from Phase 3 and Phase 4 before the definitive run.
RETRIEVAL_STRATEGY = "dense"  # Phase 3 winner
RERANKING_PROVIDER: Optional[str] = (
    None  # Phase 4 winner: None | "cross-encoder" | "cohere"
)


# ── Config builder ───────────────────────────────────────────────────────────


def _make_config(generation_strategy: str) -> Any:
    """Build a Config with Phase 3/4 winners and the given generation strategy."""
    reranking_dict: Dict[str, Any] = {"enabled": False}
    if RERANKING_PROVIDER is not None:
        reranking_dict = {"enabled": True, "provider": RERANKING_PROVIDER}

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
            "generation": {"strategy": generation_strategy},
        }
    )


# ── Variant registry ─────────────────────────────────────────────────────────

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "5a",
        "label": "Standard",
        "generation_strategy": "standard",
        "slow": False,
        "note": "baseline — retrieve context, call LLM once, return answer",
    },
    {
        "id": "5b",
        "label": "CoVe",
        "generation_strategy": "cove",
        "slow": True,
        "note": (
            "Chain-of-Verification: initial answer → verification questions "
            "→ verify each claim → refined answer (up to 6 LLM calls per query)"
        ),
    },
    {
        "id": "5c",
        "label": "Attributed",
        "generation_strategy": "attributed",
        "slow": False,
        "note": (
            "one LLM call — answer includes inline [N] citations; "
            "citation digits lower raw F1/EM scores (not a quality regression)"
        ),
    },
]

VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _experiment_name(variant_id: str) -> str:
    return f"phase5_generation_{variant_id}"


def _result_exists(variant_id: str) -> bool:
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 14
    logger.info("")
    logger.info("=" * 68)
    logger.info("Phase 5 — Generation Strategy Ablation Results")
    logger.info("=" * 68)
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
    logger.info("  " + "-" * 64)
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
    logger.info("=" * 68)
    logger.info("")
    logger.info("  Note: Attributed (5c) F1/EM may be lower due to citation digits")
    logger.info("  in the answer — this is a metric artefact, not a quality drop.")
    logger.info("")
    best_f1 = max(rows, key=lambda r: r["metrics"]["f1"])
    best_em = max(rows, key=lambda r: r["metrics"]["exact_match"])
    best_latency = min(rows, key=lambda r: r["metrics"]["avg_latency"])
    logger.info("  Best F1:      %s", best_f1["label"])
    logger.info("  Best EM:      %s", best_em["label"])
    logger.info("  Lowest latency: %s", best_latency["label"])
    logger.info("")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Generation Strategy Ablation"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Include slow CoVe variant (5b) — adds up to 5× more LLM calls",
    )
    group.add_argument(
        "--variants",
        type=str,
        metavar="IDS",
        help="Comma-separated variant IDs to run (e.g. --variants 5a,5b)",
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

    reranker_label = RERANKING_PROVIDER if RERANKING_PROVIDER else "None"
    logger.info("=" * 55)
    logger.info("Phase 5 — Generation Strategy Ablation")
    logger.info("=" * 55)
    logger.info("  Chunking:   TextSplitter(512, 50) [Phase 2 baseline]")
    logger.info(
        "  Retrieval:  %s [update RETRIEVAL_STRATEGY after Phase 3]",
        RETRIEVAL_STRATEGY,
    )
    logger.info(
        "  Reranking:  %s [update RERANKING_PROVIDER after Phase 4]",
        reranker_label,
    )
    logger.info("  Mode:       shared_corpus")
    logger.info("  Questions:  %d", NUM_EVAL_QUESTIONS)
    logger.info("")
    logger.info("  Variants scheduled:")
    for v in to_run:
        tags = []
        if v["slow"]:
            tags.append("SLOW")
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

        if variant["slow"]:
            logger.info(
                "  [%s] Warning: %s makes up to 6 LLM calls per query.",
                vid,
                variant["label"],
            )
            logger.info(
                "  [%s]   ~%d queries × 6 calls = up to %d LLM calls total.",
                vid,
                NUM_EVAL_QUESTIONS,
                NUM_EVAL_QUESTIONS * 6,
            )

        logger.info("  [%s] Running %s ...", vid, variant["label"])

        config = _make_config(variant["generation_strategy"])

        result = run_experiment(
            config=config,
            samples=samples,
            embedding_provider=embedding,
            llm_provider=llm,
            experiment_name=_experiment_name(vid),
            results_dir=RESULTS_DIR,
            mode="shared_corpus",
        )

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

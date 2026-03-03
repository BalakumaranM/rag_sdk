"""Phase 2 — Chunking Strategy Ablation (QASPER)
=================================================
Same 7 chunking variants as phase2_chunking/run.py, evaluated on QASPER.

Why QASPER transforms this experiment
--------------------------------------
HotpotQA documents are ~300–800 chars each.  With chunk_size=512 most fit in
one chunk, so chunking variants produce near-identical results — the splitter
barely fires.

QASPER papers are 6,000–12,000 tokens with 20–50 paragraphs.  A single paper
at chunk_size=512 yields 40–100 chunks.  Now chunking boundaries genuinely
matter:
- SemanticSplitter can align cuts to section/topic transitions
- PropositionSplitter extracts self-contained claims from dense Methods sections
- Different chunk_size values produce measurably different retrieval pools

Use --inspect to print a full chunk table for each variant on the first paper.
This lets you see exactly what each splitter does to real NLP paper text.

Variants
--------
  2a  TextSplitter(512, 50)    baseline
  2b  TextSplitter(256, 25)    smaller chunks
  2c  TextSplitter(1024, 100)  larger chunks
  2d  SemanticSplitter         embedding-based topic boundaries
  2e  AgenticSplitter          LLM-detected section boundaries     [slow]
  2f  PropositionSplitter      atomic fact extraction              [slow]
  2g  LateSplitter             contextual token embeddings         [needs: transformers torch]

Usage
-----
  # Fast variants only (2a–2d):
  .venv/bin/python research/phase2_chunking/run_qasper.py

  # All variants including slow:
  .venv/bin/python research/phase2_chunking/run_qasper.py --all

  # Specific variants:
  .venv/bin/python research/phase2_chunking/run_qasper.py --variants 2a,2d,2f

  # Inspect chunk output for each variant (first paper only, no full eval):
  .venv/bin/python research/phase2_chunking/run_qasper.py --inspect

  # Inspect + save CSVs for offline browsing (Excel / pandas):
  .venv/bin/python research/phase2_chunking/run_qasper.py --inspect --chunks-out research/results/chunks/

  # Re-run even if a result file exists:
  .venv/bin/python research/phase2_chunking/run_qasper.py --force
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_sdk.document import (
    AgenticSplitter,
    TextSplitter,
    inspect_chunks,
)

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

VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "2a",
        "label": "TextSplitter(512,50)",
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "slow": False,
        "needs_transformers": False,
    },
    {
        "id": "2b",
        "label": "TextSplitter(256,25)",
        "strategy": "recursive",
        "chunk_size": 256,
        "chunk_overlap": 25,
        "slow": False,
        "needs_transformers": False,
    },
    {
        "id": "2c",
        "label": "TextSplitter(1024,100)",
        "strategy": "recursive",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "slow": False,
        "needs_transformers": False,
    },
    {
        "id": "2d",
        "label": "SemanticSplitter",
        "strategy": "semantic",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": False,
        "needs_transformers": False,
    },
    {
        "id": "2e",
        "label": "AgenticSplitter",
        "strategy": "agentic",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": True,
        "needs_transformers": False,
    },
    {
        "id": "2f",
        "label": "PropositionSplitter",
        "strategy": "proposition",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": True,
        "needs_transformers": False,
    },
    {
        "id": "2g",
        "label": "LateSplitter",
        "strategy": "late",
        "chunk_size": 512,
        "chunk_overlap": 0,
        "slow": False,
        "needs_transformers": True,
    },
]
VARIANT_BY_ID: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VARIANTS}


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
    return (RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json").exists()


def _load_result(variant_id: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / f"{_experiment_name(variant_id)}_qasper.json"
    return json.loads(path.read_text()) if path.exists() else None


def _make_splitter(variant: Dict[str, Any], llm: Any) -> Any:
    """Instantiate the right splitter for --inspect mode (no full RAG needed)."""
    strategy = variant["strategy"]
    if strategy == "recursive":
        return TextSplitter(
            chunk_size=variant["chunk_size"],
            chunk_overlap=variant["chunk_overlap"],
        )
    if strategy == "semantic":
        from rag_sdk.document import SemanticSplitter as SS
        from research.shared.providers import LocalAPIEmbedding

        return SS(embedding_provider=LocalAPIEmbedding())
    if strategy == "agentic":
        return AgenticSplitter(llm_provider=llm)
    # proposition / late — skip in inspect mode (slow / needs special deps)
    return None


def _run_inspect(
    samples: Any,
    to_run: List[Dict[str, Any]],
    llm: Any,
    chunks_out_dir: Optional[Path] = None,
) -> None:
    """Print chunk inspection tables for the first paper for each variant.

    If ``chunks_out_dir`` is given, each variant's chunks are saved to
    ``<dir>/chunks_<variant_id>.csv`` for offline browsing in Excel / pandas.
    """
    first_docs = samples[0].to_documents()
    paper_id = samples[0].paper_id
    logger.info("")
    logger.info(
        "Chunk inspection on paper: %s  (%d paragraphs)", paper_id, len(first_docs)
    )
    logger.info("")

    for variant in to_run:
        splitter = _make_splitter(variant, llm)
        if splitter is None:
            logger.info(
                "  [%s] %s — skipped in inspect mode (requires full RAG context)",
                variant["id"],
                variant["label"],
            )
            continue

        report = inspect_chunks(first_docs, splitter)
        logger.info("── %s (%s) ─────────────", variant["id"], variant["label"])
        report.summary()
        report.table()

        if chunks_out_dir is not None:
            csv_path = report.to_csv(chunks_out_dir / f"chunks_{variant['id']}.csv")
            logger.info("  Saved → %s", csv_path)

        logger.info("")


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    col = 24
    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 2 — Chunking Ablation Results  [QASPER]")
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
    best_prec = max(rows, key=lambda r: r["metrics"]["context_precision"])
    best_f1 = max(rows, key=lambda r: r["metrics"]["f1"])
    logger.info("  Best context_recall:    %s", best_recall["label"])
    logger.info("  Best context_precision: %s", best_prec["label"])
    logger.info("  Best F1:                %s", best_f1["label"])
    logger.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Chunking Ablation [QASPER]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--variants", type=str, metavar="IDS")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print chunk inspection table for each variant on the first paper, then exit",
    )
    parser.add_argument(
        "--chunks-out",
        metavar="DIR",
        help="With --inspect: save each variant's chunks to DIR/chunks_<id>.csv "
        "(opens in Excel / pandas)",
    )
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
    logger.info("Phase 2 — Chunking Strategy Ablation  [QASPER]")
    logger.info("=" * 60)
    logger.info("  Retrieval:  Dense (fixed)")
    logger.info("  Generation: Standard (fixed)")
    logger.info("  Questions:  %d", NUM_QASPER_QUESTIONS)
    logger.info("  Mode:       per_question")
    logger.info("")

    samples = load_qasper(num_questions=NUM_QASPER_QUESTIONS)
    embedding = LocalAPIEmbedding()
    llm = make_llm()

    if args.inspect:
        chunks_out_dir = Path(args.chunks_out) if args.chunks_out else None
        _run_inspect(samples, to_run, llm, chunks_out_dir=chunks_out_dir)
        return

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

        if variant["needs_transformers"] and not _has_transformers():
            logger.info("  [%s] Skipping — transformers/torch not installed.", vid)
            continue

        logger.info("  [%s] Running %s …", vid, variant["label"])

        config = make_config(
            chunking_strategy=variant["strategy"],
            retrieval_strategy="dense",
            generation_strategy="standard",
            chunk_size=variant["chunk_size"],
            chunk_overlap=variant["chunk_overlap"],
        )

        try:
            result = run_qasper_experiment(
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

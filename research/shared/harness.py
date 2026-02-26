"""Evaluation harness for all research phases.

Workflow per experiment
-----------------------
1. Build a single RAG instance from (config, embedding_provider, llm_provider).
2. For each HotpotQA sample:
   a. ``rag.clear_index()``  — wipes the in-memory vector store.
   b. ``rag.ingest_documents(sample.to_documents())``  — ingest the 10 context paragraphs.
   c. ``rag.query(sample.question)``  — retrieve + generate.
   d. Compute retrieval & answer metrics against HotpotQA ground truth.
3. Aggregate and save results to ``results/<experiment_name>.json``.

Metrics
-------
context_recall     Fraction of gold supporting titles found in retrieved docs.
                   Measures whether we retrieved what we *needed*.

context_precision  Fraction of retrieved docs that are gold supporting.
                   Measures retrieval *noise*.

exact_match        Binary: normalised predicted answer == normalised gold answer.

f1                 Token-level F1 between predicted and gold answer.
                   Partial-credit metric, standard in open-domain QA.

avg_latency        Mean wall-clock seconds per query (from rag.query internals).
"""

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from rag_sdk import RAG
from rag_sdk.config import Config
from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.llm.base import LLMProvider

from .config import RESULTS_DIR
from .dataset import HotpotQASample

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        desc = kwargs.get("desc", "")
        items = list(iterable)
        for i, item in enumerate(items, 1):
            logger.info("  %s [%d/%d]", desc, i, len(items))
            yield item


# ── Metric helpers ─────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip articles, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if not num_common:
        return 0.0
    precision = num_common / len(pred_toks) if pred_toks else 0.0
    recall = num_common / len(gold_toks) if gold_toks else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _context_recall(retrieved: List[str], supporting: Set[str]) -> float:
    if not supporting:
        return 1.0
    return len(supporting & set(retrieved)) / len(supporting)


def _context_precision(retrieved: List[str], supporting: Set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for s in retrieved if s in supporting) / len(retrieved)


# ── Experiment runner ──────────────────────────────────────────────────────


def run_experiment(
    config: Config,
    samples: List[HotpotQASample],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    experiment_name: str,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Run the evaluation loop for one configuration.

    Returns the full result dict (metrics + per-sample rows) and writes it
    to ``results_dir/<experiment_name>.json``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rag = RAG(config, embedding_provider=embedding_provider, llm_provider=llm_provider)

    rows: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc=experiment_name):
        rag.clear_index()
        rag.ingest_documents(sample.to_documents())

        result = rag.query(sample.question)

        retrieved_sources = [
            doc.metadata.get("source", "") for doc in result.get("sources", [])
        ]
        supporting = sample.supporting_titles

        rows.append(
            {
                "question_id": sample.id,
                "question": sample.question,
                "level": sample.level,
                "type": sample.type,
                "answer": result["answer"],
                "ground_truth": sample.answer,
                "retrieved_sources": retrieved_sources,
                "supporting_titles": list(supporting),
                "context_recall": _context_recall(retrieved_sources, supporting),
                "context_precision": _context_precision(retrieved_sources, supporting),
                "exact_match": _exact_match(result["answer"], sample.answer),
                "f1": _token_f1(result["answer"], sample.answer),
                "latency": result.get("latency", 0.0),
            }
        )

    n = len(rows)
    metrics = {
        "context_recall": sum(r["context_recall"] for r in rows) / n,
        "context_precision": sum(r["context_precision"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "avg_latency": sum(r["latency"] for r in rows) / n,
        "num_samples": n,
    }

    output: Dict[str, Any] = {
        "experiment": experiment_name,
        "metrics": metrics,
        "rows": rows,
    }

    out_path = results_dir / f"{experiment_name}.json"
    out_path.write_text(json.dumps(output, indent=2))

    return {**output, "output_file": str(out_path)}

"""Evaluation harness for all research phases.

Two evaluation modes
--------------------
per_question (default)
    For each sample: clear index → ingest its 10 docs → query → measure.
    Search space: ~20 chunks.
    Easier but clean isolation. Good for Phase 1 (baseline) and Phase 2 (chunking),
    where we want to measure chunking quality without corpus-size interference.

shared_corpus
    Deduplicate all docs from all samples by Wikipedia title, ingest once,
    then query every sample against the shared corpus.
    Search space: ~1,500–2,000 chunks for 100 questions.
    More realistic — retriever must find 2 gold articles among ~200 unique titles.
    Recommended for Phase 3 (retrieval) and beyond.

Choosing a mode
---------------
    run_experiment(..., mode="per_question")   # default
    run_experiment(..., mode="shared_corpus")

The result JSON and all metric definitions are identical in both modes.
The only difference is the corpus the retriever sees at query time.

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
from rag_sdk.document.models import Document
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


# ── Corpus builders ────────────────────────────────────────────────────────


def _build_shared_corpus(samples: List[HotpotQASample]) -> List[Document]:
    """Deduplicate all Wikipedia paragraphs across all samples by title.

    The same Wikipedia article (e.g. "United States") can appear as a distractor
    in many different questions. We store it once. The source title is the only
    metadata we keep — is_supporting is question-specific and is computed at
    evaluation time from the sample object, not from document metadata.
    """
    seen: Dict[str, bool] = {}
    docs = []
    for sample in samples:
        for title, sentences in sample.context:
            if title not in seen:
                seen[title] = True
                docs.append(
                    Document(
                        content=title + "\n\n" + " ".join(sentences),
                        metadata={"source": title},
                    )
                )
    logger.info(
        "Shared corpus: %d unique Wikipedia articles from %d questions.",
        len(docs),
        len(samples),
    )
    return docs


# ── Inner loops ────────────────────────────────────────────────────────────


def _eval_row(sample: HotpotQASample, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build one result row from a rag.query() result and a HotpotQA sample."""
    retrieved_sources = [
        doc.metadata.get("source", "") for doc in result.get("sources", [])
    ]
    supporting = sample.supporting_titles
    return {
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


def _run_per_question(
    rag: RAG, samples: List[HotpotQASample], desc: str
) -> List[Dict[str, Any]]:
    """Clear → ingest 10 docs → query, repeated for each sample."""
    rows = []
    for sample in tqdm(samples, desc=desc):
        rag.clear_index()
        rag.ingest_documents(sample.to_documents())
        rows.append(_eval_row(sample, rag.query(sample.question)))
    return rows


def _run_shared_corpus(
    rag: RAG, samples: List[HotpotQASample], desc: str
) -> List[Dict[str, Any]]:
    """Ingest all unique docs once, query every sample against the shared corpus."""
    corpus = _build_shared_corpus(samples)
    logger.info("Ingesting shared corpus (%d documents)…", len(corpus))
    rag.ingest_documents(corpus)
    rows = []
    for sample in tqdm(samples, desc=desc):
        rows.append(_eval_row(sample, rag.query(sample.question)))
    return rows


# ── Public API ─────────────────────────────────────────────────────────────


def run_experiment(
    config: Config,
    samples: List[HotpotQASample],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    experiment_name: str,
    results_dir: Path = RESULTS_DIR,
    mode: str = "per_question",
) -> Dict[str, Any]:
    """Run the evaluation loop for one configuration.

    Args:
        config: RAG configuration (chunking, retrieval, generation strategy).
        samples: HotpotQA samples to evaluate.
        embedding_provider: Embedding model to use.
        llm_provider: LLM to use for generation (and LLM-based retrievers).
        experiment_name: Used for the output filename and result JSON key.
        results_dir: Directory where the JSON result file is written.
        mode: "per_question" — fresh 10-doc corpus per sample (easier, isolated).
              "shared_corpus" — one shared corpus from all samples (harder, realistic).

    Returns:
        Dict with "experiment", "mode", "metrics", "rows", and "output_file".
        Also writes the same dict to ``results_dir/<experiment_name>.json``.
    """
    if mode not in ("per_question", "shared_corpus"):
        raise ValueError(
            f"mode must be 'per_question' or 'shared_corpus', got {mode!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rag = RAG(config, embedding_provider=embedding_provider, llm_provider=llm_provider)

    logger.info("Running %s [mode=%s, n=%d]", experiment_name, mode, len(samples))

    if mode == "per_question":
        rows = _run_per_question(rag, samples, desc=experiment_name)
    else:
        rows = _run_shared_corpus(rag, samples, desc=experiment_name)

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
        "mode": mode,
        "metrics": metrics,
        "rows": rows,
    }

    out_path = results_dir / f"{experiment_name}.json"
    out_path.write_text(json.dumps(output, indent=2))

    return {**output, "output_file": str(out_path)}

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

Metrics
-------
The following metrics are computed for every row:

  context_recall      Fraction of gold supporting titles found in retrieved docs.
                      Delegates to rag_sdk.evaluation.metrics.retrieval.

  context_precision   Fraction of retrieved docs that are gold supporting.
                      Delegates to rag_sdk.evaluation.metrics.retrieval.

  sentence_recall     Fraction of gold supporting sentences (from HotpotQA's
                      fine-grained supporting_facts) found as substrings in the
                      retrieved chunks. HotpotQA-specific; lives in this harness.

  mrr                 Mean Reciprocal Rank — 1/rank of the first gold doc.
                      Delegates to rag_sdk.evaluation.metrics.retrieval.

  hit_rate            1.0 if at least one gold doc is retrieved, else 0.0.
                      Delegates to rag_sdk.evaluation.metrics.retrieval.

  exact_match         Binary: normalised predicted answer == normalised gold.
                      Delegates to rag_sdk.evaluation.metrics.string_match.

  f1                  Token-level F1 between predicted and gold answer.
                      Delegates to rag_sdk.evaluation.metrics.string_match.

  faithfulness        (optional, eval_faithfulness=True) LLM-as-judge score.
                      Uses the two-call claim-extraction algorithm from
                      rag_sdk.evaluation.metrics.faithfulness (more accurate
                      than the single-call version used in harness v1).

  avg_latency         Mean wall-clock seconds per query.

Aggregate breakdowns
--------------------
metrics["by_type"]["bridge"|"comparison"]   — HotpotQA question type strata
metrics["by_level"]["easy"|"medium"|"hard"] — HotpotQA difficulty strata

Note: all generic metric implementations live in rag_sdk/evaluation/metrics/.
Only HotpotQA-specific logic (sentence_recall, shared corpus building, etc.)
remains in this file.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rag_sdk import RAG
from rag_sdk.config import Config
from rag_sdk.document.models import Document
from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.evaluation.metrics.faithfulness import faithfulness as _sdk_faithfulness
from rag_sdk.evaluation.metrics.retrieval import (
    context_precision_labeled as _context_precision,
    context_recall_labeled as _context_recall,
    hit_rate as _hit_rate,
    mrr as _mrr,
)
from rag_sdk.evaluation.metrics.string_match import exact_match as _exact_match
from rag_sdk.evaluation.metrics.string_match import token_f1 as _token_f1
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


# ── Chunk logging ───────────────────────────────────────────────────────────

_CHUNK_LOG_FIELDS = [
    "doc_id",
    "question_id",
    "source",
    "chunk_index",
    "char_count",
    "word_count",
    "content",
]


def _write_chunk_log(
    chunks: List[Document],
    path: Path,
    question_id: str,
    first: bool,
) -> None:
    """Append chunk rows (tagged with question_id) to a CSV log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if first else "a"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CHUNK_LOG_FIELDS)
        if first:
            writer.writeheader()
        for chunk in chunks:
            writer.writerow(
                {
                    "doc_id": chunk.id,
                    "question_id": question_id,
                    "source": chunk.metadata.get("source", ""),
                    "chunk_index": chunk.metadata.get("chunk_index", ""),
                    "char_count": len(chunk.content),
                    "word_count": len(chunk.content.split()),
                    "content": chunk.content,
                }
            )


# ── HotpotQA-specific metric ────────────────────────────────────────────────


def _sentence_recall(
    retrieved_docs: List[Document],
    sample: HotpotQASample,
) -> float:
    """Fraction of gold supporting sentences found in retrieved chunk content.

    Uses HotpotQA's fine-grained supporting_facts = [(title, sentence_idx), ...].
    Looks up each supporting sentence verbatim and checks for substring presence
    across all retrieved chunk content joined together.

    This metric is HotpotQA-specific and lives here rather than in the SDK
    because it depends on the HotpotQASample data structure.

    Note: splitters that reformulate text (PropositionSplitter, AgenticSplitter)
    may produce lower sentence_recall than context_recall.  This is itself a
    meaningful signal — those splitters transformed away the exact supporting
    evidence.
    """
    if not sample.supporting_facts:
        return 1.0

    context_dict = {title: sents for title, sents in sample.context}
    supporting_sentences = [
        context_dict[title][sent_idx]
        for title, sent_idx in sample.supporting_facts
        if title in context_dict and sent_idx < len(context_dict[title])
    ]
    if not supporting_sentences:
        return 1.0

    retrieved_content = " ".join(doc.content for doc in retrieved_docs)
    found = sum(1 for sent in supporting_sentences if sent in retrieved_content)
    return found / len(supporting_sentences)


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


def _eval_row(
    sample: HotpotQASample,
    result: Dict[str, Any],
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
) -> Dict[str, Any]:
    """Build one result row from a rag.query() result and a HotpotQA sample."""
    retrieved_docs: List[Document] = result.get("sources", [])
    retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
    supporting: Set[str] = sample.supporting_titles

    row: Dict[str, Any] = {
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
        "sentence_recall": _sentence_recall(retrieved_docs, sample),
        "mrr": _mrr(retrieved_sources, supporting),
        "hit_rate": _hit_rate(retrieved_sources, supporting),
        "exact_match": _exact_match(result["answer"], sample.answer),
        "f1": _token_f1(result["answer"], sample.answer),
        "latency": result.get("latency", 0.0),
    }

    if eval_faithfulness and llm_provider is not None:
        contexts = [doc.content for doc in retrieved_docs]
        score, _, _ = _sdk_faithfulness(result["answer"], contexts, llm_provider)
        row["faithfulness"] = score

    return row


def _run_per_question(
    rag: RAG,
    samples: List[HotpotQASample],
    desc: str,
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
    chunk_log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Clear → ingest 10 docs → query, repeated for each sample."""
    rows = []
    for i, sample in enumerate(tqdm(samples, desc=desc)):
        rag.clear_index()
        rag.ingest_documents(sample.to_documents())

        if chunk_log_path is not None:
            try:
                chunks = rag.vector_store.dump_documents()
                _write_chunk_log(chunks, chunk_log_path, sample.id, first=(i == 0))
                logger.info(
                    "  Chunk log: %d chunks (question %s) → %s",
                    len(chunks),
                    sample.id,
                    chunk_log_path,
                )
            except NotImplementedError:
                logger.warning(
                    "Chunk logging skipped: %s does not support dump_documents().",
                    type(rag.vector_store).__name__,
                )
                chunk_log_path = None  # stop trying

        rows.append(
            _eval_row(
                sample,
                rag.query(sample.question),
                llm_provider=llm_provider,
                eval_faithfulness=eval_faithfulness,
            )
        )
    return rows


def _run_shared_corpus(
    rag: RAG,
    samples: List[HotpotQASample],
    desc: str,
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
    chunk_log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Ingest all unique docs once, query every sample against the shared corpus."""
    corpus = _build_shared_corpus(samples)
    logger.info("Ingesting shared corpus (%d documents)…", len(corpus))
    rag.ingest_documents(corpus)

    if chunk_log_path is not None:
        try:
            chunks = rag.vector_store.dump_documents()
            _write_chunk_log(chunks, chunk_log_path, question_id="shared", first=True)
            logger.info(
                "  Chunk log: %d chunks (shared corpus) → %s",
                len(chunks),
                chunk_log_path,
            )
        except NotImplementedError:
            logger.warning(
                "Chunk logging skipped: %s does not support dump_documents().",
                type(rag.vector_store).__name__,
            )

    rows = []
    for sample in tqdm(samples, desc=desc):
        rows.append(
            _eval_row(
                sample,
                rag.query(sample.question),
                llm_provider=llm_provider,
                eval_faithfulness=eval_faithfulness,
            )
        )
    return rows


# ── Aggregate helpers ───────────────────────────────────────────────────────


def _aggregate(rows: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Any]:
    """Mean over metric_keys plus avg_latency and num_samples for a row set."""
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}
    agg: Dict[str, Any] = {k: sum(r[k] for r in rows) / n for k in metric_keys}
    agg["avg_latency"] = sum(r["latency"] for r in rows) / n
    agg["num_samples"] = n
    return agg


# ── Public API ─────────────────────────────────────────────────────────────


def run_experiment(
    config: Config,
    samples: List[HotpotQASample],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    experiment_name: str,
    results_dir: Path = RESULTS_DIR,
    mode: str = "per_question",
    eval_faithfulness: bool = False,
    chunk_log_path: Optional[Path] = None,
    overwrite_chunk_log: bool = True,
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
        eval_faithfulness: If True, adds a per-row ``faithfulness`` score via
                           the two-call claim-extraction algorithm from
                           rag_sdk.evaluation.metrics.faithfulness. Costs two
                           extra LLM calls per question. Disabled by default.
                           Recommended for Phase 5 (generation strategy ablation).

    Returns:
        Dict with "experiment", "mode", "metrics", "rows", and "output_file".
        metrics includes per-type and per-level strata breakdowns.
        Also writes the same dict to ``results_dir/<experiment_name>.json``.
    """
    if mode not in ("per_question", "shared_corpus"):
        raise ValueError(
            f"mode must be 'per_question' or 'shared_corpus', got {mode!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Handle chunk log file
    log_path: Optional[Path] = None
    if chunk_log_path is not None:
        log_path = Path(chunk_log_path)
        if overwrite_chunk_log and log_path.exists():
            log_path.unlink()
            logger.info("Chunk log cleared: %s", log_path)

    rag = RAG(config, embedding_provider=embedding_provider, llm_provider=llm_provider)

    logger.info("Running %s [mode=%s, n=%d]", experiment_name, mode, len(samples))

    faith_llm = llm_provider if eval_faithfulness else None

    if mode == "per_question":
        rows = _run_per_question(
            rag,
            samples,
            desc=experiment_name,
            llm_provider=faith_llm,
            eval_faithfulness=eval_faithfulness,
            chunk_log_path=log_path,
        )
    else:
        rows = _run_shared_corpus(
            rag,
            samples,
            desc=experiment_name,
            llm_provider=faith_llm,
            eval_faithfulness=eval_faithfulness,
            chunk_log_path=log_path,
        )

    # Metric keys present in every row
    base_keys = [
        "context_recall",
        "context_precision",
        "sentence_recall",
        "mrr",
        "hit_rate",
        "exact_match",
        "f1",
    ]
    if eval_faithfulness:
        base_keys.append("faithfulness")

    metrics = _aggregate(rows, base_keys)

    # Per question-type strata (bridge / comparison)
    metrics["by_type"] = {
        qtype: _aggregate([r for r in rows if r["type"] == qtype], base_keys)
        for qtype in ("bridge", "comparison")
    }

    # Per difficulty strata (easy / medium / hard)
    metrics["by_level"] = {
        level: _aggregate([r for r in rows if r["level"] == level], base_keys)
        for level in ("easy", "medium", "hard")
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

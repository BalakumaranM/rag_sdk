"""Evaluation harness for QASPER-based research phases.

Mirrors the structure of harness.py (HotpotQA) but handles QASPER's
distinct evidence format: evidence is verbatim paragraph text, not article
titles, so all retrieval metrics use substring matching instead of set
equality.

Key design differences from HotpotQA harness
---------------------------------------------
- context_recall / context_precision: a retrieved chunk "covers" an evidence
  paragraph if ``evidence_text`` appears as a substring in
  ``chunk.metadata["paragraph_text"]`` (the original un-split paragraph stored
  at ingest time).  This works because QASPER evidence is verbatim text from
  the paper's full_text, and each Document is created from exactly one
  paragraph — so a chunk always knows which original paragraph it came from.

- No by_type / by_level strata: QASPER has no "bridge/comparison" or
  "easy/medium/hard" fields.  The harness does break results out by
  answer_type (extractive / abstractive / boolean).

- per_question mode: each question runs on its paper's full paragraph set.
  A 30-paragraph paper at chunk_size=512 → ~40–80 chunks.  This is a much
  harder retrieval problem than HotpotQA's 15–25 chunks.

- shared_corpus mode: all unique papers from all samples are ingested once.
  With 100 questions spanning ~30–50 papers × ~30 paragraphs each, the
  retriever must find 1–2 evidence paragraphs among ~1,000–2,000+ chunks.

Metrics
-------
  context_recall      Fraction of evidence paragraphs whose text appears in
                      at least one retrieved chunk (via paragraph_text metadata).

  context_precision   Fraction of retrieved chunks whose source paragraph is
                      an evidence paragraph.

  evidence_hit_rate   1.0 if at least one evidence paragraph is covered.

  mrr                 1 / rank of first chunk covering any evidence paragraph.

  exact_match         Normalised predicted answer == normalised gold answer.

  f1                  Token-level F1(predicted, gold).

  faithfulness        (optional) LLM-as-judge hallucination score.
                      Enabled with eval_faithfulness=True.

  avg_latency         Mean wall-clock seconds per query.

  by_answer_type      Breakdown of all metrics by QASPER answer type
                      (extractive / abstractive / boolean).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rag_sdk import RAG
from rag_sdk.config import Config
from rag_sdk.document.models import Document
from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.evaluation.metrics.faithfulness import faithfulness as _sdk_faithfulness
from rag_sdk.evaluation.metrics.string_match import exact_match as _exact_match
from rag_sdk.evaluation.metrics.string_match import token_f1 as _token_f1
from rag_sdk.llm.base import LLMProvider

from .config import RESULTS_DIR
from .qasper_dataset import QASPERSample

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


# ── QASPER-specific retrieval metrics ────────────────────────────────────────


def _covers_evidence(chunk: Document, evidence_set: Set[str]) -> bool:
    """Return True if this chunk's source paragraph is an evidence paragraph.

    Uses the ``paragraph_text`` metadata stored at ingest time (the verbatim
    paragraph text before splitting).  An evidence paragraph is "covered" by
    any chunk that was split from it.
    """
    para_text = chunk.metadata.get("paragraph_text", chunk.content)
    return para_text in evidence_set


def _context_recall_qasper(
    retrieved_chunks: List[Document], evidence: List[str]
) -> float:
    """Fraction of evidence paragraphs covered by at least one retrieved chunk."""
    if not evidence:
        return 1.0
    evidence_set = set(evidence)
    covered_evidence: Set[str] = set()
    for chunk in retrieved_chunks:
        para_text = chunk.metadata.get("paragraph_text", chunk.content)
        if para_text in evidence_set:
            covered_evidence.add(para_text)
    return len(covered_evidence) / len(evidence_set)


def _context_precision_qasper(
    retrieved_chunks: List[Document], evidence: List[str]
) -> float:
    """Fraction of retrieved chunks whose source paragraph is an evidence paragraph."""
    if not retrieved_chunks:
        return 0.0
    evidence_set = set(evidence)
    gold_hits = sum(1 for c in retrieved_chunks if _covers_evidence(c, evidence_set))
    return gold_hits / len(retrieved_chunks)


def _mrr_qasper(retrieved_chunks: List[Document], evidence: List[str]) -> float:
    """1 / rank of the first retrieved chunk that covers an evidence paragraph."""
    evidence_set = set(evidence)
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if _covers_evidence(chunk, evidence_set):
            return 1.0 / rank
    return 0.0


def _hit_rate_qasper(retrieved_chunks: List[Document], evidence: List[str]) -> bool:
    """True if any retrieved chunk covers an evidence paragraph."""
    evidence_set = set(evidence)
    return any(_covers_evidence(c, evidence_set) for c in retrieved_chunks)


# ── Corpus builders ───────────────────────────────────────────────────────────


def _build_shared_corpus(samples: List[QASPERSample]) -> List[Document]:
    """Deduplicate all papers by paper_id; ingest each paper's paragraphs once."""
    seen: Set[str] = set()
    docs: List[Document] = []
    for sample in samples:
        if sample.paper_id not in seen:
            seen.add(sample.paper_id)
            docs.extend(sample.to_documents())
    logger.info(
        "Shared QASPER corpus: %d paragraphs from %d papers (%d questions).",
        len(docs),
        len(seen),
        len(samples),
    )
    return docs


# ── Inner loops ───────────────────────────────────────────────────────────────


def _eval_row(
    sample: QASPERSample,
    result: Dict[str, Any],
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
) -> Dict[str, Any]:
    """Build one result row from a rag.query() result and a QASPERSample."""
    retrieved: List[Document] = result.get("sources", [])
    evidence = sample.gold_evidence

    recall = _context_recall_qasper(retrieved, evidence)
    precision = _context_precision_qasper(retrieved, evidence)
    mrr = _mrr_qasper(retrieved, evidence)
    hit = _hit_rate_qasper(retrieved, evidence)

    row: Dict[str, Any] = {
        "question_id": sample.id,
        "paper_id": sample.paper_id,
        "question": sample.question,
        "answer_type": sample.answer.answer_type,
        "answer": result["answer"],
        "ground_truth": sample.gold_answer,
        "evidence": evidence,
        "context_recall": recall,
        "context_precision": precision,
        "mrr": mrr,
        "hit_rate": float(hit),
        "exact_match": _exact_match(result["answer"], sample.gold_answer),
        "f1": _token_f1(result["answer"], sample.gold_answer),
        "latency": result.get("latency", 0.0),
    }

    if eval_faithfulness and llm_provider is not None:
        contexts = [doc.content for doc in retrieved]
        score, _, _ = _sdk_faithfulness(result["answer"], contexts, llm_provider)
        row["faithfulness"] = score

    return row


def _run_per_question(
    rag: RAG,
    samples: List[QASPERSample],
    desc: str,
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
) -> List[Dict[str, Any]]:
    rows = []
    for sample in tqdm(samples, desc=desc):
        rag.clear_index()
        rag.ingest_documents(sample.to_documents())
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
    samples: List[QASPERSample],
    desc: str,
    llm_provider: Optional[LLMProvider] = None,
    eval_faithfulness: bool = False,
) -> List[Dict[str, Any]]:
    corpus = _build_shared_corpus(samples)
    logger.info("Ingesting shared QASPER corpus (%d paragraphs)…", len(corpus))
    rag.ingest_documents(corpus)
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


# ── Aggregate helpers ─────────────────────────────────────────────────────────


def _aggregate(rows: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}
    agg: Dict[str, Any] = {k: sum(r[k] for r in rows) / n for k in metric_keys}
    agg["avg_latency"] = sum(r["latency"] for r in rows) / n
    agg["num_samples"] = n
    return agg


# ── Public API ────────────────────────────────────────────────────────────────


def run_qasper_experiment(
    config: Config,
    samples: List[QASPERSample],
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider,
    experiment_name: str,
    results_dir: Path = RESULTS_DIR,
    mode: str = "per_question",
    eval_faithfulness: bool = False,
) -> Dict[str, Any]:
    """Run the evaluation loop on QASPER for one configuration.

    API mirrors :func:`research.shared.harness.run_experiment` so phase
    scripts stay structurally identical between the two datasets.

    Args:
        config: RAG configuration (chunking, retrieval, generation).
        samples: QASPER samples from :func:`~research.shared.qasper_dataset.load_qasper`.
        embedding_provider: Embedding model.
        llm_provider: LLM for generation (and LLM-based retrievers).
        experiment_name: Used for the output filename.
        results_dir: Directory where the JSON result file is written.
        mode: ``"per_question"`` — fresh corpus (paper paragraphs) per sample.
              ``"shared_corpus"`` — all unique papers ingested once.
        eval_faithfulness: Enable per-row faithfulness scoring (opt-in, costs
                           two extra LLM calls per question).

    Returns:
        Dict with ``experiment``, ``mode``, ``metrics``, ``rows``,
        ``output_file``.  Also writes to
        ``results_dir/<experiment_name>_qasper.json``.
    """
    if mode not in ("per_question", "shared_corpus"):
        raise ValueError(
            f"mode must be 'per_question' or 'shared_corpus', got {mode!r}"
        )

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rag = RAG(config, embedding_provider=embedding_provider, llm_provider=llm_provider)

    logger.info(
        "Running %s [QASPER, mode=%s, n=%d]", experiment_name, mode, len(samples)
    )

    faith_llm = llm_provider if eval_faithfulness else None

    if mode == "per_question":
        rows = _run_per_question(
            rag,
            samples,
            desc=experiment_name,
            llm_provider=faith_llm,
            eval_faithfulness=eval_faithfulness,
        )
    else:
        rows = _run_shared_corpus(
            rag,
            samples,
            desc=experiment_name,
            llm_provider=faith_llm,
            eval_faithfulness=eval_faithfulness,
        )

    base_keys = [
        "context_recall",
        "context_precision",
        "mrr",
        "hit_rate",
        "exact_match",
        "f1",
    ]
    if eval_faithfulness:
        base_keys.append("faithfulness")

    metrics = _aggregate(rows, base_keys)

    # Breakdown by QASPER answer type (extractive / abstractive / boolean)
    metrics["by_answer_type"] = {
        atype: _aggregate([r for r in rows if r["answer_type"] == atype], base_keys)
        for atype in ("extractive", "abstractive", "boolean")
    }

    output: Dict[str, Any] = {
        "experiment": experiment_name,
        "dataset": "qasper",
        "mode": mode,
        "metrics": metrics,
        "rows": rows,
    }

    out_path = results_dir / f"{experiment_name}_qasper.json"
    out_path.write_text(json.dumps(output, indent=2))

    return {**output, "output_file": str(out_path)}

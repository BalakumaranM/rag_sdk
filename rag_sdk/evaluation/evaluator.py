"""RAGEvaluator — main evaluation class for the RAG SDK.

Orchestrates all metric modules and provides three evaluation entry points:

    evaluate()         — single question/answer pair
    evaluate_batch()   — list of pre-computed answers
    evaluate_rag()     — runs rag.query() for every sample, then evaluates

Metric selection is automatic based on what data is available:
  - String metrics (exact_match, token_f1, rouge_l) run when ground_truth exists.
  - Retrieval metrics (mrr, hit_rate, ndcg, …) run when gold_sources are provided.
  - LLM-as-judge metrics (faithfulness, answer_relevancy, context_relevancy) always run.
  - answer_correctness runs when ground_truth exists.

Pass ``metrics=[...]`` to run only a specific subset.
"""

import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..embeddings.base import EmbeddingProvider
from ..llm.base import LLMProvider
from .dataset import EvalDataset, EvalResult, EvalSample
from .metrics.correctness import answer_correctness
from .metrics.faithfulness import faithfulness as _faithfulness
from .metrics.relevancy import answer_relevancy, context_relevancy
from .metrics.retrieval import (
    context_precision_labeled,
    context_recall_labeled,
    hit_rate,
    mrr,
    ndcg,
)
from .metrics.string_match import exact_match, rouge_l, token_f1

if TYPE_CHECKING:
    from ..core import RAG

logger = logging.getLogger(__name__)

_ALL_METRICS = frozenset(
    [
        "exact_match",
        "token_f1",
        "rouge_l",
        "faithfulness",
        "answer_relevancy",
        "context_relevancy",
        "answer_correctness",
        "context_recall",
        "context_precision",
        "mrr",
        "hit_rate",
        "ndcg",
    ]
)


class RAGEvaluator:
    """Provider-agnostic RAG evaluator.

    Uses the same ``LLMProvider`` and ``EmbeddingProvider`` already in the SDK.
    No external evaluation framework is required.

    Args:
        llm_provider: LLM used for all LLM-as-judge metrics.
        embedding_provider: Optional embedding model.  When provided,
            ``answer_relevancy`` uses the more accurate embedding-based
            (RAGAS-style) approach instead of the LLM-only fallback.
        metrics: Optional whitelist of metric names to compute.  Defaults to
            all applicable metrics based on available inputs.  Valid names::

                "exact_match", "token_f1", "rouge_l",
                "faithfulness", "answer_relevancy", "context_relevancy",
                "answer_correctness",
                "context_recall", "context_precision", "mrr", "hit_rate", "ndcg"

    Example::

        evaluator = RAGEvaluator(llm_provider=llm, embedding_provider=embed)
        results   = evaluator.evaluate_rag(rag, dataset)
        summary   = evaluator.summary(results)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self._llm = llm_provider
        self._embed = embedding_provider
        if metrics is not None:
            unknown = set(metrics) - _ALL_METRICS
            if unknown:
                raise ValueError(
                    f"Unknown metrics: {sorted(unknown)}. Valid: {sorted(_ALL_METRICS)}"
                )
            self._metric_whitelist: Optional[Set[str]] = set(metrics)
        else:
            self._metric_whitelist = None  # means: all applicable

    # ── Internal helpers ────────────────────────────────────────────────────

    def _want(self, metric: str) -> bool:
        return self._metric_whitelist is None or metric in self._metric_whitelist

    # ── Core evaluation ─────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        retrieved_sources: Optional[List[str]] = None,
        gold_sources: Optional[Set[str]] = None,
    ) -> EvalResult:
        """Evaluate a single question/answer pair.

        Args:
            question: The question asked.
            answer: The RAG system's answer.
            contexts: Retrieved chunk *text* content in retrieval order.
                Used for faithfulness and context_relevancy.
            ground_truth: Gold answer string.  Required for exact_match,
                token_f1, rouge_l, and answer_correctness.
            retrieved_sources: Source identifiers (titles, paths, IDs) in
                retrieval order.  Required for retrieval metrics.
            gold_sources: Set of gold source identifiers.  Required for
                retrieval metrics.

        Returns:
            ``EvalResult`` with scores populated for all applicable metrics.
        """
        scores: Dict[str, Optional[float]] = {}

        has_truth = ground_truth is not None and bool(ground_truth.strip())
        has_sources = (
            retrieved_sources is not None
            and gold_sources is not None
            and len(retrieved_sources) > 0
        )

        # ── String metrics (no LLM) ─────────────────────────────────────────
        if has_truth:
            assert ground_truth is not None
            if self._want("exact_match"):
                scores["exact_match"] = exact_match(answer, ground_truth)
            if self._want("token_f1"):
                scores["token_f1"] = token_f1(answer, ground_truth)
            if self._want("rouge_l"):
                scores["rouge_l"] = rouge_l(answer, ground_truth)

        # ── Labeled retrieval metrics ───────────────────────────────────────
        if has_sources:
            assert retrieved_sources is not None
            assert gold_sources is not None
            if self._want("context_recall"):
                scores["context_recall"] = context_recall_labeled(
                    retrieved_sources, gold_sources
                )
            if self._want("context_precision"):
                scores["context_precision"] = context_precision_labeled(
                    retrieved_sources, gold_sources
                )
            if self._want("mrr"):
                scores["mrr"] = mrr(retrieved_sources, gold_sources)
            if self._want("hit_rate"):
                scores["hit_rate"] = hit_rate(retrieved_sources, gold_sources)
            if self._want("ndcg"):
                scores["ndcg"] = ndcg(retrieved_sources, gold_sources)

        # ── LLM-as-judge metrics ────────────────────────────────────────────
        if self._want("faithfulness"):
            score, _, _ = _faithfulness(answer, contexts, self._llm)
            scores["faithfulness"] = score

        if self._want("answer_relevancy"):
            scores["answer_relevancy"] = answer_relevancy(
                question, answer, self._llm, self._embed
            )

        if self._want("context_relevancy") and contexts:
            scores["context_relevancy"] = context_relevancy(
                question, contexts, self._llm
            )

        if has_truth and self._want("answer_correctness"):
            assert ground_truth is not None
            scores["answer_correctness"] = answer_correctness(
                answer, ground_truth, self._llm
            )

        return EvalResult(
            question=question,
            answer=answer,
            retrieved_contexts=contexts,
            ground_truth=ground_truth,
            scores=scores,
        )

    # ── Batch evaluation ────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        samples: List[EvalSample],
        answers: List[str],
        contexts_per_sample: List[List[str]],
        retrieved_sources_per_sample: Optional[List[List[str]]] = None,
        gold_sources_per_sample: Optional[List[Set[str]]] = None,
    ) -> List[EvalResult]:
        """Evaluate a batch of samples against pre-computed answers.

        Use this when you have already run inference and just want to score
        the outputs.  For end-to-end evaluation (inference + scoring) use
        :meth:`evaluate_rag` instead.

        All list arguments must have equal length.

        Args:
            samples: Evaluation samples (questions + optional ground truth).
            answers: Pre-computed answers, one per sample.
            contexts_per_sample: Retrieved chunk text per sample.
            retrieved_sources_per_sample: Source IDs per sample (optional).
            gold_sources_per_sample: Gold source sets per sample (optional).
        """
        if not (len(samples) == len(answers) == len(contexts_per_sample)):
            raise ValueError(
                "samples, answers, and contexts_per_sample must have equal length"
            )

        results = []
        for i, (sample, answer, contexts) in enumerate(
            zip(samples, answers, contexts_per_sample)
        ):
            retrieved_sources = (
                retrieved_sources_per_sample[i]
                if retrieved_sources_per_sample
                else None
            )
            gold_sources = (
                gold_sources_per_sample[i] if gold_sources_per_sample else None
            )
            result = self.evaluate(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                ground_truth=sample.ground_truth,
                retrieved_sources=retrieved_sources,
                gold_sources=gold_sources,
            )
            result.metadata.update(sample.metadata)
            results.append(result)
        return results

    # ── End-to-end RAG evaluation ───────────────────────────────────────────

    def evaluate_rag(
        self,
        rag: "RAG",
        dataset: EvalDataset,
        top_k: int = 5,
    ) -> List[EvalResult]:
        """Run ``rag.query()`` for every sample, then evaluate.

        The most convenient entry point for end-to-end evaluation.  Runs
        inference and scoring in one call.

        Reference-free metrics (faithfulness, answer_relevancy,
        context_relevancy) are always computed.  Ground-truth metrics
        (exact_match, token_f1, rouge_l, answer_correctness) are computed
        when ``sample.ground_truth`` is set.  Labeled retrieval metrics
        (mrr, hit_rate, ndcg, context_recall, context_precision) are NOT
        computed here because ``EvalSample`` stores ground-truth chunk
        *content*, not source identifiers.  Use :meth:`evaluate_batch` with
        explicit ``gold_sources_per_sample`` if you need those.

        Args:
            rag: An ingested ``RAG`` instance ready to answer queries.
            dataset: Evaluation dataset (from ``SyntheticDatasetGenerator``
                or hand-labelled).
            top_k: Number of chunks to retrieve per query.

        Returns:
            List of ``EvalResult`` objects, one per sample.
        """
        results = []
        total = len(dataset)
        for idx, sample in enumerate(dataset, start=1):
            logger.info("evaluate_rag: sample %d/%d", idx, total)
            query_result = rag.query(sample.question, top_k=top_k)
            answer = query_result["answer"]
            source_docs = query_result.get("sources", [])
            contexts = [doc.content for doc in source_docs]

            result = self.evaluate(
                question=sample.question,
                answer=answer,
                contexts=contexts,
                ground_truth=sample.ground_truth,
            )
            result.metadata.update(sample.metadata)
            results.append(result)
        return results

    # ── Aggregation ─────────────────────────────────────────────────────────

    def summary(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Aggregate mean scores across all results.

        Metrics absent in a result (``None``) are excluded from that
        sample's contribution but do not disqualify other samples.

        Returns a dict of ``{metric: mean_score, ..., "num_samples": n}``.
        Returns ``{}`` for an empty result list.
        """
        if not results:
            return {}

        totals: Dict[str, List[float]] = {}
        for r in results:
            for metric, score in r.scores.items():
                if score is not None:
                    totals.setdefault(metric, []).append(score)

        agg: Dict[str, Any] = {
            metric: sum(vals) / len(vals) for metric, vals in totals.items()
        }
        agg["num_samples"] = len(results)
        return agg

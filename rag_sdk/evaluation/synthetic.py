"""Synthetic evaluation dataset generator.

Generates Q&A pairs from any document corpus using an LLM.  No hand-labelling
required.  Generated samples include the source chunk text as
``ground_truth_contexts`` so that faithfulness and answer correctness can be
measured against them.

Supported question types
------------------------
simple      One chunk → one question answerable from that chunk alone.
multi_hop   Two chunks → one question that requires information from both.
null        One chunk → one question that is *not* answerable from the corpus
            (tests hallucination resistance).

Quality filtering
-----------------
After each (question, answer) pair is generated, a third LLM call scores
how well the answer is supported by its source chunk(s) on a 0–10 scale.
Pairs that score below ``quality_threshold`` (default 7.0) are discarded.
Up to 4× the target count of attempts are made per question type to reach
the requested ``num_questions``.

Usage::

    gen = SyntheticDatasetGenerator(llm_provider=llm)
    dataset = gen.generate(
        documents=my_docs,
        num_questions=100,
        question_types=["simple", "multi_hop"],
    )
    dataset.save(Path("my_eval_dataset.json"))

    # Or from an already-ingested RAG instance (InMemoryVectorStore):
    dataset = gen.generate_from_rag(rag, num_questions=100)
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional

from ..document.models import Document
from ..document.splitter import TextSplitter
from ..llm.base import LLMProvider
from ..embeddings.base import EmbeddingProvider
from .dataset import EvalDataset, EvalSample

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_SIMPLE_Q_PROMPT = """\
Read the following text passage carefully.
Write one specific, factual question that:
  - Can ONLY be answered using information in this passage
  - Is not trivially answered by the title alone
  - Tests a specific fact, date, name, event, or relationship

Passage:
{chunk}

Write only the question (no answer, no explanation):"""

_SIMPLE_A_PROMPT = """\
Answer the following question using ONLY information from the given passage.
If the passage does not contain enough information, respond with "I don't know."

Passage:
{chunk}

Question: {question}
Answer:"""

_MULTI_HOP_Q_PROMPT = """\
Read the following two text passages.
Write one question that:
  - REQUIRES information from BOTH passages to answer
  - Cannot be answered from either passage alone
  - Is specific (not vague or overly broad)

Passage 1:
{chunk1}

Passage 2:
{chunk2}

Write only the question (no answer, no explanation):"""

_MULTI_HOP_A_PROMPT = """\
Answer the following question using ONLY information from the two given passages.
If the passages do not contain enough information, respond with "I don't know."

Passage 1:
{chunk1}

Passage 2:
{chunk2}

Question: {question}
Answer:"""

_QUALITY_PROMPT = """\
Question: {question}
Answer: {answer}
Context: {context}

Rate from 0 to 10: Is the answer fully and correctly answerable from the context alone,
with no additional outside knowledge required?
  10 — perfectly answerable from context; answer is complete and accurate
   5 — partially answerable; minor gaps or ambiguity
   0 — not answerable from context, or requires outside knowledge

Respond with ONLY an integer from 0 to 10:"""

_NULL_Q_PROMPT = """\
Read the following passage.
Write one question about the TOPIC of this passage that CANNOT be answered
from the passage itself — answering it would require a different source or
outside knowledge.

Passage:
{chunk}

Write only the question (no answer, no explanation):"""

_VALID_TYPES = {"simple", "multi_hop", "null"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_quality_score(text: str) -> float:
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text.strip())
    if match:
        return min(10.0, max(0.0, float(match.group(1))))
    return 0.0


# ── Generator ────────────────────────────────────────────────────────────────


class SyntheticDatasetGenerator:
    """Generate Q&A evaluation datasets from any document corpus via LLM.

    Args:
        llm_provider: Used for question generation, answer generation, and
            quality scoring (three LLM calls per accepted sample).
        embedding_provider: Currently unused; reserved for future embedding-
            based deduplication of generated questions.
        quality_threshold: Minimum quality score (0–10) for a sample to be
            kept.  Default 7.0.
        seed: Optional random seed for reproducible chunk sampling.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: Optional[EmbeddingProvider] = None,
        quality_threshold: float = 7.0,
        seed: Optional[int] = None,
    ) -> None:
        self._llm = llm_provider
        self._embed = embedding_provider
        self._quality_threshold = quality_threshold
        if seed is not None:
            random.seed(seed)

    # ── Quality filter ───────────────────────────────────────────────────────

    def _quality_score(self, question: str, answer: str, context: str) -> float:
        try:
            raw = self._llm.generate(
                _QUALITY_PROMPT.format(
                    question=question, answer=answer, context=context
                )
            )
            return _parse_quality_score(raw)
        except Exception as exc:
            logger.warning("quality_score: LLM call failed (%s); accepting sample", exc)
            return 10.0  # assume acceptable when the check itself fails

    # ── Per-type generators ──────────────────────────────────────────────────

    def _generate_simple(self, chunk: Document) -> Optional[EvalSample]:
        content = chunk.content.strip()
        if not content:
            return None
        try:
            question = self._llm.generate(
                _SIMPLE_Q_PROMPT.format(chunk=content)
            ).strip()
            answer = self._llm.generate(
                _SIMPLE_A_PROMPT.format(chunk=content, question=question)
            ).strip()
        except Exception as exc:
            logger.debug("simple generation failed: %s", exc)
            return None

        if self._quality_score(question, answer, content) < self._quality_threshold:
            return None

        return EvalSample(
            question=question,
            ground_truth=answer,
            ground_truth_contexts=[content],
            metadata={
                "question_type": "simple",
                "source": chunk.metadata.get("source", ""),
            },
        )

    def _generate_multi_hop(
        self, chunk1: Document, chunk2: Document
    ) -> Optional[EvalSample]:
        c1 = chunk1.content.strip()
        c2 = chunk2.content.strip()
        if not c1 or not c2:
            return None
        try:
            question = self._llm.generate(
                _MULTI_HOP_Q_PROMPT.format(chunk1=c1, chunk2=c2)
            ).strip()
            answer = self._llm.generate(
                _MULTI_HOP_A_PROMPT.format(chunk1=c1, chunk2=c2, question=question)
            ).strip()
        except Exception as exc:
            logger.debug("multi_hop generation failed: %s", exc)
            return None

        combined = f"{c1}\n\n{c2}"
        if self._quality_score(question, answer, combined) < self._quality_threshold:
            return None

        return EvalSample(
            question=question,
            ground_truth=answer,
            ground_truth_contexts=[c1, c2],
            metadata={
                "question_type": "multi_hop",
                "sources": [
                    chunk1.metadata.get("source", ""),
                    chunk2.metadata.get("source", ""),
                ],
            },
        )

    def _generate_null(self, chunk: Document) -> Optional[EvalSample]:
        content = chunk.content.strip()
        if not content:
            return None
        try:
            question = self._llm.generate(_NULL_Q_PROMPT.format(chunk=content)).strip()
        except Exception as exc:
            logger.debug("null generation failed: %s", exc)
            return None

        return EvalSample(
            question=question,
            ground_truth=None,
            ground_truth_contexts=[content],
            metadata={
                "question_type": "null",
                "source": chunk.metadata.get("source", ""),
                "note": "unanswerable from corpus — tests hallucination resistance",
            },
        )

    # ── Core generation loop ─────────────────────────────────────────────────

    def _run_generation_loop(
        self,
        chunks: List[Document],
        num_questions: int,
        question_types: List[str],
    ) -> List[EvalSample]:
        """Distribute budget across types and generate samples."""
        if not chunks:
            raise ValueError("No chunks available for generation.")

        shuffled = chunks[:]
        random.shuffle(shuffled)

        # Distribute question budget across types
        per_type = num_questions // len(question_types)
        remainder = num_questions % len(question_types)
        type_budget: Dict[str, int] = {
            qt: per_type + (1 if i < remainder else 0)
            for i, qt in enumerate(question_types)
        }

        samples: List[EvalSample] = []
        chunk_cursor = 0

        for qtype in question_types:
            budget = type_budget[qtype]
            generated = 0
            attempts = 0
            max_attempts = budget * 4  # allow up to 4× attempts for quality filter

            while generated < budget and attempts < max_attempts:
                attempts += 1
                idx = chunk_cursor % len(shuffled)
                chunk_cursor += 1

                sample: Optional[EvalSample] = None

                if qtype == "simple":
                    sample = self._generate_simple(shuffled[idx])

                elif qtype == "multi_hop":
                    # Pick two distinct chunks
                    idx2 = chunk_cursor % len(shuffled)
                    chunk_cursor += 1
                    if idx2 == idx:
                        idx2 = (idx2 + 1) % len(shuffled)
                    sample = self._generate_multi_hop(shuffled[idx], shuffled[idx2])

                elif qtype == "null":
                    sample = self._generate_null(shuffled[idx])

                if sample is not None:
                    samples.append(sample)
                    generated += 1

            accept_rate = int(100 * generated / max(attempts, 1))
            logger.info(
                "SyntheticDatasetGenerator: type='%s' generated=%d/%d "
                "(attempts=%d, quality_accept=%d%%)",
                qtype,
                generated,
                budget,
                attempts,
                accept_rate,
            )

        random.shuffle(samples)
        return samples

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        documents: List[Document],
        num_questions: int = 100,
        question_types: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> EvalDataset:
        """Generate a synthetic evaluation dataset from a list of documents.

        Documents are chunked first (using ``TextSplitter``), then Q&A pairs
        are generated from individual chunks (simple, null) or chunk pairs
        (multi_hop).

        Args:
            documents: Source documents.  Can be raw documents before ingestion.
            num_questions: Target number of samples.  Actual count may be lower
                if quality filtering discards too many candidates.
            question_types: Subset of ``["simple", "multi_hop", "null"]``.
                Defaults to ``["simple", "multi_hop"]``.
            chunk_size: Token chunk size for splitting.
            chunk_overlap: Token overlap between chunks.

        Returns:
            ``EvalDataset`` with generated samples and generation metadata.
        """
        if question_types is None:
            question_types = ["simple", "multi_hop"]
        unknown = set(question_types) - _VALID_TYPES
        if unknown:
            raise ValueError(
                f"Unknown question types: {unknown}. Valid: {_VALID_TYPES}"
            )

        splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)

        logger.info(
            "SyntheticDatasetGenerator.generate: %d source docs → %d chunks, "
            "targeting %d questions",
            len(documents),
            len(chunks),
            num_questions,
        )

        samples = self._run_generation_loop(chunks, num_questions, question_types)

        return EvalDataset(
            samples=samples,
            metadata={
                "num_source_documents": len(documents),
                "num_chunks": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "question_types": question_types,
                "quality_threshold": self._quality_threshold,
                "num_samples": len(samples),
                "target_num_questions": num_questions,
            },
        )

    def generate_from_rag(
        self,
        rag: Any,
        num_questions: int = 100,
        question_types: Optional[List[str]] = None,
    ) -> EvalDataset:
        """Generate from a RAG instance's already-ingested document store.

        Reads chunks directly from ``rag.vector_store``.  Currently supported
        for ``InMemoryVectorStore`` (which exposes a ``.documents`` dict).
        For other vector stores, use :meth:`generate` with the original
        document list instead.

        Args:
            rag: An ingested ``RAG`` instance.
            num_questions: Target number of samples.
            question_types: See :meth:`generate`.

        Raises:
            RuntimeError: When the vector store does not expose stored documents.
            ValueError: When the store is empty.
        """
        if question_types is None:
            question_types = ["simple", "multi_hop"]
        unknown = set(question_types) - _VALID_TYPES
        if unknown:
            raise ValueError(
                f"Unknown question types: {unknown}. Valid: {_VALID_TYPES}"
            )

        store = rag.vector_store
        if not hasattr(store, "documents") or not isinstance(store.documents, dict):
            raise RuntimeError(
                "generate_from_rag() requires a vector store that exposes a "
                ".documents dict (e.g. InMemoryVectorStore). For other stores, "
                "call generate(original_documents) instead."
            )

        chunks: List[Document] = list(store.documents.values())
        if not chunks:
            raise ValueError(
                "The vector store is empty. Ingest documents into the RAG "
                "instance before calling generate_from_rag()."
            )

        logger.info(
            "SyntheticDatasetGenerator.generate_from_rag: %d chunks from store, "
            "targeting %d questions",
            len(chunks),
            num_questions,
        )

        samples = self._run_generation_loop(chunks, num_questions, question_types)

        return EvalDataset(
            samples=samples,
            metadata={
                "source": "generate_from_rag",
                "num_chunks": len(chunks),
                "question_types": question_types,
                "quality_threshold": self._quality_threshold,
                "num_samples": len(samples),
                "target_num_questions": num_questions,
            },
        )

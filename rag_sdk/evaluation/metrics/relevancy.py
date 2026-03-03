"""Relevancy metrics — LLM-as-judge with optional embedding enhancement.

answer_relevancy
    Does the answer actually address the question?
    Primary algorithm (when embedding_provider is given): generate N reverse
    questions from the answer, compute cosine similarity to the original
    question (RAGAS approach).  Falls back to a single LLM-judge call when
    no embedding provider is available.

context_relevancy
    Are the retrieved chunks relevant to the question?
    Single batched LLM call rating all chunks at once (1 call per question,
    not 1 call per chunk).
"""

import json
import logging
import re
from typing import List, Optional

from ...embeddings.base import EmbeddingProvider
from ...llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ── Prompts ─────────────────────────────────────────────────────────────────

_REVERSE_QUESTIONS_PROMPT = """\
Given the following answer, generate {n} distinct questions that this answer
directly and completely addresses.  Each question should be answerable using
only the information in the answer.

Answer: {answer}

Return ONLY a valid JSON array of {n} question strings, e.g.:
["Question 1?", "Question 2?"]

JSON array:"""

_ANSWER_RELEVANCY_LLM_PROMPT = """\
Rate how well the following answer addresses the question on a scale from 0.0 to 1.0.
  1.0 — the answer directly and completely addresses the question
  0.5 — the answer is related but only partially addresses the question
  0.0 — the answer does not address the question at all

Question: {question}
Answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0."""

_CONTEXT_RELEVANCY_BATCH_PROMPT = """\
Rate how relevant each numbered context chunk below is to answering the question.
Use a scale from 0.0 to 1.0:
  1.0 — contains information directly needed to answer the question
  0.5 — related but not essential
  0.0 — irrelevant to the question

Question: {question}

{chunks_section}

Return ONLY a valid JSON array of floats with exactly {n} elements
(one per chunk), e.g.: [0.9, 0.1, 0.7]

JSON array:"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_float(text: str) -> Optional[float]:
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _parse_json_array(text: str) -> Optional[list]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Public API ───────────────────────────────────────────────────────────────


def answer_relevancy(
    question: str,
    answer: str,
    llm_provider: LLMProvider,
    embedding_provider: Optional[EmbeddingProvider] = None,
    num_questions: int = 3,
) -> float:
    """How well does the answer address the question?

    When ``embedding_provider`` is given (recommended), uses the RAGAS
    reverse-question approach: generate ``num_questions`` paraphrases of the
    question that the answer addresses, then score by average cosine similarity
    to the original question embedding.

    Falls back to a single LLM-as-judge call (less stable) when no
    embedding provider is configured.

    Returns a float in [0, 1].  Returns 0.0 on failure.
    """
    if not answer.strip():
        return 0.0

    if embedding_provider is not None:
        try:
            raw = llm_provider.generate(
                _REVERSE_QUESTIONS_PROMPT.format(n=num_questions, answer=answer)
            )
            generated = _parse_json_array(raw)
            if not isinstance(generated, list) or not generated:
                raise ValueError("no questions parsed from LLM output")
            generated_qs = [str(q).strip() for q in generated if str(q).strip()]
            if not generated_qs:
                raise ValueError("empty question list after filtering")

            all_texts = [question] + generated_qs
            embeddings = embedding_provider.embed_documents(all_texts)
            orig_emb = embeddings[0]
            sims = [_cosine(orig_emb, emb) for emb in embeddings[1:]]
            return sum(sims) / len(sims) if sims else 0.0
        except Exception as exc:
            logger.warning(
                "answer_relevancy: embedding approach failed (%s); "
                "falling back to LLM-only judge",
                exc,
            )

    # LLM-only fallback
    try:
        raw = llm_provider.generate(
            _ANSWER_RELEVANCY_LLM_PROMPT.format(question=question, answer=answer)
        )
        score = _parse_float(raw)
        return score if score is not None else 0.0
    except Exception as exc:
        logger.warning("answer_relevancy: LLM judge failed: %s", exc)
        return 0.0


def context_relevancy(
    question: str,
    contexts: List[str],
    llm_provider: LLMProvider,
) -> float:
    """Mean relevance of retrieved chunks to the question (batched LLM call).

    Scores all chunks in a single LLM call to minimise latency.  Returns the
    mean score across chunks, in [0, 1].  Returns 0.0 on failure or empty
    context list.
    """
    if not contexts:
        return 0.0

    chunks_section = "\n\n".join(
        f"Chunk {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = _CONTEXT_RELEVANCY_BATCH_PROMPT.format(
        question=question,
        chunks_section=chunks_section,
        n=len(contexts),
    )

    try:
        raw = llm_provider.generate(prompt)
        scores = _parse_json_array(raw)
        if not isinstance(scores, list) or len(scores) != len(contexts):
            logger.warning(
                "context_relevancy: expected %d scores, got %s", len(contexts), scores
            )
            return 0.0
        valid = [max(0.0, min(1.0, float(s))) for s in scores]
        return sum(valid) / len(valid)
    except Exception as exc:
        logger.warning("context_relevancy: LLM call failed: %s", exc)
        return 0.0

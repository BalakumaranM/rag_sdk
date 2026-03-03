"""Answer correctness metric — requires a ground truth answer.

Composite score with two components:
  - String component  (weight 0.3 by default): ROUGE-L against ground truth.
  - Semantic component (weight 0.7 by default): LLM judges factual equivalence.

The string component catches trivially wrong answers without an LLM call.
The semantic component handles paraphrasing and partial credit that pure
string overlap misses.

Both weights are configurable; they must sum to 1.0.
"""

import logging
import re
from typing import Optional

from ...llm.base import LLMProvider
from .string_match import rouge_l

logger = logging.getLogger(__name__)

_CORRECTNESS_PROMPT = """\
Compare the predicted answer to the ground truth and rate semantic correctness
from 0.0 to 1.0.
  1.0 — conveys the same core facts as the ground truth (wording may differ)
  0.5 — partially correct: some facts match, others are missing or wrong
  0.0 — factually incorrect or completely different from the ground truth

Ground truth: {ground_truth}
Predicted answer: {answer}

Respond with ONLY a decimal number between 0.0 and 1.0."""


def _parse_float(text: str) -> Optional[float]:
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def answer_correctness(
    answer: str,
    ground_truth: str,
    llm_provider: LLMProvider,
    string_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> float:
    """Composite answer correctness: ROUGE-L + LLM semantic judge.

    Args:
        answer: The RAG system's predicted answer.
        ground_truth: The gold answer to compare against.
        llm_provider: Used for the semantic component.
        string_weight: Weight for the ROUGE-L component (default 0.3).
        semantic_weight: Weight for the LLM semantic component (default 0.7).
            ``string_weight + semantic_weight`` should equal 1.0.

    Returns:
        Float in [0, 1].  Falls back to pure ROUGE-L if the LLM call fails.
    """
    if not ground_truth.strip():
        return 0.0

    string_score = rouge_l(answer, ground_truth)

    try:
        raw = llm_provider.generate(
            _CORRECTNESS_PROMPT.format(ground_truth=ground_truth, answer=answer)
        )
        semantic_score = _parse_float(raw)
        if semantic_score is None:
            logger.warning(
                "answer_correctness: could not parse LLM score; using ROUGE-L only"
            )
            return string_score
    except Exception as exc:
        logger.warning(
            "answer_correctness: LLM call failed (%s); using ROUGE-L only", exc
        )
        return string_score

    return string_weight * string_score + semantic_weight * semantic_score

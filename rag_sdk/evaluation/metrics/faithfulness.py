"""Faithfulness metric — LLM-as-judge, two-call claim extraction approach.

Algorithm (adopted from the RAGAS paper, implemented without RAGAS):
  Step 1 — Claim extraction (1 LLM call):
      Ask the LLM to decompose the answer into atomic factual statements.
  Step 2 — Claim verification (1 LLM call, all claims batched):
      For each claim, ask the LLM: "Is this directly inferable from the context?"
  Score = supported_claims / total_claims

The two-call approach is more accurate than a single holistic rating and
produces interpretable intermediate results (which specific claims failed).

The returned tuple (score, claims, verdicts) lets callers log or display
exactly which claims were unsupported — actionable for debugging pipelines.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from ...llm.base import LLMProvider

logger = logging.getLogger(__name__)

_CLAIM_EXTRACTION_PROMPT = """\
Break the following answer into a list of atomic factual claims.
Each claim must be a single, self-contained statement that can be verified
independently. Omit stylistic filler (e.g. "That's interesting").
Only include factual statements.

Answer: {answer}

Return ONLY a valid JSON array of strings, e.g.:
["Claim 1.", "Claim 2.", "Claim 3."]

JSON array:"""

_CLAIM_VERIFICATION_PROMPT = """\
You are given a context and a list of claims.
For each claim, determine whether it can be directly inferred from the context.

Context:
{context}

Claims (JSON array):
{claims_json}

For each claim at index i (0-based), output 1 if it is supported by the
context, 0 if it is not. Return ONLY a JSON array of integers with exactly
the same number of elements as the claims array, e.g.:
[1, 0, 1]

JSON array:"""


def _parse_json_array(text: str) -> Optional[list]:
    """Extract the first JSON array from an LLM response."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def faithfulness(
    answer: str,
    contexts: List[str],
    llm_provider: LLMProvider,
) -> Tuple[float, List[str], List[int]]:
    """Compute faithfulness via claim extraction and verification.

    Args:
        answer: The RAG system's answer to evaluate.
        contexts: Retrieved chunk text content (used as the reference context).
        llm_provider: LLM used for both extraction and verification calls.

    Returns:
        Tuple of:
            score     — float in [0, 1]; 0.0 on failure.
            claims    — list of atomic claims extracted from the answer.
            verdicts  — list of 0/1 integers (same length as claims).

        Returns (0.0, [], []) when the answer or context is empty, or when
        LLM calls fail.
    """
    if not answer.strip() or not contexts:
        return 0.0, [], []

    # ── Step 1: extract atomic claims ──────────────────────────────────────
    try:
        raw_claims = llm_provider.generate(
            _CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        )
        parsed = _parse_json_array(raw_claims)
        if not isinstance(parsed, list) or not parsed:
            logger.warning("faithfulness: claim extraction returned no parseable array")
            return 0.0, [], []
        claims = [str(c).strip() for c in parsed if str(c).strip()]
        if not claims:
            return 0.0, [], []
    except Exception as exc:
        logger.warning("faithfulness: claim extraction failed: %s", exc)
        return 0.0, [], []

    # ── Step 2: verify claims against context (one batched call) ───────────
    context_text = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    try:
        raw_verdicts = llm_provider.generate(
            _CLAIM_VERIFICATION_PROMPT.format(
                context=context_text,
                claims_json=json.dumps(claims),
            )
        )
        parsed_v = _parse_json_array(raw_verdicts)
        if not isinstance(parsed_v, list) or len(parsed_v) != len(claims):
            logger.warning(
                "faithfulness: verdict count mismatch (claims=%d, got=%s)",
                len(claims),
                parsed_v,
            )
            return 0.0, claims, []
        verdicts = [1 if v else 0 for v in parsed_v]
    except Exception as exc:
        logger.warning("faithfulness: claim verification failed: %s", exc)
        return 0.0, claims, []

    score = sum(verdicts) / len(verdicts) if verdicts else 0.0
    return score, claims, verdicts

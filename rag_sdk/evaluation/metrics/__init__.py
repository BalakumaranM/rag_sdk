"""Evaluation metrics for the RAG SDK.

Metric categories
-----------------
String-match (no LLM required):
    exact_match, token_f1, rouge_l

Retrieval quality (require gold source labels):
    context_recall_labeled, context_precision_labeled, mrr, hit_rate, ndcg

LLM-as-judge (reference-free):
    faithfulness, answer_relevancy, context_relevancy

LLM-as-judge (require ground truth):
    answer_correctness
"""

from .correctness import answer_correctness
from .faithfulness import faithfulness
from .relevancy import answer_relevancy, context_relevancy
from .retrieval import (
    context_precision_labeled,
    context_recall_labeled,
    hit_rate,
    mrr,
    ndcg,
)
from .string_match import exact_match, rouge_l, token_f1

__all__ = [
    # string-match
    "exact_match",
    "token_f1",
    "rouge_l",
    # retrieval (labeled)
    "context_recall_labeled",
    "context_precision_labeled",
    "mrr",
    "hit_rate",
    "ndcg",
    # LLM-as-judge
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "answer_correctness",
]

"""Retrieval quality metrics — require gold source labels.

All functions take a ranked list of retrieved source identifiers and a set
of gold source identifiers.  Source identifiers can be document titles,
file paths, chunk IDs, or any hashable string — as long as they are
consistent between the retrieved list and the gold set.

Functions
---------
context_recall_labeled(retrieved, gold)     → float
context_precision_labeled(retrieved, gold)  → float
mrr(retrieved, gold)                        → float
hit_rate(retrieved, gold)                   → float
ndcg(retrieved, gold, k)                    → float
"""

import math
from typing import List, Set


def context_recall_labeled(retrieved: List[str], gold: Set[str]) -> float:
    """Fraction of gold sources found anywhere in the retrieved list.

    Measures whether we retrieved *all* the evidence we needed.
    Order-insensitive.
    """
    if not gold:
        return 1.0
    return len(gold & set(retrieved)) / len(gold)


def context_precision_labeled(retrieved: List[str], gold: Set[str]) -> float:
    """Fraction of retrieved sources that are gold.

    Measures retrieval noise: a lower score means more irrelevant chunks
    were returned alongside the useful ones.
    """
    if not retrieved:
        return 0.0
    return sum(1 for s in retrieved if s in gold) / len(retrieved)


def mrr(retrieved: List[str], gold: Set[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first gold source.

    Captures ranking quality that recall misses: two systems with identical
    recall=1.0 can differ from MRR=1.0 (gold at rank 1) to MRR=0.2 (rank 5).
    Returns 0.0 if no gold source appears in the retrieved list.
    """
    for rank, source in enumerate(retrieved, start=1):
        if source in gold:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: List[str], gold: Set[str]) -> float:
    """1.0 if at least one gold source is retrieved, else 0.0.

    A binary floor metric useful for multi-hop questions where even finding
    one of several required sources is meaningful progress.
    """
    return 1.0 if any(s in gold for s in retrieved) else 0.0


def ndcg(retrieved: List[str], gold: Set[str], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    NDCG rewards placing gold documents at early ranks. Unlike MRR it
    accounts for *all* gold documents, not just the first one, making it
    the most informative single retrieval metric when multiple gold sources
    exist (e.g. HotpotQA always has exactly 2).

    Formula (binary rel_i ∈ {0, 1}):
        DCG@k  = Σ rel_i / log2(i+1)  for i in 1..k
        IDCG@k = Σ 1 / log2(i+1)      for i in 1..min(|gold|, k)
        NDCG@k = DCG@k / IDCG@k
    """
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(top_k, start=1)
        if doc in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

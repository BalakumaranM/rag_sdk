"""String-match metrics — no LLM, no embedding model, no extra dependencies.

All functions normalise text the same way (lowercase, strip articles,
strip punctuation, collapse whitespace) before comparison so that minor
formatting differences do not penalise correct answers.

Functions
---------
exact_match(pred, gold)  → float
token_f1(pred, gold)     → float
rouge_l(pred, gold)      → float
"""

import difflib
import re
import string
from collections import Counter
from typing import List


def _normalize(text: str) -> str:
    """Lowercase, strip articles, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _tokens(text: str) -> List[str]:
    return _normalize(text).split()


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised pred equals normalised gold, else 0.0."""
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold answer.

    Standard open-domain QA metric; gives partial credit for answers that
    share most but not all tokens with the gold string.
    """
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if not num_common:
        return 0.0
    precision = num_common / len(pred_toks) if pred_toks else 0.0
    recall = num_common / len(gold_toks) if gold_toks else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 using longest common subsequence.

    Better than token_f1 for multi-sentence answers because it rewards
    contiguous token overlap rather than bag-of-words overlap.

    Implemented with ``difflib.SequenceMatcher`` — no extra dependency.
    """
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    matcher = difflib.SequenceMatcher(None, pred_toks, gold_toks)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    precision = lcs_len / len(pred_toks)
    recall = lcs_len / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

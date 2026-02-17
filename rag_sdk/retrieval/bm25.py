import math
import re
from collections import Counter
from typing import List, Dict, Tuple
from ..document import Document


class BM25:
    """BM25 (Okapi BM25) sparse retrieval implementation.

    A term-frequency based ranking function for keyword search.

    Args:
        k1: Term frequency saturation parameter. Default 1.5.
        b: Length normalization parameter. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: List[Document] = []
        self._doc_freqs: List[Counter[str]] = []
        self._idf: Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._doc_lengths: List[int] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\w+", text.lower())

    def index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of documents to index.
        """
        self._documents = documents
        self._doc_freqs = []
        self._doc_lengths = []

        # Compute term frequencies per document
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self._doc_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))

        n = len(documents)
        self._avg_dl = sum(self._doc_lengths) / n if n > 0 else 0.0

        # Compute IDF for all terms
        df: Counter[str] = Counter()
        for freq in self._doc_freqs:
            for term in freq:
                df[term] += 1

        self._idf = {}
        for term, doc_freq in df.items():
            # Standard BM25 IDF formula
            self._idf[term] = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search indexed documents using BM25 scoring.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of (document, score) tuples, ordered by BM25 score descending.
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = []

        for i, doc_freq in enumerate(self._doc_freqs):
            score = 0.0
            dl = self._doc_lengths[i]

            for term in query_tokens:
                if term not in doc_freq:
                    continue

                tf = doc_freq[term]
                idf = self._idf.get(term, 0.0)

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                score += idf * numerator / denominator

            scores.append(score)

        # Sort by score descending
        scored = list(zip(self._documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

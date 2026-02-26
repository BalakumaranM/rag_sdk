"""HotpotQA dataset loading and preprocessing.

We use the *distractor* dev split — each question ships with exactly 10
Wikipedia paragraphs (2 gold + 8 distractors), making it self-contained.
No need to download all of Wikipedia.

Distractor split URL (≈54 MB):
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

Each example has:
  - question / answer
  - context: 10 (title, [sentence, ...]) pairs
  - supporting_facts: [(title, sentence_idx), ...] — the gold evidence
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import requests

from rag_sdk.document.models import Document

from .config import DATA_DIR, HOTPOTQA_DEV_URL

logger = logging.getLogger(__name__)


@dataclass
class HotpotQASample:
    id: str
    question: str
    answer: str
    type: str  # "bridge" | "comparison"
    level: str  # "easy" | "medium" | "hard"
    context: List[Tuple[str, List[str]]]  # [(title, [sent, ...]), ...]
    supporting_facts: List[Tuple[str, int]]  # [(title, sent_idx), ...]

    @property
    def supporting_titles(self) -> Set[str]:
        return {title for title, _ in self.supporting_facts}

    def to_documents(self) -> List[Document]:
        """Convert this sample's 10 context paragraphs to Document objects."""
        supporting = self.supporting_titles
        docs = []
        for title, sentences in self.context:
            docs.append(
                Document(
                    content=title + "\n\n" + " ".join(sentences),
                    metadata={
                        "source": title,
                        "question_id": self.id,
                        "is_supporting": title in supporting,
                    },
                )
            )
        return docs


def load_hotpotqa(
    num_questions: int = 100,
    level: str = "all",
    cache_path: Path = DATA_DIR / "hotpotqa_dev_distractor.json",
) -> List[HotpotQASample]:
    """Return ``num_questions`` samples from the HotpotQA distractor dev set.

    Downloads and caches the file on first call (~54 MB).

    Args:
        num_questions: How many samples to return.
        level: Filter by difficulty — "easy", "medium", "hard", or "all".
        cache_path: Where to cache the downloaded JSON.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        logger.info("Downloading HotpotQA dev set to %s …", cache_path)
        response = requests.get(HOTPOTQA_DEV_URL, timeout=120, stream=True)
        response.raise_for_status()
        with cache_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
        logger.info("Download complete.")

    with cache_path.open() as fh:
        raw = json.load(fh)

    samples: List[HotpotQASample] = []
    for item in raw:
        if level != "all" and item.get("level") != level:
            continue
        samples.append(
            HotpotQASample(
                id=item["_id"],
                question=item["question"],
                answer=item["answer"],
                type=item["type"],
                level=item["level"],
                context=[(t, s) for t, s in item["context"]],
                supporting_facts=[(t, i) for t, i in item["supporting_facts"]],
            )
        )
        if len(samples) >= num_questions:
            break

    logger.info("Loaded %d HotpotQA samples.", len(samples))
    return samples

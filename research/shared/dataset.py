"""HotpotQA dataset loading and preprocessing.

We use the *distractor* dev split — each question ships with exactly 10
Wikipedia paragraphs (2 gold + 8 distractors), making it self-contained.

Getting the data
----------------
Download the dev JSON and place it at ``research/data/hotpotqa_dev_distractor.json``:

  https://huggingface.co/datasets/hotpot_qa  (distractor config, validation split)

  OR direct download (≈54 MB):
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

  Then rename / move the file to:
  research/data/hotpotqa_dev_distractor.json

``load_hotpotqa()`` raises ``FileNotFoundError`` with these instructions if the
file is missing.

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

from rag_sdk.document.models import Document

from .config import DATA_DIR

logger = logging.getLogger(__name__)

_HOTPOTQA_INSTRUCTIONS = """
HotpotQA dev file not found at: {path}

Download it from one of these sources and place it there:
  • HuggingFace:  https://huggingface.co/datasets/hotpot_qa
                  (distractor config, validation split → export as JSON)
  • Direct (~54 MB): http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

Then rename / copy the file to:
  {path}
""".strip()


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
    data_path: Path = DATA_DIR / "hotpotqa_dev_distractor.json",
) -> List[HotpotQASample]:
    """Return ``num_questions`` samples from the local HotpotQA distractor dev JSON.

    Args:
        num_questions: How many samples to return.
        level: Filter by difficulty — "easy", "medium", "hard", or "all".
        data_path: Path to the local JSON file.

    Raises:
        FileNotFoundError: If the JSON file is not found at ``data_path``,
            with instructions on where to download it.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(_HOTPOTQA_INSTRUCTIONS.format(path=data_path))

    with data_path.open() as fh:
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

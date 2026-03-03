"""QASPER dataset loading and preprocessing.

QASPER (Dasigi et al. 2021) is a QA dataset over NLP research papers.
Each paper is a full document (6,000–12,000 tokens with multiple sections),
making it ideal for chunking research — unlike HotpotQA whose paragraphs are
already single-chunk-sized.

Key differences from HotpotQA
------------------------------
- Documents are full papers, not short Wikipedia paragraphs.
  A paper with 30 paragraphs at chunk_size=512 → ~40–80 chunks.
  HotpotQA: 10 docs × ~1.5 chunks each = ~15 chunks.
- Evidence is verbatim paragraph text, not article titles.
  Evaluation matches retrieved chunk content against evidence paragraphs
  rather than comparing source strings.
- Answer types: extractive spans, abstractive free-form, yes/no, unanswerable.

Data format (QASPER v0.3 JSON)
------------------------------
{
  "<paper_id>": {
    "title": "...",
    "abstract": "...",
    "full_text": [
      {"section_name": "Introduction", "paragraphs": ["para1", "para2"]},
      ...
    ],
    "qas": [
      {
        "question": "...",
        "question_id": "...",
        "answers": [
          {
            "annotation_id": "...",
            "type": "extractive" | "abstractive" | "boolean" | "unanswerable",
            "free_form_answer": "...",
            "evidence": ["verbatim paragraph from full_text"],
            "highlighted_evidence": ["specific span"]
          }
        ]
      }
    ]
  }
}

Getting the data
----------------
Download the dev JSON from Allen AI's S3 bucket and place it at
``research/data/qasper_dev_v0.3.json``:

  Archive (~10.8 MB, contains train + dev):
    https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

  Extract the dev JSON:
    tar -xzf qasper-train-dev-v0.3.tgz qasper-dev-v0.3.json
    mv qasper-dev-v0.3.json research/data/

``load_qasper()`` raises ``FileNotFoundError`` with these instructions if the
file is missing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from rag_sdk.document.models import Document

from .config import DATA_DIR

logger = logging.getLogger(__name__)

_QASPER_INSTRUCTIONS = """
QASPER dev file not found at: {path}

Download and extract it, then place the JSON at that path:

  1. Download archive (~10.8 MB):
     https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz

  2. Extract the dev JSON:
     tar -xzf qasper-train-dev-v0.3.tgz qasper-dev-v0.3.json

  3. Move it into place:
     mv qasper-dev-v0.3.json {path}
""".strip()


@dataclass
class QASPERAnswer:
    """A single annotator's answer for a QASPER question."""

    answer_type: str  # extractive | abstractive | boolean | unanswerable
    free_form_answer: str  # gold answer string
    evidence: List[str]  # verbatim paragraphs from the paper
    highlighted_evidence: List[str]  # specific answer spans (extractive only)

    @property
    def gold_answer(self) -> str:
        """Canonical string answer for EM/F1 evaluation."""
        if self.answer_type == "unanswerable":
            return ""
        if self.free_form_answer:
            return self.free_form_answer
        if self.highlighted_evidence:
            return self.highlighted_evidence[0]
        return ""


@dataclass
class QASPERSample:
    """One QA pair from QASPER, with the full paper as its corpus.

    ``paragraphs`` is a flat list of (section_name, paragraph_text) tuples
    covering the entire paper. Each tuple becomes one Document for ingestion.
    The evidence paragraphs in ``answer.evidence`` are verbatim substrings of
    paragraphs in this list, enabling exact substring matching for evaluation.
    """

    id: str  # "<paper_id>__<question_id>"
    paper_id: str
    question_id: str
    question: str
    answer: QASPERAnswer  # first non-unanswerable annotation
    paragraphs: List[Tuple[str, str]] = field(default_factory=list)
    # [(section_name, paragraph_text), ...] — full paper flattened

    @property
    def gold_answer(self) -> str:
        return self.answer.gold_answer

    @property
    def gold_evidence(self) -> List[str]:
        """Evidence paragraph texts verbatim from the paper."""
        return self.answer.evidence

    @property
    def supporting_paragraph_set(self) -> Set[str]:
        return set(self.answer.evidence)

    def to_documents(self) -> List[Document]:
        """Convert the full paper into one Document per paragraph.

        Each Document carries:
          - content         : the paragraph text
          - source          : "<section_name> [<para_idx>]" for unique identification
          - paper_id        : paper identifier
          - paragraph_text  : verbatim copy for evidence matching in the harness
        """
        docs = []
        for para_idx, (section_name, para_text) in enumerate(self.paragraphs):
            docs.append(
                Document(
                    content=para_text,
                    metadata={
                        "source": f"{section_name} [{para_idx}]",
                        "paper_id": self.paper_id,
                        "section_name": section_name,
                        "paragraph_text": para_text,
                    },
                )
            )
        return docs


def _pick_answer(raw_answers: List[dict]) -> Optional[QASPERAnswer]:
    """Select the first non-unanswerable answer annotation."""
    for ann in raw_answers:
        if ann.get("type") == "unanswerable":
            continue
        return QASPERAnswer(
            answer_type=ann.get("type", "abstractive"),
            free_form_answer=ann.get("free_form_answer", ""),
            evidence=ann.get("evidence", []),
            highlighted_evidence=ann.get("highlighted_evidence", []),
        )
    return None


def _flatten_paragraphs(full_text: List[dict]) -> List[Tuple[str, str]]:
    """Flatten full_text sections into a list of (section_name, paragraph) tuples."""
    result: List[Tuple[str, str]] = []
    for section in full_text:
        section_name = section.get("section_name", "")
        for para in section.get("paragraphs", []):
            if para.strip():
                result.append((section_name, para))
    return result


def load_qasper(
    num_questions: int = 100,
    data_path: Path = DATA_DIR / "qasper_dev_v0.3.json",
) -> List[QASPERSample]:
    """Return up to ``num_questions`` non-unanswerable QA pairs from the local QASPER dev JSON.

    Each returned sample contains the full paper as ``paragraphs`` — a flat
    list of (section_name, paragraph_text) pairs.  Unlike HotpotQA, one paper
    may contribute multiple questions; all share the same paragraph list.

    Args:
        num_questions: Maximum number of QA pairs to return.
        data_path: Path to the local ``qasper-dev-v0.3.json`` file.

    Raises:
        FileNotFoundError: If the JSON file is not found at ``data_path``,
            with instructions on where to download it.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(_QASPER_INSTRUCTIONS.format(path=data_path))

    with data_path.open() as fh:
        raw = json.load(fh)

    samples: List[QASPERSample] = []

    for paper_id, paper in raw.items():
        paragraphs = _flatten_paragraphs(paper.get("full_text", []))

        for qa in paper.get("qas", []):
            answer = _pick_answer(qa.get("answers", []))
            if answer is None:
                continue  # all annotations are unanswerable — skip

            question_id = qa.get("question_id", "")
            samples.append(
                QASPERSample(
                    id=f"{paper_id}__{question_id}",
                    paper_id=paper_id,
                    question_id=question_id,
                    question=qa.get("question", ""),
                    answer=answer,
                    paragraphs=paragraphs,
                )
            )

            if len(samples) >= num_questions:
                logger.info("Loaded %d QASPER samples.", len(samples))
                return samples

    logger.info("Loaded %d QASPER samples.", len(samples))
    return samples

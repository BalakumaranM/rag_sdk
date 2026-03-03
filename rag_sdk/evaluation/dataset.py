"""Data model for the evaluation module.

EvalSample   — one evaluation example (question + optional ground truth).
EvalResult   — one evaluated example (scores populated after evaluation).
EvalDataset  — a collection of samples with serialization helpers.

All three are plain dataclasses. EvalDataset serialises to / from JSON so that
synthetic datasets can be saved, versioned, and shared across experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class EvalSample:
    """One evaluation example — provider-agnostic, dataset-agnostic.

    Args:
        question: The question to ask the RAG system.
        ground_truth: Gold answer string, if available.  Required for
            ``exact_match``, ``token_f1``, ``rouge_l``, and
            ``answer_correctness``.
        ground_truth_contexts: Gold source chunk *text strings*, if available.
            Used to identify which chunks were the correct retrieval targets.
            Not required for reference-free metrics.
        metadata: Arbitrary key-value pairs (question type, difficulty level,
            source file name, etc.).  Passed through to ``EvalResult``.
    """

    question: str
    ground_truth: Optional[str] = None
    ground_truth_contexts: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation output for one question.

    Args:
        question: The question that was evaluated.
        answer: The RAG system's answer.
        retrieved_contexts: Retrieved chunk *text* content in retrieval order.
        ground_truth: Gold answer, forwarded from ``EvalSample``.
        scores: Metric name → score.  ``None`` means the metric was not
            computed (e.g. ``answer_correctness`` when no ground truth exists).
        metadata: Forwarded from ``EvalSample`` plus any runtime additions.
    """

    question: str
    answer: str
    retrieved_contexts: List[str]
    ground_truth: Optional[str]
    scores: Dict[str, Optional[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """An ordered collection of ``EvalSample`` objects with generation metadata.

    Supports iteration, ``len()``, JSON save/load, and slicing via list index::

        dataset = EvalDataset.load(Path("my_eval.json"))
        for sample in dataset:
            ...
        subset = EvalDataset(dataset.samples[:20], dataset.metadata)
    """

    samples: List[EvalSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvalSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> EvalSample:
        return self.samples[index]

    def save(self, path: Path) -> None:
        """Serialise to JSON.  Parent directories are created if missing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": self.metadata,
            "samples": [asdict(s) for s in self.samples],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        """Deserialise from a JSON file written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        samples = [
            EvalSample(
                question=s["question"],
                ground_truth=s.get("ground_truth"),
                ground_truth_contexts=s.get("ground_truth_contexts"),
                metadata=s.get("metadata", {}),
            )
            for s in data["samples"]
        ]
        return cls(samples=samples, metadata=data.get("metadata", {}))

"""RAG Evaluation module.

Provides three building blocks for evaluating any RAG pipeline:

1. **Data model** — ``EvalSample``, ``EvalDataset``, ``EvalResult``
2. **Evaluator** — ``RAGEvaluator`` (runs metrics, aggregates results)
3. **Synthetic data** — ``SyntheticDatasetGenerator`` (creates Q&A pairs
   from any document corpus without hand-labelling)

Quick start::

    from rag_sdk.evaluation import (
        RAGEvaluator,
        SyntheticDatasetGenerator,
        EvalSample,
        EvalDataset,
    )

    # Generate a synthetic ground-truth dataset from your documents
    gen = SyntheticDatasetGenerator(llm_provider=llm, embedding_provider=embed)
    dataset = gen.generate(my_docs, num_questions=100)
    dataset.save(Path("my_eval.json"))

    # Evaluate your RAG pipeline end-to-end
    evaluator = RAGEvaluator(llm_provider=llm, embedding_provider=embed)
    results   = evaluator.evaluate_rag(rag, dataset)
    summary   = evaluator.summary(results)

Individual metrics are importable from ``rag_sdk.evaluation.metrics``::

    from rag_sdk.evaluation.metrics import faithfulness, mrr, rouge_l
"""

from .dataset import EvalDataset, EvalResult, EvalSample
from .evaluator import RAGEvaluator
from .synthetic import SyntheticDatasetGenerator

__all__ = [
    "EvalSample",
    "EvalResult",
    "EvalDataset",
    "RAGEvaluator",
    "SyntheticDatasetGenerator",
]

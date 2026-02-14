from typing import List, Dict, Any
from .base import GenerationStrategy
from ..document import Document
from ..llm import LLMProvider


class StandardGeneration(GenerationStrategy):
    """
    Standard RAG generation: concatenate context and prompt LLM.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        context_text = "\n\n".join([doc.content for doc in documents])

        system_prompt = (
            "You are a helpful assistant. Use the following pieces of context to answer the user's question.\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context_text}"
        )

        answer = self.llm_provider.generate(prompt=query, system_prompt=system_prompt)

        return {"answer": answer}

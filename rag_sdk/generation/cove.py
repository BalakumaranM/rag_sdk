import json
import logging
import re
from typing import List, Dict, Any
from .base import GenerationStrategy
from ..document import Document
from ..llm import LLMProvider

logger = logging.getLogger(__name__)


class ChainOfVerificationGeneration(GenerationStrategy):
    """Generates an answer, verifies claims via follow-up questions, then refines."""

    def __init__(self, llm_provider: LLMProvider, max_verification_questions: int = 3):
        self.llm_provider = llm_provider
        self.max_verification_questions = max_verification_questions

    def _generate_initial_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer the user's question.\n"
            "If you don't know the answer, say so.\n\n"
            f"Context:\n{context}"
        )
        return self.llm_provider.generate(prompt=query, system_prompt=system_prompt)

    def _generate_verification_questions(self, answer: str) -> List[str]:
        prompt = (
            "Given the following answer, generate verification questions that check "
            "the factual claims made in the answer. Each question should be answerable "
            "using the original source material.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Return ONLY a JSON array of up to {self.max_verification_questions} question strings.\n"
            "Response:"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                return [q for q in questions if isinstance(q, str)][
                    : self.max_verification_questions
                ]
        except Exception as e:
            logger.warning(f"Verification question generation failed: {e}")

        return []

    def _answer_verification_question(self, question: str, context: str) -> str:
        system_prompt = (
            "Answer the following verification question using ONLY the provided context. "
            "Be concise and factual.\n\n"
            f"Context:\n{context}"
        )
        return self.llm_provider.generate(prompt=question, system_prompt=system_prompt)

    def _generate_refined_answer(
        self,
        query: str,
        context: str,
        initial_answer: str,
        verification_qa: List[Dict[str, str]],
    ) -> str:
        verification_text = "\n".join(
            [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in verification_qa]
        )

        system_prompt = (
            "You are a helpful assistant. You previously generated an answer, then verified "
            "its claims. Use the verification results to produce an improved, more accurate answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Initial answer:\n{initial_answer}\n\n"
            f"Verification results:\n{verification_text}"
        )

        return self.llm_provider.generate(prompt=query, system_prompt=system_prompt)

    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        context = "\n\n".join([doc.content for doc in documents])

        # Step 1: Initial answer
        initial_answer = self._generate_initial_answer(query, context)

        # Step 2: Generate verification questions
        questions = self._generate_verification_questions(initial_answer)

        if not questions:
            return {
                "answer": initial_answer,
                "initial_answer": initial_answer,
                "verification_qa": [],
            }

        # Step 3: Answer each verification question independently
        verification_qa = []
        for question in questions:
            answer = self._answer_verification_question(question, context)
            verification_qa.append({"question": question, "answer": answer})

        # Step 4: Refined answer
        refined_answer = self._generate_refined_answer(
            query, context, initial_answer, verification_qa
        )

        return {
            "answer": refined_answer,
            "initial_answer": initial_answer,
            "verification_qa": verification_qa,
        }

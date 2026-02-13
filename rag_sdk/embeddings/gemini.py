from typing import List
import google.generativeai as genai
from .base import EmbeddingProvider
from ..config import GeminiEmbeddingConfig


class GeminiEmbedding(EmbeddingProvider):
    """
    Gemini embedding provider.
    """

    def __init__(self, config: GeminiEmbeddingConfig):
        self.config = config
        genai.configure(api_key=config.get_api_key())  # type: ignore

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Gemini expects 'content' and task_type
            result = genai.embed_content(  # type: ignore
                model=self.config.model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(  # type: ignore
            model=self.config.model,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

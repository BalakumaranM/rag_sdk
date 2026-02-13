from typing import List
import google.generativeai as genai
from .base import EmbeddingProvider
from ..config import GeminiEmbeddingConfig


class GeminiEmbedding(EmbeddingProvider):
    """
    Google Gemini embedding provider.
    """

    def __init__(self, config: GeminiEmbeddingConfig):
        self.config = config
        genai.configure(api_key=config.get_api_key())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Gemini batch embedding usually requires iterating or specific batch calls
        # For simplicity in this SDK version, we iterate.
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=self.config.model, content=text, task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model=self.config.model, content=text, task_type="retrieval_query"
        )
        return result["embedding"]

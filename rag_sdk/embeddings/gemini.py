from typing import List
from google import genai
from google.genai import types
from .base import EmbeddingProvider
from ..config import GeminiEmbeddingConfig


class GeminiEmbedding(EmbeddingProvider):
    """
    Gemini embedding provider.
    """

    def __init__(self, config: GeminiEmbeddingConfig):
        self.config = config
        self.client = genai.Client(api_key=config.get_api_key())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=self.config.model,
            contents=texts,  # type: ignore[arg-type]
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        if not response.embeddings:
            return []
        return [emb.values or [] for emb in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.config.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        if not response.embeddings:
            return []
        return response.embeddings[0].values or []

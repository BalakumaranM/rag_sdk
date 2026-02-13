from typing import List
import cohere
from .base import EmbeddingProvider
from ..config import CohereEmbeddingConfig


class CohereEmbedding(EmbeddingProvider):
    """
    Cohere embedding provider.
    """

    def __init__(self, config: CohereEmbeddingConfig):
        self.config = config
        self.client = cohere.Client(api_key=config.get_api_key())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embed(
            texts=texts, model=self.config.model, input_type=self.config.input_type
        )
        return response.embeddings  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embed(
            texts=[text], model=self.config.model, input_type="search_query"
        )
        return list(response.embeddings)[0]  # type: ignore

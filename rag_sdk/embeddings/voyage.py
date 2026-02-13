from typing import List
import voyageai
from .base import EmbeddingProvider
from ..config import VoyageEmbeddingConfig


class VoyageEmbedding(EmbeddingProvider):
    """
    Voyage AI embedding provider.
    """

    def __init__(self, config: VoyageEmbeddingConfig):
        self.config = config
        self.client = voyageai.Client(api_key=config.get_api_key())  # type: ignore

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.client.embed(
            texts=texts, model=self.config.model, input_type="document"
        )
        return result.embeddings  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embed(
            texts=[text], model=self.config.model, input_type="query"
        )
        return response.embeddings[0]  # type: ignore

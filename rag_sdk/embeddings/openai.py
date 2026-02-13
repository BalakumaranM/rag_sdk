from typing import List
import openai
from .base import EmbeddingProvider
from ..config import OpenAIEmbeddingConfig


class OpenAIEmbedding(EmbeddingProvider):
    """
    OpenAI embedding provider.
    """

    def __init__(self, config: OpenAIEmbeddingConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=config.get_api_key())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using OpenAI API.
        Currently processes in batches if needed, but for simplicity here we assume
        the user handles batching or we do a simple loop.
        """
        # Simple implementation: process in batches defined in config
        batch_size = self.config.batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Clean newlines as recommended by OpenAI for some models,
            # though less critical for text-embedding-3
            batch = [text.replace("\n", " ") for text in batch]

            response = self.client.embeddings.create(
                input=batch,
                model=self.config.model,
                dimensions=self.config.dimensions or openai.NOT_GIVEN,
            )

            # Sort by index to ensure order is preserved (OpenAI usually preserves it)
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=self.config.model,
            dimensions=self.config.dimensions or openai.NOT_GIVEN,
        )
        return response.data[0].embedding

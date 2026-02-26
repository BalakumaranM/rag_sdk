"""Factory helpers for the local LLM + embedding API used in all experiments."""

from typing import List

import requests

from rag_sdk.config import Config, ConfigLoader, OpenAIConfig
from rag_sdk.embeddings.base import EmbeddingProvider
from rag_sdk.llm.openai import OpenAILLM

from .config import (
    LOCAL_EMBED_ENDPOINT,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    TOP_K,
)


class LocalAPIEmbedding(EmbeddingProvider):
    """Wraps the locally-hosted embedding REST API."""

    def __init__(self, endpoint: str = LOCAL_EMBED_ENDPOINT):
        self.endpoint = endpoint

    def _call(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(self.endpoint, json={"texts": texts}, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", data) if isinstance(data, dict) else data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


def make_llm() -> OpenAILLM:
    """Local LLM via OpenAI-compatible Ollama endpoint."""
    return OpenAILLM(
        OpenAIConfig(
            model=LOCAL_LLM_MODEL,
            base_url=LOCAL_LLM_BASE_URL,
            temperature=0.0,
            max_tokens=512,
        )
    )


def make_config(
    chunking_strategy: str = "recursive",
    retrieval_strategy: str = "dense",
    generation_strategy: str = "standard",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    top_k: int = TOP_K,
) -> Config:
    """Build a Config object programmatically.

    The embedding provider is always overridden at RAG construction time via
    the explicit ``embedding_provider`` kwarg, so the embeddings section here
    is a no-op placeholder.  The LLM config points to the local server.
    """
    return ConfigLoader.from_dict(
        {
            "document_processing": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking": {"strategy": chunking_strategy},
            },
            "embeddings": {"provider": "openai"},  # overridden at RAG init
            "vectorstore": {"provider": "memory"},  # fresh in-memory per experiment
            "llm": {
                "provider": "openai",
                "openai": {
                    "model": LOCAL_LLM_MODEL,
                    "base_url": LOCAL_LLM_BASE_URL,
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            },
            "retrieval": {
                "strategy": retrieval_strategy,
                "top_k": top_k,
            },
            "generation": {"strategy": generation_strategy},
        }
    )

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .embeddings.base import EmbeddingProvider
    from .llm.base import LLMProvider


class _Settings:
    """Module-level singleton for global provider defaults.

    Components resolve providers lazily against this object at *call time*,
    not at construction time.  That means swapping a provider here is
    immediately visible everywhere in the SDK — no re-initialisation needed.

    Priority chain (highest → lowest):
      1. Explicit kwarg passed directly to RAG() or a component constructor
      2. Settings.embedding_provider / Settings.llm_provider   ← this object
      3. Config-driven initialisation (provider name in config.yaml)

    Example — custom local embedding + local LLM::

        from rag_sdk import RAG, Settings
        from my_server import MyEmbedding, MyLLM

        Settings.embedding_provider = MyEmbedding("http://localhost:8080")
        Settings.llm_provider       = MyLLM("http://localhost:11434")

        rag = RAG(config)   # uses your providers automatically
    """

    def __init__(self) -> None:
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._llm_provider: Optional[LLMProvider] = None

    # ------------------------------------------------------------------
    # embedding_provider
    # ------------------------------------------------------------------

    @property
    def embedding_provider(self) -> Optional[EmbeddingProvider]:
        return self._embedding_provider

    @embedding_provider.setter
    def embedding_provider(self, value: EmbeddingProvider) -> None:
        self._embedding_provider = value

    # ------------------------------------------------------------------
    # llm_provider
    # ------------------------------------------------------------------

    @property
    def llm_provider(self) -> Optional[LLMProvider]:
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, value: LLMProvider) -> None:
        self._llm_provider = value

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all overrides back to None.  Useful in tests."""
        self._embedding_provider = None
        self._llm_provider = None

    def __repr__(self) -> str:
        emb = (
            type(self._embedding_provider).__name__
            if self._embedding_provider
            else None
        )
        llm = type(self._llm_provider).__name__ if self._llm_provider else None
        return f"Settings(embedding_provider={emb}, llm_provider={llm})"


# One instance lives at module level.  Python caches modules in sys.modules,
# so every `from rag_sdk.settings import Settings` resolves to the exact same
# object — the global singleton, without any Borg/metaclass tricks.
Settings = _Settings()

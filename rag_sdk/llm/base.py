from abc import ABC, abstractmethod
from typing import Optional, Iterator


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for the prompt.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """
        Stream the response for the prompt.
        """
        pass

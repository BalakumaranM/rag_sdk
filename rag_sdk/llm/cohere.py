from typing import Optional, Iterator
import cohere
from .base import LLMProvider
from ..config import CohereConfig


class CohereLLM(LLMProvider):
    """
    Cohere LLM provider.
    """

    def __init__(self, config: CohereConfig):
        self.config = config
        self.client = cohere.Client(api_key=config.get_api_key())

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.chat(
            message=prompt,
            preamble=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
        )
        return response.text

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        stream = self.client.chat_stream(
            message=prompt,
            preamble=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        for event in stream:
            if event.event_type == "text-generation":
                yield event.text

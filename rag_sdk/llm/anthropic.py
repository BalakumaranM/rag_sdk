from typing import Optional, Iterator
import anthropic
from .base import LLMProvider
from ..config import AnthropicConfig


class AnthropicLLM(LLMProvider):
    """
    Anthropic LLM provider.
    """

    def __init__(self, config: AnthropicConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.get_api_key())

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt or "",
        )
        return str(response.content[0].text)

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = [{"role": "user", "content": prompt}]

        with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

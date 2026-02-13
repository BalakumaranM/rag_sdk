from typing import Optional, Iterator
import openai
from .base import LLMProvider
from ..config import OpenAIConfig


class OpenAILLM(LLMProvider):
    """
    OpenAI LLM provider.
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.get_api_key(), base_url=config.base_url
        )

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._prepare_messages(prompt, system_prompt)

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = self._prepare_messages(prompt, system_prompt)

        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

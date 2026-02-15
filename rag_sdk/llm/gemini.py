from typing import Optional, Iterator
from google import genai
from google.genai import types
from .base import LLMProvider
from ..config import GeminiConfig


class GeminiLLM(LLMProvider):
    """
    Google Gemini LLM provider.
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = genai.Client(api_key=config.get_api_key())

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        return str(response.text)

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        for chunk in self.client.models.generate_content_stream(
            model=self.config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        ):
            if chunk.text:
                yield chunk.text

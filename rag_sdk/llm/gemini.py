from typing import Optional, Iterator
import google.generativeai as genai
from .base import LLMProvider
from ..config import GeminiConfig


class GeminiLLM(LLMProvider):
    """
    Google Gemini LLM provider.
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.get_api_key())
        self.model = genai.GenerativeModel(config.model)
        self.generation_config = genai.types.GenerationConfig(
            temperature=config.temperature, max_output_tokens=config.max_output_tokens
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Gemini handles system prompts differently or via the model config,
        # but for simple chat, we can prepend it or use a chat session.
        # For simplicity in this base implementation:

        full_prompt = prompt
        if system_prompt:
            # Note: Gemini 1.5 allows system instructions in model init,
            # but we initialized once. We can pass it if we re-instantiate or just prepend.
            # Best practice for single turn:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        response = self.model.generate_content(
            full_prompt, generation_config=self.generation_config
        )
        return response.text

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        response = self.model.generate_content(
            full_prompt, generation_config=self.generation_config, stream=True
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

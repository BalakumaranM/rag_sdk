from typing import Optional, Iterator
import google.generativeai as genai
from .base import LLMProvider
from ..config import GeminiConfig  # type: ignore


class GeminiLLM(LLMProvider):
    """
    Google Gemini LLM provider.
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.get_api_key())  # type: ignore
        self.model = genai.GenerativeModel(config.model)  # type: ignore
        self.generation_config = genai.types.GenerationConfig(
            temperature=config.temperature, max_output_tokens=config.max_output_tokens
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # construct prompt with system instruction if provided
        if system_prompt:
            # Gemini supports system instructions in newer models, but for simplicity
            # we'll prepend it to the prompt or use chat history if we were doing chat.
            # Here we just prepend.
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        else:
            full_prompt = prompt

        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(  # type: ignore
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        return str(response.text)

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

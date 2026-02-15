import json
import logging
import re
from typing import Any, Callable, Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)

# Patterns for extracting JSON from LLM responses
_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _build_retry_prompt(original_prompt: str, prior_response: str, error: str) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response was invalid.\n"
        f"Previous response: {prior_response}\n"
        f"Error: {error}\n\n"
        "Please return ONLY valid JSON. Response:"
    )


def extract_json_from_llm(
    llm_provider: LLMProvider,
    prompt: str,
    system_prompt: Optional[str] = None,
    expected_type: type = list,
    max_retries: int = 2,
    validate: Optional[Callable[[Any], str]] = None,
) -> Optional[Any]:
    """Call an LLM, extract JSON from its response, and retry on failure.

    Args:
        llm_provider: The LLM to call.
        prompt: The prompt to send.
        system_prompt: Optional system prompt.
        expected_type: Expected JSON root type (list or dict).
        max_retries: Total attempts (initial + retries). Defaults to 2.
        validate: Optional callback that returns "" if valid, else an error message.

    Returns:
        Parsed JSON (list or dict), or None if all attempts fail.
    """
    pattern = _OBJECT_PATTERN if expected_type is dict else _ARRAY_PATTERN
    current_prompt = prompt

    for attempt in range(max_retries):
        response = ""
        try:
            response = llm_provider.generate(
                prompt=current_prompt, system_prompt=system_prompt
            )
            match = pattern.search(response)
            if not match:
                error = f"No JSON {expected_type.__name__} found in response."
                logger.warning(
                    "JSON extraction failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error,
                )
                current_prompt = _build_retry_prompt(prompt, response, error)
                continue

            parsed = json.loads(match.group())
            if not isinstance(parsed, expected_type):
                error = (
                    f"Expected {expected_type.__name__}, got {type(parsed).__name__}."
                )
                logger.warning(
                    "JSON type mismatch (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    error,
                )
                current_prompt = _build_retry_prompt(prompt, response, error)
                continue

            if validate:
                validation_error = validate(parsed)
                if validation_error:
                    logger.warning(
                        "JSON validation failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_retries,
                        validation_error,
                    )
                    current_prompt = _build_retry_prompt(
                        prompt, response, validation_error
                    )
                    continue

            return parsed

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "JSON extraction error (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            current_prompt = _build_retry_prompt(prompt, response, str(e))

    return None

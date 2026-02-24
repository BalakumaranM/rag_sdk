# LLM Providers

The SDK supports 4 LLM providers. Select one via config:

```yaml
llm:
  provider: "openai"  # "openai" | "gemini" | "anthropic" | "cohere"
```

All providers implement the `LLMProvider` interface:

```python
class LLMProvider(ABC):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str: ...
    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]: ...
```

## OpenAI

```yaml
llm:
  provider: "openai"
  openai:
    model: "gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 1000
    base_url: null  # custom API endpoint (e.g., Azure OpenAI)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `OPENAI_API_KEY` |
| `base_url` | `Optional[str]` | `None` | Custom base URL for compatible APIs |
| `model` | `str` | `"gpt-4-turbo-preview"` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `int` | `1000` | Max output tokens |

Uses the chat completions API. The `base_url` parameter enables Azure OpenAI or any OpenAI-compatible API.

## Gemini

```yaml
llm:
  provider: "gemini"
  gemini:
    model: "gemini-2.5-flash"
    temperature: 0.7
    max_output_tokens: 1000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `GOOGLE_API_KEY` |
| `model` | `str` | `"gemini-2.5-flash"` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_output_tokens` | `int` | `1000` | Max output tokens |

Uses the `google-genai` SDK with `generate_content` / `generate_content_stream`.

## Anthropic

```yaml
llm:
  provider: "anthropic"
  anthropic:
    model: "claude-3-5-sonnet-20240620"
    temperature: 0.7
    max_tokens: 1024
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `ANTHROPIC_API_KEY` |
| `model` | `str` | `"claude-3-5-sonnet-20240620"` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `1024` | Max output tokens |

Uses the Anthropic messages API with streaming via `messages.stream()`.

## Cohere

```yaml
llm:
  provider: "cohere"
  cohere:
    model: "command-r-plus"
    temperature: 0.7
    max_tokens: 1000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `Optional[SecretStr]` | `None` | Falls back to `COHERE_API_KEY` |
| `model` | `str` | `"command-r-plus"` | Model name |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `1000` | Max output tokens |

Uses `client.generate()` for non-streaming and `client.chat_stream()` for streaming.

## Streaming

All providers support streaming via the `stream()` method:

```python
for chunk in rag.llm_provider.stream("Tell me about RAG"):
    print(chunk, end="", flush=True)
```

## JSON Extraction Utility

The SDK includes `extract_json_from_llm`, used internally by agentic components (agentic splitter, self-RAG, corrective RAG, etc.) to reliably extract structured JSON from LLM responses.

```python
from rag_sdk.llm import extract_json_from_llm

result = extract_json_from_llm(
    llm_provider=rag.llm_provider,
    prompt="Return a JSON array of 3 colors",
    expected_type=list,  # list or dict
    max_retries=2,       # total attempts (initial + retries)
    validate=None,       # optional validation callback
)
# result: ["red", "blue", "green"] or None on failure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | `LLMProvider` | required | The LLM to call |
| `prompt` | `str` | required | The prompt |
| `system_prompt` | `Optional[str]` | `None` | Optional system prompt |
| `expected_type` | `type` | `list` | Expected root type (`list` or `dict`) |
| `max_retries` | `int` | `2` | Total attempts |
| `validate` | `Optional[Callable]` | `None` | Returns `""` if valid, else error message |

On failure (invalid JSON, wrong type, validation error), it automatically retries with an error-correcting prompt. Returns `None` if all attempts fail.

## See Also

- [Generation Strategies](27-generation-strategies.md) — how LLMs are used for answer generation
- [API: Providers](33-api-providers.md) — `LLMProvider` ABC

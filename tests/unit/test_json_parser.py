from unittest.mock import MagicMock
import pytest
from rag_sdk.llm.json_parser import extract_json_from_llm


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestExtractJsonFromLlm:
    def test_extracts_list_from_clean_response(self, mock_llm):
        mock_llm.generate.return_value = '["a", "b", "c"]'
        result = extract_json_from_llm(mock_llm, "prompt")
        assert result == ["a", "b", "c"]

    def test_extracts_list_from_wrapped_response(self, mock_llm):
        mock_llm.generate.return_value = (
            'Here is the result: ["x", "y"] hope that helps!'
        )
        result = extract_json_from_llm(mock_llm, "prompt")
        assert result == ["x", "y"]

    def test_extracts_dict_from_response(self, mock_llm):
        mock_llm.generate.return_value = '{"key": "value", "num": 42}'
        result = extract_json_from_llm(mock_llm, "prompt", expected_type=dict)
        assert result == {"key": "value", "num": 42}

    def test_returns_none_on_no_json(self, mock_llm):
        mock_llm.generate.return_value = "No JSON here at all."
        result = extract_json_from_llm(mock_llm, "prompt", max_retries=1)
        assert result is None

    def test_returns_none_on_wrong_type(self, mock_llm):
        mock_llm.generate.return_value = '{"key": "value"}'
        result = extract_json_from_llm(
            mock_llm, "prompt", expected_type=list, max_retries=1
        )
        assert result is None

    def test_returns_none_on_invalid_json(self, mock_llm):
        mock_llm.generate.return_value = "[invalid json"
        result = extract_json_from_llm(mock_llm, "prompt", max_retries=1)
        assert result is None

    def test_retries_on_failure_then_succeeds(self, mock_llm):
        mock_llm.generate.side_effect = [
            "no json here",
            '["a", "b"]',
        ]
        result = extract_json_from_llm(mock_llm, "prompt", max_retries=2)
        assert result == ["a", "b"]
        assert mock_llm.generate.call_count == 2

    def test_retry_prompt_includes_error_context(self, mock_llm):
        mock_llm.generate.side_effect = [
            "bad response",
            '["ok"]',
        ]
        extract_json_from_llm(mock_llm, "original prompt", max_retries=2)
        retry_call = mock_llm.generate.call_args_list[1]
        retry_prompt = retry_call.kwargs.get(
            "prompt", retry_call.args[0] if retry_call.args else ""
        )
        assert "previous response was invalid" in retry_prompt.lower()

    def test_passes_system_prompt(self, mock_llm):
        mock_llm.generate.return_value = "[1, 2]"
        extract_json_from_llm(mock_llm, "prompt", system_prompt="be helpful")
        mock_llm.generate.assert_called_once_with(
            prompt="prompt", system_prompt="be helpful"
        )

    def test_validate_callback_rejects_invalid(self, mock_llm):
        mock_llm.generate.return_value = '["not", "ints"]'

        def must_be_ints(parsed):
            if not all(isinstance(x, int) for x in parsed):
                return "All elements must be integers."
            return ""

        result = extract_json_from_llm(
            mock_llm, "prompt", validate=must_be_ints, max_retries=1
        )
        assert result is None

    def test_validate_callback_accepts_valid(self, mock_llm):
        mock_llm.generate.return_value = "[1, 2, 3]"

        def must_be_ints(parsed):
            if not all(isinstance(x, int) for x in parsed):
                return "All elements must be integers."
            return ""

        result = extract_json_from_llm(mock_llm, "prompt", validate=must_be_ints)
        assert result == [1, 2, 3]

    def test_retries_on_validate_failure_then_succeeds(self, mock_llm):
        mock_llm.generate.side_effect = [
            '["not_int"]',
            "[1, 2]",
        ]

        def must_be_ints(parsed):
            if not all(isinstance(x, int) for x in parsed):
                return "All elements must be integers."
            return ""

        result = extract_json_from_llm(
            mock_llm, "prompt", validate=must_be_ints, max_retries=2
        )
        assert result == [1, 2]

    def test_handles_llm_exception(self, mock_llm):
        mock_llm.generate.side_effect = RuntimeError("API down")
        result = extract_json_from_llm(mock_llm, "prompt", max_retries=1)
        assert result is None

    def test_retries_after_exception_then_succeeds(self, mock_llm):
        mock_llm.generate.side_effect = [
            RuntimeError("API down"),
            '{"ok": true}',
        ]
        result = extract_json_from_llm(
            mock_llm, "prompt", expected_type=dict, max_retries=2
        )
        assert result == {"ok": True}

    def test_extracts_nested_json(self, mock_llm):
        mock_llm.generate.return_value = (
            '```json\n{"entities": [{"name": "Alice"}], "relationships": []}\n```'
        )
        result = extract_json_from_llm(mock_llm, "prompt", expected_type=dict)
        assert result["entities"] == [{"name": "Alice"}]

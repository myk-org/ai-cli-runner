from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ai_cli_runner.models import AITokenUsage

from ai_cli_runner.parsers import (
    _extract_json,
    parse_claude_json,
    parse_cursor_json,
    parse_gemini_json,
    parse_json_output,
)

CLAUDE_JSON = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 2647,
        "duration_api_ms": 2581,
        "result": "\n\nHi!",
        "total_cost_usd": 0.0807275,
        "usage": {
            "input_tokens": 3,
            "cache_creation_input_tokens": 11936,
            "cache_read_input_tokens": 11925,
            "output_tokens": 6,
        },
        "modelUsage": {
            "claude-opus-4-6[1m]": {
                "inputTokens": 3,
                "outputTokens": 6,
                "cacheReadInputTokens": 11925,
                "cacheCreationInputTokens": 11936,
                "costUSD": 0.0807275,
            }
        },
    }
)

CURSOR_JSON = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 6661,
        "duration_api_ms": 6661,
        "result": "Hi — good to meet you. How can I help you today?",
        "session_id": "abc123",
        "usage": {
            "inputTokens": 3602,
            "outputTokens": 61,
            "cacheReadTokens": 9728,
            "cacheWriteTokens": 0,
        },
    }
)

GEMINI_JSON_BODY = json.dumps(
    {
        "session_id": "sess456",
        "response": "Hi!",
        "stats": {
            "models": {
                "gemini-3.1-pro-preview": {
                    "api": {
                        "totalRequests": 1,
                        "totalErrors": 0,
                        "totalLatencyMs": 5371,
                    },
                    "tokens": {
                        "input": 10288,
                        "candidates": 2,
                        "total": 10638,
                        "cached": 0,
                        "thoughts": 348,
                        "tool": 0,
                    },
                }
            }
        },
    }
)

GEMINI_JSON_WITH_NOISE = (
    'Skill "something" is overriding the built-in skill.\nAnother warning line\n' + GEMINI_JSON_BODY
)


class TestExtractJson:
    def test_clean_json(self) -> None:
        data = _extract_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_with_noise_prefix(self) -> None:
        raw = 'Some warning line\nAnother line\n{"key": "value"}'
        data = _extract_json(raw)
        assert data == {"key": "value"}

    def test_no_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _extract_json("no json here at all")

    def test_closing_brace_only_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _extract_json("text with } but no opening brace")

    def test_invalid_json_after_brace_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _extract_json("noise {not valid json")

    def test_json_with_trailing_noise(self) -> None:
        raw = '{"key": "value"}\nSome trailing text'
        data = _extract_json(raw)
        assert data == {"key": "value"}

    def test_json_with_both_prefix_and_suffix_noise(self) -> None:
        raw = 'Warning line\n{"key": "value"}\nDone.'
        data = _extract_json(raw)
        assert data == {"key": "value"}


class TestParseClaudeJson:
    @pytest.fixture()
    def parsed(self) -> tuple[str, AITokenUsage | None]:
        return parse_claude_json(CLAUDE_JSON, "claude")

    def test_parses_text(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        text, _usage = parsed
        assert text == "\n\nHi!"

    def test_parses_input_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.input_tokens == 3

    def test_parses_output_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.output_tokens == 6

    def test_parses_cache_read_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_read_tokens == 11925

    def test_parses_cache_write_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_write_tokens == 11936

    def test_parses_cost(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cost_usd == 0.0807275

    def test_parses_duration(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.duration_ms == 2647

    def test_parses_model(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.model == "claude-opus-4-6[1m]"

    def test_parses_provider(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.provider == "claude"


class TestParseCursorJson:
    @pytest.fixture()
    def parsed(self) -> tuple[str, AITokenUsage | None]:
        return parse_cursor_json(CURSOR_JSON, "cursor")

    def test_parses_text(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        text, _usage = parsed
        assert text == "Hi — good to meet you. How can I help you today?"

    def test_parses_input_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.input_tokens == 3602

    def test_parses_output_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.output_tokens == 61

    def test_parses_cache_read_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_read_tokens == 9728

    def test_parses_cache_write_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_write_tokens == 0

    def test_no_cost(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cost_usd is None

    def test_parses_duration(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.duration_ms == 6661

    def test_parses_provider(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.provider == "cursor"

    def test_model_empty_by_design(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.model == ""


class TestParseGeminiJson:
    @pytest.fixture()
    def parsed(self) -> tuple[str, AITokenUsage | None]:
        return parse_gemini_json(GEMINI_JSON_BODY, "gemini")

    def test_parses_text(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        text, _usage = parsed
        assert text == "Hi!"

    def test_parses_text_with_noise(self) -> None:
        text, _usage = parse_gemini_json(GEMINI_JSON_WITH_NOISE, "gemini")
        assert text == "Hi!"

    def test_parses_input_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.input_tokens == 10288

    def test_parses_output_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.output_tokens == 350

    def test_parses_cache_read_tokens(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_read_tokens == 0

    def test_cache_write_tokens_zero(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.cache_write_tokens == 0

    def test_parses_duration(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.duration_ms == 5371

    def test_parses_model(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.model == "gemini-3.1-pro-preview"

    def test_parses_provider(self, parsed: tuple[str, AITokenUsage | None]) -> None:
        _, usage = parsed
        assert usage is not None
        assert usage.provider == "gemini"

    def test_aggregates_multiple_models(self) -> None:
        """Gemini may use multiple models (router + main); verify aggregation."""
        multi_model_json = json.dumps(
            {
                "response": "Hello!",
                "stats": {
                    "models": {
                        "gemini-2.5-flash-lite": {
                            "api": {"totalLatencyMs": 1386},
                            "tokens": {"input": 1197, "candidates": 50, "cached": 0, "thoughts": 127},
                        },
                        "gemini-3-flash-preview": {
                            "api": {"totalLatencyMs": 1124},
                            "tokens": {"input": 8468, "candidates": 24, "cached": 0, "thoughts": 0},
                        },
                    }
                },
            }
        )
        text, usage = parse_gemini_json(multi_model_json, "gemini")
        assert text == "Hello!"
        assert usage is not None
        assert usage.input_tokens == 1197 + 8468  # sum across models
        assert usage.output_tokens == (50 + 127) + (24 + 0)  # candidates + thoughts per model
        assert usage.cache_read_tokens == 0
        assert usage.duration_ms == 1386 + 1124  # sum of latencies
        assert usage.model == "gemini-2.5-flash-lite"
        assert usage.provider == "gemini"


class TestParseJsonOutput:
    def test_routes_to_claude(self) -> None:
        text, usage = parse_json_output(CLAUDE_JSON, "claude")
        assert text == "\n\nHi!"
        assert usage is not None
        assert usage.provider == "claude"

    def test_routes_to_cursor(self) -> None:
        text, usage = parse_json_output(CURSOR_JSON, "cursor")
        assert text == "Hi — good to meet you. How can I help you today?"
        assert usage is not None
        assert usage.provider == "cursor"

    def test_routes_to_gemini(self) -> None:
        text, usage = parse_json_output(GEMINI_JSON_BODY, "gemini")
        assert text == "Hi!"
        assert usage is not None
        assert usage.provider == "gemini"

    def test_unknown_provider_returns_raw(self) -> None:
        text, usage = parse_json_output("raw output", "unknown_provider")
        assert text == "raw output"
        assert usage is None

    def test_invalid_json_returns_raw(self) -> None:
        text, usage = parse_json_output("not json at all", "claude")
        assert text == "not json at all"
        assert usage is None

    def test_best_effort_no_exception(self) -> None:
        # Should never raise, even with garbage input
        text, usage = parse_json_output("", "gemini")
        assert text == ""
        assert usage is None

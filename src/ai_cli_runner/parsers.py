from __future__ import annotations

import json
import os
from typing import Any

from simple_logger.logger import get_logger

from ai_cli_runner.models import AITokenUsage, ParsedOutput

logger = get_logger(name=__name__, level=os.environ.get("LOG_LEVEL", "INFO"))


def _extract_json(raw_output: str) -> dict[str, Any]:
    """Extract JSON from raw CLI output, handling noise lines (e.g., Gemini warnings).

    First tries to parse the whole string. If that fails, finds the first '{'
    and last '}' to extract the JSON block, handling both prefix and suffix noise.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
    """
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise
        return json.loads(raw_output[start : end + 1])


def parse_claude_json(raw_output: str, provider: str) -> tuple[str, AITokenUsage | None, str]:
    """Parse Claude CLI JSON output. Returns (text, usage, thinking)."""
    data = _extract_json(raw_output)
    text = data.get("result", "")

    usage_data = data.get("usage", {})
    model_usage = data.get("modelUsage", {})
    model_name = next(iter(model_usage), "")

    usage = AITokenUsage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
        cost_usd=data.get("total_cost_usd"),
        duration_ms=data.get("duration_ms"),
        model=model_name,
        provider=provider,
        session_id=data.get("session_id", ""),
    )
    return text, usage, ""


def parse_cursor_json(raw_output: str, provider: str) -> tuple[str, AITokenUsage | None, str]:
    """Parse Cursor CLI stream-json output. Returns (text, usage, thinking).

    Cursor uses stream-json format (NDJSON) to separate intermediate tool-use
    reasoning from the final answer. We extract only the last assistant message
    as the result text, and usage from the final result line.
    """
    all_assistant_texts: list[str] = []
    usage: AITokenUsage | None = None
    result_text = ""

    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON line in Cursor stream: %s", line[:120])
            continue

        msg_type = data.get("type", "")

        if msg_type == "assistant":
            # Extract text from assistant message content blocks
            content = data.get("message", {}).get("content", [])
            texts = [block["text"] for block in content if block.get("type") == "text" and block.get("text")]
            joined = "\n".join(texts)
            if joined:
                all_assistant_texts.append(joined)

        elif msg_type == "result":
            result_text = data.get("result", "")
            # Extract usage from the result line
            usage_data = data.get("usage", {})
            usage = AITokenUsage(
                input_tokens=usage_data.get("inputTokens", 0),
                output_tokens=usage_data.get("outputTokens", 0),
                cache_read_tokens=usage_data.get("cacheReadTokens", 0),
                cache_write_tokens=usage_data.get("cacheWriteTokens", 0),
                duration_ms=data.get("duration_ms"),
                provider=provider,
                session_id=data.get("session_id", ""),
            )

    if usage is None:
        logger.warning("No result line found in Cursor stream-json output; returning best-effort text")

    # Use last assistant message if available, otherwise fall back to result text
    if all_assistant_texts:
        text = all_assistant_texts[-1]
        thinking = "\n\n".join(all_assistant_texts[:-1]) if len(all_assistant_texts) >= 2 else ""
    else:
        text = result_text
        thinking = ""
    return text, usage, thinking


def parse_gemini_json(raw_output: str, provider: str) -> tuple[str, AITokenUsage | None, str]:
    """Parse Gemini CLI JSON output. Returns (text, usage, thinking).

    Note: text field is 'response' not 'result'.
    Note: Gemini may use multiple models (e.g., router + main); tokens and
          duration are aggregated across all models. The primary model (highest
          output tokens) is used for cost calculation, so multi-model costs are
          approximate (single rate card applied to summed tokens).
    Note: output tokens are 'candidates' + 'thoughts' (thinking model tokens).
    """
    data = _extract_json(raw_output)
    text = data.get("response", "")

    stats = data.get("stats", {})
    models = stats.get("models", {})

    total_input = 0
    total_output = 0
    total_cached = 0
    total_duration = 0
    primary_model = ""
    primary_output = -1

    for model_name, model_data in models.items():
        tokens = model_data.get("tokens", {})
        api = model_data.get("api", {})

        model_output = tokens.get("candidates", 0) + tokens.get("thoughts", 0)
        total_input += tokens.get("input", 0)
        total_output += model_output
        total_cached += tokens.get("cached", 0)
        total_duration += api.get("totalLatencyMs", 0)

        # Track the primary model (highest output tokens) for pricing lookup.
        # On tie, prefer lexicographically smaller name for determinism.
        if model_output > primary_output or (model_output == primary_output and model_name < primary_model):
            primary_output = model_output
            primary_model = model_name

    usage = AITokenUsage(
        input_tokens=total_input,
        output_tokens=total_output,
        cache_read_tokens=total_cached,
        cache_write_tokens=0,
        duration_ms=total_duration if total_duration > 0 else None,
        model=primary_model,
        provider=provider,
        session_id=data.get("session_id", ""),
    )
    return text, usage, ""


def parse_json_output(raw_output: str, provider: str) -> ParsedOutput:
    """Route to the correct provider parser.

    Best-effort: if parsing fails, log warning and return ParsedOutput with raw text.

    Returns:
        ParsedOutput supporting backward-compatible tuple unpacking:
            text, usage = parse_json_output(...)   # still works
            result = parse_json_output(...)         # access .thinking too
    """
    # Lazy import to avoid circular dependency (providers imports parsers at module level)
    from ai_cli_runner.providers import PROVIDERS

    config = PROVIDERS.get(provider)
    if config is None or config.parse_json is None:
        logger.warning("No JSON parser for provider '%s'; returning raw output", provider)
        return ParsedOutput(text=raw_output, usage=None, thinking="")

    try:
        text, usage, thinking = config.parse_json(raw_output, provider)
        return ParsedOutput(text=text, usage=usage, thinking=thinking)
    except Exception:  # noqa: BLE001 — best-effort: never raise to caller
        logger.warning("Failed to parse JSON output from '%s'; returning raw output", provider, exc_info=True)
        return ParsedOutput(text=raw_output, usage=None, thinking="")

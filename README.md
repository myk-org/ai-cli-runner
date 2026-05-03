# ai-cli-runner

Shared async Python package for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess.

## Usage

```python
import asyncio
from pathlib import Path

from ai_cli_runner import call_ai_cli, check_ai_cli_available, run_parallel_with_limit, AIResult, AITokenUsage


async def main() -> None:
    # Check if AI CLI is available
    available, msg = await check_ai_cli_available(
        ai_provider="claude",
        ai_model="claude-sonnet-4-6",
    )

    # Call Claude
    success, output = await call_ai_cli(
        prompt="Analyze this code",
        cwd=Path("/path/to/repo"),
        ai_provider="claude",
        ai_model="claude-sonnet-4-6",
        cli_flags=["--dangerously-skip-permissions"],
    )

    # Call Gemini
    success, output = await call_ai_cli(
        prompt="Summarize this PR",
        cwd=Path("/path/to/repo"),
        ai_provider="gemini",
        ai_model="gemini-2.5-flash",
        cli_flags=["--yolo"],
    )

    # Call Cursor
    success, output = await call_ai_cli(
        prompt="Fix the failing test",
        cwd=Path("/path/to/repo"),
        ai_provider="cursor",
        ai_model="sonnet-4.6",
        cli_flags=["--force"],
    )

    # Run multiple calls in parallel with bounded concurrency
    prompts = ["Analyze module A", "Analyze module B", "Analyze module C"]
    results = await run_parallel_with_limit(
        [
            call_ai_cli(
                prompt=p,
                ai_provider="claude",
                ai_model="claude-sonnet-4-6",
            )
            for p in prompts
        ],
        max_concurrency=5,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

## JSON Output / Token Usage

Pass `output_format="json"` to get structured token usage metadata from providers that support it. Parsing is best-effort — if the provider output can't be parsed, usage fields fall back to defaults.

```python
result = await call_ai_cli(
    prompt="Analyze this code",
    cwd=Path("/path/to/repo"),
    ai_provider="claude",
    ai_model="claude-sonnet-4-6",
    output_format="json",
)

# Structured result
print(result.success)              # True
print(result.text)                 # "Analysis text..."
print(result.usage.input_tokens)   # 1234
print(result.usage.output_tokens)  # 567
print(result.usage.cost_usd)       # 0.05
print(result.usage.duration_ms)    # 3000
print(result.usage.model)          # "claude-sonnet-4-6"
```

`AIResult` supports tuple unpacking and boolean evaluation for backward compatibility:

```python
# Tuple unpacking still works
success, text = await call_ai_cli(
    prompt="Hello",
    ai_provider="claude",
    ai_model="claude-sonnet-4-6",
)

# Boolean evaluation reflects success
result = await call_ai_cli(...)
if result:
    print(result.text)
```

### `AIResult`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the CLI call succeeded |
| `text` | `str` | Output text from the AI |
| `usage` | `AITokenUsage \| None` | Token usage metadata (when `output_format="json"`) |

### `AITokenUsage`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_tokens` | `int` | `0` | Tokens in the prompt |
| `output_tokens` | `int` | `0` | Tokens in the response |
| `cache_read_tokens` | `int` | `0` | Tokens read from cache |
| `cache_write_tokens` | `int` | `0` | Tokens written to cache |
| `cost_usd` | `float \| None` | `None` | Estimated cost in USD |
| `duration_ms` | `int \| None` | `None` | Wall-clock duration in ms |
| `model` | `str` | `""` | Model used |
| `provider` | `str` | `""` | Provider name |

### Provider Support Matrix

Not all providers return the same metadata. Fields that a provider does not report will be `None` or `0`.

| Field | Claude | Gemini | Cursor |
|-------|--------|--------|--------|
| `input_tokens` | ✅ | ✅ | ✅ |
| `output_tokens` | ✅ | ✅ | ✅ |
| `cache_read_tokens` | ✅ | ✅ | ✅ |
| `cache_write_tokens` | ✅ | ❌ | ✅ |
| `cost_usd` | ✅ | ✅ (calculated) | ✅ (calculated) |
| `duration_ms` | ✅ | ✅ | ✅ |
| `model` | ✅ | ✅ | ❌ |

> **Note:** Claude reports `cost_usd` natively. For Gemini and Cursor, costs are calculated
> using [LiteLLM pricing data](https://github.com/BerriAI/litellm). Call
> `await pricing_cache.load()` at application startup to enable cost calculation.

## Supported Providers

| Provider | Binary | Notes |
|----------|--------|-------|
| `claude` | `claude` | Uses stdin prompt |
| `gemini` | `gemini` | Uses stdin prompt |
| `cursor` | `agent` | Uses `--workspace` for cwd |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_PROVIDER` | `claude` | Which AI CLI to use |
| `AI_MODEL` | (empty) | Model name to pass to the CLI |
| `AI_CLI_TIMEOUT` | `10` | Timeout in minutes for AI CLI calls |

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

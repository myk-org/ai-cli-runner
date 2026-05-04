# ai-cli-runner

Async Python package for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess. Includes LLM cost calculation via [LiteLLM pricing data](https://github.com/BerriAI/litellm) and model listing/validation.

## Install

```bash
uv add ai-cli-runner
```

## Quick Start

```python
import asyncio
from ai_cli_runner import call_ai_cli

async def main():
    result = await call_ai_cli(
        prompt="What is the capital of France?",
        ai_provider="claude",
        ai_model="claude-haiku-4-20250514",
        output_format="json",
    )

    if result.success:
        print(result.text)
        if result.usage:
            print(f"Tokens: in={result.usage.input_tokens} out={result.usage.output_tokens}")
            if result.usage.cost_usd is not None:
                print(f"Cost: ${result.usage.cost_usd:.6f}")

asyncio.run(main())
```

See [`examples/`](examples/) for complete usage:

| Example | What it shows |
|---------|---------------|
| [`basic_call.py`](examples/basic_call.py) | Parallel calls to all 3 providers with token usage |
| [`with_pricing.py`](examples/with_pricing.py) | LLM cost tracking via LiteLLM pricing |
| [`model_listing.py`](examples/model_listing.py) | List models, validate names, check CLI availability |

Run any example: `uv run examples/basic_call.py`

## API

### `call_ai_cli(prompt, cwd, ai_provider, ai_model, ai_cli_timeout, cli_flags, output_format) → AIResult`

Call an AI CLI tool. Pass `output_format="json"` to get structured token usage.

### `check_ai_cli_available(ai_provider, ai_model, cli_flags) → AIResult`

Send a trivial prompt to verify the CLI is installed and working.

### `AIResult`

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the call succeeded |
| `text` | `str` | Response text |
| `usage` | `AITokenUsage \| None` | Token usage (when `output_format="json"`) |

Supports tuple unpacking (`success, text = await call_ai_cli(...)`) and boolean evaluation (`if result: ...`).

### `AITokenUsage`

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Tokens in the prompt |
| `output_tokens` | `int` | Tokens in the response |
| `cache_read_tokens` | `int` | Tokens read from cache |
| `cache_write_tokens` | `int` | Tokens written to cache |
| `cost_usd` | `float \| None` | Cost in USD (native or LiteLLM calculated) |
| `duration_ms` | `int \| None` | Wall-clock duration |
| `model` | `str` | Model used |
| `provider` | `str` | Provider name |

### Cost Calculation

Claude reports cost natively. For Gemini and Cursor, costs are calculated using LiteLLM pricing data:

```python
from ai_cli_runner import pricing_cache

await pricing_cache.load()  # call once at startup
# cost_usd is now auto-populated on all output_format="json" calls
```

### Model Listing & Validation

```python
from ai_cli_runner import model_cache, pricing_cache

await pricing_cache.load()
models = await model_cache.list_models("claude")
is_valid = model_cache.is_valid_model("claude", "claude-haiku-4-20250514")
```

## Supported Providers

| Provider | Binary | Notes |
|----------|--------|-------|
| `claude` | `claude` | `-p` flag for non-interactive mode |
| `gemini` | `gemini` | Stdin prompt |
| `cursor` | `agent` | `--workspace` for cwd |

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_CLI_TIMEOUT` | `10` | Timeout in minutes for AI CLI calls |

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

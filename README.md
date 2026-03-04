# ai-cli-runner

Shared async Python package for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess.

## Usage

```python
import asyncio
from pathlib import Path

from ai_cli_runner import call_ai_cli, check_ai_cli_available, run_parallel_with_limit


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

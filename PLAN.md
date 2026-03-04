# ai-cli-runner — Implementation Plan

## Overview

Extract the shared AI CLI calling functionality from 4 existing projects into a standalone, pip-installable Python package. This package provides a unified async interface for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess.

## GitHub Organization

Repository: `myk-org/ai-cli-runner` (already created at `/home/myakove/git/ai-cli-runner`)

## Why This Package Exists

4 projects currently duplicate the same AI CLI integration code:

1. `/home/myakove/git/jenkins-job-insight/src/jenkins_job_insight/analyzer.py`
2. `/home/myakove/git/pr-test-oracle/src/pr_test_oracle/ai_client.py`
3. `/home/myakove/git/github-webhook-server/webhook_server/libs/ai_cli.py`
4. `/home/myakove/git/docsfy/src/docsfy/ai_client.py`

All 4 implement the same pattern: build a CLI command for an AI provider, run it via subprocess, handle timeouts and errors, return (success, output). This package centralizes that logic.

## What Goes Into This Package

### 1. `call_ai_cli()` — The core function

Canonical signature (from jenkins-job-insight, pr-test-oracle, docsfy):

```python
async def call_ai_cli(
    prompt: str,
    cwd: Path | None = None,
    ai_provider: str = "",
    ai_model: str = "",
    ai_cli_timeout: int | None = None,
) -> tuple[bool, str]:
```

- Runs an AI CLI tool (claude, gemini, or cursor) via subprocess
- Returns `(True, output)` on success, `(False, error_message)` on failure
- Uses `asyncio.to_thread()` to wrap `subprocess.run()`
- Timeout configurable per-call or via `AI_CLI_TIMEOUT` env var (default: 10 minutes)
- Handles: FileNotFoundError (CLI not found), TimeoutExpired, non-zero exit codes, asyncio.CancelledError

### 2. `ProviderConfig` — Provider configuration

```python
@dataclass(frozen=True)
class ProviderConfig:
    binary: str
    build_cmd: Callable[[str, str, str], list[str]]
    uses_own_cwd: bool = False
```

Three providers configured:
- **Claude**: `claude --model <model> --dangerously-skip-permissions -p` (stdin prompt)
- **Gemini**: `gemini --model <model> --yolo` (stdin prompt)
- **Cursor**: `agent --force --model <model> --print --workspace <cwd>` (stdin prompt, uses_own_cwd=True)

When `uses_own_cwd=True`, the subprocess `cwd` is set to `None` because the CLI handles its own working directory via the `--workspace` argument.

### 3. `run_parallel_with_limit()` — Bounded parallel execution

```python
async def run_parallel_with_limit(
    coroutines: list,
    max_concurrency: int = 10,
) -> list:
```

- Uses `asyncio.Semaphore` for bounded concurrency
- Returns results in order
- Handles failures gracefully (one failure doesn't crash all)

Use the pr-test-oracle version which has max_concurrency validation:
```python
if max_concurrency < 1:
    max_concurrency = MAX_CONCURRENT_AI_CALLS
```

### 4. `check_ai_cli_available()` — Sanity check

```python
async def check_ai_cli_available(
    ai_provider: str = "",
    ai_model: str = "",
) -> tuple[bool, str]:
```

- Sends a trivial "Hi" prompt to verify the CLI is installed and working
- Returns (True, output) or (False, error_message)

### 5. Timeout helper

```python
def get_ai_cli_timeout(default_minutes: int = 10) -> int:
    """Get timeout in minutes from AI_CLI_TIMEOUT env var or use default."""
    raw = os.getenv("AI_CLI_TIMEOUT", str(default_minutes))
    try:
        return max(1, int(raw))
    except ValueError:
        return default_minutes
```

The `default_minutes` parameter allows each consumer project to set its own default (e.g., docsfy uses 60, others use 10).

## Project Structure

```
ai-cli-runner/
├── src/ai_cli_runner/
│   ├── __init__.py          # Public API: call_ai_cli, run_parallel_with_limit, check_ai_cli_available
│   ├── client.py            # call_ai_cli(), check_ai_cli_available(), get_ai_cli_timeout()
│   ├── providers.py         # ProviderConfig dataclass, PROVIDERS dict, command builders
│   └── parallel.py          # run_parallel_with_limit()
├── tests/
│   ├── conftest.py
│   ├── test_client.py       # Tests for call_ai_cli, check_ai_cli_available
│   ├── test_providers.py    # Tests for provider command building
│   └── test_parallel.py     # Tests for run_parallel_with_limit
├── pyproject.toml
├── ruff.toml
├── CLAUDE.md
├── README.md
└── PLAN.md
```

## Reference Code — COPY, DO NOT REWRITE

**CRITICAL: Do not reinvent the wheel. Copy the working, tested code from the reference projects and adapt it.**

### Primary reference: `/home/myakove/git/jenkins-job-insight/src/jenkins_job_insight/analyzer.py`
Copy from here:
- `call_ai_cli()` function (the complete implementation)
- `ProviderConfig` dataclass
- `PROVIDERS` dict with all 3 provider configurations
- Provider command builder functions (`_build_claude_cmd`, etc.)
- Timeout handling logic (`_get_ai_cli_timeout`)

### Secondary reference: `/home/myakove/git/pr-test-oracle/src/pr_test_oracle/ai_client.py`
Copy from here:
- `run_parallel_with_limit()` (enhanced version with max_concurrency validation)

### Tertiary reference: `/home/myakove/git/docsfy/src/docsfy/ai_client.py`
Copy from here:
- `check_ai_cli_available()` (more detailed version with better error messages)

## What Does NOT Go Into This Package

These are project-specific and stay in their respective projects:

- `_parse_json_response()` — JSON recovery logic (jenkins-job-insight specific)
- `get_failure_signature()` — Failure deduplication (jenkins-job-insight specific)
- `analyze_failure_group()` — Analysis orchestration (jenkins-job-insight specific)
- `get_ai_config()` — Config dict parsing (github-webhook-server specific)
- Any project-specific prompt building or response parsing

## Public API (`__init__.py`)

```python
from ai_cli_runner.client import call_ai_cli, check_ai_cli_available, get_ai_cli_timeout
from ai_cli_runner.parallel import run_parallel_with_limit
from ai_cli_runner.providers import ProviderConfig, PROVIDERS

__all__ = [
    "call_ai_cli",
    "check_ai_cli_available",
    "get_ai_cli_timeout",
    "run_parallel_with_limit",
    "ProviderConfig",
    "PROVIDERS",
]
```

## Dependencies

Minimal — this package should have very few dependencies:

- `python-simple-logger` — for logging (same as all 4 consumer projects)
- Python stdlib only for everything else (asyncio, subprocess, dataclasses, os, pathlib)

No AI SDKs. No HTTP libraries. The whole point is CLI-based.

## pyproject.toml

Copy structure from `/home/myakove/git/jenkins-job-insight/pyproject.toml` and adapt:
- Name: `ai-cli-runner`
- Package dir: `src/ai_cli_runner`
- Dependencies: only `python-simple-logger`
- Dev dependencies: `pytest`, `pytest-asyncio`
- Same ruff, mypy, pytest config sections

## ruff.toml

Copy as-is from `/home/myakove/git/jenkins-job-insight/ruff.toml`

## CLAUDE.md

Include these coding principles (from jenkins-job-insight):
- Never truncate data arbitrarily
- Use everything you create (no dead code)
- Run independent operations in parallel with graceful failure handling
- AI providers are CLI-based (subprocess), not SDK
- Logging via python-simple-logger
- Use `uv` for dependency management

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_PROVIDER` | `claude` | Which AI CLI to use (claude, gemini, cursor) |
| `AI_MODEL` | (empty) | Model name to pass to the CLI |
| `AI_CLI_TIMEOUT` | `10` | Timeout in minutes for AI CLI calls |

## Consumer Projects — How They Will Use This Package

After this package is built, each consumer project will:

1. Add `ai-cli-runner` as a dependency in their `pyproject.toml`
2. Replace their local AI CLI code with imports from `ai_cli_runner`
3. Remove their duplicated code

### jenkins-job-insight (`/home/myakove/git/jenkins-job-insight`)
- Replace: `call_ai_cli`, `run_parallel_with_limit`, `check_ai_cli_available`, `ProviderConfig`, timeout logic from `analyzer.py`
- Keep: `_parse_json_response`, `get_failure_signature`, `analyze_failure_group` and all analysis-specific code
- Import: `from ai_cli_runner import call_ai_cli, run_parallel_with_limit, check_ai_cli_available`

### pr-test-oracle (`/home/myakove/git/pr-test-oracle`)
- Replace: entire `ai_client.py` contents
- Keep: analysis orchestration in `analyzer.py`
- Import: `from ai_cli_runner import call_ai_cli, run_parallel_with_limit`

### github-webhook-server (`/home/myakove/git/github-webhook-server`)
- Replace: `call_ai_cli` from `webhook_server/libs/ai_cli.py`
- NOTE: This project currently has a different signature (takes `logger` param, uses `str` for cwd). It will need a small adapter or refactor to use the standardized signature.
- Keep: `get_ai_config()` helper (project-specific config parsing)
- Import: `from ai_cli_runner import call_ai_cli`

### docsfy (`/home/myakove/git/docsfy`)
- Replace: entire `ai_client.py` contents
- Note: docsfy uses a 60-minute default timeout. It should call `call_ai_cli(..., ai_cli_timeout=60)` or set `AI_CLI_TIMEOUT=60` in its environment.
- Import: `from ai_cli_runner import call_ai_cli, run_parallel_with_limit, check_ai_cli_available`

## Testing Strategy

Tests should mock `subprocess.run` — do NOT actually call AI CLIs in tests.

Test cases:
- `call_ai_cli` with each provider (claude, gemini, cursor)
- `call_ai_cli` with unknown provider (should fail gracefully)
- `call_ai_cli` timeout handling
- `call_ai_cli` CLI not found (FileNotFoundError)
- `call_ai_cli` non-zero exit code
- `call_ai_cli` with custom cwd
- `call_ai_cli` cursor provider ignores cwd (uses_own_cwd)
- `run_parallel_with_limit` with multiple coroutines
- `run_parallel_with_limit` with failure in one coroutine
- `run_parallel_with_limit` concurrency limiting
- `check_ai_cli_available` success and failure
- `get_ai_cli_timeout` env var parsing
- Provider command building for each provider

## Implementation Order

1. Set up project scaffolding (pyproject.toml, ruff.toml, CLAUDE.md, directory structure)
2. Implement `providers.py` (ProviderConfig, PROVIDERS, command builders)
3. Implement `client.py` (call_ai_cli, check_ai_cli_available, get_ai_cli_timeout)
4. Implement `parallel.py` (run_parallel_with_limit)
5. Write `__init__.py` with public exports
6. Write tests
7. Write README.md

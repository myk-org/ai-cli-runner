# ai-cli-runner

## What This Is
Shared async Python package for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess. Includes LLM cost calculation via LiteLLM pricing data with disk-based caching.

## Coding Principles
- Never truncate data arbitrarily
- Use everything you create (no dead code)
- Run independent operations in parallel with graceful failure handling
- AI providers are CLI-based (subprocess), not SDK
- Logging via python-simple-logger
- Use `uv` for dependency management

## Project Structure
- `src/ai_cli_runner/` — package source
  - `client.py` — main `call_ai_cli()` function, subprocess management
  - `models.py` — `AIResult` (with `thinking` field for chain-of-thought), `AITokenUsage`, `ParsedOutput` dataclasses
  - `providers.py` — provider configs (claude, gemini, cursor)
  - `parsers.py` — JSON output parsers per provider (returns `ParsedOutput`)
  - `parallel.py` — `run_parallel_with_limit()` concurrency helper
  - `llm_pricing.py` — LLM cost calculation via LiteLLM pricing data with disk cache
  - `ai_models.py` — `AIModelCache` for model listing and validation per provider
- `tests/` — pytest test suite
- `examples/` — usage examples (basic_call, with_pricing, model_listing)

## Dependencies
- `python-simple-logger` — logging
- `httpx` — HTTP client for fetching LiteLLM pricing data

## Commands
- Run tests: `uv run pytest`
- Run linter: `uv run ruff check .`
- Format: `uv run ruff format .`

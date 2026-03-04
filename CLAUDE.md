# ai-cli-runner

## What This Is
Shared async Python package for calling AI CLI tools (Claude, Gemini, Cursor) via subprocess.

## Coding Principles
- Never truncate data arbitrarily
- Use everything you create (no dead code)
- Run independent operations in parallel with graceful failure handling
- AI providers are CLI-based (subprocess), not SDK
- Logging via python-simple-logger
- Use `uv` for dependency management

## Project Structure
- `src/ai_cli_runner/` — package source
- `tests/` — pytest test suite

## Commands
- Run tests: `uv run pytest`
- Run linter: `uv run ruff check .`
- Format: `uv run ruff format .`

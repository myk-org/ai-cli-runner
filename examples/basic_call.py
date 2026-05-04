# Run: uv run examples/basic_call.py

"""Basic example: call AI CLI tools."""

import asyncio

from ai_cli_runner import call_ai_cli

CLAUDE_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
CURSOR_MODEL = "gpt-5.4-nano-none"

# CLI flags needed for non-interactive/headless execution
CLAUDE_FLAGS = ["--dangerously-skip-permissions"]
GEMINI_FLAGS = ["--skip-trust"]
CURSOR_FLAGS = ["--trust"]

TIMEOUT_MINUTES = 2


async def main() -> None:
    # Call all 3 providers in parallel
    results = await asyncio.gather(
        call_ai_cli(
            prompt="What is the capital of France?",
            ai_provider="claude",
            ai_model=CLAUDE_MODEL,
            ai_cli_timeout=TIMEOUT_MINUTES,
            cli_flags=CLAUDE_FLAGS,
            output_format="json",
        ),
        call_ai_cli(
            prompt="What is 2 + 2?",
            ai_provider="gemini",
            ai_model=GEMINI_MODEL,
            ai_cli_timeout=TIMEOUT_MINUTES,
            cli_flags=GEMINI_FLAGS,
            output_format="json",
        ),
        call_ai_cli(
            prompt="What color is the sky?",
            ai_provider="cursor",
            ai_model=CURSOR_MODEL,
            ai_cli_timeout=TIMEOUT_MINUTES,
            cli_flags=CURSOR_FLAGS,
            output_format="json",
        ),
    )

    for result in results:
        provider = result.usage.provider if result.usage else "unknown"
        model = result.usage.model if result.usage else "unknown"
        prefix = f"[{provider}/{model}]"

        if result.success:
            print(f"{prefix} {result.text}")
            if result.usage:
                print(f"{prefix} Tokens: in={result.usage.input_tokens} out={result.usage.output_tokens}")
        else:
            print(f"{prefix} ERROR: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())

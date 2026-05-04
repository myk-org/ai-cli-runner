# Run: uv run examples/with_pricing.py

"""Example: AI CLI calls with LLM cost tracking."""

import asyncio

from ai_cli_runner import call_ai_cli, pricing_cache

CLAUDE_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
CURSOR_MODEL = "gpt-5.4-nano-none"

# CLI flags needed for non-interactive/headless execution
CLAUDE_FLAGS = ["--dangerously-skip-permissions"]
GEMINI_FLAGS = ["--skip-trust"]
CURSOR_FLAGS = ["--trust"]

TIMEOUT_MINUTES = 2


async def main() -> None:
    # Load pricing data at startup (uses disk cache if fresh < 24h)
    await pricing_cache.load()

    # Call all providers in parallel
    providers = [
        ("claude", CLAUDE_MODEL, CLAUDE_FLAGS),
        ("gemini", GEMINI_MODEL, GEMINI_FLAGS),
        ("cursor", CURSOR_MODEL, CURSOR_FLAGS),
    ]

    results = await asyncio.gather(
        *(
            call_ai_cli(
                prompt="What is 1 + 1?",
                ai_provider=provider,
                ai_model=model,
                ai_cli_timeout=TIMEOUT_MINUTES,
                cli_flags=flags,
                output_format="json",
            )
            for provider, model, flags in providers
        )
    )

    for result in results:
        provider = result.usage.provider if result.usage else "unknown"
        model = result.usage.model if result.usage else "unknown"

        print(f"\n--- {provider}/{model} ---")
        if result.success and result.usage:
            print(f"Response:      {result.text}")
            print(f"Input tokens:  {result.usage.input_tokens}")
            print(f"Output tokens: {result.usage.output_tokens}")
            if result.usage.cost_usd is not None:
                print(f"Cost:          ${result.usage.cost_usd:.6f}")
            else:
                print("Cost:          N/A")
        elif result.success:
            print(f"Response: {result.text}")
        else:
            print(f"Error: {result.text}")

    # For long-running apps, refresh pricing in the background:
    #
    #   await pricing_cache.start_background_refresh()
    #   try:
    #       await run_app()  # your app logic here
    #   finally:
    #       await pricing_cache.stop_background_refresh()


if __name__ == "__main__":
    asyncio.run(main())

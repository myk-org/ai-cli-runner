# Run: uv run examples/model_listing.py

"""Example: list available models and validate model names."""

import asyncio

from ai_cli_runner import check_ai_cli_available, model_cache, pricing_cache

CLAUDE_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
CURSOR_MODEL = "gpt-5.4-nano-none"

# CLI flags needed for non-interactive/headless execution
CLAUDE_FLAGS = ["--dangerously-skip-permissions"]
GEMINI_FLAGS = ["--skip-trust"]
CURSOR_FLAGS = ["--trust"]


async def main() -> None:
    # Load pricing data (needed for Claude/Gemini model lists)
    await pricing_cache.load()

    # List available models per provider
    for provider in ("claude", "gemini", "cursor"):
        models = await model_cache.list_models(provider)
        print(f"\n{provider.upper()} models ({len(models)}):")
        for model in models[:10]:
            print(f"  {model.get('id', 'unknown')} — {model.get('name', 'unknown')}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

    # Validate model names
    validations = [
        ("claude", CLAUDE_MODEL),
        ("gemini", GEMINI_MODEL),
        ("cursor", CURSOR_MODEL),
        ("claude", "nonexistent-model"),
    ]
    print("\nModel validation:")
    for provider, model_name in validations:
        is_valid = model_cache.is_valid_model(provider, model_name)
        print(f"  {provider}/{model_name}: {is_valid}")

    # Check CLI availability for each provider
    providers = [
        ("claude", CLAUDE_MODEL, CLAUDE_FLAGS),
        ("gemini", GEMINI_MODEL, GEMINI_FLAGS),
        ("cursor", CURSOR_MODEL, CURSOR_FLAGS),
    ]
    print("\nCLI availability:")
    for ai_provider, ai_model, flags in providers:
        result = await check_ai_cli_available(
            ai_provider=ai_provider,
            ai_model=ai_model,
            cli_flags=flags,
        )
        status = "✅" if result.success else f"❌ {result.text}"
        print(f"  {ai_provider}/{ai_model}: {status}")


if __name__ == "__main__":
    asyncio.run(main())

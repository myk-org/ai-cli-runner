# Run: uv run examples/session_usage.py

"""Example: session management — start, resume, and continue sessions."""

import asyncio

from ai_cli_runner import call_ai_cli

CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_FLAGS = ["--dangerously-skip-permissions"]
TIMEOUT_MINUTES = 2


async def main() -> None:
    # Step 1: Start a new session
    print("--- Starting new session ---")
    result = await call_ai_cli(
        prompt="My name is Alice. Remember that.",
        ai_provider="claude",
        ai_model=CLAUDE_MODEL,
        ai_cli_timeout=TIMEOUT_MINUTES,
        cli_flags=CLAUDE_FLAGS,
        output_format="json",
    )

    if not result.success:
        print(f"Error: {result.text}")
        return

    print(f"Response: {result.text}")
    print(f"Session ID: {result.session_id}")

    if not result.session_id:
        print("No session ID returned — session features require output_format='json'")
        return

    # Step 2: Resume the session by ID with a follow-up question
    print("\n--- Resuming session by ID ---")
    followup = await call_ai_cli(
        prompt="What is my name?",
        ai_provider="claude",
        ai_model=CLAUDE_MODEL,
        ai_cli_timeout=TIMEOUT_MINUTES,
        cli_flags=CLAUDE_FLAGS,
        output_format="json",
        session_id=result.session_id,
    )

    if followup.success:
        print(f"Response: {followup.text}")
        print(f"Session ID: {followup.session_id}")
    else:
        print(f"Error: {followup.text}")

    # Step 3: Continue the most recent session (no ID needed)
    print("\n--- Continuing most recent session ---")
    continued = await call_ai_cli(
        prompt="And what did I ask you to remember?",
        ai_provider="claude",
        ai_model=CLAUDE_MODEL,
        ai_cli_timeout=TIMEOUT_MINUTES,
        cli_flags=CLAUDE_FLAGS,
        output_format="json",
        continue_session=True,
    )

    if continued.success:
        print(f"Response: {continued.text}")
        print(f"Session ID: {continued.session_id}")
    else:
        print(f"Error: {continued.text}")


if __name__ == "__main__":
    asyncio.run(main())

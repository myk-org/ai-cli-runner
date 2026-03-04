import asyncio
import os
import subprocess
from pathlib import Path

from simple_logger.logger import get_logger

from ai_cli_runner.providers import PROVIDERS, VALID_AI_PROVIDERS

logger = get_logger(name=__name__, level=os.environ.get("LOG_LEVEL", "INFO"))

DEFAULT_TIMEOUT_MINUTES = 10
SANITY_CHECK_TIMEOUT_SECONDS = 60


def get_ai_cli_timeout(default_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> int:
    """Get timeout in minutes from AI_CLI_TIMEOUT env var or use default."""
    raw = os.getenv("AI_CLI_TIMEOUT", str(default_minutes))
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid AI_CLI_TIMEOUT=%s; defaulting to %d", raw, default_minutes)
        return default_minutes
    if value <= 0:
        logger.warning("Non-positive AI_CLI_TIMEOUT=%s; defaulting to %d", raw, default_minutes)
        return default_minutes
    return value


async def call_ai_cli(
    prompt: str,
    cwd: Path | None = None,
    ai_provider: str = "",
    ai_model: str = "",
    ai_cli_timeout: int | None = None,
    cli_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Call AI CLI (Claude, Gemini, or Cursor) with given prompt.

    Args:
        prompt: The prompt to send to the AI CLI.
        cwd: Working directory for AI to explore (typically repo path).
        ai_provider: AI provider to use.
        ai_model: AI model to use.
        ai_cli_timeout: Timeout in minutes (overrides AI_CLI_TIMEOUT env var).
        cli_flags: Extra CLI flags to pass to the provider command.

    Returns:
        Tuple of (success, output). success is True with AI output, False with error message.
    """
    config = PROVIDERS.get(ai_provider)
    if not config:
        return (
            False,
            f"Unknown AI provider: '{ai_provider}'. Valid providers: {', '.join(sorted(VALID_AI_PROVIDERS))}",
        )

    if not ai_model:
        return (
            False,
            "No AI model configured. Set AI_MODEL env var or pass ai_model in request body.",
        )

    provider_info = f"{ai_provider.upper()} ({ai_model})"
    cmd = config.build_cmd(config.binary, ai_model, cwd, cli_flags or [])

    subprocess_cwd = None if config.uses_own_cwd else cwd

    if ai_cli_timeout is None:
        effective_timeout = get_ai_cli_timeout()
    elif ai_cli_timeout <= 0:
        return False, f"Invalid ai_cli_timeout: {ai_cli_timeout}. Must be a positive integer (minutes)."
    else:
        effective_timeout = ai_cli_timeout
    timeout = effective_timeout * 60  # Convert minutes to seconds

    logger.info("Calling %s CLI", provider_info)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            cwd=subprocess_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=prompt,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"{provider_info} CLI error: Analysis timed out after {effective_timeout} minutes",
        )
    except FileNotFoundError:
        return (
            False,
            f"{provider_info} CLI error: '{config.binary}' not found. Ensure the CLI is installed and on PATH.",
        )

    if result.returncode != 0:
        error_detail = result.stderr or result.stdout or "unknown error (no output)"
        return False, f"{provider_info} CLI error: {error_detail}"

    logger.debug("%s CLI response length: %d chars", provider_info, len(result.stdout))
    return True, result.stdout


async def check_ai_cli_available(
    ai_provider: str = "",
    ai_model: str = "",
    cli_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if an AI CLI tool is available and working.

    Sends a trivial "Hi" prompt to verify the CLI is installed and working.

    Args:
        ai_provider: AI provider to check.
        ai_model: AI model to check.
        cli_flags: Extra CLI flags to pass to the provider command.

    Returns:
        Tuple of (available, message). available is True if working, False with error message.
    """
    config = PROVIDERS.get(ai_provider)
    if not config:
        return False, f"Unknown AI provider: '{ai_provider}'"
    if not ai_model:
        return False, "No AI model configured"

    provider_info = f"{ai_provider.upper()} ({ai_model})"
    sanity_cmd = config.build_cmd(config.binary, ai_model, None, cli_flags or [])

    try:
        sanity_result = await asyncio.to_thread(
            subprocess.run,
            sanity_cmd,
            cwd=None,
            capture_output=True,
            text=True,
            timeout=SANITY_CHECK_TIMEOUT_SECONDS,
            input="Hi",
        )
        if sanity_result.returncode != 0:
            error_detail = sanity_result.stderr or sanity_result.stdout or "unknown"
            return False, f"{provider_info} sanity check failed: {error_detail}"
    except subprocess.TimeoutExpired:
        return False, f"{provider_info} sanity check timed out"
    except FileNotFoundError:
        return False, f"{provider_info}: '{config.binary}' not found in PATH"

    return True, ""

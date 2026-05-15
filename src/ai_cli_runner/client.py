import asyncio
import contextlib
import os
import signal
import subprocess
from pathlib import Path
from typing import Literal

from simple_logger.logger import get_logger

from ai_cli_runner.llm_pricing import pricing_cache
from ai_cli_runner.models import AIResult
from ai_cli_runner.parsers import parse_json_output
from ai_cli_runner.providers import PROVIDERS, VALID_AI_PROVIDERS, ProviderConfig

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


def _validate_provider_and_model(ai_provider: str, ai_model: str) -> tuple[bool, str, ProviderConfig | None]:
    """Validate AI provider and model configuration.

    Returns:
        Tuple of (valid, error_message, config). If valid is False, error_message explains why.
    """
    config = PROVIDERS.get(ai_provider)
    if not config:
        return (
            False,
            f"Unknown AI provider: '{ai_provider}'. Valid providers: {', '.join(sorted(VALID_AI_PROVIDERS))}",
            None,
        )
    if not ai_model:
        return False, "No AI model configured. Set AI_MODEL env var or pass ai_model.", None
    return True, "", config


def _run_with_process_group(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | float | None = None,
    input_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess in its own process group for reliable timeout enforcement.

    Using ``start_new_session=True`` places the child (and any processes it
    spawns) in a new process group.  On timeout we send SIGTERM to the entire
    group, wait briefly, then escalate to SIGKILL -- ensuring no orphan
    processes survive.

    Args:
        cmd: Command to execute.
        cwd: Working directory for the subprocess.
        timeout: Maximum wall-clock seconds to allow.
        input_data: Data to write to the subprocess's stdin.

    Returns:
        A ``CompletedProcess`` with captured stdout/stderr.

    Raises:
        subprocess.TimeoutExpired: If the process exceeds *timeout*.
        FileNotFoundError: If the command binary is not found.
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        raise
    except BaseException:
        _kill_process_group(process)
        raise

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """Kill a subprocess and its entire process group.

    Sends SIGTERM first for graceful shutdown, waits up to 5 seconds,
    then escalates to SIGKILL if the process group is still alive.

    Args:
        process: The Popen instance whose process group should be killed.
    """
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        # Process already gone; nothing to kill
        process.communicate()
        return

    # Graceful shutdown attempt
    with contextlib.suppress(OSError):
        os.killpg(pgid, signal.SIGTERM)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Escalate to SIGKILL
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)
        process.wait()

    # Drain any remaining output to avoid ResourceWarning
    process.communicate()


async def call_ai_cli(
    prompt: str,
    cwd: Path | None = None,
    ai_provider: str = "",
    ai_model: str = "",
    ai_cli_timeout: int | None = None,
    cli_flags: list[str] | None = None,
    output_format: Literal["json"] | None = None,
    session_id: str | None = None,
    continue_session: bool = False,
) -> AIResult:
    """Call AI CLI (Claude, Gemini, or Cursor) with given prompt.

    Args:
        prompt: The prompt to send to the AI CLI.
        cwd: Working directory for AI to explore (typically repo path).
        ai_provider: AI provider to use.
        ai_model: AI model to use.
        ai_cli_timeout: Timeout in minutes (overrides AI_CLI_TIMEOUT env var).
        cli_flags: Extra CLI flags to pass to the provider command.
        output_format: Output format ("json" or None). When set, parses structured output.
        session_id: Resume a specific session by ID. Mutually exclusive with continue_session.
        continue_session: Continue the most recent session. Mutually exclusive with session_id.

    Returns:
        AIResult with success status, text output, optional usage metadata, and session_id.
        Supports tuple unpacking: success, text = await call_ai_cli(...)

    Note:
        Cost calculation via LiteLLM pricing requires ``await pricing_cache.load()``
        at application startup. Without it, ``cost_usd`` will only be populated
        for providers that report it natively (e.g., Claude).

        Session parameters (``session_id``, ``continue_session``) work with any
        output format — the CLI will resume/continue the conversation regardless.
        However, ``AIResult.session_id`` is only populated when ``output_format="json"``
        is set, since session IDs are extracted from the JSON response.
    """
    valid, error_msg, config = _validate_provider_and_model(ai_provider, ai_model)
    if not valid:
        return AIResult(success=False, text=error_msg)

    if config is None:  # defensive: guaranteed by _validate_provider_and_model
        raise RuntimeError("ProviderConfig unexpectedly None after validation")

    if session_id is not None and continue_session:
        return AIResult(
            success=False,
            text=(
                "Cannot use both session_id and continue_session."
                " Use session_id to resume a specific session,"
                " or continue_session to continue the most recent session."
            ),
        )

    provider_info = f"{ai_provider.upper()} ({ai_model})"

    effective_cli_flags = list(cli_flags or [])
    if output_format:
        # Remove any existing --output-format flag (both "--output-format value" and "--output-format=value" forms)
        cleaned_flags: list[str] = []
        skip_next = False
        found_existing = False
        for i, flag in enumerate(effective_cli_flags):
            if skip_next:
                skip_next = False
                continue
            if flag == "--output-format":
                found_existing = True
                # Skip the next element (the value), if present
                if i + 1 < len(effective_cli_flags):
                    skip_next = True
                continue
            if flag.startswith("--output-format="):
                found_existing = True
                continue
            cleaned_flags.append(flag)
        if found_existing:
            logger.warning(
                "Caller-supplied --output-format in cli_flags will be overridden by output_format=%r",
                output_format,
            )
        wire_format = config.json_wire_format

        # Strip partial-streaming flags — partial assistant deltas would
        # break parsers that expect complete turns.
        # Known flags: Cursor --stream-partial-output, Claude --include-partial-messages
        partial_flags = {"--stream-partial-output", "--include-partial-messages"}
        found_partial = partial_flags & set(cleaned_flags)
        if found_partial:
            for flag in sorted(found_partial):
                logger.warning(
                    "Stripping %s from cli_flags — incompatible with structured output parsing",
                    flag,
                )
            cleaned_flags = [f for f in cleaned_flags if f not in partial_flags]

        effective_cli_flags = ["--output-format", wire_format, *cleaned_flags]

    if continue_session:
        effective_cli_flags.extend(config.continue_flags)
    elif session_id is not None:
        effective_cli_flags.extend([config.resume_flag, session_id])

    cmd = config.build_cmd(config.binary, ai_model, cwd, effective_cli_flags)

    if ai_cli_timeout is None:
        effective_timeout = get_ai_cli_timeout()
    elif ai_cli_timeout <= 0:
        return AIResult(
            success=False,
            text=f"Invalid ai_cli_timeout: {ai_cli_timeout}. Must be a positive integer (minutes).",
        )
    else:
        effective_timeout = ai_cli_timeout
    timeout = effective_timeout * 60  # Convert minutes to seconds

    logger.info("Calling %s CLI", provider_info)

    try:
        result = await asyncio.to_thread(
            _run_with_process_group,
            cmd,
            cwd=cwd,
            timeout=timeout,
            input_data=prompt,
        )
    except subprocess.TimeoutExpired:
        return AIResult(
            success=False,
            text=f"{provider_info} CLI error: Analysis timed out after {effective_timeout} minutes",
        )
    except FileNotFoundError:
        return AIResult(
            success=False,
            text=f"{provider_info} CLI error: '{config.binary}' not found. Ensure the CLI is installed and on PATH.",
        )

    if result.returncode != 0:
        error_detail = result.stderr or result.stdout or "unknown error (no output)"
        return AIResult(success=False, text=f"{provider_info} CLI error: {error_detail}")

    logger.debug("%s CLI response length: %d chars", provider_info, len(result.stdout))

    if output_format:
        parsed = parse_json_output(result.stdout, ai_provider)
        if parsed.usage is not None and parsed.usage.cost_usd is None:
            parsed.usage.cost_usd = pricing_cache.calculate_cost(
                provider=parsed.usage.provider or ai_provider,
                model=parsed.usage.model or ai_model,
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
                cache_read_tokens=parsed.usage.cache_read_tokens,
                cache_write_tokens=parsed.usage.cache_write_tokens,
            )
        if parsed.usage is None:
            logger.debug(
                "%s: output_format=%r requested but no usage parsed; raw output length=%d",
                provider_info,
                output_format,
                len(result.stdout),
            )
        # Bridge AITokenUsage.session_id (str, defaults "") → AIResult.session_id (str | None)
        session = parsed.usage.session_id if parsed.usage else None
        return AIResult(
            success=True,
            text=parsed.text,
            usage=parsed.usage,
            session_id=session or None,
            thinking=parsed.thinking,
        )

    return AIResult(success=True, text=result.stdout)


async def check_ai_cli_available(
    ai_provider: str = "",
    ai_model: str = "",
    cli_flags: list[str] | None = None,
) -> AIResult:
    """Check if an AI CLI tool is available and working.

    Sends a trivial "Hi" prompt to verify the CLI is installed and working.

    Args:
        ai_provider: AI provider to check.
        ai_model: AI model to check.
        cli_flags: Extra CLI flags to pass to the provider command.

    Returns:
        AIResult with success=True if working, success=False with error message.
        Supports tuple unpacking: available, msg = await check_ai_cli_available(...)
    """
    valid, error_msg, config = _validate_provider_and_model(ai_provider, ai_model)
    if not valid:
        return AIResult(success=False, text=error_msg)

    if config is None:  # defensive: guaranteed by _validate_provider_and_model
        raise RuntimeError("ProviderConfig unexpectedly None after validation")

    provider_info = f"{ai_provider.upper()} ({ai_model})"
    sanity_cmd = config.build_cmd(config.binary, ai_model, None, cli_flags or [])

    try:
        sanity_result = await asyncio.to_thread(
            _run_with_process_group,
            sanity_cmd,
            cwd=None,
            timeout=SANITY_CHECK_TIMEOUT_SECONDS,
            input_data="Hi",
        )
        if sanity_result.returncode != 0:
            error_detail = sanity_result.stderr or sanity_result.stdout or "unknown"
            return AIResult(success=False, text=f"{provider_info} sanity check failed: {error_detail}")
    except subprocess.TimeoutExpired:
        return AIResult(success=False, text=f"{provider_info} sanity check timed out")
    except FileNotFoundError:
        return AIResult(success=False, text=f"{provider_info}: '{config.binary}' not found in PATH")

    return AIResult(success=True, text="")

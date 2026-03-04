import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from simple_logger.logger import get_logger

logger = get_logger(name=__name__, level=os.environ.get("LOG_LEVEL", "INFO"))


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI CLI provider."""

    binary: str
    build_cmd: Callable[[str, str, Path | None, list[str]], list[str]]
    uses_own_cwd: bool = False


def _build_claude_cmd(binary: str, model: str, _cwd: Path | None, cli_flags: list[str]) -> list[str]:
    return [binary, "--model", model, *cli_flags]


def _build_gemini_cmd(binary: str, model: str, _cwd: Path | None, cli_flags: list[str]) -> list[str]:
    return [binary, "--model", model, *cli_flags]


def _build_cursor_cmd(binary: str, model: str, cwd: Path | None, cli_flags: list[str]) -> list[str]:
    cmd = [binary, "--model", model, *cli_flags]
    if cwd:
        cmd.extend(["--workspace", str(cwd)])
    return cmd


PROVIDERS: dict[str, ProviderConfig] = {
    "claude": ProviderConfig(binary="claude", build_cmd=_build_claude_cmd),
    "gemini": ProviderConfig(binary="gemini", build_cmd=_build_gemini_cmd),
    "cursor": ProviderConfig(binary="agent", uses_own_cwd=True, build_cmd=_build_cursor_cmd),
}

VALID_AI_PROVIDERS: frozenset[str] = frozenset(PROVIDERS.keys())

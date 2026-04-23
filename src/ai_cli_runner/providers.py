from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_cli_runner.parsers import parse_claude_json, parse_cursor_json, parse_gemini_json

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ai_cli_runner.models import AITokenUsage


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI CLI provider."""

    binary: str
    build_cmd: Callable[[str, str, Path | None, list[str]], list[str]]
    parse_json: Callable[[str, str], tuple[str, AITokenUsage | None]] | None = None


def _build_claude_cmd(binary: str, model: str, _cwd: Path | None, cli_flags: list[str]) -> list[str]:
    # -p: non-interactive print mode, required for subprocess piping
    return [binary, "--model", model, "-p", *cli_flags]


def _build_gemini_cmd(binary: str, model: str, _cwd: Path | None, cli_flags: list[str]) -> list[str]:
    return [binary, "--model", model, *cli_flags]


def _build_cursor_cmd(binary: str, model: str, cwd: Path | None, cli_flags: list[str]) -> list[str]:
    # --print: non-interactive print mode, required for subprocess piping
    cmd = [binary, "--model", model, "--print", *cli_flags]
    if cwd:
        cmd.extend(["--workspace", str(cwd)])
    return cmd


PROVIDERS: dict[str, ProviderConfig] = {
    "claude": ProviderConfig(binary="claude", build_cmd=_build_claude_cmd, parse_json=parse_claude_json),
    "gemini": ProviderConfig(binary="gemini", build_cmd=_build_gemini_cmd, parse_json=parse_gemini_json),
    "cursor": ProviderConfig(binary="agent", build_cmd=_build_cursor_cmd, parse_json=parse_cursor_json),
}

VALID_AI_PROVIDERS: frozenset[str] = frozenset(PROVIDERS.keys())

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an AI CLI provider."""

    binary: str
    build_cmd: Callable[[str, str, Path | None, list[str]], list[str]]


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
    "claude": ProviderConfig(binary="claude", build_cmd=_build_claude_cmd),
    "gemini": ProviderConfig(binary="gemini", build_cmd=_build_gemini_cmd),
    "cursor": ProviderConfig(binary="agent", build_cmd=_build_cursor_cmd),
}

VALID_AI_PROVIDERS: frozenset[str] = frozenset(PROVIDERS.keys())

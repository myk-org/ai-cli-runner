from pathlib import Path

import pytest

from ai_cli_runner.providers import (
    PROVIDERS,
    VALID_AI_PROVIDERS,
    ProviderConfig,
    _build_claude_cmd,
    _build_cursor_cmd,
    _build_gemini_cmd,
)


class TestProviderConfig:
    async def test_provider_config_attributes(self) -> None:
        config = ProviderConfig(binary="test-bin", build_cmd=_build_claude_cmd)
        assert config.binary == "test-bin"
        assert config.build_cmd is _build_claude_cmd
        assert config.uses_own_cwd is False

    async def test_provider_config_uses_own_cwd_default_false(self) -> None:
        config = ProviderConfig(binary="x", build_cmd=_build_claude_cmd)
        assert config.uses_own_cwd is False

    async def test_provider_config_uses_own_cwd_explicit_true(self) -> None:
        config = ProviderConfig(binary="x", build_cmd=_build_claude_cmd, uses_own_cwd=True)
        assert config.uses_own_cwd is True

    async def test_provider_config_is_frozen(self) -> None:
        config = ProviderConfig(binary="x", build_cmd=_build_claude_cmd)
        with pytest.raises(AttributeError):
            config.binary = "y"  # type: ignore[misc]

    async def test_build_cmd_callable_with_four_args(self) -> None:
        config = ProviderConfig(binary="claude", build_cmd=_build_claude_cmd)
        cmd = config.build_cmd(config.binary, "opus-4", None, [])
        assert cmd == ["claude", "--model", "opus-4", "-p"]


class TestBuildClaudeCmd:
    async def test_basic_command_no_flags(self) -> None:
        cmd = _build_claude_cmd("claude", "opus-4", None, [])
        assert cmd == ["claude", "--model", "opus-4", "-p"]

    async def test_with_flags(self) -> None:
        cmd = _build_claude_cmd("claude", "opus-4", None, ["--dangerously-skip-permissions"])
        assert cmd == ["claude", "--model", "opus-4", "-p", "--dangerously-skip-permissions"]

    async def test_ignores_cwd(self) -> None:
        cwd = Path("/some/path")
        cmd_with_cwd = _build_claude_cmd("claude", "opus-4", cwd, [])
        cmd_without_cwd = _build_claude_cmd("claude", "opus-4", None, [])
        assert cmd_with_cwd == cmd_without_cwd

    async def test_custom_binary(self) -> None:
        cmd = _build_claude_cmd("/usr/local/bin/claude", "sonnet", None, [])
        assert cmd[0] == "/usr/local/bin/claude"
        assert cmd[2] == "sonnet"
        assert "-p" in cmd


class TestBuildGeminiCmd:
    async def test_basic_command_no_flags(self) -> None:
        cmd = _build_gemini_cmd("gemini", "gemini-2.0-flash", None, [])
        assert cmd == ["gemini", "--model", "gemini-2.0-flash"]

    async def test_with_flags(self) -> None:
        cmd = _build_gemini_cmd("gemini", "flash", None, ["--yolo"])
        assert cmd == ["gemini", "--model", "flash", "--yolo"]

    async def test_ignores_cwd(self) -> None:
        cwd = Path("/workspace")
        cmd_with_cwd = _build_gemini_cmd("gemini", "flash", cwd, [])
        cmd_without_cwd = _build_gemini_cmd("gemini", "flash", None, [])
        assert cmd_with_cwd == cmd_without_cwd

    async def test_custom_binary(self) -> None:
        cmd = _build_gemini_cmd("/opt/gemini", "pro", None, [])
        assert cmd[0] == "/opt/gemini"


class TestBuildCursorCmd:
    async def test_without_cwd_no_flags(self) -> None:
        cmd = _build_cursor_cmd("agent", "gpt-4", None, [])
        assert cmd == ["agent", "--model", "gpt-4", "--print"]

    async def test_with_cwd_and_flags(self) -> None:
        cwd = Path("/my/project")
        cmd = _build_cursor_cmd("agent", "gpt-4", cwd, ["--force"])
        assert cmd == ["agent", "--model", "gpt-4", "--print", "--force", "--workspace", str(cwd)]

    async def test_without_cwd_no_workspace_flag(self) -> None:
        cmd = _build_cursor_cmd("agent", "gpt-4", None, [])
        assert "--workspace" not in cmd
        assert "--print" in cmd

    async def test_custom_binary(self) -> None:
        cmd = _build_cursor_cmd("/usr/bin/agent", "model-x", Path("/workspace"), [])
        assert cmd[0] == "/usr/bin/agent"
        assert "--print" in cmd


class TestProvidersDict:
    async def test_has_three_providers(self) -> None:
        assert len(PROVIDERS) == 3

    async def test_claude_provider(self) -> None:
        config = PROVIDERS["claude"]
        assert config.binary == "claude"
        assert config.uses_own_cwd is False
        assert config.build_cmd is _build_claude_cmd

    async def test_gemini_provider(self) -> None:
        config = PROVIDERS["gemini"]
        assert config.binary == "gemini"
        assert config.uses_own_cwd is False
        assert config.build_cmd is _build_gemini_cmd

    async def test_cursor_provider(self) -> None:
        config = PROVIDERS["cursor"]
        assert config.binary == "agent"
        assert config.uses_own_cwd is True
        assert config.build_cmd is _build_cursor_cmd

    async def test_valid_ai_providers_matches_keys(self) -> None:
        assert set(PROVIDERS.keys()) == VALID_AI_PROVIDERS
        assert {"claude", "gemini", "cursor"} == VALID_AI_PROVIDERS

    async def test_only_cursor_uses_own_cwd(self) -> None:
        for name, config in PROVIDERS.items():
            if name == "cursor":
                assert config.uses_own_cwd is True
            else:
                assert config.uses_own_cwd is False, f"{name} should not use own cwd"

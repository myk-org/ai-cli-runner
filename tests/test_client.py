import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ai_cli_runner.client import (
    DEFAULT_TIMEOUT_MINUTES,
    call_ai_cli,
    check_ai_cli_available,
    get_ai_cli_timeout,
)


async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def _successful_run_result() -> MagicMock:
    return MagicMock(returncode=0, stdout="AI response", stderr="")


class TestCallAiCli:
    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_claude_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is True
        assert output == "AI response"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["claude", "--model", "opus-4"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_gemini_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        success, output = await call_ai_cli(prompt="hello", ai_provider="gemini", ai_model="flash")
        assert success is True
        assert output == "AI response"
        call_args = mock_run.call_args
        assert call_args[0][0] == ["gemini", "--model", "flash"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_cursor_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/my/project")
        success, output = await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="cursor", ai_model="gpt-4")
        assert success is True
        assert output == "AI response"
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["agent", "--model", "gpt-4", "--workspace", "/my/project"]

    async def test_unknown_provider(self) -> None:
        success, output = await call_ai_cli(prompt="hello", ai_provider="unknown", ai_model="model")
        assert success is False
        assert "Unknown AI provider" in output
        assert "unknown" in output
        for provider in ("claude", "cursor", "gemini"):
            assert provider in output

    async def test_empty_model(self) -> None:
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="")
        assert success is False
        assert "No AI model configured" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_timeout_expired(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=600)
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "timed out" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_file_not_found(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "not found" in output
        assert "claude" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_nonzero_exit_code(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "some error" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_nonzero_exit_code_no_output(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "unknown error" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_custom_cwd_passed_to_subprocess(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/custom/path")
        await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="claude", ai_model="opus-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == cwd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_cursor_subprocess_cwd_is_none(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/my/project")
        await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="cursor", ai_model="gpt-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] is None

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_custom_timeout(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", ai_cli_timeout=5)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 300  # 5 minutes * 60

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_prompt_passed_as_input(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="my prompt text", ai_provider="claude", ai_model="opus-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["input"] == "my prompt text"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_cli_flags_passed_to_command(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Verify cli_flags are included in the subprocess command."""
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            cli_flags=["--dangerously-skip-permissions", "-p"],
        )
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4", "--dangerously-skip-permissions", "-p"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_no_cli_flags_default(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Without cli_flags, command has no extra flags."""
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4"]


class TestCheckAiCliAvailable:
    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is True
        assert msg == ""

    async def test_unknown_provider(self) -> None:
        available, msg = await check_ai_cli_available(ai_provider="bad", ai_model="model")
        assert available is False
        assert "Unknown AI provider" in msg

    async def test_no_model(self) -> None:
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="")
        assert available is False
        assert "No AI model configured" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_nonzero_exit(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "sanity check failed" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_timeout(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "timed out" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_file_not_found(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "not found in PATH" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client.subprocess.run")
    async def test_cli_flags_passed_to_sanity_command(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Verify cli_flags are included in the sanity check command."""
        mock_run.return_value = _successful_run_result()
        await check_ai_cli_available(
            ai_provider="claude",
            ai_model="opus-4",
            cli_flags=["--dangerously-skip-permissions", "-p"],
        )
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4", "--dangerously-skip-permissions", "-p"]


class TestGetAiCliTimeout:
    async def test_default_value(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            if "AI_CLI_TIMEOUT" in __import__("os").environ:
                del __import__("os").environ["AI_CLI_TIMEOUT"]
            result = get_ai_cli_timeout()
            assert result == DEFAULT_TIMEOUT_MINUTES

    async def test_valid_env_var(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "20"}):
            result = get_ai_cli_timeout()
            assert result == 20

    async def test_invalid_env_var_non_numeric(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "abc"}):
            result = get_ai_cli_timeout()
            assert result == DEFAULT_TIMEOUT_MINUTES

    async def test_non_positive_env_var_zero(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "0"}):
            result = get_ai_cli_timeout()
            assert result == DEFAULT_TIMEOUT_MINUTES

    async def test_non_positive_env_var_negative(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "-5"}):
            result = get_ai_cli_timeout()
            assert result == DEFAULT_TIMEOUT_MINUTES

    async def test_custom_default_minutes(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            if "AI_CLI_TIMEOUT" in __import__("os").environ:
                del __import__("os").environ["AI_CLI_TIMEOUT"]
            result = get_ai_cli_timeout(default_minutes=30)
            assert result == 30

    async def test_custom_default_minutes_used_on_invalid(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "bad"}):
            result = get_ai_cli_timeout(default_minutes=25)
            assert result == 25

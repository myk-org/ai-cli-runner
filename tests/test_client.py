import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_cli_runner.client import (
    DEFAULT_TIMEOUT_MINUTES,
    _kill_process_group,
    _run_with_process_group,
    call_ai_cli,
    check_ai_cli_available,
    get_ai_cli_timeout,
)
from ai_cli_runner.models import AIResult


async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def _successful_run_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="AI response", stderr="")


class TestCallAiCli:
    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_claude_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is True
        assert output == "AI response"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["claude", "--model", "opus-4", "-p"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_gemini_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        success, output = await call_ai_cli(prompt="hello", ai_provider="gemini", ai_model="flash")
        assert success is True
        assert output == "AI response"
        call_args = mock_run.call_args
        assert call_args[0][0] == ["gemini", "--model", "flash"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cursor_success(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/my/project")
        success, output = await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="cursor", ai_model="gpt-4")
        assert success is True
        assert output == "AI response"
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["agent", "--model", "gpt-4", "--print", "--workspace", str(cwd)]

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
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_timeout_expired(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=600)
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "timed out" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_file_not_found(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "not found" in output
        assert "claude" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_nonzero_exit_code(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "some error" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_nonzero_exit_code_no_output(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert success is False
        assert "unknown error" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_custom_cwd_passed_to_subprocess(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/custom/path")
        await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="claude", ai_model="opus-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == cwd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cursor_subprocess_cwd_is_passed(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        cwd = Path("/my/project")
        await call_ai_cli(prompt="hello", cwd=cwd, ai_provider="cursor", ai_model="gpt-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == cwd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cursor_subprocess_cwd_none_when_not_provided(
        self, mock_run: MagicMock, _mock_thread: MagicMock
    ) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="cursor", ai_model="gpt-4")
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        call_kwargs = call_args[1]
        assert call_kwargs["cwd"] is None
        assert "--workspace" not in cmd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_custom_timeout(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", ai_cli_timeout=5)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 300  # 5 minutes * 60

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_prompt_passed_as_input(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="my prompt text", ai_provider="claude", ai_model="opus-4")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["input_data"] == "my prompt text"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cli_flags_passed_to_command(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Verify cli_flags are included in the subprocess command."""
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            cli_flags=["--dangerously-skip-permissions"],
        )
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4", "-p", "--dangerously-skip-permissions"]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_no_cli_flags_default(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Without cli_flags, command has no extra flags beyond structural ones."""
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4", "-p"]

    async def test_invalid_timeout_zero(self) -> None:
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", ai_cli_timeout=0)
        assert success is False
        assert "Invalid ai_cli_timeout" in output

    async def test_invalid_timeout_negative(self) -> None:
        success, output = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", ai_cli_timeout=-5)
        assert success is False
        assert "Invalid ai_cli_timeout" in output

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_returns_ai_result(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        result = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert isinstance(result, AIResult)
        assert result.success is True
        assert result.text == "AI response"
        assert result.usage is None

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_output_format_json_adds_flag(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        import json

        claude_response = json.dumps(
            {
                "type": "result",
                "result": "Hello!",
                "duration_ms": 100,
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "modelUsage": {"opus-4": {}},
            }
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=claude_response, stderr="")
        result = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", output_format="json")
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"
        assert result.success is True
        assert result.text == "Hello!"
        assert result.usage is not None
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_output_format_json_with_cli_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        import json

        claude_response = json.dumps({"result": "ok", "usage": {}, "modelUsage": {}})
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=claude_response, stderr="")
        await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            output_format="json",
            cli_flags=["--dangerously-skip-permissions"],
        )
        cmd = mock_run.call_args[0][0]
        # --output-format json should be before the user's cli_flags
        assert cmd == [
            "claude",
            "--model",
            "opus-4",
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ]

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cursor_output_format_uses_stream_json_wire_format(
        self, mock_run: MagicMock, _mock_thread: MagicMock
    ) -> None:
        import json

        cursor_response = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
                        "session_id": "s1",
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "duration_ms": 100,
                        "result": "Hello!",
                        "session_id": "s1",
                        "usage": {"inputTokens": 5, "outputTokens": 3, "cacheReadTokens": 0, "cacheWriteTokens": 0},
                    }
                ),
            ]
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=cursor_response, stderr="")
        result = await call_ai_cli(prompt="hello", ai_provider="cursor", ai_model="gpt-4", output_format="json")
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "stream-json"  # cursor uses stream-json on the wire
        assert result.success is True
        assert result.text == "Hello!"

    async def test_error_returns_ai_result(self) -> None:
        result = await call_ai_cli(prompt="hello", ai_provider="unknown", ai_model="model")
        assert isinstance(result, AIResult)
        assert result.success is False
        assert "Unknown AI provider" in result.text


class TestCallAiCliSession:
    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_continue_session_claude_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", continue_session=True)
        cmd = mock_run.call_args[0][0]
        assert "--continue" in cmd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_continue_session_gemini_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="gemini", ai_model="flash", continue_session=True)
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        # gemini uses --resume with no arg for continue; verify no session id follows
        resume_idx = cmd.index("--resume")
        # --resume should be the last element (no arg after it)
        assert resume_idx == len(cmd) - 1

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_continue_session_cursor_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="cursor", ai_model="gpt-4", continue_session=True)
        cmd = mock_run.call_args[0][0]
        assert "--continue" in cmd

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_claude_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", session_id="sess-abc")
        cmd = mock_run.call_args[0][0]
        resume_idx = cmd.index("--resume")
        assert cmd[resume_idx + 1] == "sess-abc"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_gemini_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="gemini", ai_model="flash", session_id="sess-def")
        cmd = mock_run.call_args[0][0]
        resume_idx = cmd.index("--resume")
        assert cmd[resume_idx + 1] == "sess-def"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_cursor_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(prompt="hello", ai_provider="cursor", ai_model="gpt-4", session_id="sess-ghi")
        cmd = mock_run.call_args[0][0]
        resume_idx = cmd.index("--resume")
        assert cmd[resume_idx + 1] == "sess-ghi"

    async def test_session_id_and_continue_session_mutually_exclusive(self) -> None:
        result = await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            session_id="sess-abc",
            continue_session=True,
        )
        assert result.success is False
        assert "Cannot use both session_id and continue_session" in result.text

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_returned_from_json_output(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        import json

        claude_response = json.dumps(
            {
                "type": "result",
                "result": "Hello!",
                "session_id": "test-session-abc",
                "duration_ms": 100,
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "modelUsage": {"opus-4": {}},
            }
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=claude_response, stderr="")
        result = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", output_format="json")
        assert result.session_id == "test-session-abc"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_none_without_json_output(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        result = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4")
        assert result.session_id is None

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_none_when_not_in_json(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        import json

        claude_response = json.dumps(
            {
                "type": "result",
                "result": "Hello!",
                "duration_ms": 100,
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "modelUsage": {"opus-4": {}},
            }
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=claude_response, stderr="")
        result = await call_ai_cli(prompt="hello", ai_provider="claude", ai_model="opus-4", output_format="json")
        # session_id defaults to "" in AITokenUsage, which becomes None via `or None`
        assert result.session_id is None

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_continue_session_with_cli_flags(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = _successful_run_result()
        await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            cli_flags=["--dangerously-skip-permissions"],
            continue_session=True,
        )
        cmd = mock_run.call_args[0][0]
        # Session flags should be appended AFTER other cli_flags
        skip_idx = cmd.index("--dangerously-skip-permissions")
        continue_idx = cmd.index("--continue")
        assert continue_idx > skip_idx

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_id_with_output_format(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        import json

        claude_response = json.dumps(
            {
                "type": "result",
                "result": "Hello!",
                "session_id": "sess-combo",
                "duration_ms": 100,
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "modelUsage": {"opus-4": {}},
            }
        )
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=claude_response, stderr="")
        result = await call_ai_cli(
            prompt="hello",
            ai_provider="claude",
            ai_model="opus-4",
            output_format="json",
            session_id="sess-prev",
        )
        cmd = mock_run.call_args[0][0]
        # Both --output-format json and --resume sess-prev should be in the command
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--resume" in cmd
        assert "sess-prev" in cmd
        # Result should have session_id from the JSON response
        assert result.session_id == "sess-combo"

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_session_flags_without_output_format(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Session flags are sent to CLI even without output_format; result.session_id is None."""
        mock_run.return_value = _successful_run_result()
        result = await call_ai_cli(prompt="hi", ai_provider="claude", ai_model="opus-4", session_id="sess-123")
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert "sess-123" in cmd
        assert "--output-format" not in cmd
        assert result.session_id is None


class TestCheckAiCliAvailable:
    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
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
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_nonzero_exit(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "sanity check failed" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_timeout(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "timed out" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_file_not_found(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        available, msg = await check_ai_cli_available(ai_provider="claude", ai_model="opus-4")
        assert available is False
        assert "not found in PATH" in msg

    @patch("ai_cli_runner.client.asyncio.to_thread", side_effect=fake_to_thread)
    @patch("ai_cli_runner.client._run_with_process_group")
    async def test_cli_flags_passed_to_sanity_command(self, mock_run: MagicMock, _mock_thread: MagicMock) -> None:
        """Verify cli_flags are included in the sanity check command."""
        mock_run.return_value = _successful_run_result()
        await check_ai_cli_available(
            ai_provider="claude",
            ai_model="opus-4",
            cli_flags=["--dangerously-skip-permissions"],
        )
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--model", "opus-4", "-p", "--dangerously-skip-permissions"]


class TestGetAiCliTimeout:
    async def test_default_value(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("AI_CLI_TIMEOUT", None)
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
            os.environ.pop("AI_CLI_TIMEOUT", None)
            result = get_ai_cli_timeout(default_minutes=30)
            assert result == 30

    async def test_custom_default_minutes_used_on_invalid(self) -> None:
        with patch.dict("os.environ", {"AI_CLI_TIMEOUT": "bad"}):
            result = get_ai_cli_timeout(default_minutes=25)
            assert result == 25


class TestRunWithProcessGroup:
    @patch("ai_cli_runner.client.subprocess.Popen")
    def test_success(self, mock_popen: MagicMock) -> None:
        proc = MagicMock()
        proc.communicate.return_value = ("output", "")
        proc.returncode = 0
        mock_popen.return_value = proc

        result = _run_with_process_group(["echo", "hi"], input_data="hello")

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        mock_popen.assert_called_once_with(
            ["echo", "hi"],
            cwd=None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        proc.communicate.assert_called_once_with(input="hello", timeout=None)

    @patch("ai_cli_runner.client._kill_process_group")
    @patch("ai_cli_runner.client.subprocess.Popen")
    def test_timeout_kills_process_group(self, mock_popen: MagicMock, mock_kill_pg: MagicMock) -> None:
        proc = MagicMock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="cmd", timeout=10)
        mock_popen.return_value = proc

        with contextlib.suppress(subprocess.TimeoutExpired):
            _run_with_process_group(["cmd"], timeout=10, input_data="prompt")

        mock_kill_pg.assert_called_once_with(proc)

    @patch("ai_cli_runner.client.subprocess.Popen")
    def test_file_not_found_propagates(self, mock_popen: MagicMock) -> None:
        mock_popen.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError):
            _run_with_process_group(["nonexistent"])

    @patch("ai_cli_runner.client.subprocess.Popen")
    def test_cwd_and_timeout_forwarded(self, mock_popen: MagicMock) -> None:
        proc = MagicMock()
        proc.communicate.return_value = ("out", "err")
        proc.returncode = 0
        mock_popen.return_value = proc
        cwd = Path("/some/path")

        _run_with_process_group(["cmd"], cwd=cwd, timeout=300, input_data="data")

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["cwd"] == cwd
        proc.communicate.assert_called_once_with(input="data", timeout=300)

    @patch("ai_cli_runner.client._kill_process_group")
    @patch("ai_cli_runner.client.subprocess.Popen")
    def test_base_exception_kills_process_group(self, mock_popen: MagicMock, mock_kill_pg: MagicMock) -> None:
        proc = MagicMock()
        proc.communicate.side_effect = KeyboardInterrupt()
        mock_popen.return_value = proc

        with contextlib.suppress(KeyboardInterrupt):
            _run_with_process_group(["cmd"], input_data="prompt")

        mock_kill_pg.assert_called_once_with(proc)


class TestKillProcessGroup:
    @patch("ai_cli_runner.client.os.killpg")
    @patch("ai_cli_runner.client.os.getpgid", return_value=12345)
    def test_sigterm_then_wait(self, _mock_getpgid: MagicMock, mock_killpg: MagicMock) -> None:
        import signal

        proc = MagicMock()
        proc.pid = 1000
        proc.wait.return_value = 0

        _kill_process_group(proc)

        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=5)

    @patch("ai_cli_runner.client.os.killpg")
    @patch("ai_cli_runner.client.os.getpgid", return_value=12345)
    def test_sigkill_on_timeout(self, _mock_getpgid: MagicMock, mock_killpg: MagicMock) -> None:
        import signal

        proc = MagicMock()
        proc.pid = 1000
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="cmd", timeout=5), None]

        _kill_process_group(proc)

        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(12345, signal.SIGTERM)
        mock_killpg.assert_any_call(12345, signal.SIGKILL)

    @patch("ai_cli_runner.client.os.killpg")
    @patch("ai_cli_runner.client.os.getpgid", return_value=12345)
    def test_oserror_on_killpg_ignored(self, _mock_getpgid: MagicMock, mock_killpg: MagicMock) -> None:
        proc = MagicMock()
        proc.pid = 1000
        mock_killpg.side_effect = OSError("No such process")

        # Should not raise
        _kill_process_group(proc)

import pytest

from ai_cli_runner.models import AIResult, AITokenUsage, ParsedOutput


class TestAITokenUsage:
    def test_defaults(self) -> None:
        usage = AITokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.cost_usd is None
        assert usage.duration_ms is None
        assert usage.model == ""
        assert usage.provider == ""

    def test_custom_values(self) -> None:
        usage = AITokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=300,
            cost_usd=0.05,
            duration_ms=1234,
            model="opus-4",
            provider="claude",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 200
        assert usage.cache_write_tokens == 300
        assert usage.cost_usd == 0.05
        assert usage.duration_ms == 1234
        assert usage.model == "opus-4"
        assert usage.provider == "claude"

    def test_session_id_default(self) -> None:
        usage = AITokenUsage()
        assert usage.session_id == ""

    def test_session_id_custom(self) -> None:
        usage = AITokenUsage(session_id="sess-abc-123")
        assert usage.session_id == "sess-abc-123"


class TestParsedOutput:
    def test_default_thinking(self) -> None:
        parsed = ParsedOutput(text="hello")
        assert parsed.thinking == ""

    def test_default_usage(self) -> None:
        parsed = ParsedOutput(text="hello")
        assert parsed.usage is None

    def test_tuple_unpacking(self) -> None:
        usage = AITokenUsage(input_tokens=10)
        parsed = ParsedOutput(text="hello", usage=usage, thinking="reasoning")
        text, u = parsed
        assert text == "hello"
        assert u is usage

    def test_tuple_unpacking_no_usage(self) -> None:
        parsed = ParsedOutput(text="hello")
        text, usage = parsed
        assert text == "hello"
        assert usage is None

    def test_thinking_accessible(self) -> None:
        parsed = ParsedOutput(text="answer", thinking="intermediate")
        assert parsed.thinking == "intermediate"

    def test_thinking_not_in_tuple_unpacking(self) -> None:
        parsed = ParsedOutput(text="answer", thinking="intermediate")
        text, usage = parsed
        assert text == "answer"
        assert usage is None
        # thinking is still accessible via attribute
        assert parsed.thinking == "intermediate"

    def test_len(self) -> None:
        parsed = ParsedOutput(text="hello", thinking="some thinking")
        assert len(parsed) == 2

    def test_index_access(self) -> None:
        usage = AITokenUsage(output_tokens=5)
        parsed = ParsedOutput(text="hello", usage=usage)
        assert parsed[0] == "hello"
        assert parsed[1] is usage

    def test_index_out_of_range(self) -> None:
        parsed = ParsedOutput(text="hello")
        with pytest.raises(IndexError):
            parsed[2]


class TestAIResult:
    def test_tuple_unpacking(self) -> None:
        result = AIResult(success=True, text="hello")
        success, text = result
        assert success is True
        assert text == "hello"

    def test_tuple_unpacking_failure(self) -> None:
        result = AIResult(success=False, text="error msg")
        success, text = result
        assert success is False
        assert text == "error msg"

    def test_index_access(self) -> None:
        result = AIResult(success=True, text="output")
        assert result[0] is True
        assert result[1] == "output"

    def test_len(self) -> None:
        result = AIResult(success=True, text="hi")
        assert len(result) == 2

    def test_with_usage(self) -> None:
        usage = AITokenUsage(input_tokens=10, output_tokens=5)
        result = AIResult(success=True, text="hi", usage=usage)
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_without_usage(self) -> None:
        result = AIResult(success=True, text="hi")
        assert result.usage is None

    def test_bool_true_on_success(self) -> None:
        result = AIResult(success=True, text="ok")
        assert bool(result) is True

    def test_bool_false_on_failure(self) -> None:
        result = AIResult(success=False, text="error")
        assert bool(result) is False

    def test_usage_not_in_tuple_unpacking(self) -> None:
        usage = AITokenUsage(input_tokens=10)
        result = AIResult(success=True, text="hi", usage=usage)
        success, text = result
        assert success is True
        assert text == "hi"
        # usage is still accessible via attribute
        assert result.usage is usage

    def test_session_id_default_none(self) -> None:
        result = AIResult(success=True, text="hi")
        assert result.session_id is None

    def test_session_id_custom(self) -> None:
        result = AIResult(success=True, text="hi", session_id="sess-xyz")
        assert result.session_id == "sess-xyz"

    def test_session_id_not_in_tuple_unpacking(self) -> None:
        result = AIResult(success=True, text="hi", session_id="sess-xyz")
        success, text = result
        assert success is True
        assert text == "hi"
        # session_id is still accessible via attribute
        assert result.session_id == "sess-xyz"

    def test_thinking_default_empty(self) -> None:
        result = AIResult(success=True, text="hi")
        assert result.thinking == ""

    def test_thinking_custom_value(self) -> None:
        result = AIResult(success=True, text="answer", thinking="intermediate reasoning")
        assert result.thinking == "intermediate reasoning"

    def test_thinking_not_in_tuple_unpacking(self) -> None:
        result = AIResult(success=True, text="hi", thinking="some thinking")
        success, text = result
        assert success is True
        assert text == "hi"
        # thinking is still accessible via attribute
        assert result.thinking == "some thinking"

    def test_thinking_not_in_getitem(self) -> None:
        result = AIResult(success=True, text="hi", thinking="some thinking")
        assert result[0] is True
        assert result[1] == "hi"
        with pytest.raises(IndexError):
            result[2]

    def test_thinking_not_in_len(self) -> None:
        result = AIResult(success=True, text="hi", thinking="some thinking")
        assert len(result) == 2

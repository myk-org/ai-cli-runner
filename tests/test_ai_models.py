"""Tests for ai_cli_runner.ai_models module."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from ai_cli_runner.ai_models import _MODEL_CACHE_TTL_SECONDS, AIModelCache, _model_id_to_display_name, model_cache
from ai_cli_runner.llm_pricing import LLMPricingCache

SAMPLE_PRICING_DATA = {
    "claude-sonnet-4-20250514": {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015},
    "claude-opus-4-6": {"input_cost_per_token": 0.000015, "output_cost_per_token": 0.000075},
    "anthropic/claude-sonnet-4-20250514": {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015},
    "gemini-2.5-flash": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006},
    "gemini/gemini-2.5-pro": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000004},
    "vertex_ai/gemini-2.5-flash": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006},
    "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
}

CURSOR_OUTPUT = """\
Available models
claude-sonnet-4-6 - Claude Sonnet 4.6
gemini-2.5-flash - Gemini 2.5 Flash
gpt-4o - GPT 4o
Tip: use --model flag to select a model
"""


@pytest.fixture
def cache() -> AIModelCache:
    return AIModelCache()


@pytest.fixture
def cache_with_pricing(cache: AIModelCache) -> AIModelCache:
    pricing = LLMPricingCache()
    pricing._data = SAMPLE_PRICING_DATA.copy()
    cache.set_pricing_cache(pricing)
    return cache


def test_model_id_to_display_name() -> None:
    assert _model_id_to_display_name("claude-sonnet-4") == "Claude Sonnet 4"
    assert _model_id_to_display_name("gemini-2.5-pro") == "Gemini 2.5 Pro"
    assert _model_id_to_display_name("gpt-4o") == "Gpt 4O"


# -- Cursor models -----------------------------------------------------------


class TestCursorModels:
    @pytest.mark.asyncio
    async def test_list_cursor_models(self, cache: AIModelCache) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(CURSOR_OUTPUT.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            models = await cache.list_models("cursor")

        assert len(models) == 3
        assert models[0] == {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"}
        assert models[1] == {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"}
        assert models[2] == {"id": "gpt-4o", "name": "GPT 4o"}

    @pytest.mark.asyncio
    async def test_binary_not_found(self, cache: AIModelCache) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            models = await cache.list_models("cursor")
        assert models == []

    @pytest.mark.asyncio
    async def test_timeout(self, cache: AIModelCache) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("ai_cli_runner.ai_models._kill_process") as mock_kill,
        ):
            models = await cache.list_models("cursor")

        assert models == []
        mock_kill.assert_called_once_with(mock_proc)

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, cache: AIModelCache) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            models = await cache.list_models("cursor")

        assert models == []


# -- Parse cursor output ------------------------------------------------------


class TestParseCursorOutput:
    def test_parse_normal(self) -> None:
        models = AIModelCache._parse_cursor_output(CURSOR_OUTPUT)
        assert len(models) == 3
        assert models[0]["id"] == "claude-sonnet-4-6"
        assert models[0]["name"] == "Claude Sonnet 4.6"

    def test_parse_empty(self) -> None:
        assert AIModelCache._parse_cursor_output("") == []
        assert AIModelCache._parse_cursor_output("Available models\nTip: use --model") == []

    def test_parse_no_separator(self) -> None:
        output = "Available models\nsome-model\nTip: use --model"
        models = AIModelCache._parse_cursor_output(output)
        assert len(models) == 1
        assert models[0]["id"] == "some-model"
        assert models[0]["name"] == "Some Model"


# -- Claude models ------------------------------------------------------------


class TestClaudeModels:
    @pytest.mark.asyncio
    async def test_list(self, cache_with_pricing: AIModelCache) -> None:
        models = await cache_with_pricing.list_models("claude")
        model_ids = [m["id"] for m in models]
        assert "claude-sonnet-4-20250514" in model_ids
        assert "claude-opus-4-6" in model_ids

    @pytest.mark.asyncio
    async def test_excludes_provider_prefixed(self, cache_with_pricing: AIModelCache) -> None:
        models = await cache_with_pricing.list_models("claude")
        model_ids = [m["id"] for m in models]
        # anthropic/ prefixed key should be excluded
        assert "anthropic/claude-sonnet-4-20250514" not in model_ids

    @pytest.mark.asyncio
    async def test_no_pricing_cache(self, cache: AIModelCache) -> None:
        models = await cache.list_models("claude")
        assert models == []


# -- Gemini models ------------------------------------------------------------


class TestGeminiModels:
    @pytest.mark.asyncio
    async def test_list(self, cache_with_pricing: AIModelCache) -> None:
        models = await cache_with_pricing.list_models("gemini")
        model_ids = [m["id"] for m in models]
        assert "gemini-2.5-flash" in model_ids

    @pytest.mark.asyncio
    async def test_handles_prefixed_keys(self, cache_with_pricing: AIModelCache) -> None:
        models = await cache_with_pricing.list_models("gemini")
        model_ids = [m["id"] for m in models]
        # gemini/ prefix should be stripped
        assert "gemini-2.5-pro" in model_ids

    @pytest.mark.asyncio
    async def test_deduplication(self, cache_with_pricing: AIModelCache) -> None:
        """gemini-2.5-flash appears as both bare and gemini/ prefixed — should deduplicate."""
        pricing = LLMPricingCache()
        pricing._data = {
            "gemini-2.5-flash": {"input_cost_per_token": 0.00000015},
            "gemini/gemini-2.5-flash": {"input_cost_per_token": 0.00000015},
        }
        cache = AIModelCache()
        cache.set_pricing_cache(pricing)

        models = await cache.list_models("gemini")
        model_ids = [m["id"] for m in models]
        assert model_ids.count("gemini-2.5-flash") == 1

    @pytest.mark.asyncio
    async def test_no_pricing_cache(self, cache: AIModelCache) -> None:
        models = await cache.list_models("gemini")
        assert models == []


# -- Validation ---------------------------------------------------------------


class TestModelValidation:
    @pytest.mark.asyncio
    async def test_valid(self, cache_with_pricing: AIModelCache) -> None:
        await cache_with_pricing.list_models("claude")
        assert cache_with_pricing.is_valid_model("claude", "claude-sonnet-4-20250514") is True

    @pytest.mark.asyncio
    async def test_not_found(self, cache_with_pricing: AIModelCache) -> None:
        await cache_with_pricing.list_models("claude")
        assert cache_with_pricing.is_valid_model("claude", "nonexistent-model") is False

    def test_cache_empty(self, cache: AIModelCache) -> None:
        assert cache.is_valid_model("claude", "claude-sonnet-4-20250514") is None


# -- Caching ------------------------------------------------------------------


class TestCaching:
    @pytest.mark.asyncio
    async def test_uses_cache(self, cache_with_pricing: AIModelCache) -> None:
        models1 = await cache_with_pricing.list_models("claude")
        # Mutate pricing data — cached result should still be returned
        cache_with_pricing._pricing_cache._data = {}  # type: ignore[union-attr]
        models2 = await cache_with_pricing.list_models("claude")
        assert models1 == models2

    @pytest.mark.asyncio
    async def test_refreshes_expired(self, cache_with_pricing: AIModelCache) -> None:
        await cache_with_pricing.list_models("claude")
        # Manually expire the cache
        cache_with_pricing._cache["claude"]["fetched_at"] = time.monotonic() - _MODEL_CACHE_TTL_SECONDS - 1

        # Now pricing data is empty, so re-fetch returns []
        cache_with_pricing._pricing_cache._data = {}  # type: ignore[union-attr]
        models = await cache_with_pricing.list_models("claude")
        assert models == []

    @pytest.mark.asyncio
    async def test_refresh_clears(self, cache_with_pricing: AIModelCache) -> None:
        await cache_with_pricing.list_models("claude")
        assert "claude" in cache_with_pricing._cache

        await cache_with_pricing.refresh("claude")
        # After refresh, cache should be repopulated
        assert "claude" in cache_with_pricing._cache


# -- set_pricing_cache --------------------------------------------------------


class TestSetPricingCache:
    @pytest.mark.asyncio
    async def test_invalidates_claude_gemini(self) -> None:
        cache = AIModelCache()
        pricing = LLMPricingCache()
        pricing._data = SAMPLE_PRICING_DATA.copy()
        cache.set_pricing_cache(pricing)

        await cache.list_models("claude")
        await cache.list_models("gemini")

        # Set up cursor cache manually
        cache._cache["cursor"] = {"models": [{"id": "test", "name": "Test"}], "fetched_at": time.monotonic()}

        assert "claude" in cache._cache
        assert "gemini" in cache._cache
        assert "cursor" in cache._cache

        # Setting a new pricing cache should invalidate claude and gemini but NOT cursor
        new_pricing = LLMPricingCache()
        new_pricing._data = SAMPLE_PRICING_DATA.copy()
        cache.set_pricing_cache(new_pricing)

        assert "claude" not in cache._cache
        assert "gemini" not in cache._cache
        assert "cursor" in cache._cache


# -- Singleton ----------------------------------------------------------------


class TestSingleton:
    def test_model_cache_singleton(self) -> None:
        assert isinstance(model_cache, AIModelCache)


# -- Unknown provider ---------------------------------------------------------


class TestUnknownProvider:
    @pytest.mark.asyncio
    async def test_unknown_provider_returns_empty(self, cache: AIModelCache) -> None:
        models = await cache.list_models("unknown_provider")
        assert models == []


# -- Excludes vertex_ai prefixed gemini models --------------------------------


class TestGeminiExcludesVertexAI:
    @pytest.mark.asyncio
    async def test_excludes_vertex_ai(self, cache_with_pricing: AIModelCache) -> None:
        models = await cache_with_pricing.list_models("gemini")
        model_ids = [m["id"] for m in models]
        assert "vertex_ai/gemini-2.5-flash" not in model_ids

"""Tests for ai_cli_runner.llm_pricing module."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_cli_runner.llm_pricing import (
    _REFRESH_INTERVAL_SECONDS,
    LLMPricingCache,
    pricing_cache,
)

SAMPLE_PRICING_DATA = {
    "claude-sonnet-4-20250514": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_read_input_token_cost": 0.0000003,
        "cache_creation_input_token_cost": 0.00000375,
    },
    "anthropic/claude-sonnet-4-20250514": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_read_input_token_cost": 0.0000003,
        "cache_creation_input_token_cost": 0.00000375,
    },
    "gemini-2.5-flash": {
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000006,
    },
    "claude-opus-4-6": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
    "gpt-4o": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.00001,
    },
    "model-2024-01-15": {
        "input_cost_per_token": 0.000001,
        "output_cost_per_token": 0.000002,
    },
    "no-cost-model": {
        "some_field": "value",
    },
}


def _make_mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── Disk caching ──────────────────────────────────────────────────────


class TestDiskCaching:
    async def test_load_from_fresh_disk_cache(self) -> None:
        """Disk cache exists and is fresh — loads from disk, no HTTP."""
        cache = LLMPricingCache()

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=True),
            patch.object(cache, "_read_disk_cache", return_value=SAMPLE_PRICING_DATA),
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient") as mock_client_cls,
        ):
            await cache.load()

            assert cache._data == SAMPLE_PRICING_DATA
            assert len(cache._data) == len(SAMPLE_PRICING_DATA)
            mock_client_cls.assert_not_called()

    async def test_load_fetches_when_disk_cache_stale(self) -> None:
        """Disk cache > 24h old — fetches from HTTP."""
        cache = LLMPricingCache()
        mock_response = _make_mock_response(SAMPLE_PRICING_DATA)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=False),
            patch.object(cache, "_write_disk_cache") as mock_write,
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache.load()

            assert cache._data == SAMPLE_PRICING_DATA
            mock_client.get.assert_called_once()
            mock_write.assert_called_once_with(SAMPLE_PRICING_DATA)

    async def test_load_fetches_when_no_disk_cache(self) -> None:
        """No disk cache — fetches from HTTP."""
        cache = LLMPricingCache()
        mock_response = _make_mock_response(SAMPLE_PRICING_DATA)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=False),
            patch.object(cache, "_read_disk_cache", return_value=None),
            patch.object(cache, "_write_disk_cache") as mock_write,
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache.load()

            assert cache._data == SAMPLE_PRICING_DATA
            mock_write.assert_called_once_with(SAMPLE_PRICING_DATA)

    async def test_fetch_writes_to_disk_cache(self) -> None:
        """After HTTP fetch, data is written to disk."""
        cache = LLMPricingCache()
        mock_response = _make_mock_response(SAMPLE_PRICING_DATA)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=False),
            patch.object(cache, "_write_disk_cache") as mock_write,
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache._fetch()

            mock_write.assert_called_once_with(SAMPLE_PRICING_DATA)

    async def test_fetch_falls_back_to_stale_disk_cache(self) -> None:
        """HTTP fails, loads stale disk cache."""
        cache = LLMPricingCache()

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("network error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=False),
            patch.object(cache, "_read_disk_cache", return_value=SAMPLE_PRICING_DATA),
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache._fetch()

            assert cache._data == SAMPLE_PRICING_DATA

    async def test_fetch_fails_no_disk_cache(self) -> None:
        """HTTP fails, no disk cache — data stays empty."""
        cache = LLMPricingCache()

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("network error")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=False),
            patch.object(cache, "_read_disk_cache", return_value=None),
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache._fetch()

            assert cache._data == {}

    async def test_refresh_bypasses_fresh_disk_cache(self) -> None:
        """refresh() fetches from HTTP even when disk cache is fresh."""
        cache = LLMPricingCache()
        mock_response = _make_mock_response(SAMPLE_PRICING_DATA)

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(cache, "_is_disk_cache_fresh", return_value=True),
            patch.object(cache, "_write_disk_cache"),
            patch("ai_cli_runner.llm_pricing.httpx.AsyncClient", return_value=mock_client),
        ):
            await cache.refresh()

            # HTTP should be called even though disk cache is fresh
            mock_client.get.assert_called_once()
            assert cache._data == SAMPLE_PRICING_DATA


# ── Cost calculation ──────────────────────────────────────────────────


class TestCalculateCost:
    def test_calculate_cost_basic(self) -> None:
        """Simple input/output tokens cost calculation."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        cost = cache.calculate_cost(
            provider="claude",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is not None
        # 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105)

    def test_calculate_cost_with_cache_tokens(self) -> None:
        """Cost includes cache_read and cache_write costs."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        cost = cache.calculate_cost(
            provider="claude",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=2000,
            cache_write_tokens=1500,
        )
        assert cost is not None
        # input:       1000 * 0.000003 = 0.003
        # output:      500  * 0.000015 = 0.0075
        # cache_read:  2000 * 0.0000003 = 0.0006
        # cache_write: 1500 * 0.00000375 = 0.005625
        expected = 0.003 + 0.0075 + 0.0006 + 0.005625
        assert cost == pytest.approx(expected)

    def test_calculate_cost_model_not_found(self) -> None:
        """Returns None when model not in pricing data."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        cost = cache.calculate_cost(
            provider="claude",
            model="nonexistent-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None

    def test_calculate_cost_missing_cost_fields(self) -> None:
        """Returns None when model entry lacks cost fields."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        cost = cache.calculate_cost(
            provider="claude",
            model="no-cost-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None


# ── Model lookup ──────────────────────────────────────────────────────


class TestModelLookup:
    def test_lookup_direct_match(self) -> None:
        """Model name matches directly."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        result = cache._lookup_model("claude", "claude-sonnet-4-20250514")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000003

    def test_lookup_provider_prefixed(self) -> None:
        """Matches as anthropic/model-name."""
        cache = LLMPricingCache()
        cache._data = {
            "anthropic/claude-test-model": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000025,
            }
        }

        result = cache._lookup_model("claude", "claude-test-model")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000005

    def test_lookup_bracket_stripping(self) -> None:
        """model[1m] → model."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        result = cache._lookup_model("claude", "claude-sonnet-4-20250514[1m]")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000003

    def test_lookup_at_replacement(self) -> None:
        """model@date → model-date."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        result = cache._lookup_model("claude", "model@2024-01-15")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000001

    def test_lookup_cursor_suffix_stripping(self) -> None:
        """Strips -xhigh-fast, -fast, etc. for cursor provider."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        # claude-sonnet-4-20250514-xhigh-fast → claude-sonnet-4-20250514
        result = cache._lookup_model("cursor", "claude-sonnet-4-20250514-xhigh-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000003

        # Also test -fast suffix
        result = cache._lookup_model("cursor", "claude-sonnet-4-20250514-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000003

    def test_resolve_cursor_claude_model(self) -> None:
        """claude-4.6-opus-max-thinking → claude-opus-4-6."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        result = cache._lookup_model("cursor", "claude-4.6-opus-max-thinking")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000015

    def test_resolve_cursor_gpt4o_model(self) -> None:
        """gpt-4o-mini-fast → gpt-4o-mini (not gpt-4)."""
        cache = LLMPricingCache()
        cache._data = {
            "gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,
                "output_cost_per_token": 0.0000006,
            },
        }

        result = cache._lookup_model("cursor", "gpt-4o-mini-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.00000015

    def test_resolve_cursor_gemini_model(self) -> None:
        """gemini-2.5-flash-fast → gemini-2.5-flash."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA

        result = cache._lookup_model("cursor", "gemini-2.5-flash-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.00000015

    def test_resolve_cursor_gpt_with_variant(self) -> None:
        """gpt-5.4-nano-none → gpt-5.4-nano via variant extraction."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-5.4-nano-none")
        assert resolved == "gpt-5.4-nano"

    # --- Direct _resolve_cursor_model: GPT ---

    def test_resolve_cursor_gpt_version_only(self) -> None:
        """gpt-4-fast → gpt-4 (version only, no variant, strip suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4-fast")
        assert resolved == "gpt-4"

    def test_resolve_cursor_gpt_version_variant(self) -> None:
        """gpt-4-mini-fast → gpt-4-mini (version + variant)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4-mini-fast")
        assert resolved == "gpt-4-mini"

    def test_resolve_cursor_gpt_decimal_version(self) -> None:
        """gpt-5.4-nano-none → gpt-5.4-nano (decimal version + variant)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-5.4-nano-none")
        assert resolved == "gpt-5.4-nano"

    def test_resolve_cursor_gpt_no_variant_no_suffix(self) -> None:
        """gpt-4 → gpt-4 (bare, no variant, no suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4")
        assert resolved == "gpt-4"

    def test_resolve_cursor_gpt_variant_no_suffix(self) -> None:
        """gpt-4-mini → gpt-4-mini (variant, no routing suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4-mini")
        assert resolved == "gpt-4-mini"

    def test_resolve_cursor_gpt_xhigh_fast_suffix(self) -> None:
        """gpt-5.4-nano-xhigh-fast → gpt-5.4-nano."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-5.4-nano-xhigh-fast")
        assert resolved == "gpt-5.4-nano"

    def test_resolve_cursor_gpt_max_thinking_suffix(self) -> None:
        """gpt-5.4-nano-max-thinking → gpt-5.4-nano."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-5.4-nano-max-thinking")
        assert resolved == "gpt-5.4-nano"

    # --- Direct _resolve_cursor_model: Claude ---

    def test_resolve_cursor_claude_version_variant(self) -> None:
        """claude-4.6-opus-max-thinking → claude-opus-4-6."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-4.6-opus-max-thinking")
        assert resolved == "claude-opus-4-6"

    def test_resolve_cursor_claude_version_only(self) -> None:
        """claude-4.6-fast → claude-4-6 (no variant, strip suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-4.6-fast")
        assert resolved == "claude-4-6"

    def test_resolve_cursor_claude_sonnet(self) -> None:
        """claude-4-sonnet-xhigh → claude-sonnet-4."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-4-sonnet-xhigh")
        assert resolved == "claude-sonnet-4"

    def test_resolve_cursor_claude_haiku(self) -> None:
        """claude-4.5-haiku → claude-haiku-4-5."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-4.5-haiku")
        assert resolved == "claude-haiku-4-5"

    def test_resolve_cursor_claude_bare(self) -> None:
        """claude-4 → claude-4."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-4")
        assert resolved == "claude-4"

    # --- Direct _resolve_cursor_model: Gemini ---

    def test_resolve_cursor_gemini_version_variant(self) -> None:
        """gemini-2.5-flash-fast → gemini-2.5-flash."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gemini-2.5-flash-fast")
        assert resolved == "gemini-2.5-flash"

    def test_resolve_cursor_gemini_pro(self) -> None:
        """gemini-2.5-pro-xhigh → gemini-2.5-pro."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gemini-2.5-pro-xhigh")
        assert resolved == "gemini-2.5-pro"

    def test_resolve_cursor_gemini_bare(self) -> None:
        """gemini-2.5 → gemini-2.5."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gemini-2.5")
        assert resolved == "gemini-2.5"

    # --- Direct _resolve_cursor_model: Edge cases ---

    def test_resolve_cursor_gpt4o(self) -> None:
        """gpt-4o → gpt-4o (alphanumeric version)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4o")
        assert resolved == "gpt-4o"

    def test_resolve_cursor_gpt4o_mini(self) -> None:
        """gpt-4o-mini → gpt-4o-mini (alphanumeric version + variant)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4o-mini")
        assert resolved == "gpt-4o-mini"

    def test_resolve_cursor_gpt4o_fast(self) -> None:
        """gpt-4o-fast → gpt-4o (strip routing suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4o-fast")
        assert resolved == "gpt-4o"

    def test_resolve_cursor_gpt4o_mini_fast(self) -> None:
        """gpt-4o-mini-fast → gpt-4o-mini (variant + strip suffix)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("gpt-4o-mini-fast")
        assert resolved == "gpt-4o-mini"

    def test_resolve_cursor_unknown_prefix(self) -> None:
        """unknown-model → None."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("unknown-model")
        assert resolved is None

    def test_resolve_cursor_no_version(self) -> None:
        """claude-abc → None (no version number)."""
        cache = LLMPricingCache()
        resolved = cache._resolve_cursor_model("claude-abc")
        assert resolved is None

    # --- End-to-end _lookup_model for Cursor ---

    def test_lookup_cursor_gpt_via_suffix_stripping(self) -> None:
        """gpt-4o-mini-fast matches gpt-4o-mini via suffix stripping (not resolver)."""
        cache = LLMPricingCache()
        cache._data = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006},
        }
        result = cache._lookup_model("cursor", "gpt-4o-mini-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.00000015

    def test_lookup_cursor_gpt_via_resolver(self) -> None:
        """gpt-5.4-nano-none matches gpt-5.4-nano via resolver variant extraction."""
        cache = LLMPricingCache()
        cache._data = {
            "gpt-5.4-nano": {"input_cost_per_token": 0.0000001, "output_cost_per_token": 0.0000004},
        }
        result = cache._lookup_model("cursor", "gpt-5.4-nano-none")
        assert result is not None
        assert result["input_cost_per_token"] == 0.0000001

    def test_lookup_cursor_claude_via_resolver(self) -> None:
        """claude-4.5-haiku-max-thinking matches claude-haiku-4-5 via resolver."""
        cache = LLMPricingCache()
        cache._data = {
            "claude-haiku-4-5": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000005},
        }
        result = cache._lookup_model("cursor", "claude-4.5-haiku-max-thinking")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000001

    def test_lookup_cursor_gemini_via_resolver(self) -> None:
        """gemini-2.5-pro-xhigh-fast matches gemini-2.5-pro via suffix strip + resolver."""
        cache = LLMPricingCache()
        cache._data = {
            "gemini-2.5-pro": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000004},
        }
        result = cache._lookup_model("cursor", "gemini-2.5-pro-xhigh-fast")
        assert result is not None
        assert result["input_cost_per_token"] == 0.000001

    def test_lookup_cursor_no_match(self) -> None:
        """Completely unknown cursor model returns None."""
        cache = LLMPricingCache()
        cache._data = SAMPLE_PRICING_DATA
        result = cache._lookup_model("cursor", "totally-unknown-model-xyz")
        assert result is None


# ── Background refresh ───────────────────────────────────────────────


class TestBackgroundRefresh:
    async def test_start_stop_background_refresh(self) -> None:
        """Starts and stops cleanly."""
        cache = LLMPricingCache()

        with patch.object(cache, "refresh", new_callable=AsyncMock) as mock_refresh:
            await cache.start_background_refresh()
            assert cache._refresh_task is not None
            assert not cache._refresh_task.done()

            # Give the loop a moment to call refresh at least once
            await asyncio.sleep(0.05)

            await cache.stop_background_refresh()
            assert cache._refresh_task is None

            # The refresh should have been called at least once
            mock_refresh.assert_called()


# ── Singleton ─────────────────────────────────────────────────────────


class TestSingleton:
    def test_pricing_cache_singleton(self) -> None:
        """pricing_cache is an instance of LLMPricingCache."""
        assert isinstance(pricing_cache, LLMPricingCache)


# ── Disk cache helper methods ─────────────────────────────────────────


class TestDiskCacheHelpers:
    def test_is_disk_cache_fresh_no_file(self) -> None:
        """Returns False when cache file doesn't exist."""
        cache = LLMPricingCache()
        with patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file:
            mock_file.exists.return_value = False
            assert not cache._is_disk_cache_fresh()

    def test_is_disk_cache_fresh_old_file(self) -> None:
        """Returns False when cache file is older than 24h."""
        cache = LLMPricingCache()
        mock_stat = MagicMock()
        mock_stat.st_mtime = time.time() - (_REFRESH_INTERVAL_SECONDS + 100)

        with patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.stat.return_value = mock_stat
            assert not cache._is_disk_cache_fresh()

    def test_is_disk_cache_fresh_recent_file(self) -> None:
        """Returns True when cache file is less than 24h old."""
        cache = LLMPricingCache()
        mock_stat = MagicMock()
        mock_stat.st_mtime = time.time() - 100

        with patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.stat.return_value = mock_stat
            assert cache._is_disk_cache_fresh()

    def test_read_disk_cache_valid(self, tmp_path: object) -> None:
        """Reads valid JSON from disk cache."""
        cache = LLMPricingCache()
        with patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file:
            mock_file.exists.return_value = True
            mock_file.read_text.return_value = json.dumps(SAMPLE_PRICING_DATA)
            result = cache._read_disk_cache()
            assert result == SAMPLE_PRICING_DATA

    def test_read_disk_cache_missing(self) -> None:
        """Returns None when file doesn't exist."""
        cache = LLMPricingCache()
        with patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file:
            mock_file.exists.return_value = False
            result = cache._read_disk_cache()
            assert result is None

    def test_write_disk_cache(self, tmp_path: object) -> None:
        """Writes data to disk cache using atomic write."""
        cache = LLMPricingCache()
        mock_tmp_file = MagicMock()
        with (
            patch("ai_cli_runner.llm_pricing._CACHE_DIR") as mock_dir,
            patch("ai_cli_runner.llm_pricing._CACHE_FILE") as mock_file,
        ):
            mock_file.with_suffix.return_value = mock_tmp_file
            cache._write_disk_cache(SAMPLE_PRICING_DATA)
            mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True, mode=0o700)
            mock_tmp_file.write_text.assert_called_once()
            mock_tmp_file.replace.assert_called_once_with(mock_file)


# ── Client integration ────────────────────────────────────────────────


class TestClientIntegration:
    async def test_call_ai_cli_populates_cost_from_pricing_cache(self) -> None:
        """call_ai_cli auto-populates cost_usd when provider doesn't report it."""
        from ai_cli_runner.client import call_ai_cli

        fake_gemini_output = json.dumps(
            {
                "response": "hello",
                "stats": {
                    "models": {
                        "gemini-2.5-flash": {
                            "tokens": {"input": 100, "candidates": 50, "thoughts": 0, "cached": 0},
                            "api": {"totalLatencyMs": 500},
                        }
                    }
                },
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_gemini_output
        mock_result.stderr = ""

        with (
            patch("ai_cli_runner.client._run_with_process_group", return_value=mock_result),
            patch("ai_cli_runner.client.pricing_cache") as mock_cache,
        ):
            mock_cache.calculate_cost.return_value = 0.000045

            result = await call_ai_cli(
                prompt="test",
                ai_provider="gemini",
                ai_model="gemini-2.5-flash",
                output_format="json",
            )

            assert result.success
            assert result.usage is not None
            assert result.usage.cost_usd == 0.000045
            mock_cache.calculate_cost.assert_called_once()

    async def test_call_ai_cli_skips_pricing_when_cost_already_set(self) -> None:
        """call_ai_cli does NOT override cost_usd when provider reports it."""
        from ai_cli_runner.client import call_ai_cli

        fake_claude_output = json.dumps(
            {
                "result": "hello",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
                "modelUsage": {"claude-sonnet-4-6": {}},
                "total_cost_usd": 0.123,
                "duration_ms": 1000,
            }
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_claude_output
        mock_result.stderr = ""

        with (
            patch("ai_cli_runner.client._run_with_process_group", return_value=mock_result),
            patch("ai_cli_runner.client.pricing_cache") as mock_cache,
        ):
            result = await call_ai_cli(
                prompt="test",
                ai_provider="claude",
                ai_model="claude-sonnet-4-6",
                output_format="json",
            )

            assert result.success
            assert result.usage is not None
            assert result.usage.cost_usd == 0.123
            mock_cache.calculate_cost.assert_not_called()

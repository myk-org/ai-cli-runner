import asyncio

import pytest

from ai_cli_runner.parallel import MAX_CONCURRENT_AI_CALLS, run_parallel_with_limit


class TestRunParallelWithLimit:
    async def test_all_succeeding(self) -> None:
        async def coro(i: int) -> int:
            return i * 2

        results = await run_parallel_with_limit([coro(i) for i in range(5)])
        assert results == [0, 2, 4, 6, 8]

    async def test_preserves_result_order(self) -> None:
        async def delayed(i: int) -> int:
            await asyncio.sleep(0.01 * (5 - i))
            return i

        results = await run_parallel_with_limit([delayed(i) for i in range(5)])
        assert results == [0, 1, 2, 3, 4]

    async def test_failure_in_one_coroutine(self) -> None:
        async def maybe_fail(i: int) -> int:
            if i == 2:
                raise ValueError("boom")
            return i

        results = await run_parallel_with_limit([maybe_fail(i) for i in range(5)])
        assert results[0] == 0
        assert results[1] == 1
        assert isinstance(results[2], ValueError)
        assert str(results[2]) == "boom"
        assert results[3] == 3
        assert results[4] == 4

    async def test_concurrency_limiting(self) -> None:
        active = 0
        max_active = 0

        async def tracked_coro(i: int) -> int:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return i

        results = await run_parallel_with_limit(
            [tracked_coro(i) for i in range(20)],
            max_concurrency=3,
        )
        assert max_active <= 3
        assert len(results) == 20
        assert results == list(range(20))

    async def test_max_concurrency_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await run_parallel_with_limit(
                [],
                max_concurrency=0,
            )

    async def test_empty_list(self) -> None:
        results = await run_parallel_with_limit([])
        assert results == []

    async def test_single_coroutine(self) -> None:
        async def single() -> str:
            return "done"

        results = await run_parallel_with_limit([single()])
        assert results == ["done"]

    async def test_max_concurrent_ai_calls_constant(self) -> None:
        assert MAX_CONCURRENT_AI_CALLS == 10

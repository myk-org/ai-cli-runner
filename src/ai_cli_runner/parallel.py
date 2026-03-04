import asyncio
from collections.abc import Coroutine
from typing import Any

MAX_CONCURRENT_AI_CALLS = 10


async def run_parallel_with_limit(
    coroutines: list[Coroutine[Any, Any, Any]],
    max_concurrency: int = MAX_CONCURRENT_AI_CALLS,
) -> list[Any]:
    """Run coroutines in parallel with bounded concurrency.

    Args:
        coroutines: List of coroutines to execute.
        max_concurrency: Maximum concurrent executions.

    Returns:
        List of results (including exceptions if any failed).
    """
    if max_concurrency < 1:
        max_concurrency = MAX_CONCURRENT_AI_CALLS
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(coro: Coroutine[Any, Any, Any]) -> object:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[bounded(c) for c in coroutines],
        return_exceptions=True,
    )

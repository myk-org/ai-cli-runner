from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class AITokenUsage:
    """Token usage metadata from an AI CLI call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    duration_ms: int | None = None
    model: str = ""
    provider: str = ""


@dataclass
class AIResult:
    """Result from an AI CLI call.

    Supports tuple unpacking for backward compatibility:
        success, text = await call_ai_cli(...)
    """

    success: bool
    text: str
    usage: AITokenUsage | None = None

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking for backward compatibility.

        Allows: success, text = await call_ai_cli(...)
        """
        return iter((self.success, self.text))

    def __getitem__(self, index: int) -> Any:
        """Support index access for backward compatibility."""
        return (self.success, self.text)[index]

    def __len__(self) -> int:
        """Support len() for backward compatibility with tuple."""
        return 2

    def __bool__(self) -> bool:
        """Return success status for boolean evaluation.

        Allows: if result: ...
        """
        return self.success

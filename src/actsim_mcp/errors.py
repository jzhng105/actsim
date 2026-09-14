"""Error type carrying agent-actionable remediation text."""

from __future__ import annotations

from typing import Iterable, Sequence

try:  # MCP Python SDK >= 2.0
    from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError
except ModuleNotFoundError:  # pragma: no cover - exercised only on SDK 1.x
    from mcp.server.fastmcp.exceptions import ToolError as _SdkToolError  # type: ignore[no-redef]


class ActsimToolError(_SdkToolError):
    """A tool failure the caller can act on.

    Subclasses the SDK's ToolError deliberately: the SDK treats any other
    exception as a server-side crash and replaces the text with a generic
    message, which would hide exactly the remediation the agent needs. Carrying a
    ``hint`` means the message says what to do next, not just what went wrong.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


def did_you_mean(value: str, options: Iterable[str], limit: int = 3) -> str:
    """Return a ' Did you mean: a, b?' fragment, or '' when nothing is close."""
    import difflib

    close = difflib.get_close_matches(value, list(options), n=limit, cutoff=0.6)
    return f" Did you mean: {', '.join(close)}?" if close else ""


def unknown_value(kind: str, value: str, options: Sequence[str]) -> ActsimToolError:
    """Build the standard 'unknown X' error listing every accepted value."""
    return ActsimToolError(
        f"Unknown {kind}: {value!r}.{did_you_mean(value, options)}",
        hint=f"Valid {kind}s: {', '.join(sorted(options))}.",
    )

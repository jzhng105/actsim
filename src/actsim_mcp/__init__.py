"""MCP server exposing the actsim actuarial modelling package to LLM agents.

Run it over stdio with ``actsim-mcp`` or ``python -m actsim_mcp``.
"""

from .version import SERVER_VERSION

__all__ = ["SERVER_VERSION", "build_server", "main"]
__version__ = SERVER_VERSION


def build_server():
    """Return a configured MCP server instance (imports the SDK lazily)."""
    from .server import build_server as _build

    return _build()


def main() -> None:
    """Console-script entry point."""
    from .__main__ import main as _main

    _main()

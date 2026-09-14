#!/usr/bin/env python3
"""Compatibility shim for the old single-file actsim MCP server.

The server now lives in the installable ``actsim_mcp`` package under ``src/`` and
is launched with ``actsim-mcp`` or ``python -m actsim_mcp``. This file stays so
that client configurations pointing at ``mcp/actsim_server.py`` keep working; it
does nothing but hand off to the package.

Note that this directory is named ``mcp``, which shadows the MCP SDK package
whenever the repository root ends up on ``sys.path``. That is why the server
itself is no longer kept here - see docs/mcp_server.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _prepare_path() -> None:
    """Allow running from a source checkout without installing the package.

    The repository root is deliberately kept off ``sys.path``: this file's own
    directory is named ``mcp``, and having the root importable would make
    ``import mcp`` resolve to it instead of the SDK.
    """
    repo_root = str(_REPO_SRC.parent)
    for entry in (repo_root, os.curdir, ""):
        while entry in sys.path:
            sys.path.remove(entry)
    if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
        sys.path.insert(0, str(_REPO_SRC))


def main() -> int:
    _prepare_path()
    try:
        from actsim_mcp.__main__ import main as _main
    except ModuleNotFoundError as exc:  # pragma: no cover - install guidance only
        print(
            "Could not import actsim_mcp. Install the server with:\n"
            "    pip install 'actsim[mcp]'\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "Note: mcp/actsim_server.py is a compatibility shim. "
        "Point your MCP client at the 'actsim-mcp' command instead.",
        file=sys.stderr,
    )
    return _main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

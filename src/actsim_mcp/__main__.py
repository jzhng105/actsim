"""Command-line entry point: ``actsim-mcp`` / ``python -m actsim_mcp``."""

from __future__ import annotations

import argparse
import sys

from .settings import SETTINGS
from .version import SERVER_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actsim-mcp",
        description="Model Context Protocol server for the actsim actuarial toolkit.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport to serve on (default: stdio, for local MCP clients).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the registered tools, resources and prompts, then exit.",
    )
    parser.add_argument("--version", action="store_true", help="Print the server version and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.version:
        print(SERVER_VERSION)
        return 0

    from .server import build_server

    server = build_server()

    if args.list_tools:
        import asyncio
        import json

        async def describe() -> dict:
            tools = await server.list_tools()
            resources = await server.list_resources()
            prompts = await server.list_prompts()
            return {
                "server_version": SERVER_VERSION,
                "workspace": str(SETTINGS.workspace),
                "tools": [
                    {"name": t.name, "description": (t.description or "").split("\n")[0]}
                    for t in tools
                ],
                "resources": [str(r.uri) for r in resources],
                "prompts": [p.name for p in prompts],
            }

        print(json.dumps(asyncio.run(describe()), indent=2))
        return 0

    # stdio transport owns stdout; anything else printed there corrupts the stream.
    print(
        f"actsim MCP server {SERVER_VERSION} starting on {args.transport} "
        f"(workspace: {SETTINGS.workspace})",
        file=sys.stderr,
    )
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Adapter over the MCP Python SDK so the server runs on both 1.x and 2.x.

The SDK renamed ``mcp.server.fastmcp.FastMCP`` to ``mcp.server.mcpserver.MCPServer``
in 2.0. Both expose the same decorator surface (``tool`` / ``resource`` / ``prompt``)
but with slightly different keyword arguments, so every decorator call in this
package goes through :func:`tool_decorator` rather than touching the SDK directly.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

try:  # MCP Python SDK >= 2.0
    from mcp.server.mcpserver import Context, Image, MCPServer

    SDK_MAJOR = 2
except ModuleNotFoundError:  # pragma: no cover - exercised only on SDK 1.x
    from mcp.server.fastmcp import Context, FastMCP as MCPServer, Image  # type: ignore[no-redef]

    SDK_MAJOR = 1

from mcp.types import ToolAnnotations

__all__ = [
    "SDK_MAJOR",
    "Context",
    "Image",
    "MCPServer",
    "ToolAnnotations",
    "annotations",
    "make_server",
    "prompt_decorator",
    "resource_decorator",
    "run_server",
    "tool_decorator",
]


def _supported(method: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keyword arguments the installed SDK's decorator does not accept."""
    params = inspect.signature(method).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def annotations(
    *,
    title: str,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool = True,
    open_world: bool = False,
) -> ToolAnnotations:
    """Build tool annotations with actsim's defaults (read-only, closed-world).

    Almost every actsim tool is a pure computation over data the caller supplies,
    so the defaults describe that case and only the few tools that write files or
    mutate stored artifacts need to override them.
    """
    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


def make_server(name: str, *, instructions: str, version: str) -> MCPServer:
    """Instantiate the SDK server object, passing only supported kwargs."""
    kwargs = _supported(
        MCPServer.__init__,
        {
            "name": name,
            "instructions": instructions,
            "version": version,
            "log_level": "WARNING",
        },
    )
    return MCPServer(**kwargs)


def tool_decorator(
    server: MCPServer,
    *,
    name: str,
    title: str,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool = True,
    open_world: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a tool.

    ``structured_output=False`` is requested where supported: every tool in this
    package returns a pre-rendered string, and letting the SDK also emit a
    ``{"result": ...}`` structured block would duplicate the whole payload in the
    model's context.
    """
    kwargs = _supported(
        MCPServer.tool,
        {
            "name": name,
            "title": title,
            "annotations": annotations(
                title=title,
                read_only=read_only,
                destructive=destructive,
                idempotent=idempotent,
                open_world=open_world,
            ),
            "structured_output": False,
        },
    )
    return server.tool(**kwargs)


def resource_decorator(
    server: MCPServer,
    uri: str,
    *,
    name: str,
    description: str,
    mime_type: str = "application/json",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a resource, passing only kwargs the installed SDK accepts."""
    kwargs = _supported(
        MCPServer.resource,
        {"name": name, "description": description, "mime_type": mime_type},
    )
    return server.resource(uri, **kwargs)


def prompt_decorator(
    server: MCPServer, *, name: str, description: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a prompt, passing only kwargs the installed SDK accepts."""
    kwargs = _supported(MCPServer.prompt, {"name": name, "description": description})
    return server.prompt(**kwargs)


def run_server(server: MCPServer, transport: str, *, host: str, port: int) -> None:
    """Start the server, routing host/port the way the installed SDK expects.

    SDK 2.x takes them as ``run()`` keywords; 1.x's ``run()`` accepts only
    ``transport`` and ``mount_path`` and reads the bind address off the server's
    settings object. Passing them unconditionally raises TypeError on 1.x.
    """
    if transport == "stdio":
        server.run(transport="stdio")
        return

    accepted = _supported(type(server).run, {"host": host, "port": port})
    if not accepted:
        settings = getattr(server, "settings", None)
        for attribute, value in (("host", host), ("port", port)):
            if settings is not None and hasattr(settings, attribute):
                setattr(settings, attribute, value)
    server.run(transport=transport, **accepted)

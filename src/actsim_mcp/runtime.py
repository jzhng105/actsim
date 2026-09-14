"""Process-level guards that let actsim run safely inside an MCP stdio server.

Two things in the actsim library are incompatible with a stdio transport if left
alone:

* ``DistributionFitter.fit`` prints one line per candidate distribution, and
  ``@timing_decorator`` prints a timing line from every simulation entry point.
  Anything written to stdout lands in the middle of the JSON-RPC stream and
  corrupts the session, so :func:`captured` redirects stdout to stderr.
* ``StochasticSimulator.__init__`` calls ``logging.basicConfig(level=INFO)`` and
  the plotting helpers call ``plt.show()``, which needs an interactive backend.

Importing this module before ``actsim`` pins matplotlib to the headless Agg
backend; :func:`captured` handles the rest.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from typing import Iterator

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

LOGGER = logging.getLogger("actsim_mcp")


@contextlib.contextmanager
def captured() -> Iterator[io.StringIO]:
    """Run library code with its stdout chatter diverted away from the transport.

    Yields the buffer so callers can surface the captured text as diagnostics.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            yield buffer
        finally:
            text = buffer.getvalue()
            if text.strip():
                LOGGER.debug("actsim stdout: %s", text.strip())


@contextlib.contextmanager
def captured_figure() -> Iterator[list[bytes]]:
    """Collect PNG bytes from actsim plotting helpers that end in ``plt.show()``.

    ``plt.show`` is swapped for a hook that serialises the active figure, so the
    library's own plotting code is reused as-is instead of being reimplemented.
    """
    images: list[bytes] = []
    original_show = plt.show

    def _capture(*_args: object, **_kwargs: object) -> None:
        figure = plt.gcf()
        if not figure.get_axes():
            return
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        images.append(buffer.getvalue())

    plt.show = _capture  # type: ignore[assignment]
    try:
        with captured():
            yield images
        # Some helpers build a figure without ever calling show(); capture it too.
        if not images and plt.gcf().get_axes():
            _capture()
    finally:
        plt.show = original_show  # type: ignore[assignment]
        plt.close("all")


def configure_logging() -> None:
    """Send all logging to stderr so it can never reach the JSON-RPC stream."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is sys.stdout:
            root.removeHandler(handler)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)

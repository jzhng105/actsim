"""Response rendering: one payload, two presentations.

Every tool builds a plain dict and hands it here. ``markdown`` is the default
because it costs far fewer tokens than pretty-printed JSON for the same numbers;
``json`` is available for callers that post-process the result.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-serialisable types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.reset_index(drop=True).to_dict("records")]
    if isinstance(value, pd.Series):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return str(value)


def number(value: Any, digits: int = 2) -> str:
    """Format one number for markdown: thousands separators, NaN as '-'."""
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "-"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    if numeric != 0 and abs(numeric) < 1e-4:
        return f"{numeric:.3e}"
    if numeric.is_integer() and abs(numeric) < 1e15:
        # Counts, seeds and trial numbers read badly as '3,000.00'.
        return f"{numeric:,.0f}"
    return f"{numeric:,.{digits}f}"


def table(frame: pd.DataFrame, *, digits: int = 2, index: bool = True) -> str:
    """Render a DataFrame as a compact markdown table."""
    if frame.empty:
        return "_(no rows)_"
    display = frame.reset_index() if index else frame.copy()
    header = list(display.columns)
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in display.itertuples(index=False):
        cells = [
            number(v, digits) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
            else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bullets(data: Mapping[str, Any], *, digits: int = 2) -> str:
    """Render a flat mapping as a markdown bullet list."""
    lines = []
    for key, value in data.items():
        label = key.replace("_", " ")
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            lines.append(f"- **{label}**: {number(value, digits)}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"- **{label}**: {', '.join(str(v) for v in value)}")
        elif isinstance(value, Mapping):
            inner = ", ".join(f"{k}={number(v, digits) if isinstance(v, (int, float)) else v}"
                              for k, v in value.items())
            lines.append(f"- **{label}**: {inner}")
        elif value is not None and value != "":
            lines.append(f"- **{label}**: {value}")
    return "\n".join(lines)


def histogram(values: Sequence[float], bins: int = 12, width: int = 28) -> str:
    """A text histogram, so the shape of a distribution costs ~15 lines not 10k."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return "_(no data)_"
    counts, edges = np.histogram(array, bins=bins)
    peak = counts.max() or 1
    lines = ["```"]
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * count / peak))
        lines.append(f"{low:>12,.0f} .. {high:>12,.0f} | {bar:<{width}} {count}")
    lines.append("```")
    return "\n".join(lines)


def document(title: str, sections: Sequence[tuple[str, str]], *, footer: str = "") -> str:
    """Assemble a markdown document from (heading, body) pairs, dropping empties."""
    parts = [f"# {title}"]
    for heading, body in sections:
        if not body:
            continue
        parts.append(f"## {heading}\n\n{body}" if heading else body)
    if footer:
        parts.append(footer)
    return "\n\n".join(parts)


def respond(
    response_format: ResponseFormat | str,
    payload: Mapping[str, Any],
    markdown: str,
) -> str:
    """Return the markdown rendering or the JSON payload, per the caller's choice."""
    fmt = ResponseFormat(response_format)
    if fmt is ResponseFormat.JSON:
        return json.dumps(jsonable(payload), indent=2)
    return markdown


def next_steps(*suggestions: str) -> str:
    """A short 'what to call next' block; keeps multi-step workflows discoverable."""
    items = [s for s in suggestions if s]
    if not items:
        return ""
    return "**Next steps**\n" + "\n".join(f"- {s}" for s in items)

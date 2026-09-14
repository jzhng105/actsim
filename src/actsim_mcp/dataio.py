"""Input resolution: files, inline values and artifact handles.

Every tool that accepts data funnels through here so that path sandboxing, size
limits and parameter parsing behave identically everywhere.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .errors import ActsimToolError
from .settings import SETTINGS
from .store import STORE

READABLE_SUFFIXES = (".csv", ".json", ".tsv", ".txt")


def resolve_path(path: str, *, must_exist: bool = True) -> Path:
    """Resolve a caller-supplied path inside the configured workspace.

    The server refuses to touch anything outside ``ACTSIM_MCP_WORKSPACE`` so a
    prompt-injected path cannot read ``~/.ssh`` or overwrite files elsewhere.
    """
    if not path or not path.strip():
        raise ActsimToolError("Empty path.", hint="Give a path relative to the workspace root.")
    candidate = Path(path).expanduser()
    root = SETTINGS.workspace
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ActsimToolError(
            f"Path {path!r} is outside the server workspace ({root}).",
            hint=(
                "Pass a path inside the workspace, or restart the server with "
                "ACTSIM_MCP_WORKSPACE pointing at the directory you want to use."
            ),
        ) from exc
    if must_exist and not resolved.is_file():
        raise ActsimToolError(
            f"File not found: {resolved}.",
            hint=f"Workspace root is {root}. Supported formats: {', '.join(READABLE_SUFFIXES)}.",
        )
    return resolved


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV/TSV/JSON table from the workspace into a DataFrame."""
    resolved = resolve_path(path)
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(resolved)
        if suffix == ".tsv":
            return pd.read_csv(resolved, sep="\t")
        if suffix == ".txt":
            return pd.read_csv(resolved, sep=None, engine="python")
        if suffix == ".json":
            return pd.read_json(resolved)
    except Exception as exc:
        raise ActsimToolError(
            f"Could not parse {resolved.name}: {exc}",
            hint="Check the file is a well-formed table with a header row.",
        ) from exc
    raise ActsimToolError(
        f"Unsupported file type {suffix!r}.",
        hint=f"Supported: {', '.join(READABLE_SUFFIXES)}.",
    )


def read_json_file(path: str) -> Any:
    """Read and parse a JSON document from the workspace."""
    resolved = resolve_path(path)
    try:
        return json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ActsimToolError(
            f"{resolved.name} is not valid JSON: {exc}",
            hint="Check for trailing commas or unquoted keys.",
        ) from exc


def pick_column(frame: pd.DataFrame, column: str | None, *, source: str) -> pd.Series:
    """Select one numeric column, requiring an explicit choice when ambiguous."""
    if column:
        if column not in frame.columns:
            raise ActsimToolError(
                f"Column {column!r} not found in {source}.",
                hint=f"Available columns: {', '.join(map(str, frame.columns))}.",
            )
        series = frame[column]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        raise ActsimToolError(
            f"{source} has {frame.shape[1]} columns, so the column to use is ambiguous.",
            hint=f"Pass column= one of: {', '.join(map(str, frame.columns))}.",
        )
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ActsimToolError(
            f"Column {series.name!r} in {source} has no numeric values.",
            hint="Pick a numeric column, or clean the source data first.",
        )
    return numeric.reset_index(drop=True)


def resolve_series(
    *,
    values: Sequence[float] | None = None,
    file_path: str | None = None,
    artifact_id: str | None = None,
    column: str | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve one numeric sample from exactly one of the three input styles.

    Returns the values and a short human-readable description of where they came
    from, which tools echo back so the agent can confirm it used the right input.
    """
    # Treat a blank string as absent, then dispatch on the same normalised values
    # the guard counted - otherwise file_path="" passes the count and then fails
    # downstream with a confusing "Empty path".
    file_path = file_path.strip() or None if isinstance(file_path, str) else file_path
    artifact_id = artifact_id.strip() or None if isinstance(artifact_id, str) else artifact_id

    provided = [name for name, val in
                (("values", values), ("file_path", file_path), ("artifact_id", artifact_id))
                if val is not None]
    if len(provided) != 1:
        raise ActsimToolError(
            f"Provide exactly one data source; got {len(provided)} ({', '.join(provided) or 'none'}).",
            hint=(
                "Use values=[...] for small inline samples, file_path='losses.csv' for files "
                "in the workspace, or artifact_id='data_ab12cd34' for data already loaded."
            ),
        )

    if values is not None:
        if len(values) > SETTINGS.max_inline_values:
            raise ActsimToolError(
                f"{len(values)} inline values exceeds the limit of {SETTINGS.max_inline_values}.",
                hint="Write the data to a CSV in the workspace and pass file_path instead.",
            )
        array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
        source = f"{array.size} inline values"
    elif file_path is not None:
        frame = read_table(file_path)
        array = pick_column(frame, column, source=file_path).to_numpy(float)
        source = f"{file_path}" + (f" [{column}]" if column else "")
    else:
        artifact = STORE.get(str(artifact_id))
        array = _series_from_artifact(artifact.payload, column, str(artifact_id)).to_numpy(float)
        source = f"artifact {artifact_id}" + (f" [{column}]" if column else "")

    if array.size == 0:
        raise ActsimToolError(
            f"No finite numeric values in {source}.",
            hint="Check the column choice and that the data is not all NaN.",
        )
    return array, source


def _series_from_artifact(payload: Any, column: str | None, artifact_id: str) -> pd.Series:
    """Coerce whatever an artifact holds into a single numeric series."""
    if isinstance(payload, pd.DataFrame):
        return pick_column(payload, column, source=f"artifact {artifact_id}")
    if isinstance(payload, pd.Series):
        return pd.to_numeric(payload, errors="coerce").dropna().reset_index(drop=True)
    if isinstance(payload, np.ndarray):
        return pd.Series(payload.ravel().astype(float))
    if isinstance(payload, dict):
        for key in ("results", "aggregate", "values"):
            if key in payload:
                return _series_from_artifact(payload[key], column, artifact_id)
    raise ActsimToolError(
        f"Artifact {artifact_id!r} does not hold numeric data this tool can read.",
        hint="Use actsim_describe_artifact to see what it contains.",
    )


def parse_params(value: Any) -> tuple[float, ...]:
    """Parse a distribution parameter cell into a tuple of floats - never eval().

    Policy CSVs commonly store parameters as ``"(9.0, 0.5)"`` or ``"9.0 0.5"``.
    The original server ran ``eval()`` on these cells, which executes whatever the
    file contains; ``ast.literal_eval`` plus a numeric fallback is equivalent for
    well-formed data and inert for hostile data.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(float(v) for v in np.asarray(value).ravel())
    if isinstance(value, (int, float, np.number)):
        return (float(value),)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (tuple, list)):
            return tuple(float(v) for v in parsed)
        if isinstance(parsed, (int, float)):
            return (float(parsed),)
    except (ValueError, SyntaxError):
        pass
    cleaned = text.strip("()[]{}").replace(",", " ")
    parts = [p for p in cleaned.split() if p]
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ActsimToolError(
            f"Could not read distribution parameters from {text!r}.",
            hint="Use a JSON array like [9.0, 0.5], or a CSV cell like '(9.0, 0.5)'.",
        ) from exc


def write_export(frame: pd.DataFrame, path: str, *, fmt: str = "csv") -> Path:
    """Write a DataFrame to the workspace, creating parent directories."""
    resolved = resolve_path(path, must_exist=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        frame.to_csv(resolved, index=False)
    elif fmt == "json":
        resolved.write_text(frame.to_json(orient="records", indent=2, date_format="iso"))
    elif fmt == "parquet":
        frame.to_parquet(resolved, index=False)
    else:
        raise ActsimToolError(
            f"Unsupported export format {fmt!r}.", hint="Use 'csv', 'json' or 'parquet'."
        )
    return resolved

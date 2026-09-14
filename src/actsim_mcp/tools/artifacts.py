"""Tools for loading data and managing stored artifacts."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import moments
from ..compat import tool_decorator
from ..dataio import read_table, resolve_series, write_export
from ..params import (
    ArtifactId,
    Column,
    FilePath,
    Format,
    InlineValues,
    Limit,
    Offset,
    paginate,
)
from ..render import ResponseFormat, bullets, document, histogram, next_steps, respond, table
from ..store import STORE
from ..settings import SETTINGS


def _payload_frame(payload) -> pd.DataFrame:
    """Best-effort view of an artifact payload as a DataFrame for preview/export."""
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, pd.Series):
        return payload.to_frame(payload.name or "value")
    if isinstance(payload, np.ndarray):
        if payload.ndim == 1:
            return pd.DataFrame({"value": payload})
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        # Most specific view first: per-event detail beats the annual aggregate,
        # which in turn beats a bare results vector.
        for key in ("events", "claims", "development", "marginals", "annual", "aggregate", "results"):
            if key in payload and payload[key] is not None:
                return _payload_frame(payload[key])
        flat = {k: v for k, v in payload.items() if np.isscalar(v)}
        if flat:
            return pd.DataFrame([flat])
    return pd.DataFrame()


def register(server) -> None:
    @tool_decorator(server, name="actsim_load_dataset", title="Load Dataset")
    async def actsim_load_dataset(
        values: InlineValues = None,
        file_path: FilePath = None,
        column: Column = None,
        label: Annotated[str, Field(default="dataset", description="Short name for the stored artifact.", max_length=80)] = "dataset",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Load a numeric loss sample into a reusable dataset artifact.

        Use this once per dataset, then pass the returned artifact_id to fitting,
        risk-metric and stress-testing tools instead of re-sending the numbers.
        Accepts inline values or a CSV/TSV/JSON file inside the server workspace.

        Returns: artifact id, source description and summary statistics.
        """
        array, source = resolve_series(values=values, file_path=file_path, column=column)
        stats = moments(array)
        artifact = STORE.put(
            "dataset",
            label,
            pd.Series(array, name=column or "value"),
            {"source": source, "observations": stats["count"]},
        )
        payload = {"artifact_id": artifact.id, "source": source, "statistics": stats}
        markdown = document(
            f"Dataset `{artifact.id}`",
            [
                ("Source", f"{source} - {stats['count']:,} observations"),
                ("Summary", bullets(stats)),
                ("Shape", histogram(array)),
                ("", next_steps(
                    f"actsim_fit_distributions(artifact_id='{artifact.id}') to find the best-fitting model",
                    f"actsim_risk_metrics(artifact_id='{artifact.id}') for VaR/TVaR on the raw data",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_list_artifacts", title="List Artifacts")
    async def actsim_list_artifacts(
        kind: Annotated[
            Literal["any", "dataset", "fit", "simulation", "portfolio", "policies", "claims", "triangle"],
            Field(default="any", description="Filter to one artifact kind."),
        ] = "any",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """List artifacts currently held in server memory.

        Artifacts are produced by the fitting, simulation and claims tools and are
        lost when the server restarts. Use this to recover a handle you no longer
        have in context.

        Returns: one row per artifact with its id, kind, label and key metadata.
        """
        items = [a for a in STORE if kind == "any" or a.kind == kind]
        rows = [a.summary() for a in sorted(items, key=lambda a: a.created_at, reverse=True)]
        payload = {
            "artifacts": rows,
            "count": len(rows),
            "capacity": SETTINGS.max_artifacts,
        }
        if not rows:
            markdown = "No artifacts stored. Produce one with actsim_load_dataset, actsim_fit_distributions or actsim_simulate_aggregate."
        else:
            frame = pd.DataFrame(rows).set_index("artifact_id")
            markdown = document(
                "Stored artifacts",
                [("", table(frame)), ("", f"{len(rows)} of {SETTINGS.max_artifacts} slots used.")],
            )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_describe_artifact", title="Describe Artifact")
    async def actsim_describe_artifact(
        artifact_id: ArtifactId,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Describe a stored artifact: kind, metadata, columns and summary statistics.

        Use before previewing or exporting so you know which columns exist.

        Returns: artifact metadata plus per-column dtypes and numeric summaries.
        """
        artifact = STORE.get(artifact_id)
        frame = _payload_frame(artifact.payload)
        columns = [
            {"name": str(c), "dtype": str(frame[c].dtype)} for c in frame.columns
        ]
        describe = (
            frame.describe(include=[np.number]).round(4)
            if not frame.empty and frame.select_dtypes("number").shape[1]
            else pd.DataFrame()
        )
        payload = {
            **artifact.summary(),
            "rows": int(frame.shape[0]),
            "columns": columns,
            "statistics": describe,
        }
        markdown = document(
            f"Artifact `{artifact.id}` ({artifact.kind})",
            [
                ("Metadata", bullets(artifact.summary())),
                ("Columns", ", ".join(f"`{c['name']}` ({c['dtype']})" for c in columns) or "_(scalar payload)_"),
                ("Statistics", table(describe) if not describe.empty else ""),
                ("", next_steps(
                    f"actsim_preview_artifact(artifact_id='{artifact.id}') to see rows",
                    f"actsim_export_artifact(artifact_id='{artifact.id}', path='out.csv') to write it to disk",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_preview_artifact", title="Preview Artifact Rows")
    async def actsim_preview_artifact(
        artifact_id: ArtifactId,
        limit: Limit = 20,
        offset: Offset = 0,
        columns: Annotated[
            list[str] | None,
            Field(default=None, description="Subset of columns to return.", max_length=40),
        ] = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Return a paginated window of rows from a stored artifact.

        Simulations hold tens of thousands of rows; this is the safe way to look at
        them without flooding the context. Page with offset/next_offset.

        Returns: the requested rows plus total/has_more/next_offset pagination fields.
        """
        artifact = STORE.get(artifact_id)
        frame = _payload_frame(artifact.payload)
        if columns:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                from ..errors import ActsimToolError

                raise ActsimToolError(
                    f"Columns not in artifact {artifact_id!r}: {', '.join(missing)}.",
                    hint=f"Available: {', '.join(map(str, frame.columns))}.",
                )
            frame = frame[columns]
        window = frame.iloc[offset : offset + limit]
        payload = {
            "artifact_id": artifact_id,
            **paginate(int(frame.shape[0]), offset, int(window.shape[0])),
            "rows": window,
        }
        markdown = document(
            f"`{artifact_id}` rows {offset}-{offset + len(window)} of {len(frame):,}",
            [("", table(window, index=False))],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(
        server,
        name="actsim_export_artifact",
        title="Export Artifact To File",
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def actsim_export_artifact(
        artifact_id: ArtifactId,
        path: Annotated[
            str,
            Field(description="Destination path relative to the server workspace, e.g. 'output/simulation.csv'.", min_length=1),
        ],
        fmt: Annotated[
            Literal["csv", "json", "parquet"],
            Field(default="csv", description="Output file format."),
        ] = "csv",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Write a stored artifact to a file in the server workspace.

        Overwrites the destination if it exists. Paths outside the workspace root
        are rejected.

        Returns: the resolved path and the number of rows written.
        """
        artifact = STORE.get(artifact_id)
        frame = _payload_frame(artifact.payload)
        if frame.empty:
            from ..errors import ActsimToolError

            raise ActsimToolError(
                f"Artifact {artifact_id!r} holds no tabular data to export.",
                hint="Use actsim_describe_artifact to see what it contains.",
            )
        resolved = write_export(frame, path, fmt=fmt)
        payload = {
            "artifact_id": artifact_id,
            "path": str(resolved),
            "format": fmt,
            "rows": int(frame.shape[0]),
            "columns": [str(c) for c in frame.columns],
        }
        markdown = document(
            "Export complete",
            [("", bullets(payload))],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(
        server,
        name="actsim_delete_artifact",
        title="Delete Artifact",
        read_only=False,
        destructive=True,
    )
    async def actsim_delete_artifact(
        artifact_id: ArtifactId,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Drop a stored artifact to free server memory.

        Returns: whether the artifact existed and how many remain.
        """
        deleted = STORE.delete(artifact_id)
        payload = {"artifact_id": artifact_id, "deleted": deleted, "remaining": len(STORE)}
        markdown = (
            f"Deleted `{artifact_id}`. {len(STORE)} artifact(s) remain."
            if deleted
            else f"No artifact `{artifact_id}` to delete. {len(STORE)} artifact(s) stored."
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_inspect_file", title="Inspect Workspace File")
    async def actsim_inspect_file(
        file_path: Annotated[
            str, Field(description="CSV/TSV/JSON file inside the server workspace.", min_length=1)
        ],
        limit: Limit = 10,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Preview a data file's columns, dtypes and first rows before using it.

        Use this to discover the right `column` argument for fitting tools, or to
        confirm a policy CSV has the schema actsim_simulate_claims requires.

        Returns: shape, column dtypes and a sample of rows.
        """
        frame = read_table(file_path)
        head = frame.head(limit)
        payload = {
            "file_path": file_path,
            "rows": int(frame.shape[0]),
            "columns": {str(c): str(frame[c].dtype) for c in frame.columns},
            "sample": head,
        }
        markdown = document(
            f"`{file_path}`",
            [
                ("", f"{frame.shape[0]:,} rows x {frame.shape[1]} columns"),
                ("Columns", ", ".join(f"`{c}` ({frame[c].dtype})" for c in frame.columns)),
                ("Sample", table(head, index=False)),
            ],
        )
        return respond(response_format, payload, markdown)

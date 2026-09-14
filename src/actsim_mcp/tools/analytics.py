"""Portfolio-level risk metrics, stress testing and reporting."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import moments, return_periods, risk_table, tvar, validate_quantiles, var
from ..compat import tool_decorator
from ..dataio import resolve_series
from ..errors import ActsimToolError
from ..params import (
    Column,
    FilePath,
    Format,
    InlineValues,
    OptionalArtifactId,
    Quantiles,
)
from ..render import ResponseFormat, bullets, document, histogram, next_steps, respond, table
from ..settings import DEFAULT_QUANTILES, SETTINGS
from ..store import STORE
from ..version import SERVER_VERSION


def _loss_vector(
    values: list[float] | None, file_path: str | None, artifact_id: str | None, column: str | None
) -> tuple[np.ndarray, str]:
    """Resolve a loss vector, preferring a simulation artifact's aggregate column."""
    if artifact_id:
        artifact = STORE.get(artifact_id)
        if artifact.kind in ("simulation", "portfolio"):
            aggregate = np.asarray(artifact.payload["aggregate"], dtype=float)
            return aggregate, f"{artifact.kind} {artifact_id}"
    return resolve_series(values=values, file_path=file_path, artifact_id=artifact_id, column=column)


def register(server) -> None:
    @tool_decorator(server, name="actsim_risk_metrics", title="Risk Metrics")
    async def actsim_risk_metrics(
        values: InlineValues = None,
        file_path: FilePath = None,
        artifact_id: OptionalArtifactId = None,
        column: Column = None,
        quantiles: Quantiles = None,
        return_period_years: Annotated[
            list[float] | None,
            Field(default=None, description="Return periods in years, e.g. [100, 250].", max_length=20),
        ] = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Compute VaR, TVaR, moments and return periods for any loss vector.

        Works on a simulation or portfolio artifact, a stored dataset, a workspace
        file, or inline values, so empirical and simulated losses are measured the
        same way.

        Returns: moments, a VaR/TVaR table, return-period losses and a shape histogram.
        """
        array, source = _loss_vector(values, file_path, artifact_id, column)
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)
        stats = moments(array)
        measures = risk_table(array, qs)
        periods = return_periods(array, return_period_years or (10, 25, 50, 100, 200, 250))
        payload = {
            "source": source,
            "statistics": stats,
            "risk_measures": measures,
            "return_periods": periods,
        }
        markdown = document(
            f"Risk metrics: {source}",
            [
                ("Statistics", bullets(stats)),
                ("Risk measures", table(measures)),
                ("Return periods", bullets(periods)),
                ("Shape", histogram(array)),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_stress_test", title="Stress Test")
    async def actsim_stress_test(
        scenarios: Annotated[
            list[dict[str, Any]],
            Field(
                description=(
                    "Scenarios to apply to the loss vector. Each is "
                    "{'name': str, 'severity_multiplier': float, 'shift': float, 'frequency_multiplier': float}; "
                    "omitted keys default to no change. Example: "
                    "[{'name':'inflation 10%','severity_multiplier':1.1}, {'name':'cat load','shift':250000}]."
                ),
                min_length=1,
                max_length=25,
            ),
        ],
        values: InlineValues = None,
        file_path: FilePath = None,
        artifact_id: OptionalArtifactId = None,
        column: Column = None,
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Re-measure a loss distribution under multiplicative and additive stresses.

        `severity_multiplier` scales every loss (claims inflation), `shift` adds a
        flat amount (a known exposure change), and `frequency_multiplier` scales the
        aggregate as a proxy for more claims of the same size. Each scenario is
        reported against the unstressed base so the marginal capital impact is
        explicit.

        Returns: base and per-scenario mean/std/VaR/TVaR with changes versus base.
        """
        array, source = _loss_vector(values, file_path, artifact_id, column)
        qs = validate_quantiles(quantiles, (0.95, 0.99, 0.995))

        base_row = {
            "scenario": "base",
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            **{f"VaR {q:g}": var(array, q) for q in qs},
            **{f"TVaR {q:g}": tvar(array, q) for q in qs},
        }
        rows = [base_row]
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                raise ActsimToolError(
                    f"Scenario {index} is not an object.",
                    hint="Each scenario looks like {'name':'inflation','severity_multiplier':1.1}.",
                )
            unknown = set(scenario) - {"name", "severity_multiplier", "shift", "frequency_multiplier"}
            if unknown:
                raise ActsimToolError(
                    f"Scenario {scenario.get('name', index)!r} has unsupported key(s): {', '.join(sorted(unknown))}.",
                    hint="Supported keys: name, severity_multiplier, shift, frequency_multiplier.",
                )
            name = str(scenario.get("name") or f"scenario_{index + 1}")
            sev_mult = float(scenario.get("severity_multiplier", 1.0))
            freq_mult = float(scenario.get("frequency_multiplier", 1.0))
            shift = float(scenario.get("shift", 0.0))
            if sev_mult < 0 or freq_mult < 0:
                raise ActsimToolError(
                    f"Scenario {name!r} has a negative multiplier.",
                    hint="Multipliers scale losses, so they must be >= 0 (1.0 means no change).",
                )
            stressed = array * sev_mult * freq_mult + shift
            rows.append(
                {
                    "scenario": name,
                    "mean": float(stressed.mean()),
                    "std": float(stressed.std(ddof=1)) if stressed.size > 1 else 0.0,
                    **{f"VaR {q:g}": var(stressed, q) for q in qs},
                    **{f"TVaR {q:g}": tvar(stressed, q) for q in qs},
                }
            )

        frame = pd.DataFrame(rows).set_index("scenario")
        deltas = (frame / frame.loc["base"] - 1.0).drop(index="base").round(4) * 100
        payload = {
            "source": source,
            "scenarios": frame,
            "change_vs_base_pct": deltas,
        }
        markdown = document(
            f"Stress test: {source}",
            [
                ("Absolute", table(frame)),
                ("Change versus base (%)", table(deltas)),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_compare_simulations", title="Compare Simulations")
    async def actsim_compare_simulations(
        artifact_ids: Annotated[
            list[str],
            Field(description="Two or more simulation/portfolio/dataset artifact ids to compare.", min_length=2, max_length=8),
        ],
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Put several simulations side by side on the same risk measures.

        The usual way to answer "what did that reinsurance structure buy us" or
        "how much does the copula assumption move the 1-in-200".

        Returns: one column per artifact with mean, std, CoV and VaR/TVaR at each quantile.
        """
        qs = validate_quantiles(quantiles, (0.9, 0.99, 0.995))
        columns: dict[str, dict[str, float]] = {}
        for artifact_id in artifact_ids:
            array, label = _loss_vector(None, None, artifact_id, None)
            stats = moments(array)
            columns[artifact_id] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "coefficient_of_variation": stats["coefficient_of_variation"],
                **{f"VaR {q:g}": var(array, q) for q in qs},
                **{f"TVaR {q:g}": tvar(array, q) for q in qs},
            }
        frame = pd.DataFrame(columns)
        first = frame.columns[0]
        relative = (frame.div(frame[first], axis=0) - 1.0).round(4) * 100
        payload = {"comparison": frame, "change_vs_first_pct": relative, "baseline": first}
        markdown = document(
            "Simulation comparison",
            [
                ("Absolute", table(frame)),
                (f"Change versus `{first}` (%)", table(relative)),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_generate_risk_report", title="Generate Risk Report")
    async def actsim_generate_risk_report(
        artifact_ids: Annotated[
            list[str],
            Field(description="Artifacts to include: fits, simulations, portfolios, claims or triangles.", min_length=1, max_length=12),
        ],
        title: Annotated[str, Field(default="Risk Analysis Report", description="Report title.", max_length=120)] = "Risk Analysis Report",
        quantiles: Quantiles = None,
        include_observations: Annotated[
            bool,
            Field(default=True, description="Append data-driven observations on tail heaviness, volatility and layer efficiency."),
        ] = True,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Assemble stored artifacts into one markdown risk report.

        Pulls each artifact's headline numbers - fitted models, aggregate risk
        measures, claim summaries, development triangles - into a single document.
        Observations are derived from the numbers (skewness, coefficient of
        variation, TVaR-to-VaR ratio); they are descriptive, not advice.

        Returns: a markdown report, or the structured sections when response_format='json'.
        """
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)
        sections: list[tuple[str, str]] = []
        structured: list[dict[str, Any]] = []
        observations: list[str] = []

        for artifact_id in artifact_ids:
            artifact = STORE.get(artifact_id)
            heading = f"{artifact.kind.title()} `{artifact_id}` - {artifact.label}"
            entry: dict[str, Any] = {"artifact_id": artifact_id, "kind": artifact.kind}

            if artifact.kind in ("simulation", "portfolio"):
                array = np.asarray(artifact.payload["aggregate"], dtype=float)
                stats = moments(array)
                measures = risk_table(array, qs, artifact.payload.get("events"))
                entry.update({"statistics": stats, "risk_measures": measures})
                sections.append((heading, bullets(artifact.summary()) + "\n\n" + bullets(stats) + "\n\n" + table(measures)))
                if include_observations:
                    observations.extend(_observe(artifact_id, array, stats, measures))
            elif artifact.kind == "fit":
                fitter = artifact.payload
                selected = fitter.selected_fit
                info = {
                    "selected": selected["name"] if selected else None,
                    "params": [round(float(p), 4) for p in selected["params"]] if selected else None,
                    "aic": round(float(selected["aic"]), 2) if selected else None,
                    "ks": round(float(selected["ks"]), 4) if selected else None,
                    "observations": artifact.meta.get("observations"),
                }
                entry.update(info)
                sections.append((heading, bullets(info)))
            elif artifact.kind == "claims":
                claims = artifact.payload
                loss_column = "ultimate_loss" if "ultimate_loss" in claims.columns else "amount"
                stats = moments(claims[loss_column].to_numpy(float))
                entry.update({"claims": int(len(claims)), "statistics": stats})
                sections.append((heading, bullets({"claims": len(claims), **stats})))
            elif artifact.kind == "triangle":
                development = artifact.payload
                triangle = development.pivot_table(
                    index="accident_year", columns="development_month", values="incurred_loss", aggfunc="sum"
                ).round(2)
                entry["triangle"] = triangle
                sections.append((heading, table(triangle)))
            else:
                array, _ = _loss_vector(None, None, artifact_id, None)
                stats = moments(array)
                entry["statistics"] = stats
                sections.append((heading, bullets(stats)))
            structured.append(entry)

        if include_observations and observations:
            sections.append(("Observations", "\n".join(f"- {o}" for o in observations)))

        markdown = document(
            title,
            sections,
            footer="_Generated by the actsim MCP server "
            f"(v{SERVER_VERSION}). Figures are simulation output, not a statement of required capital._",
        )
        payload = {"title": title, "artifacts": structured, "observations": observations}
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_server_info", title="Server Info")
    async def actsim_server_info(response_format: Format = ResponseFormat.MARKDOWN) -> str:
        """Report server version, actsim version, configured limits and stored artifacts.

        Use this first in a session to confirm the workspace root (which bounds every
        file path) and how much simulation capacity is available.

        Returns: versions, limits, the workspace path and current artifact usage.
        """
        from importlib import metadata

        try:
            actsim_version = metadata.version("actsim")
        except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
            actsim_version = "unknown"

        info = {
            "server_version": SERVER_VERSION,
            "actsim_version": actsim_version,
            "workspace": str(SETTINGS.workspace),
            "default_simulations": SETTINGS.default_simulations,
            "max_simulations": SETTINGS.max_simulations,
            "max_policies": SETTINGS.max_policies,
            "max_artifacts": SETTINGS.max_artifacts,
            "artifacts_stored": len(STORE),
        }
        payload = {**info, "artifacts": [a.summary() for a in STORE]}
        markdown = document(
            "actsim MCP server",
            [
                ("Environment", bullets(info)),
                ("Stored artifacts", table(pd.DataFrame([a.summary() for a in STORE]).set_index("artifact_id")) if len(STORE) else "_none_"),
                ("", next_steps(
                    "actsim_list_distributions() to see supported distributions and their parameterisations",
                    "actsim_load_dataset(file_path='losses.csv') to start from your own data",
                )),
            ],
        )
        return respond(response_format, payload, markdown)


def _observe(artifact_id: str, array: np.ndarray, stats: dict[str, float], measures: pd.DataFrame) -> list[str]:
    """Derive plain-language observations from the numbers, without giving advice."""
    notes: list[str] = []
    if stats["skewness"] > 2:
        notes.append(
            f"`{artifact_id}`: skewness {stats['skewness']:.1f} - the distribution is heavily right-tailed, "
            "so the mean sits well below the high quantiles."
        )
    cov = stats["coefficient_of_variation"]
    if np.isfinite(cov) and cov > 1:
        notes.append(f"`{artifact_id}`: coefficient of variation {cov:.2f} - annual results vary by more than their mean.")
    if "0.99" in measures.index and "TVaR" in measures.columns:
        var_99 = float(measures.loc["0.99", "VaR"])
        tvar_99 = float(measures.loc["0.99", "TVaR"])
        if var_99 and tvar_99 / var_99 > 1.5:
            notes.append(
                f"`{artifact_id}`: TVaR(99%) is {tvar_99 / var_99:.2f}x VaR(99%) - losses beyond the 99th percentile "
                "are much larger than the percentile itself."
            )
    if stats["zero_share"] > 0.2:
        notes.append(f"`{artifact_id}`: {stats['zero_share'] * 100:.0f}% of trials produced no loss at all.")
    return notes

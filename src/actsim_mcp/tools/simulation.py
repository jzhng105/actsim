"""Monte Carlo simulation tools built on actsim.StochasticSimulator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import apply_layer, moments, return_periods, risk_table, validate_quantiles
from ..catalog import FREQUENCY_DISTRIBUTIONS, normalize, validate_params
from ..compat import Image, tool_decorator
from ..dataio import read_json_file, read_table
from ..errors import ActsimToolError, unknown_value
from ..params import ArtifactId, DistributionName, DistributionParams, Format, NumSimulations, Quantiles, Seed
from ..render import ResponseFormat, bullets, document, histogram, next_steps, respond, table
from ..runtime import captured, captured_figure
from ..settings import COPULA_TYPES, DEFAULT_QUANTILES, SETTINGS
from ..store import STORE

# statsmodels' copulas reject out-of-domain theta with opaque errors, so validate here.
_THETA_DOMAIN = {
    "clayton": (0.0, False, "theta must be > 0"),
    "gumbel": (1.0, True, "theta must be >= 1"),
    "frank": (None, None, "theta must be non-zero"),
}


def _expected_events(freq_dist: str, freq_params: tuple[float, ...], num_sim: int) -> float:
    """Expected number of stored event rows, used to refuse runaway keep_events runs."""
    from ..catalog import make_frozen

    try:
        with captured():
            mean = float(make_frozen(freq_dist, list(freq_params)).mean())
    except Exception:
        mean = float("nan")
    return mean * num_sim if np.isfinite(mean) else float("nan")


def _validate_dependence(
    correlation: float | None, copula: str | None, theta: float
) -> tuple[float | None, str | None, float]:
    """Check the frequency/severity dependence arguments hang together.

    actsim picks its dependence branch on ``correlation is not None`` and
    ``copula_type is not None``. A copula name with no correlation silently falls
    through to the independent branch, so require the combination to be explicit.
    """
    if copula is None:
        return correlation, None, theta
    name = copula.lower()
    if name not in COPULA_TYPES:
        raise unknown_value("copula", copula, COPULA_TYPES)
    if name == "gaussian":
        if correlation is None:
            raise ActsimToolError(
                "The gaussian copula needs a correlation.",
                hint="Pass correlation between -1 and 1, e.g. correlation=0.4.",
            )
        return correlation, name, theta
    bound, inclusive, message = _THETA_DOMAIN[name]
    if name == "frank":
        if theta == 0:
            raise ActsimToolError(f"The frank copula requires {message}.", hint="Use theta=2 for moderate positive dependence.")
    elif bound is not None and not (theta >= bound if inclusive else theta > bound):
        raise ActsimToolError(
            f"The {name} copula requires {message}; got theta={theta}.",
            hint="Try theta=1.5 for the gumbel copula or theta=2 for the clayton copula.",
        )
    # Archimedean copulas read only theta, but actsim still tests `correlation is
    # not None` to enter the copula branch, so supply a neutral placeholder.
    return (0.0 if correlation is None else correlation), name, theta


def _simulation_payload(artifact_id: str, aggregate: np.ndarray, events: pd.DataFrame | None, quantiles: list[float]):
    stats = moments(aggregate)
    measures = risk_table(aggregate, quantiles, events)
    return stats, measures


def register(server) -> None:
    @tool_decorator(server, name="actsim_simulate_aggregate", title="Simulate Aggregate Loss")
    async def actsim_simulate_aggregate(
        freq_dist: Annotated[
            str,
            Field(description=f"Claim-count distribution, one of {list(FREQUENCY_DISTRIBUTIONS)}.", min_length=2),
        ],
        freq_params: Annotated[
            list[float],
            Field(description="Frequency parameters, e.g. poisson [lambda] or negative binomial [r, p].", min_length=1, max_length=3),
        ],
        sev_dist: DistributionName,
        sev_params: DistributionParams,
        num_simulations: NumSimulations = SETTINGS.default_simulations,
        seed: Seed = SETTINGS.default_seed,
        correlation: Annotated[
            float | None,
            Field(default=None, description="Frequency/severity rank correlation in [-1, 1]. Omit for independence.", ge=-1.0, le=1.0),
        ] = None,
        copula: Annotated[
            Literal["gaussian", "frank", "gumbel", "clayton"] | None,
            Field(default=None, description="Copula for frequency/severity dependence. Omit to use a Gaussian correlation matrix when correlation is given."),
        ] = None,
        theta: Annotated[
            float,
            Field(default=0.0, description="Archimedean copula parameter (frank/gumbel/clayton). Ignored by the gaussian copula."),
        ] = 0.0,
        keep_events: Annotated[
            bool,
            Field(default=True, description="Keep per-event losses. Required for OEP and for actsim_apply_layer; costs memory proportional to expected claim count."),
        ] = True,
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Run a frequency/severity Monte Carlo and summarise the aggregate loss distribution.

        Each trial draws a claim count from the frequency distribution and that many
        severities, then sums them into one annual loss. Dependence between
        frequency and severity is optional: give `correlation` alone for a Gaussian
        correlation structure, or `copula` with `correlation`/`theta` for gaussian,
        frank, gumbel or clayton dependence.

        The full simulation is stored as an artifact; the response carries summary
        statistics, a VaR/TVaR/AEP/OEP table and a shape histogram rather than the
        raw trials.

        Returns: simulation artifact id, moments, risk-measure table, return periods
        and event counts.
        """
        freq_name = normalize(freq_dist)
        if freq_name not in FREQUENCY_DISTRIBUTIONS:
            raise ActsimToolError(
                f"{freq_name!r} is a severity distribution, not a claim-count distribution.",
                hint=f"Use one of {', '.join(FREQUENCY_DISTRIBUTIONS)} for freq_dist.",
            )
        freq_values = validate_params(freq_name, freq_params)
        sev_name = normalize(sev_dist)
        sev_values = validate_params(sev_name, sev_params)
        corr, copula_name, theta_value = _validate_dependence(correlation, copula, theta)
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)

        if keep_events:
            expected = _expected_events(freq_name, freq_values, num_simulations)
            if np.isfinite(expected) and expected > SETTINGS.max_artifact_cells / 4:
                raise ActsimToolError(
                    f"This run would store about {expected:,.0f} event rows, beyond the server's limit.",
                    hint=(
                        "Set keep_events=false (you lose OEP and layer treatment), reduce num_simulations, "
                        "or raise ACTSIM_MCP_MAX_ARTIFACT_CELLS."
                    ),
                )

        from actsim import StochasticSimulator

        with captured():
            np.random.seed(seed)  # the correlated branches draw from the global RNG
            simulator = StochasticSimulator(
                freq_dist=freq_name,
                freq_params=freq_values,
                sev_dist=sev_name,
                sev_params=sev_values,
                num_sim=num_simulations,
                keep_all=keep_events,
                seed=seed,
                correlation=corr,
                copula_type=copula_name,
                theta=theta_value,
            )
            simulator.gen_agg_simulations()

        aggregate = np.asarray(simulator.results, dtype=float)
        events = simulator.all_simulations if keep_events else None
        if events is not None and events.empty:
            events = None

        dependence = (
            f"{copula_name} copula"
            if copula_name
            else ("gaussian correlation" if corr is not None else "independent")
        )
        artifact = STORE.put(
            "simulation",
            f"{freq_name}/{sev_name} aggregate",
            {"aggregate": pd.Series(aggregate, name="aggregate_loss"), "events": events},
            {
                "frequency": f"{freq_name}{list(freq_values)}",
                "severity": f"{sev_name}{list(sev_values)}",
                "simulations": num_simulations,
                "dependence": dependence,
                "seed": seed,
                "events_stored": int(events.shape[0]) if events is not None else 0,
            },
        )

        stats, measures = _simulation_payload(artifact.id, aggregate, events, qs)
        periods = return_periods(aggregate, (10, 25, 50, 100, 200, 250))
        payload = {
            "artifact_id": artifact.id,
            "configuration": {
                "frequency": {"distribution": freq_name, "params": list(freq_values)},
                "severity": {"distribution": sev_name, "params": list(sev_values)},
                "simulations": num_simulations,
                "dependence": dependence,
                "correlation": corr,
                "theta": theta_value if copula_name and copula_name != "gaussian" else None,
                "seed": seed,
                "events_stored": int(events.shape[0]) if events is not None else 0,
            },
            "statistics": stats,
            "risk_measures": measures,
            "return_periods": periods,
        }
        markdown = document(
            f"Aggregate simulation `{artifact.id}`",
            [
                ("Configuration", bullets({
                    "frequency": f"{freq_name}{list(freq_values)}",
                    "severity": f"{sev_name}{list(sev_values)}",
                    "trials": num_simulations,
                    "dependence": dependence,
                    "events stored": payload["configuration"]["events_stored"],
                    "seed": seed,
                })),
                ("Aggregate statistics", bullets(stats)),
                ("Risk measures", table(measures)),
                ("Return periods", bullets(periods)),
                ("Shape", histogram(aggregate)),
                ("", next_steps(
                    f"actsim_apply_layer(artifact_id='{artifact.id}', per_occurrence_deductible=..., aggregate_limit=...)" if events is not None else "re-run with keep_events=true to enable layer treatment and OEP",
                    f"actsim_plot_simulation(artifact_id='{artifact.id}')",
                    f"actsim_stress_test(artifact_id='{artifact.id}', scenarios=[...])",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_analyze_simulation", title="Analyze Simulation")
    async def actsim_analyze_simulation(
        artifact_id: ArtifactId,
        quantiles: Quantiles = None,
        return_period_years: Annotated[
            list[float] | None,
            Field(default=None, description="Return periods in years, e.g. [100, 250]. Defaults to 10/25/50/100/200/250.", max_length=20),
        ] = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Recompute risk measures for a stored simulation at different quantiles.

        VaR and AEP are the aggregate annual loss at a quantile; TVaR averages the
        losses beyond it; OEP is the largest single event in a year and is only
        available when the simulation kept per-event data.

        Returns: moments, the VaR/TVaR/AEP/OEP table and return-period losses.
        """
        artifact = STORE.get(artifact_id, kind="simulation")
        aggregate = np.asarray(artifact.payload["aggregate"], dtype=float)
        events = artifact.payload.get("events")
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)
        periods = return_periods(aggregate, return_period_years or (10, 25, 50, 100, 200, 250))
        stats = moments(aggregate)
        measures = risk_table(aggregate, qs, events)
        payload = {
            "artifact_id": artifact_id,
            **artifact.summary(),
            "statistics": stats,
            "risk_measures": measures,
            "return_periods": periods,
        }
        markdown = document(
            f"Simulation `{artifact_id}`",
            [
                ("Configuration", bullets(artifact.summary())),
                ("Aggregate statistics", bullets(stats)),
                ("Risk measures", table(measures)),
                ("Return periods", bullets(periods)),
                ("", "" if events is not None else "_OEP omitted: this simulation was run with keep_events=false._"),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_apply_layer", title="Apply Deductibles And Limits")
    async def actsim_apply_layer(
        artifact_id: ArtifactId,
        per_occurrence_deductible: Annotated[
            float, Field(default=0.0, description="Deductible applied to each individual claim.", ge=0.0)
        ] = 0.0,
        per_occurrence_limit: Annotated[
            float | None, Field(default=None, description="Cap on each claim after its deductible. Null means unlimited.", ge=0.0)
        ] = None,
        aggregate_deductible: Annotated[
            float, Field(default=0.0, description="Annual aggregate deductible applied to the summed layer losses.", ge=0.0)
        ] = 0.0,
        aggregate_limit: Annotated[
            float | None, Field(default=None, description="Annual aggregate limit. Null means unlimited.", ge=0.0)
        ] = None,
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Apply per-occurrence and aggregate deductibles/limits to a stored simulation.

        Losses are trimmed claim by claim (excess of the per-occurrence deductible,
        capped at the per-occurrence limit), summed by year, then trimmed again by
        the annual aggregate deductible and limit. Years with no qualifying events
        are retained as zeros so the means are not biased upwards.

        Requires a simulation run with keep_events=true.

        Returns: ground-up, ceded-to-layer and retained statistics, a risk-measure
        table for the ceded losses, and an artifact id holding the annual detail.
        """
        artifact = STORE.get(artifact_id, kind="simulation")
        events = artifact.payload.get("events")
        if events is None or events.empty:
            raise ActsimToolError(
                f"Simulation {artifact_id!r} has no per-event data, so a layer cannot be applied.",
                hint="Re-run actsim_simulate_aggregate with keep_events=true.",
            )
        years = int(artifact.meta.get("simulations") or events["year"].max())
        annual = apply_layer(
            events,
            per_occurrence_deductible=per_occurrence_deductible,
            per_occurrence_limit=per_occurrence_limit,
            aggregate_deductible=aggregate_deductible,
            aggregate_limit=aggregate_limit,
            years=years,
        )
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)
        ceded = annual["ceded"].to_numpy(float)
        ground_up = annual["ground_up"].to_numpy(float)

        terms = {
            "per_occurrence_deductible": per_occurrence_deductible,
            "per_occurrence_limit": per_occurrence_limit,
            "aggregate_deductible": aggregate_deductible,
            "aggregate_limit": aggregate_limit,
        }
        stored = STORE.put(
            "simulation",
            f"layered {artifact_id}",
            {"aggregate": annual["ceded"].rename("ceded_loss"), "events": None, "annual": annual},
            {"source_simulation": artifact_id, "simulations": years, "terms": json.dumps(terms)},
        )
        comparison = pd.DataFrame(
            {
                "ground_up": moments(ground_up),
                "ceded_to_layer": moments(ceded),
                "retained": moments(annual["retained"].to_numpy(float)),
            }
        )
        measures = risk_table(ceded, qs)
        loss_cost_ratio = float(ceded.mean() / ground_up.mean()) if ground_up.mean() else float("nan")
        payload = {
            "artifact_id": stored.id,
            "source_simulation": artifact_id,
            "terms": terms,
            "years": years,
            "comparison": comparison,
            "layer_risk_measures": measures,
            "expected_ceded_loss": float(ceded.mean()),
            "share_of_ground_up": loss_cost_ratio,
            "years_hitting_layer": int((ceded > 0).sum()),
            "years_exhausting_limit": (
                int((ceded >= float(aggregate_limit)).sum()) if aggregate_limit is not None else None
            ),
        }
        markdown = document(
            f"Layer applied to `{artifact_id}`",
            [
                ("Terms", bullets(terms)),
                ("Ground-up vs layer", table(comparison)),
                ("Layer risk measures", table(measures)),
                ("Layer behaviour", bullets({
                    "expected ceded loss": payload["expected_ceded_loss"],
                    "share of ground-up": loss_cost_ratio,
                    "years hitting the layer": f"{payload['years_hitting_layer']:,} of {years:,}",
                    "years exhausting the limit": payload["years_exhausting_limit"],
                })),
                ("", f"Annual detail stored as `{stored.id}`."),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_simulate_correlated_portfolio", title="Simulate Correlated Portfolio")
    async def actsim_simulate_correlated_portfolio(
        lines: Annotated[
            list[dict[str, Any]] | None,
            Field(
                default=None,
                description=(
                    "Lines of business as objects: "
                    "[{'dist_name':'Property','dist_type':'lognormal','dist_param':[12,1.2]}, ...]. "
                    "Mutually exclusive with lines_file."
                ),
                max_length=50,
            ),
        ] = None,
        correlation_matrix: Annotated[
            list[list[float]] | None,
            Field(default=None, description="Square correlation matrix, rows in the same order as `lines`. Mutually exclusive with correlation_matrix_file.", max_length=50),
        ] = None,
        lines_file: Annotated[
            str | None, Field(default=None, description="JSON file in the workspace holding the lines list.")
        ] = None,
        correlation_matrix_file: Annotated[
            str | None, Field(default=None, description="CSV file in the workspace holding the correlation matrix, with row labels in the first column.")
        ] = None,
        num_simulations: NumSimulations = SETTINGS.default_simulations,
        seed: Seed = SETTINGS.default_seed,
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Simulate several correlated lines of business and aggregate them.

        Marginals are joined through a Gaussian copula built from the supplied
        correlation matrix (Cholesky decomposition of correlated normals, mapped to
        each line's distribution by inverse CDF), so the aggregate captures
        diversification benefit rather than simply adding the lines.

        Returns: portfolio artifact id, aggregate statistics and risk measures, the
        per-line marginal summary, the realised correlation matrix, and the
        diversification benefit versus the sum of standalone VaRs.
        """
        if (lines is None) == (lines_file is None):
            raise ActsimToolError(
                "Provide exactly one of `lines` or `lines_file`.",
                hint="Inline lines are easiest: [{'dist_name':'Property','dist_type':'lognormal','dist_param':[12,1.2]}].",
            )
        if (correlation_matrix is None) == (correlation_matrix_file is None):
            raise ActsimToolError(
                "Provide exactly one of `correlation_matrix` or `correlation_matrix_file`.",
                hint="Inline: [[1,0.4],[0.4,1]]. File: a CSV whose first column holds the row labels.",
            )

        spec_list = lines if lines is not None else read_json_file(str(lines_file))
        if not isinstance(spec_list, list) or not spec_list:
            raise ActsimToolError(
                "The lines definition must be a non-empty list of objects.",
                hint="Each entry needs dist_name, dist_type and dist_param.",
            )

        cleaned: list[dict[str, Any]] = []
        for index, entry in enumerate(spec_list):
            if not isinstance(entry, dict):
                raise ActsimToolError(
                    f"Line {index} is not an object.",
                    hint="Each entry must look like {'dist_name':'Property','dist_type':'lognormal','dist_param':[12,1.2]}.",
                )
            missing = {"dist_name", "dist_type", "dist_param"} - set(entry)
            if missing:
                raise ActsimToolError(
                    f"Line {index} ({entry.get('dist_name', '?')}) is missing: {', '.join(sorted(missing))}.",
                    hint="Required keys: dist_name (label), dist_type (distribution), dist_param (parameter list).",
                )
            dist_type = normalize(str(entry["dist_type"]))
            params = validate_params(dist_type, list(entry["dist_param"]))
            cleaned.append(
                {
                    "index": index + 1,
                    "dist_name": str(entry["dist_name"]),
                    "dist_type": dist_type,
                    "dist_param": list(params),
                }
            )

        if correlation_matrix is not None:
            matrix = np.asarray(correlation_matrix, dtype=float)
            labels = [line["dist_name"] for line in cleaned]
        else:
            frame = read_table(str(correlation_matrix_file))
            matrix = frame.iloc[:, 1:].to_numpy(float)
            labels = [str(v) for v in frame.iloc[:, 0]]

        n = len(cleaned)
        if matrix.shape != (n, n):
            raise ActsimToolError(
                f"Correlation matrix is {matrix.shape[0]}x{matrix.shape[1]} but {n} lines were given.",
                hint=f"Supply an {n}x{n} matrix with rows in the same order as the lines.",
            )
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ActsimToolError(
                "Correlation matrix is not symmetric.",
                hint="Mirror the upper triangle into the lower triangle.",
            )
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
            raise ActsimToolError(
                "Correlation matrix must have 1.0 on the diagonal.",
                hint="A correlation (not covariance) matrix is required.",
            )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -1e-8:
            raise ActsimToolError(
                f"Correlation matrix is not positive semi-definite (smallest eigenvalue {eigenvalues.min():.4g}).",
                hint="Shrink the off-diagonal correlations towards zero until the matrix is admissible.",
            )

        from actsim import StochasticSimulator

        with tempfile.TemporaryDirectory() as tmp:
            corr_path = Path(tmp) / "corr.csv"
            dist_path = Path(tmp) / "lines.json"
            # actsim reads the matrix with index_col=0, so write the labels as an index.
            pd.DataFrame(matrix, index=labels, columns=labels).to_csv(corr_path)
            dist_path.write_text(json.dumps(cleaned))

            with captured():
                np.random.seed(seed)
                simulator = StochasticSimulator(
                    freq_dist="poisson",
                    freq_params=(1,),
                    sev_dist="normal",
                    sev_params=(0, 1),
                    num_sim=num_simulations,
                    keep_all=True,
                    seed=seed,
                )
                aggregate = np.asarray(
                    simulator.gen_multivariate_corr_simulations(str(corr_path), str(dist_path), True),
                    dtype=float,
                )
                marginals = np.asarray(simulator._all_simulations_data, dtype=float)

        names = [line["dist_name"] for line in cleaned]
        marginal_frame = pd.DataFrame(marginals.T, columns=names)
        qs = validate_quantiles(quantiles, DEFAULT_QUANTILES)

        realized = marginal_frame.corr().round(4)
        marginal_stats = pd.DataFrame({name: moments(marginal_frame[name].to_numpy()) for name in names})
        standalone_var = {name: float(np.quantile(marginal_frame[name], 0.995)) for name in names}
        portfolio_var = float(np.quantile(aggregate, 0.995))
        diversification = sum(standalone_var.values()) - portfolio_var

        artifact = STORE.put(
            "portfolio",
            f"{n}-line correlated portfolio",
            {"aggregate": pd.Series(aggregate, name="aggregate_loss"), "marginals": marginal_frame},
            {"lines": n, "simulations": num_simulations, "seed": seed, "line_names": ", ".join(names)},
        )
        payload = {
            "artifact_id": artifact.id,
            "lines": cleaned,
            "simulations": num_simulations,
            "aggregate_statistics": moments(aggregate),
            "aggregate_risk_measures": risk_table(aggregate, qs),
            "marginal_statistics": marginal_stats,
            "input_correlation": pd.DataFrame(matrix, index=labels, columns=labels),
            "realized_correlation": realized,
            "diversification": {
                "sum_of_standalone_var_99.5": sum(standalone_var.values()),
                "portfolio_var_99.5": portfolio_var,
                "benefit": diversification,
                "benefit_pct": (diversification / sum(standalone_var.values()) * 100) if sum(standalone_var.values()) else float("nan"),
            },
        }
        markdown = document(
            f"Correlated portfolio `{artifact.id}`",
            [
                ("Lines", table(pd.DataFrame(cleaned), index=False)),
                ("Aggregate statistics", bullets(payload["aggregate_statistics"])),
                ("Aggregate risk measures", table(payload["aggregate_risk_measures"])),
                ("Per-line statistics", table(marginal_stats)),
                ("Realised correlation", table(realized, digits=3)),
                ("Diversification at 99.5%", bullets(payload["diversification"])),
                ("Shape", histogram(aggregate)),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_plot_simulation", title="Plot Simulation Distribution")
    async def actsim_plot_simulation(
        artifact_id: ArtifactId,
        bins: Annotated[int | None, Field(default=None, description="Histogram bins. Defaults to sqrt(trials).", ge=5, le=500)] = None,
        log_scale: Annotated[bool, Field(default=False, description="Log-scale the density axis, which makes a heavy tail legible.")] = False,
    ) -> Image:
        """Render the aggregate loss distribution of a stored simulation as a PNG.

        Returns: a PNG image.
        """
        artifact = STORE.get(artifact_id)
        if artifact.kind not in ("simulation", "portfolio"):
            raise ActsimToolError(
                f"Artifact {artifact_id!r} is a {artifact.kind}; this tool plots simulations and portfolios.",
                hint="Use actsim_plot_fit for fit artifacts.",
            )
        aggregate = np.asarray(artifact.payload["aggregate"], dtype=float)

        from actsim import StochasticSimulator

        with captured():
            plotter = StochasticSimulator.__new__(StochasticSimulator)
            plotter._results = aggregate
        with captured_figure() as images:
            plotter.plot_distribution(bins=bins, log_option=log_scale)
        if not images:
            raise ActsimToolError("Plot rendering produced no figure.", hint="Check the simulation artifact still holds results.")
        return Image(data=images[0], format="png")

    @tool_decorator(server, name="actsim_plot_frequency_severity", title="Plot Frequency vs Severity")
    async def actsim_plot_frequency_severity(artifact_id: ArtifactId) -> Image:
        """Plot yearly event count against mean severity for a stored simulation.

        This is how you check that a requested copula or correlation actually
        produced the dependence you asked for; the realised correlation is printed
        on the chart. Requires a simulation run with keep_events=true.

        Returns: a PNG image.
        """
        artifact = STORE.get(artifact_id, kind="simulation")
        events = artifact.payload.get("events")
        if events is None or events.empty:
            raise ActsimToolError(
                f"Simulation {artifact_id!r} has no per-event data to correlate.",
                hint="Re-run actsim_simulate_aggregate with keep_events=true.",
            )

        from actsim import StochasticSimulator

        with captured():
            plotter = StochasticSimulator.__new__(StochasticSimulator)
            plotter._all_simulations_data = events.to_dict("records")
        with captured_figure() as images:
            plotter.plot_correlated_variables()
        if not images:
            raise ActsimToolError("Plot rendering produced no figure.", hint="The simulation may contain too few events to plot.")
        return Image(data=images[0], format="png")

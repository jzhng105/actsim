"""Distribution-fitting tools built on actsim.DistributionFitter."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import moments
from ..catalog import ALL_DISTRIBUTIONS, SEVERITY_DISTRIBUTIONS, normalize
from ..compat import Image, tool_decorator
from ..dataio import resolve_series
from ..errors import ActsimToolError
from ..params import (
    ArtifactId,
    Column,
    FilePath,
    Format,
    InlineValues,
    OptionalArtifactId,
    Seed,
)
from ..render import ResponseFormat, bullets, document, histogram, next_steps, respond, table
from ..runtime import captured, captured_figure
from ..settings import FIT_METRICS, SETTINGS
from ..store import STORE

# Columns of the ranking table, in the order actuaries read them.
_SUMMARY_COLUMNS = ("name", "params", "log_likelihood", "aic", "bic", "chisquare", "ks")

_METRIC_DIRECTION = {
    "aic": "lower is better",
    "bic": "lower is better (penalises parameters more than AIC)",
    "chisquare": "lower is better (binned goodness of fit)",
    "ks": "lower is better (max CDF gap, 0-1)",
    "log_likelihood": "higher is better (no penalty for extra parameters - prefer aic or bic to choose between models)",
}

# Every other metric is minimised. actsim's own select_best_fit takes the min of
# whatever metric it is given, which picks the WORST model for log-likelihood, so
# ranking and selection are done here instead of trusting fitter.best_fits.
_MAXIMISED_METRICS = frozenset({"log_likelihood"})


def _rank(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Order candidates best-first for the given metric."""
    return frame.sort_values(metric, ascending=metric not in _MAXIMISED_METRICS).reset_index(drop=True)


def _best_name(frame: pd.DataFrame, metric: str) -> str | None:
    """Name of the best candidate on a metric, ignoring degenerate fits.

    A failed optimisation can still land in the results with a log-likelihood of
    -inf and an AIC of +inf. Those rows are kept in the reported table so the
    failure is visible, but they must never be selected.
    """
    finite = frame[np.isfinite(frame[metric])]
    if finite.empty:
        return None
    return str(_rank(finite, metric).iloc[0]["name"])


def _validate_metrics(metrics: list[str] | None) -> list[str]:
    """Reject metric names DistributionFitter does not put on its result rows.

    ``select_best_fit`` indexes each result dict by metric name, so an unknown
    metric (the commonly attempted 'ks_statistic') raises a bare KeyError deep in
    the library. Catch it here with a message naming the valid set.
    """
    if not metrics:
        return list(FIT_METRICS)
    unknown = [m for m in metrics if m not in FIT_METRICS]
    if unknown:
        raise ActsimToolError(
            f"Unsupported fit metric(s): {', '.join(unknown)}.",
            hint=f"Valid metrics: {', '.join(FIT_METRICS)}. Note the KS statistic is named 'ks', not 'ks_statistic'.",
        )
    return list(metrics)


def _validate_distributions(names: list[str] | None) -> list[str] | None:
    if not names:
        return None
    return [normalize(n) for n in names]


def _fit_frame(fitter: Any) -> pd.DataFrame:
    """Ranking table of every candidate that fitted successfully."""
    rows = [
        {
            "name": r["name"],
            "params": tuple(round(float(p), 6) for p in r["params"]),
            "log_likelihood": float(r["log_likelihood"]),
            "aic": float(r["aic"]),
            "bic": float(r["bic"]),
            "chisquare": float(r["chisquare"]),
            "ks": float(r["ks"]),
        }
        for r in fitter.results
    ]
    return pd.DataFrame(rows, columns=list(_SUMMARY_COLUMNS))


def _selected(fitter: Any) -> dict[str, Any]:
    fit = fitter.selected_fit
    if fit is None:
        return {}
    return {
        "distribution": fit["name"],
        "params": [float(p) for p in fit["params"]],
        "aic": float(fit["aic"]),
        "bic": float(fit["bic"]),
        "ks": float(fit["ks"]),
        "chisquare": float(fit["chisquare"]),
        "log_likelihood": float(fit["log_likelihood"]),
    }


def _get_fitter(artifact_id: str):
    artifact = STORE.get(artifact_id, kind="fit")
    return artifact, artifact.payload


def register(server) -> None:
    @tool_decorator(server, name="actsim_fit_distributions", title="Fit Distributions")
    async def actsim_fit_distributions(
        values: InlineValues = None,
        file_path: FilePath = None,
        artifact_id: OptionalArtifactId = None,
        column: Column = None,
        distributions: Annotated[
            list[str] | None,
            Field(
                default=None,
                description="Candidates to try, e.g. ['lognormal','gamma','pareto']. Defaults to all severity distributions.",
                max_length=len(ALL_DISTRIBUTIONS),
            ),
        ] = None,
        metrics: Annotated[
            list[str] | None,
            Field(
                default=None,
                description=f"Goodness-of-fit metrics to compute. One of {list(FIT_METRICS)}; defaults to all.",
                max_length=len(FIT_METRICS),
            ),
        ] = None,
        rank_by: Annotated[
            Literal["aic", "bic", "chisquare", "ks", "log_likelihood"],
            Field(default="aic", description="Metric used to pick the selected distribution."),
        ] = "aic",
        remove_values: Annotated[
            list[float] | None,
            Field(default=None, description="Values to drop before fitting, e.g. [0] to exclude nil claims.", max_length=20),
        ] = None,
        lower: Annotated[float | None, Field(default=None, description="Drop observations below this value.")] = None,
        upper: Annotated[float | None, Field(default=None, description="Drop observations above this value.")] = None,
        q_low: Annotated[float | None, Field(default=None, description="Drop below this quantile, e.g. 0.01.", ge=0.0, lt=1.0)] = None,
        q_high: Annotated[float | None, Field(default=None, description="Drop above this quantile, e.g. 0.99.", gt=0.0, le=1.0)] = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Fit candidate distributions to a loss sample and rank them by goodness of fit.

        Optionally truncates the data first (drop nil claims, cap outliers, trim
        quantiles) - truncation is applied before fitting and is reported back.
        Every candidate gets log-likelihood, AIC, BIC, chi-square and KS.

        Returns: a fit artifact id, the ranked candidate table, the selected
        distribution with its fitted parameters, and any candidates that failed.

        Use actsim_sample_from_fit or actsim_simulate_aggregate with the selected
        parameters to carry the fit forward.
        """
        array, source = resolve_series(
            values=values, file_path=file_path, artifact_id=artifact_id, column=column
        )
        candidates = _validate_distributions(distributions) or list(SEVERITY_DISTRIBUTIONS)
        chosen_metrics = _validate_metrics(metrics)
        if rank_by not in chosen_metrics:
            chosen_metrics.append(rank_by)

        from actsim import DistributionFitter

        raw_count = int(array.size)
        with captured() as noise:
            fitter = DistributionFitter(array, distributions=candidates, metrics=chosen_metrics)
            if any(v is not None for v in (remove_values, lower, upper, q_low, q_high)):
                fitter.truncate_data(
                    remove_values=list(remove_values) if remove_values else None,
                    lower=lower,
                    upper=upper,
                    q_low=q_low,
                    q_high=q_high,
                )
            fitter.fit()

        frame = _fit_frame(fitter)
        if frame.empty:
            raise ActsimToolError(
                "No candidate distribution could be fitted to this data.",
                hint=(
                    "Check the data range against each distribution's support "
                    "(beta needs [0,1], poisson needs non-negative integers). "
                    f"Fitter output: {noise.getvalue().strip()[:400]}"
                ),
            )
        frame = _rank(frame, rank_by)
        failed = sorted(set(candidates) - set(frame["name"]))

        best = _best_name(frame, rank_by)
        if best is None:
            raise ActsimToolError(
                f"Every candidate produced a non-finite {rank_by}, so none can be selected.",
                hint=(
                    "The optimiser failed on all of them - check the data against each "
                    "distribution's support, or try a different candidate list."
                ),
            )
        with captured():
            fitter.select_distribution(best)

        fitted_count = int(np.asarray(fitter.data).size)
        artifact = STORE.put(
            "fit",
            f"fit of {source}",
            fitter,
            {
                "source": source,
                "observations": fitted_count,
                "candidates": len(frame),
                "selected": _selected(fitter).get("distribution"),
                "ranked_by": rank_by,
            },
        )

        payload = {
            "artifact_id": artifact.id,
            "source": source,
            "observations_in": raw_count,
            "observations_fitted": fitted_count,
            "ranked_by": rank_by,
            "metric_note": _METRIC_DIRECTION[rank_by],
            "selected": _selected(fitter),
            "candidates": frame,
            "failed_candidates": failed,
            "data_statistics": moments(np.asarray(fitter.data, dtype=float)),
        }
        selected = payload["selected"]
        markdown = document(
            f"Distribution fit `{artifact.id}`",
            [
                (
                    "Data",
                    f"{source} - {fitted_count:,} observations used"
                    + (f" ({raw_count - fitted_count:,} removed by truncation)" if fitted_count != raw_count else ""),
                ),
                (
                    "Selected",
                    bullets(selected) + f"\n\n_Ranked by {rank_by}: {_METRIC_DIRECTION[rank_by]}._",
                ),
                ("Candidates", table(frame, index=False, digits=4)),
                ("Failed to fit", ", ".join(failed) if failed else ""),
                ("", next_steps(
                    f"actsim_plot_fit(artifact_id='{artifact.id}') to see fitted densities against the data",
                    f"actsim_select_fit(artifact_id='{artifact.id}', distribution='...') to override the choice",
                    f"actsim_simulate_aggregate(sev_dist='{selected.get('distribution')}', sev_params={selected.get('params')}, ...)",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_get_fit_summary", title="Get Fit Summary")
    async def actsim_get_fit_summary(
        artifact_id: ArtifactId,
        rank_by: Annotated[
            Literal["aic", "bic", "chisquare", "ks", "log_likelihood"],
            Field(default="aic", description="Metric to sort the candidate table by."),
        ] = "aic",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Re-read a stored fit's candidate ranking under a different metric.

        Useful for checking whether AIC and KS agree on the best model - when they
        disagree, the tail fit and the body fit are telling different stories.

        Returns: the ranked candidate table and the currently selected distribution.
        """
        artifact, fitter = _get_fitter(artifact_id)
        frame = _rank(_fit_frame(fitter), rank_by)
        best_by = {
            metric: _best_name(frame, metric)
            for metric in FIT_METRICS
            if metric in frame.columns
        }
        payload = {
            "artifact_id": artifact_id,
            "source": artifact.meta.get("source"),
            "ranked_by": rank_by,
            "candidates": frame,
            "best_by_metric": best_by,
            "selected": _selected(fitter),
        }
        markdown = document(
            f"Fit `{artifact_id}` ranked by {rank_by}",
            [
                ("Candidates", table(frame, index=False, digits=4)),
                ("Best by metric", bullets(best_by)),
                ("Selected", bullets(_selected(fitter))),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(
        server,
        name="actsim_select_fit",
        title="Select Fitted Distribution",
        read_only=False,
        destructive=False,
    )
    async def actsim_select_fit(
        artifact_id: ArtifactId,
        distribution: Annotated[
            str, Field(description="Name of a candidate in the stored fit to make the selected one.", min_length=2)
        ],
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Override which fitted distribution a stored fit uses downstream.

        Actuarial judgement often overrules the metric - for instance choosing a
        pareto tail over a marginally better-scoring lognormal. This changes what
        actsim_sample_from_fit and actsim_fit_statistics operate on.

        Returns: the newly selected distribution with its fitted parameters.
        """
        artifact, fitter = _get_fitter(artifact_id)
        available = [r["name"] for r in fitter.results]
        name = normalize(distribution)
        if name not in available:
            raise ActsimToolError(
                f"{name!r} is not among the fitted candidates in {artifact_id!r}.",
                hint=f"Fitted candidates: {', '.join(available)}. Re-run actsim_fit_distributions including {name!r} to add it.",
            )
        with captured():
            fitter.select_distribution(name)
        artifact.meta["selected"] = name
        payload = {"artifact_id": artifact_id, "selected": _selected(fitter)}
        markdown = document(
            f"Selected `{name}` for fit `{artifact_id}`", [("", bullets(payload["selected"]))]
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_sample_from_fit", title="Sample From Fit")
    async def actsim_sample_from_fit(
        artifact_id: ArtifactId,
        size: Annotated[
            int, Field(default=1000, description="Number of draws.", ge=1, le=SETTINGS.max_inline_values)
        ] = 1000,
        zero_proportion: Annotated[
            float,
            Field(default=0.0, description="Share of draws forced to 0, for zero-inflated (nil claim) data.", ge=0.0, le=1.0),
        ] = 0.0,
        one_proportion: Annotated[
            float,
            Field(default=0.0, description="Share of draws forced to 1, for point mass at 1 (e.g. full-limit loss ratios).", ge=0.0, le=1.0),
        ] = 0.0,
        seed: Seed = SETTINGS.default_seed,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Draw a sample from a stored fit's selected distribution.

        Supports zero/one inflation via actsim's sample_mixed, which replaces the
        given share of draws with exact 0s and 1s - the standard treatment for
        books with a mass of nil claims.

        Returns: sample statistics, a shape histogram and a dataset artifact id.
        """
        if zero_proportion + one_proportion > 1.0:
            raise ActsimToolError(
                f"zero_proportion + one_proportion = {zero_proportion + one_proportion:g}, which exceeds 1.",
                hint="These are shares of the same sample, so together they must be at most 1.",
            )
        artifact, fitter = _get_fitter(artifact_id)
        if fitter.selected_fit is None:
            raise ActsimToolError(
                f"Fit {artifact_id!r} has no selected distribution.",
                hint="Call actsim_select_fit first.",
            )
        with captured():
            np.random.seed(seed)
            if zero_proportion or one_proportion:
                draws = np.asarray(
                    fitter.sample_mixed(zero_prop=zero_proportion, one_prop=one_proportion, size=size),
                    dtype=float,
                )
            else:
                draws = np.asarray(fitter.sample(size=size), dtype=float)

        selected = _selected(fitter)
        stored = STORE.put(
            "dataset",
            f"sample from {selected['distribution']}",
            pd.Series(draws.ravel(), name="sample"),
            {
                "source": f"fit {artifact_id} ({selected['distribution']})",
                "observations": int(draws.size),
                "seed": seed,
            },
        )
        payload = {
            "source_fit": artifact_id,
            "selected": selected,
            "zero_proportion": zero_proportion,
            "one_proportion": one_proportion,
            "statistics": moments(draws.ravel()),
            "artifact_id": stored.id,
        }
        markdown = document(
            f"{size:,} draws from `{selected['distribution']}` (fit `{artifact_id}`)",
            [
                ("Fitted parameters", bullets({"distribution": selected["distribution"], "params": selected["params"]})),
                ("Summary", bullets(payload["statistics"])),
                ("Shape", histogram(draws.ravel())),
                ("", f"Stored as `{stored.id}`."),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_fit_statistics", title="Compare Fit To Data")
    async def actsim_fit_statistics(
        artifact_id: ArtifactId,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Compare the empirical data against the selected fitted distribution.

        Runs actsim's calculate_statistics, which puts the observed mean, standard
        deviation and 5/25/50/75/95th percentiles beside the fitted model's, so a
        systematic bias in the body or the tail is visible at a glance.

        Returns: side-by-side data vs fitted statistics and the selected distribution.
        """
        artifact, fitter = _get_fitter(artifact_id)
        if fitter.selected_fit is None:
            raise ActsimToolError(
                f"Fit {artifact_id!r} has no selected distribution.",
                hint="Call actsim_select_fit first.",
            )
        with captured():
            # Populates fitter.statistics; the returned frame reshapes the same data.
            fitter.calculate_statistics()

        data_stats = fitter.statistics["data"]
        pred_stats = fitter.statistics["predicted"]
        comparison = pd.DataFrame(
            {
                "data": [data_stats["mean"], data_stats["std"], *np.asarray(data_stats["percentiles"], dtype=float)],
                "fitted": [pred_stats["mean"], pred_stats["std"], *np.asarray(pred_stats["percentiles"], dtype=float)],
            },
            index=["mean", "std", "p5", "p25", "p50", "p75", "p95"],
        )
        comparison["difference"] = comparison["fitted"] - comparison["data"]
        payload = {
            "artifact_id": artifact_id,
            "selected": _selected(fitter),
            "comparison": comparison,
        }
        markdown = document(
            f"Fit `{artifact_id}`: data vs fitted",
            [
                ("Selected", bullets(_selected(fitter))),
                ("Comparison", table(comparison, digits=4)),
                ("", "_Fitted percentiles come from actsim's density-grid approximation, so treat them as indicative rather than exact quantiles._"),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_plot_fit", title="Plot Fitted Distributions")
    async def actsim_plot_fit(
        artifact_id: ArtifactId,
        distributions: Annotated[
            list[str] | None,
            Field(default=None, description="Subset of fitted candidates to draw. Defaults to all.", max_length=len(ALL_DISTRIBUTIONS)),
        ] = None,
    ) -> Image:
        """Render fitted densities over a histogram of the data as a PNG image.

        Visual inspection catches tail misfits that AIC alone hides.

        Returns: a PNG image.
        """
        artifact, fitter = _get_fitter(artifact_id)
        available = [r["name"] for r in fitter.results]
        names = [normalize(d) for d in distributions] if distributions else available
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ActsimToolError(
                f"Not fitted in {artifact_id!r}: {', '.join(unknown)}.",
                hint=f"Fitted candidates: {', '.join(available)}.",
            )
        with captured_figure() as images:
            fitter.plot_predictions(distribution_names=names)
        if not images:
            raise ActsimToolError(
                "Plot rendering produced no figure.",
                hint="Check the fit artifact still holds successfully fitted candidates.",
            )
        return Image(data=images[0], format="png")

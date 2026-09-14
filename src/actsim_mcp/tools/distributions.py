"""Tools for exploring and sampling actsim's parametric distributions."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import moments
from ..catalog import catalog_payload, make_frozen, spec
from ..compat import tool_decorator
from ..errors import ActsimToolError
from ..params import DistributionName, DistributionParams, Format, Quantiles, Seed
from ..render import ResponseFormat, bullets, document, histogram, next_steps, respond, table
from ..runtime import captured
from ..settings import SETTINGS
from ..store import STORE

_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995)


def _safe(callable_, *args):
    """Evaluate a distribution method, returning NaN when it is undefined."""
    try:
        value = float(callable_(*args))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def register(server) -> None:
    @tool_decorator(server, name="actsim_list_distributions", title="List Distributions")
    async def actsim_list_distributions(
        role: Annotated[
            Literal["all", "severity", "frequency"],
            Field(default="all", description="Filter to severity (loss size) or frequency (claim count) distributions."),
        ] = "all",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """List every distribution actsim can fit, simulate or sample, with its parameters.

        Read this before calling any tool that takes a distribution name, because
        actsim uses actuarial parameterisations that differ from scipy's: lognormal
        takes (mu, sigma) on the LOG scale, gamma takes (alpha, theta) with theta a
        scale not a rate, and pareto/weibull follow Bahnemann.

        Returns: name, ordered parameter names, example parameters, support,
        discrete/continuous and a parameterisation note for each distribution.
        """
        items = catalog_payload(None if role == "all" else role)
        payload = {"distributions": items, "count": len(items)}
        frame = pd.DataFrame(
            [
                {
                    "distribution": d["name"],
                    "parameters": ", ".join(d["parameters"]),
                    "example": d["example_params"],
                    "kind": d["kind"],
                    "roles": "/".join(d["roles"]) or "-",
                    "support": d["support"],
                }
                for d in items
            ]
        )
        notes = "\n".join(f"- **{d['name']}**: {d['note']}" for d in items if d["note"])
        markdown = document(
            "actsim distributions",
            [("", table(frame, index=False)), ("Parameterisation notes", notes)],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_distribution_stats", title="Distribution Statistics")
    async def actsim_distribution_stats(
        distribution: DistributionName,
        params: DistributionParams,
        evaluate_at: Annotated[
            list[float] | None,
            Field(default=None, description="Optional points at which to report pdf/pmf and cdf.", max_length=50),
        ] = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Report the moments and percentiles of a parameterised distribution.

        Use this to sanity-check parameters before running a simulation - for
        example to confirm a lognormal(mu, sigma) really has the mean severity you
        expect, since mu is on the log scale.

        Returns: mean, std, variance, median, a percentile table, and pdf/cdf values
        at any requested points.
        """
        info = spec(distribution)
        frozen = make_frozen(distribution, params)
        with captured():
            stats = {
                "mean": _safe(frozen.mean),
                "std": _safe(frozen.std),
                "variance": _safe(frozen.var),
                "median": _safe(frozen.median),
            }
            percentiles = {f"{p:g}": _safe(frozen.ppf, p) for p in _PERCENTILES}
            points: list[dict[str, float]] = []
            for x in evaluate_at or []:
                density = _safe(frozen.pmf if info.discrete else frozen.pdf, x)
                points.append({"x": float(x), "density": density, "cdf": _safe(frozen.cdf, x)})

        payload = {
            "distribution": info.name,
            "signature": info.signature(),
            "params": list(params),
            "kind": "discrete" if info.discrete else "continuous",
            "moments": stats,
            "percentiles": percentiles,
            "evaluated_points": points,
        }
        pct_frame = pd.DataFrame({"value": percentiles}).rename_axis("quantile")
        markdown = document(
            f"{info.signature()} = {list(params)}",
            [
                ("Moments", bullets(stats)),
                ("Percentiles", table(pct_frame)),
                ("Evaluated points", table(pd.DataFrame(points), index=False) if points else ""),
                ("", f"_{info.note}_" if info.note else ""),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_distribution_quantile", title="Distribution Quantiles")
    async def actsim_distribution_quantile(
        distribution: DistributionName,
        params: DistributionParams,
        quantiles: Quantiles = None,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Invert a distribution: return the loss level at each requested quantile.

        This is the analytic counterpart to VaR on simulated data - use it when the
        distribution is known rather than simulated.

        Returns: one value per quantile, plus the implied return period in years.
        """
        from ..analytics import validate_quantiles

        info = spec(distribution)
        frozen = make_frozen(distribution, params)
        qs = validate_quantiles(quantiles, (0.5, 0.9, 0.95, 0.99, 0.995))
        with captured():
            rows = [
                {
                    "quantile": q,
                    "value": _safe(frozen.ppf, q),
                    "return_period_years": round(1.0 / (1.0 - q), 1),
                }
                for q in qs
            ]
        payload = {"distribution": info.name, "params": list(params), "quantiles": rows}
        markdown = document(
            f"Quantiles of {info.signature()} = {list(params)}",
            [("", table(pd.DataFrame(rows), index=False))],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_sample_distribution", title="Sample From Distribution")
    async def actsim_sample_distribution(
        distribution: DistributionName,
        params: DistributionParams,
        size: Annotated[
            int,
            Field(default=1000, description="Number of draws.", ge=1, le=SETTINGS.max_inline_values),
        ] = 1000,
        seed: Seed = SETTINGS.default_seed,
        store: Annotated[
            bool,
            Field(default=True, description="Store the draws as a dataset artifact for reuse."),
        ] = True,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Draw a random sample from a parameterised distribution.

        Returns summary statistics and a text histogram rather than the raw draws;
        set store=true (default) to keep the full sample as an artifact that other
        tools can consume.

        Returns: sample statistics, a shape histogram and the dataset artifact id.
        """
        info = spec(distribution)
        frozen = make_frozen(distribution, params)
        with captured():
            np.random.seed(seed)
            draws = np.asarray(frozen.np_rvs(size=size), dtype=float).ravel()

        stats = moments(draws)
        artifact_id = None
        if store:
            artifact = STORE.put(
                "dataset",
                f"{info.name} sample",
                pd.Series(draws, name=info.name),
                {"source": f"{info.signature()}={list(params)}", "observations": int(draws.size), "seed": seed},
            )
            artifact_id = artifact.id

        payload = {
            "distribution": info.name,
            "params": list(params),
            "size": int(draws.size),
            "seed": seed,
            "statistics": stats,
            "artifact_id": artifact_id,
        }
        markdown = document(
            f"{size:,} draws from {info.signature()} = {list(params)}",
            [
                ("Summary", bullets(stats)),
                ("Shape", histogram(draws)),
                ("", next_steps(
                    f"actsim_risk_metrics(artifact_id='{artifact_id}')" if artifact_id else "",
                    f"actsim_preview_artifact(artifact_id='{artifact_id}')" if artifact_id else "",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_mix_distributions", title="Mix Two Distributions")
    async def actsim_mix_distributions(
        first_distribution: DistributionName,
        first_params: DistributionParams,
        second_distribution: DistributionName,
        second_params: DistributionParams,
        first_weight: Annotated[
            float, Field(default=0.5, description="Mixing weight on the first component.", gt=0.0, lt=1.0)
        ] = 0.5,
        size: Annotated[
            int, Field(default=1000, description="Total number of draws.", ge=2, le=SETTINGS.max_inline_values)
        ] = 1000,
        seed: Seed = SETTINGS.default_seed,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Sample from a two-component mixture, e.g. attritional plus large losses.

        Components are drawn in proportion to first_weight (the second component
        takes 1 - first_weight) and shuffled together.

        Returns: mixture statistics, per-component statistics and a dataset artifact id.
        """
        first_frozen = make_frozen(first_distribution, first_params)
        second_frozen = make_frozen(second_distribution, second_params)
        n_first = int(round(size * first_weight))
        n_second = size - n_first
        if n_first == 0 or n_second == 0:
            raise ActsimToolError(
                f"first_weight={first_weight} yields {n_first}/{n_second} draws, leaving one component empty.",
                hint="Increase size, or move first_weight closer to 0.5.",
            )
        with captured():
            np.random.seed(seed)
            first = np.asarray(first_frozen.np_rvs(size=n_first), dtype=float)
            second = np.asarray(second_frozen.np_rvs(size=n_second), dtype=float)
        mixed = np.concatenate([first.ravel(), second.ravel()])
        np.random.default_rng(seed).shuffle(mixed)

        artifact = STORE.put(
            "dataset",
            f"{first_distribution}/{second_distribution} mixture",
            pd.Series(mixed, name="mixture"),
            {
                "source": f"{first_weight:g}*{first_distribution} + {1 - first_weight:g}*{second_distribution}",
                "observations": int(mixed.size),
                "seed": seed,
            },
        )
        payload = {
            "components": [
                {"distribution": spec(first_distribution).name, "params": list(first_params), "weight": first_weight, "draws": n_first},
                {"distribution": spec(second_distribution).name, "params": list(second_params), "weight": round(1 - first_weight, 6), "draws": n_second},
            ],
            "statistics": moments(mixed),
            "artifact_id": artifact.id,
        }
        markdown = document(
            f"Mixture sample `{artifact.id}`",
            [
                ("Components", table(pd.DataFrame(payload["components"]), index=False)),
                ("Summary", bullets(payload["statistics"])),
                ("Shape", histogram(mixed)),
            ],
        )
        return respond(response_format, payload, markdown)

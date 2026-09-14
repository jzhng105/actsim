"""Risk measures over simulated or empirical loss data.

These are computed here rather than delegated to
``StochasticSimulator.analyze_results`` because that method raises
``UnboundLocalError`` whenever the simulator was built with ``keep_all=False``
(it reads ``quantile_values``/``tvar_values`` that are only bound on the
``keep_all=True`` branch) and its OEP index can run off the end of the array at
q=1.0. The definitions below match the library's intent and work for both modes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .errors import ActsimToolError


def validate_quantiles(quantiles: Sequence[float] | None, default: Sequence[float]) -> list[float]:
    """Coerce and range-check a quantile list."""
    values = list(quantiles) if quantiles else list(default)
    bad = [q for q in values if not 0.0 < float(q) < 1.0]
    if bad:
        raise ActsimToolError(
            f"Quantiles must be strictly between 0 and 1; got {bad}.",
            hint="Use fractions such as 0.95, not percentages such as 95.",
        )
    return sorted(float(q) for q in values)


def var(values: np.ndarray, q: float) -> float:
    """Value-at-Risk: the q-quantile of the loss distribution."""
    return float(np.quantile(values, q))


def tvar(values: np.ndarray, q: float) -> float:
    """Tail VaR / Expected Shortfall: mean of losses above the q-quantile.

    Falls back to the quantile itself when no observation exceeds it, which
    happens for high q on small or heavily tied samples.
    """
    cutoff = np.quantile(values, q)
    tail = values[values > cutoff]
    return float(tail.mean()) if tail.size else float(cutoff)


def moments(values: np.ndarray) -> dict[str, float]:
    """Summary moments and shape statistics for a loss sample."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ActsimToolError("No finite values to summarise.", hint="Check the input data.")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    centered = array - mean
    denom = array.size * (std**3) if std > 0 else 0.0
    skew = float((centered**3).sum() / denom) if denom else 0.0
    denom4 = array.size * (std**4) if std > 0 else 0.0
    kurt = float((centered**4).sum() / denom4 - 3.0) if denom4 else 0.0
    return {
        "count": int(array.size),
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "coefficient_of_variation": float(std / mean) if mean else float("nan"),
        "total": float(array.sum()),
        "zero_share": float((array == 0).mean()),
    }


def largest_event_per_year(aggregate: np.ndarray, events: pd.DataFrame) -> np.ndarray:
    """Largest single event in each simulated year, including years with none.

    The event table only has rows for years that produced a claim, so grouping it
    alone silently drops every claim-free year. For a low-frequency book that is
    most of the sample and it inflates OEP badly - at poisson(0.5) over 20,000
    trials, OEP(90%) came out near double its true value. The aggregate vector has
    exactly one entry per simulated year, so its length is the true year count.
    """
    years = np.asarray(aggregate, dtype=float).size
    per_year = np.zeros(years, dtype=float)
    grouped = events.groupby("year")["amount"].max()
    index = grouped.index.to_numpy(dtype=int) - 1  # actsim numbers years from 1
    inside = (index >= 0) & (index < years)
    per_year[index[inside]] = grouped.to_numpy(dtype=float)[inside]
    return per_year


def risk_table(
    aggregate: np.ndarray,
    quantiles: Sequence[float],
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the VaR / TVaR / AEP / OEP table at the requested quantiles.

    ``aggregate`` holds one total loss per simulated year, so VaR and AEP are the
    same statistic viewed two ways (AEP is the actuarial name for the aggregate
    exceedance curve) and both are reported for continuity with the library's
    output. OEP - the largest single event in a year - needs per-event data and is
    filled with NaN when the simulation did not keep it.
    """
    rows: dict[str, list[float]] = {"VaR": [], "TVaR": [], "AEP": []}
    for q in quantiles:
        rows["VaR"].append(var(aggregate, q))
        rows["TVaR"].append(tvar(aggregate, q))
        rows["AEP"].append(var(aggregate, q))

    if events is not None and not events.empty and "amount" in events.columns:
        per_year = largest_event_per_year(aggregate, events)
        rows["OEP"] = [var(per_year, q) for q in quantiles]
        rows["OEP_TVaR"] = [tvar(per_year, q) for q in quantiles]

    table = pd.DataFrame(rows, index=[f"{q:g}" for q in quantiles])
    table.index.name = "quantile"
    return table


def return_periods(aggregate: np.ndarray, periods: Sequence[float]) -> dict[str, float]:
    """Loss level associated with each return period in years (e.g. 1-in-200)."""
    out: dict[str, float] = {}
    for period in periods:
        if period <= 1:
            continue
        out[f"1-in-{int(period)}"] = var(aggregate, 1.0 - 1.0 / float(period))
    return out


def apply_layer(
    events: pd.DataFrame,
    *,
    per_occurrence_deductible: float = 0.0,
    per_occurrence_limit: float | None = None,
    aggregate_deductible: float = 0.0,
    aggregate_limit: float | None = None,
    years: int | None = None,
) -> pd.DataFrame:
    """Apply per-occurrence then aggregate deductible/limit to event-level losses.

    Mirrors ``StochasticSimulator.apply_deductible_and_limit`` but keeps the
    ground-up, ceded-to-layer and retained amounts side by side, and reinstates
    years in which every event fell below the deductible (a plain groupby drops
    them, which biases every downstream mean downwards).
    """
    occ_limit = float("inf") if per_occurrence_limit is None else float(per_occurrence_limit)
    agg_limit_value = float("inf") if aggregate_limit is None else float(aggregate_limit)
    occ_ded = float(per_occurrence_deductible or 0.0)
    agg_ded = float(aggregate_deductible or 0.0)

    if occ_limit < 0 or agg_limit_value < 0 or occ_ded < 0 or agg_ded < 0:
        raise ActsimToolError(
            "Deductibles and limits must be non-negative.",
            hint="Use null for 'unlimited' rather than a negative number.",
        )

    frame = events.copy()
    frame["excess_of_deductible"] = (frame["amount"] - occ_ded).clip(lower=0.0)
    frame["in_layer"] = frame["excess_of_deductible"].clip(upper=occ_limit)

    grouped = frame.groupby("year", as_index=False).agg(
        ground_up=("amount", "sum"),
        events=("amount", "size"),
        layer_before_aggregate=("in_layer", "sum"),
    )

    if years:
        # Years with zero events never appear in the event table; add them back so
        # aggregate statistics are computed over the full simulated period.
        full = pd.DataFrame({"year": np.arange(1, int(years) + 1)})
        grouped = full.merge(grouped, on="year", how="left").fillna(
            {"ground_up": 0.0, "events": 0, "layer_before_aggregate": 0.0}
        )

    grouped["ceded"] = (grouped["layer_before_aggregate"] - agg_ded).clip(lower=0.0).clip(
        upper=agg_limit_value
    )
    grouped["retained"] = grouped["ground_up"] - grouped["ceded"]
    return grouped

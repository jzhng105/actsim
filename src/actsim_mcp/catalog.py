"""Catalog of the distributions actsim can fit, simulate and sample from.

Parameter names and defaults are read out of ``actstats`` itself rather than
hard-coded, so the catalog cannot drift from the installed library. Agents get
the actuarial parameter names (e.g. lognormal is ``(mu, sigma)`` on the log
scale, pareto is ``(alpha, beta)`` Bahnemann-style) which is the single most
common source of silently wrong results when calling these tools.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from actstats import actuarial as act

from .errors import unknown_value

# actsim's own registries key negative binomial with a space; actstats uses an
# underscore. Tools accept either spelling and normalise here.
_ALIASES = {
    "negative_binomial": "negative binomial",
    "negbinom": "negative binomial",
    "nbinom": "negative binomial",
    "log-normal": "lognormal",
    "log normal": "lognormal",
    "gaussian": "normal",
    "expon": "exponential",
}

# Names as actsim's DistributionFitter / StochasticSimulator registries spell them.
SEVERITY_DISTRIBUTIONS = (
    "uniform",
    "normal",
    "logistic",
    "exponential",
    "gamma",
    "beta",
    "pareto",
    "weibull",
    "lognormal",
)
FREQUENCY_DISTRIBUTIONS = ("poisson", "negative binomial")
ALL_DISTRIBUTIONS = tuple(sorted({*SEVERITY_DISTRIBUTIONS, *FREQUENCY_DISTRIBUTIONS}))

DISCRETE_DISTRIBUTIONS = frozenset({"poisson", "negative binomial"})

_SUPPORT = {
    "uniform": "[a, b]",
    "normal": "(-inf, inf)",
    "logistic": "(-inf, inf)",
    "exponential": "[0, inf)",
    "gamma": "[0, inf)",
    "beta": "[0, 1]",
    "pareto": "[0, inf)",
    "weibull": "[0, inf)",
    "lognormal": "(0, inf)",
    "poisson": "{0, 1, 2, ...}",
    "negative binomial": "{0, 1, 2, ...}",
}

_NOTES = {
    "lognormal": "mu and sigma are the mean and sd of the underlying NORMAL (log) scale, not of the losses.",
    "pareto": "Bahnemann parameterisation: alpha is the shape, beta the scale; maps to scipy lomax (unshifted).",
    "weibull": "Bahnemann parameterisation: delta is the shape, beta the scale.",
    "gamma": "alpha is the shape, theta the SCALE (not the rate).",
    "exponential": "beta is the mean (scale), not the rate.",
    "uniform": "a and b are the lower and upper bounds.",
    "negative binomial": "r is the number of failures, p the success probability.",
}


@dataclass(frozen=True)
class DistributionSpec:
    """Everything an agent needs to call a distribution correctly."""

    name: str
    parameters: tuple[str, ...]
    defaults: tuple[Any, ...]
    discrete: bool
    roles: tuple[str, ...]
    support: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "example_params": list(self.defaults),
            "kind": "discrete" if self.discrete else "continuous",
            "roles": list(self.roles),
            "support": self.support,
            "note": self.note,
        }

    def signature(self) -> str:
        return f"{self.name}({', '.join(self.parameters)})"


def _actstats_name(name: str) -> str:
    return "negative_binomial" if name == "negative binomial" else name


def _introspect(name: str) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """Pull actuarial parameter names/defaults from the actstats converter."""
    frozen = getattr(act, _actstats_name(name))
    signature = inspect.signature(frozen.to_numpy)
    names = tuple(signature.parameters)
    defaults = tuple(
        p.default if p.default is not inspect.Parameter.empty else 1.0
        for p in signature.parameters.values()
    )
    return names, defaults


def _build() -> dict[str, DistributionSpec]:
    catalog: dict[str, DistributionSpec] = {}
    for name in ALL_DISTRIBUTIONS:
        names, defaults = _introspect(name)
        roles = tuple(
            role
            for role, members in (
                ("severity", SEVERITY_DISTRIBUTIONS),
                ("frequency", FREQUENCY_DISTRIBUTIONS),
            )
            if name in members
        )
        catalog[name] = DistributionSpec(
            name=name,
            parameters=names,
            defaults=defaults,
            discrete=name in DISCRETE_DISTRIBUTIONS,
            roles=roles,
            support=_SUPPORT.get(name, "unspecified"),
            note=_NOTES.get(name, ""),
        )
    return catalog


CATALOG: dict[str, DistributionSpec] = _build()


def normalize(name: str) -> str:
    """Map a user-supplied distribution name onto actsim's registry spelling."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in CATALOG:
        raise unknown_value("distribution", name, ALL_DISTRIBUTIONS)
    return key


def spec(name: str) -> DistributionSpec:
    """Look up a distribution, accepting aliases."""
    return CATALOG[normalize(name)]


def _param_error(info: DistributionSpec, values: tuple[float, ...]):
    from .errors import ActsimToolError

    return ActsimToolError(
        f"{info.name} takes {len(info.parameters)} parameter(s) but {len(values)} were given.",
        hint=f"Expected {info.signature()} - for example {info.name} params={list(info.defaults)}. {info.note}".strip(),
    )


def validate_params(name: str, params: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Validate parameter arity for a distribution and return it as a tuple."""
    info = spec(name)
    values = tuple(float(v) for v in params)
    if len(values) != len(info.parameters):
        raise _param_error(info, values)
    return values


def make_frozen(name: str, params: list[float] | tuple[float, ...]) -> Any:
    """Validate parameters and return the corresponding frozen actstats object."""
    values = validate_params(name, params)
    return getattr(act, _actstats_name(normalize(name)))(*values)


def catalog_payload(role: str | None = None) -> list[dict[str, Any]]:
    """Serialise the catalog, optionally filtered to 'severity' or 'frequency'."""
    items = CATALOG.values()
    if role:
        items = [item for item in items if role in item.roles]
    return [item.as_dict() for item in sorted(items, key=lambda i: i.name)]

"""Environment-driven settings and guardrails for the actsim MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "ACTSIM_MCP_"


def _env_int(suffix: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(ENV_PREFIX + suffix)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_path(suffix: str, default: Path) -> Path:
    raw = os.environ.get(ENV_PREFIX + suffix)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    """Runtime limits. Every field is overridable via an ``ACTSIM_MCP_*`` variable.

    The workspace root is the only directory the server will read data files from
    or write exports to; it defaults to the process working directory so a client
    launched inside a project can use relative paths naturally.
    """

    workspace: Path
    max_simulations: int
    max_inline_values: int
    max_preview_rows: int
    max_policies: int
    max_artifacts: int
    max_artifact_cells: int
    default_simulations: int
    default_seed: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            workspace=_env_path("WORKSPACE", Path.cwd().resolve()),
            max_simulations=_env_int("MAX_SIMULATIONS", 1_000_000),
            max_inline_values=_env_int("MAX_INLINE_VALUES", 100_000),
            max_preview_rows=_env_int("MAX_PREVIEW_ROWS", 200),
            max_policies=_env_int("MAX_POLICIES", 100_000),
            max_artifacts=_env_int("MAX_ARTIFACTS", 64),
            max_artifact_cells=_env_int("MAX_ARTIFACT_CELLS", 25_000_000),
            default_simulations=_env_int("DEFAULT_SIMULATIONS", 10_000),
            default_seed=_env_int("DEFAULT_SEED", 1, minimum=0),
        )


SETTINGS = Settings.from_env()

# Quantiles reported by default across simulation and risk-metric tools. Chosen to
# span working (0.5-0.9) and capital (0.99-0.995) views of a loss distribution.
DEFAULT_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99, 0.995)

# Metric names DistributionFitter puts on every fit result. 'ks_statistic' is NOT
# one of them - passing it raises KeyError inside select_best_fit - so tools
# validate against this tuple before calling the library.
FIT_METRICS: tuple[str, ...] = ("aic", "bic", "log_likelihood", "chisquare", "ks")

COPULA_TYPES: tuple[str, ...] = ("gaussian", "frank", "gumbel", "clayton")

"""Reusable Annotated parameter types shared across tools.

Declaring these once keeps descriptions, bounds and defaults identical in every
tool schema, which is what lets an agent carry a value from one tool to the next
without re-reading documentation.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .render import ResponseFormat
from .settings import SETTINGS

Format = Annotated[
    ResponseFormat,
    Field(description="'markdown' for a compact readable summary, 'json' for machine-readable output."),
]

ArtifactId = Annotated[
    str,
    Field(
        description="Artifact handle returned by an earlier tool, e.g. 'sim_7f3a1c2b'.",
        min_length=3,
        max_length=64,
    ),
]

OptionalArtifactId = Annotated[
    str | None,
    Field(
        default=None,
        description="Artifact handle holding the data, e.g. 'data_1a2b3c4d'. Mutually exclusive with values/file_path.",
    ),
]

InlineValues = Annotated[
    list[float] | None,
    Field(
        default=None,
        description=f"Inline numeric sample (max {SETTINGS.max_inline_values} values). Mutually exclusive with file_path/artifact_id.",
    ),
]

FilePath = Annotated[
    str | None,
    Field(
        default=None,
        description="CSV/JSON/TSV file inside the server workspace. Mutually exclusive with values/artifact_id.",
    ),
]

Column = Annotated[
    str | None,
    Field(default=None, description="Column to read when the source has more than one column."),
]

DistributionName = Annotated[
    str,
    Field(
        description="Distribution name as listed by actsim_list_distributions, e.g. 'lognormal', 'poisson', 'negative binomial'.",
        min_length=2,
        max_length=40,
    ),
]

DistributionParams = Annotated[
    list[float],
    Field(
        description="Actuarial parameters in catalog order, e.g. lognormal [mu, sigma] on the LOG scale, gamma [alpha, theta].",
        min_length=1,
        max_length=4,
    ),
]

Quantiles = Annotated[
    list[float] | None,
    Field(
        default=None,
        description="Quantiles as fractions strictly between 0 and 1, e.g. [0.9, 0.99]. Defaults to 0.5/0.75/0.9/0.95/0.99/0.995.",
        max_length=20,
    ),
]

NumSimulations = Annotated[
    int,
    Field(
        default=SETTINGS.default_simulations,
        description="Number of simulated years/trials.",
        ge=10,
        le=SETTINGS.max_simulations,
    ),
]

Seed = Annotated[
    int,
    Field(default=SETTINGS.default_seed, description="Random seed for reproducibility.", ge=0),
]

Limit = Annotated[
    int,
    Field(default=20, description="Maximum rows to return.", ge=1, le=SETTINGS.max_preview_rows),
]

Offset = Annotated[
    int, Field(default=0, description="Rows to skip, for pagination.", ge=0)
]


def paginate(total: int, offset: int, count: int) -> dict[str, object]:
    """Standard pagination envelope returned by every listing tool."""
    has_more = offset + count < total
    return {
        "total": total,
        "count": count,
        "offset": offset,
        "has_more": has_more,
        "next_offset": offset + count if has_more else None,
    }

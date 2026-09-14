"""actsim MCP server: assembly point for tools, resources and prompts."""

from __future__ import annotations

# Import order matters: runtime pins matplotlib to a headless backend and moves
# logging off stdout before any actsim module is imported.
from . import runtime  # noqa: F401  (imported for its import-time side effects)
from .compat import make_server
from .settings import SETTINGS
from .version import SERVER_VERSION

INSTRUCTIONS = f"""\
actsim exposes actuarial risk modelling: distribution fitting, frequency/severity
Monte Carlo simulation, correlated multi-line portfolios, policy-level claim
simulation and loss development triangles.

How to work with this server:

* Large results are stored server-side and referenced by an artifact id such as
  `sim_7f3a1c2b`. Producing tools return a summary plus the id; pass that id to
  downstream tools instead of re-sending data. `actsim_list_artifacts` recovers
  ids; artifacts do not survive a server restart.
* Distributions use ACTUARIAL parameterisations, not scipy's. lognormal takes
  (mu, sigma) on the log scale, gamma takes (alpha, theta) with theta a scale,
  and pareto/weibull follow Bahnemann. Call `actsim_list_distributions` before
  choosing parameters.
* `actsim_simulate_aggregate` needs `keep_events=true` for per-occurrence layer
  treatment and OEP; without it only aggregate measures are available.
* File paths are resolved inside the server workspace ({SETTINGS.workspace}) and
  paths outside it are refused.
* Every tool takes `response_format`: 'markdown' (default, compact) or 'json'.

Suggested entry points: `actsim_server_info`, the `actsim://workflows` resource,
or one of the prompts (price_excess_layer, portfolio_capital_review,
reserving_triangle_build).
"""


def build_server():
    """Create and fully configure the MCP server instance."""
    runtime.configure_logging()
    server = make_server("actsim_mcp", instructions=INSTRUCTIONS, version=SERVER_VERSION)

    from . import prompts, resources
    from .tools import register_all

    register_all(server)
    resources.register(server)
    prompts.register(server)
    return server


mcp = build_server()

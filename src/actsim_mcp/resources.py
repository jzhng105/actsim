"""MCP resources: reference material an agent can read without spending a tool call."""

from __future__ import annotations

import json
from typing import Any

from .catalog import catalog_payload
from .compat import resource_decorator
from .render import jsonable
from .settings import COPULA_TYPES, DEFAULT_QUANTILES, FIT_METRICS, SETTINGS
from .store import STORE
from .version import SERVER_VERSION

WORKFLOWS: dict[str, Any] = {
    "price_an_excess_layer": {
        "goal": "Price an excess-of-loss layer from historical claim amounts.",
        "steps": [
            "actsim_load_dataset(file_path='claims.csv', column='loss')",
            "actsim_fit_distributions(artifact_id=<dataset>, rank_by='aic')",
            "actsim_simulate_aggregate(freq_dist='poisson', freq_params=[<claims per year>], sev_dist=<selected>, sev_params=<fitted>, keep_events=true)",
            "actsim_apply_layer(artifact_id=<simulation>, per_occurrence_deductible=<attachment>, per_occurrence_limit=<limit>)",
            "actsim_generate_risk_report(artifact_ids=[<fit>, <simulation>, <layered>])",
        ],
    },
    "capital_from_correlated_lines": {
        "goal": "Estimate portfolio capital across correlated lines of business.",
        "steps": [
            "actsim_simulate_correlated_portfolio(lines=[...], correlation_matrix=[[1,0.4],[0.4,1]])",
            "actsim_risk_metrics(artifact_id=<portfolio>, return_period_years=[200])",
            "actsim_stress_test(artifact_id=<portfolio>, scenarios=[{'name':'inflation','severity_multiplier':1.1}])",
        ],
    },
    "reserving_triangle_from_a_policy_book": {
        "goal": "Build a loss development triangle from a book of policies.",
        "steps": [
            "actsim_build_policy_book(file_path='policies.csv')",
            "actsim_simulate_claims(artifact_id=<book>)",
            "actsim_assign_claim_dates(artifact_id=<claims>, alpha=0.3)",
            "actsim_simulate_claim_development(artifact_id=<dated claims>, base_ldfs={'12':2.5,'24':1.4,'36':1.1,'48':1.0})",
            "actsim_build_loss_triangle(artifact_id=<development>, basis='cumulative')",
        ],
    },
    "check_frequency_severity_dependence": {
        "goal": "Quantify how a frequency/severity copula moves the tail.",
        "steps": [
            "actsim_simulate_aggregate(..., keep_events=true)  # independent base",
            "actsim_simulate_aggregate(..., copula='gumbel', theta=1.8, keep_events=true)",
            "actsim_compare_simulations(artifact_ids=[<base>, <copula>])",
            "actsim_plot_frequency_severity(artifact_id=<copula>)",
        ],
    },
}

EXAMPLES: dict[str, Any] = {
    "loss_sample": [
        4200, 1150, 38000, 780, 2600, 9100, 530, 15400, 3300, 1900,
        620, 47000, 2100, 8800, 1250, 350, 26000, 5400, 970, 3100,
    ],
    "policy_book_csv": (
        "policy_id,freq_dist,freq_params,sev_dist,sev_params,start_date,end_date\n"
        "1,poisson,(0.8),lognormal,\"(9.0, 0.5)\",2024-01-01,2024-12-31\n"
        "2,poisson,(1.4),lognormal,\"(9.4, 0.7)\",2024-01-01,2024-12-31\n"
    ),
    "correlated_lines": [
        {"dist_name": "Property", "dist_type": "lognormal", "dist_param": [12.0, 1.2]},
        {"dist_name": "Liability", "dist_type": "gamma", "dist_param": [2.0, 90000.0]},
        {"dist_name": "Motor", "dist_type": "lognormal", "dist_param": [11.5, 0.8]},
    ],
    "correlation_matrix": [[1.0, 0.35, 0.2], [0.35, 1.0, 0.5], [0.2, 0.5, 1.0]],
    "base_ldfs": {"12": 2.5, "24": 1.45, "36": 1.15, "48": 1.05, "60": 1.0},
}


def register(server) -> None:
    @resource_decorator(
        server,
        "actsim://distributions",
        name="Supported distributions",
        description="Every distribution actsim can fit or simulate, with its actuarial parameter names, support and parameterisation notes.",
    )
    async def distributions() -> str:
        return json.dumps(
            {
                "distributions": catalog_payload(),
                "note": "actsim uses actuarial parameterisations; lognormal mu/sigma are on the log scale.",
            },
            indent=2,
        )

    @resource_decorator(
        server,
        "actsim://config",
        name="actsim package configuration",
        description="The distributions and metrics configured in the installed actsim package's config.yaml.",
    )
    async def config() -> str:
        from actsim import load_config

        from .runtime import captured

        with captured():
            loaded = load_config()
        return json.dumps(
            {
                "config_path": getattr(loaded, "_file_path", None),
                "distributions": loaded.distributions,
                "metrics": loaded.metrics,
                "server_fit_metrics": list(FIT_METRICS),
            },
            indent=2,
            default=str,
        )

    @resource_decorator(
        server,
        "actsim://capabilities",
        name="Server capabilities and limits",
        description="Server version, workspace root, simulation and artifact limits, supported copulas, default quantiles.",
    )
    async def capabilities() -> str:
        return json.dumps(
            {
                "server_version": SERVER_VERSION,
                "workspace": str(SETTINGS.workspace),
                "limits": {
                    "default_simulations": SETTINGS.default_simulations,
                    "max_simulations": SETTINGS.max_simulations,
                    "max_inline_values": SETTINGS.max_inline_values,
                    "max_policies": SETTINGS.max_policies,
                    "max_artifacts": SETTINGS.max_artifacts,
                    "max_preview_rows": SETTINGS.max_preview_rows,
                },
                "copulas": list(COPULA_TYPES),
                "fit_metrics": list(FIT_METRICS),
                "default_quantiles": list(DEFAULT_QUANTILES),
            },
            indent=2,
        )

    @resource_decorator(
        server,
        "actsim://workflows",
        name="Worked workflows",
        description="Tool-call sequences for the common actuarial tasks this server supports.",
    )
    async def workflows() -> str:
        return json.dumps(WORKFLOWS, indent=2)

    @resource_decorator(
        server,
        "actsim://examples",
        name="Example inputs",
        description="Ready-to-paste example data: a loss sample, a policy CSV, correlated line definitions and a set of LDFs.",
    )
    async def examples() -> str:
        return json.dumps(EXAMPLES, indent=2)

    @resource_decorator(
        server,
        "actsim://artifacts",
        name="Stored artifacts",
        description="Live index of the artifacts currently held in server memory.",
    )
    async def artifacts() -> str:
        return json.dumps(
            {"count": len(STORE), "artifacts": [jsonable(a.summary()) for a in STORE]}, indent=2
        )

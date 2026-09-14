"""End-to-end exercise of every actsim MCP tool through the SDK's dispatcher.

Each test drives ``server.call_tool`` exactly as a client would, so schema
validation, argument coercion and the error path are all covered.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

WORKSPACE = tempfile.mkdtemp(prefix="actsim-mcp-test-")
os.environ["ACTSIM_MCP_WORKSPACE"] = WORKSPACE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actsim_mcp.server import build_server  # noqa: E402

SERVER = build_server()


def call(name: str, arguments: dict | None = None):
    """Invoke a tool and return (text, is_error, first_content_block).

    The SDK reports a deliberate tool failure by raising, and the server kernel
    turns that into an is_error result for the client; this mirrors that step so
    tests can assert on the message an agent would actually see.
    """
    try:
        result = asyncio.run(SERVER.call_tool(name, arguments or {}))
    except Exception as exc:  # ToolError and friends
        return str(exc), True, None
    is_error = bool(getattr(result, "is_error", False))
    content = getattr(result, "content", result)
    first = content[0] if content else None
    text = getattr(first, "text", None)
    if text is None:
        text = getattr(first, "data", "") and "<binary>"
    return text, is_error, first


def mime_of(content) -> str:
    """ImageContent spells the field mimeType on SDK 1.x and mime_type on 2.x."""
    return getattr(content, "mime_type", None) or getattr(content, "mimeType", "")


def read_only_hint(tool):
    """ToolAnnotations is camelCase on SDK 1.x and snake_case on 2.x."""
    annotations = tool.annotations
    hint = getattr(annotations, "read_only_hint", None)
    return getattr(annotations, "readOnlyHint", None) if hint is None else hint


def ok(name: str, arguments: dict | None = None) -> str:
    text, is_error, _ = call(name, arguments)
    assert not is_error, f"{name} failed: {text}"
    return text


def as_json(name: str, arguments: dict) -> dict:
    return json.loads(ok(name, {**arguments, "response_format": "json"}))


def fails(name: str, arguments: dict) -> str:
    text, is_error, _ = call(name, arguments)
    assert is_error, f"{name} unexpectedly succeeded: {text}"
    return text


@pytest.fixture(scope="module")
def workspace() -> Path:
    root = Path(WORKSPACE)
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    pd.DataFrame({"loss": rng.lognormal(9.0, 0.8, 600)}).to_csv(root / "losses.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_id": i,
                "freq_dist": "poisson",
                "freq_params": "(1.2)",
                "sev_dist": "lognormal",
                "sev_params": "(9.0, 0.6)",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
            for i in range(1, 26)
        ]
    ).to_csv(root / "policies.csv", index=False)
    return root


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_every_tool_is_annotated_and_documented():
    tools = asyncio.run(SERVER.list_tools())
    assert len(tools) >= 30
    for tool in tools:
        assert tool.name.startswith("actsim_"), tool.name
        assert tool.description, f"{tool.name} has no description"
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert read_only_hint(tool) is not None, tool.name


def test_resources_and_prompts_registered():
    assert len(asyncio.run(SERVER.list_resources())) == 6
    assert len(asyncio.run(SERVER.list_prompts())) == 5


# --------------------------------------------------------------------------- #
# Reference and data tools
# --------------------------------------------------------------------------- #

def test_server_info_and_distribution_catalog():
    info = as_json("actsim_server_info", {})
    assert info["workspace"] == WORKSPACE

    catalog = as_json("actsim_list_distributions", {"role": "severity"})
    names = {d["name"] for d in catalog["distributions"]}
    assert {"lognormal", "gamma", "pareto"} <= names
    lognormal = next(d for d in catalog["distributions"] if d["name"] == "lognormal")
    assert lognormal["parameters"] == ["mu", "sigma"]


def test_distribution_stats_and_quantiles():
    stats = as_json("actsim_distribution_stats", {"distribution": "lognormal", "params": [9.0, 0.8]})
    assert stats["moments"]["mean"] == pytest.approx(11158.7, rel=1e-3)

    quantiles = as_json(
        "actsim_distribution_quantile",
        {"distribution": "poisson", "params": [3.0], "quantiles": [0.5, 0.99]},
    )
    assert quantiles["quantiles"][0]["value"] == 3.0


def test_unknown_distribution_lists_valid_names():
    message = fails("actsim_distribution_stats", {"distribution": "loggnormal", "params": [1, 1]})
    assert "lognormal" in message
    assert "Valid distribution" in message


def test_wrong_parameter_count_names_the_signature():
    message = fails("actsim_distribution_stats", {"distribution": "lognormal", "params": [1.0]})
    assert "lognormal(mu, sigma)" in message


def test_inspect_file_and_load_dataset(workspace):
    preview = as_json("actsim_inspect_file", {"file_path": "losses.csv"})
    assert preview["rows"] == 600

    dataset = as_json("actsim_load_dataset", {"file_path": "losses.csv", "column": "loss"})
    assert dataset["statistics"]["count"] == 600
    assert dataset["artifact_id"].startswith("data_")


def test_path_traversal_is_refused():
    message = fails("actsim_inspect_file", {"file_path": "../../../etc/passwd"})
    assert "outside the server workspace" in message


def test_ambiguous_data_source_is_refused():
    message = fails("actsim_load_dataset", {"values": [1.0, 2.0], "file_path": "losses.csv"})
    assert "exactly one data source" in message


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fit_id(workspace) -> str:
    dataset = as_json("actsim_load_dataset", {"file_path": "losses.csv", "column": "loss"})
    result = as_json(
        "actsim_fit_distributions",
        {
            "artifact_id": dataset["artifact_id"],
            "distributions": ["lognormal", "gamma", "weibull", "pareto"],
            "rank_by": "aic",
        },
    )
    assert result["selected"]["distribution"] == "lognormal"
    assert result["observations_fitted"] == 600
    return result["artifact_id"]


def test_fit_rejects_the_ks_statistic_metric_name(workspace):
    message = fails(
        "actsim_fit_distributions",
        {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "metrics": ["aic", "ks_statistic"]},
    )
    assert "'ks', not 'ks_statistic'" in message


def test_fit_truncation_drops_rows(workspace):
    result = as_json(
        "actsim_fit_distributions",
        {
            "values": [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "distributions": ["gamma"],
            "remove_values": [0.0],
        },
    )
    assert result["observations_in"] == 10
    assert result["observations_fitted"] == 8


def test_fit_summary_select_sample_and_statistics(fit_id):
    summary = as_json("actsim_get_fit_summary", {"artifact_id": fit_id, "rank_by": "ks"})
    assert len(summary["candidates"]) == 4

    selected = as_json("actsim_select_fit", {"artifact_id": fit_id, "distribution": "gamma"})
    assert selected["selected"]["distribution"] == "gamma"
    as_json("actsim_select_fit", {"artifact_id": fit_id, "distribution": "lognormal"})

    sample = as_json("actsim_sample_from_fit", {"artifact_id": fit_id, "size": 500, "zero_proportion": 0.2})
    assert sample["statistics"]["zero_share"] == pytest.approx(0.2, abs=0.01)

    stats = as_json("actsim_fit_statistics", {"artifact_id": fit_id})
    assert "comparison" in stats


def test_select_fit_rejects_uncandidate_distribution(fit_id):
    message = fails("actsim_select_fit", {"artifact_id": fit_id, "distribution": "beta"})
    assert "not among the fitted candidates" in message


def test_plot_fit_returns_png(fit_id):
    _, is_error, content = call("actsim_plot_fit", {"artifact_id": fit_id})
    assert not is_error
    assert mime_of(content) == "image/png"


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def simulation_id() -> str:
    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [4.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.8],
            "num_simulations": 2000,
            "keep_events": True,
        },
    )
    assert result["configuration"]["events_stored"] > 0
    assert result["statistics"]["mean"] > 0
    return result["artifact_id"]


def test_simulation_is_reproducible():
    args = {
        "freq_dist": "poisson",
        "freq_params": [3.0],
        "sev_dist": "gamma",
        "sev_params": [2.0, 5000.0],
        "num_simulations": 500,
        "seed": 11,
        "keep_events": False,
    }
    first = as_json("actsim_simulate_aggregate", args)
    second = as_json("actsim_simulate_aggregate", args)
    assert first["statistics"]["mean"] == pytest.approx(second["statistics"]["mean"])


def test_severity_distribution_rejected_as_frequency():
    message = fails(
        "actsim_simulate_aggregate",
        {"freq_dist": "lognormal", "freq_params": [1, 1], "sev_dist": "gamma", "sev_params": [2, 100]},
    )
    assert "claim-count distribution" in message


def test_copula_without_correlation_is_refused():
    message = fails(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [3.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.5],
            "copula": "gaussian",
        },
    )
    assert "needs a correlation" in message


def test_gumbel_theta_domain_is_checked():
    message = fails(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [3.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.5],
            "copula": "gumbel",
            "theta": 0.5,
        },
    )
    assert "must be >= 1" in message


def test_copula_simulation_runs():
    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [3.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.6],
            "num_simulations": 800,
            "copula": "gumbel",
            "theta": 2.0,
            "keep_events": True,
        },
    )
    assert result["configuration"]["dependence"] == "gumbel copula"
    _, is_error, content = call("actsim_plot_frequency_severity", {"artifact_id": result["artifact_id"]})
    assert not is_error and mime_of(content) == "image/png"


def test_analyze_simulation_reports_oep(simulation_id):
    analysis = as_json("actsim_analyze_simulation", {"artifact_id": simulation_id, "quantiles": [0.9, 0.99]})
    columns = {row for row in analysis["risk_measures"][0]}
    assert {"VaR", "TVaR", "AEP", "OEP"} <= columns
    assert analysis["risk_measures"][0]["TVaR"] >= analysis["risk_measures"][0]["VaR"]


def test_analyze_simulation_without_events_still_works():
    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [2.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.5],
            "num_simulations": 400,
            "keep_events": False,
        },
    )
    analysis = as_json("actsim_analyze_simulation", {"artifact_id": result["artifact_id"]})
    assert "OEP" not in analysis["risk_measures"][0]


def test_apply_layer(simulation_id):
    layered = as_json(
        "actsim_apply_layer",
        {
            "artifact_id": simulation_id,
            "per_occurrence_deductible": 5000,
            "per_occurrence_limit": 100000,
            "aggregate_limit": 500000,
        },
    )
    assert layered["years"] == 2000
    assert 0 <= layered["share_of_ground_up"] <= 1
    assert layered["years_exhausting_limit"] is not None


def test_apply_layer_requires_events():
    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [2.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.5],
            "num_simulations": 300,
            "keep_events": False,
        },
    )
    message = fails("actsim_apply_layer", {"artifact_id": result["artifact_id"]})
    assert "keep_events=true" in message


def test_plot_simulation_returns_png(simulation_id):
    _, is_error, content = call("actsim_plot_simulation", {"artifact_id": simulation_id, "log_scale": True})
    assert not is_error and mime_of(content) == "image/png"


# --------------------------------------------------------------------------- #
# Correlated portfolio
# --------------------------------------------------------------------------- #

LINES = [
    {"dist_name": "Property", "dist_type": "lognormal", "dist_param": [12.0, 1.0]},
    {"dist_name": "Liability", "dist_type": "gamma", "dist_param": [2.0, 90000.0]},
]


def test_correlated_portfolio_reports_diversification():
    result = as_json(
        "actsim_simulate_correlated_portfolio",
        {"lines": LINES, "correlation_matrix": [[1.0, 0.4], [0.4, 1.0]], "num_simulations": 3000},
    )
    assert result["diversification"]["benefit"] > 0
    realized = result["realized_correlation"]
    assert 0.2 < realized[0]["Liability"] < 0.6


def test_non_psd_correlation_matrix_is_refused():
    message = fails(
        "actsim_simulate_correlated_portfolio",
        {"lines": LINES, "correlation_matrix": [[1.0, 1.6], [1.6, 1.0]], "num_simulations": 100},
    )
    assert "positive semi-definite" in message


def test_correlation_matrix_shape_mismatch_is_refused():
    message = fails(
        "actsim_simulate_correlated_portfolio",
        {"lines": LINES, "correlation_matrix": [[1.0]], "num_simulations": 100},
    )
    assert "2 lines were given" in message


def test_line_missing_keys_names_the_line():
    message = fails(
        "actsim_simulate_correlated_portfolio",
        {
            "lines": [{"dist_name": "Property"}],
            "correlation_matrix": [[1.0]],
            "num_simulations": 100,
        },
    )
    assert "missing: dist_param, dist_type" in message


# --------------------------------------------------------------------------- #
# Claims pipeline
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def claims_chain(workspace) -> dict:
    book = as_json("actsim_build_policy_book", {"file_path": "policies.csv"})
    assert book["policies"] == 25

    claims = as_json("actsim_simulate_claims", {"artifact_id": book["artifact_id"]})
    assert claims["claims"] > 0

    dated = as_json("actsim_assign_claim_dates", {"artifact_id": claims["artifact_id"], "alpha": 0.3})
    assert dated["claims"] == claims["claims"]

    development = as_json(
        "actsim_simulate_claim_development",
        {
            "artifact_id": dated["artifact_id"],
            "base_ldfs": {"12": 2.5, "24": 1.4, "36": 1.1, "48": 1.0},
            "volatility": 0.05,
        },
    )
    return {"book": book, "claims": claims, "dated": dated, "development": development}


def test_synthetic_policy_book_is_valid():
    book = as_json("actsim_build_policy_book", {"synthesize": 40, "seed": 3})
    assert book["policies"] == 40
    assert book["exposure_years"] > 0


def test_policy_book_missing_columns_names_them(workspace):
    message = fails("actsim_build_policy_book", {"policies": [{"policy_id": 1}]})
    assert "missing required policy column" in message


def test_policy_book_rejects_bad_distribution(workspace):
    message = fails(
        "actsim_build_policy_book",
        {
            "policies": [
                {
                    "policy_id": 1,
                    "freq_dist": "lognormal",
                    "freq_params": [1.0],
                    "sev_dist": "lognormal",
                    "sev_params": [9.0, 0.5],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                }
            ]
        },
    )
    assert "as its frequency distribution" in message


def test_policy_params_are_parsed_without_eval(workspace):
    """A parameter cell holding code must not execute; it must fail cleanly."""
    message = fails(
        "actsim_build_policy_book",
        {
            "policies": [
                {
                    "policy_id": 1,
                    "freq_dist": "poisson",
                    "freq_params": "__import__('os').system('touch /tmp/pwned')",
                    "sev_dist": "lognormal",
                    "sev_params": [9.0, 0.5],
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                }
            ]
        },
    )
    assert "Could not read distribution parameters" in message
    assert not Path("/tmp/pwned").exists()


def test_claims_chain_produces_a_triangle(claims_chain):
    development = claims_chain["development"]
    assert development["rows"] > 0

    triangle = as_json(
        "actsim_build_loss_triangle", {"artifact_id": development["artifact_id"], "basis": "cumulative"}
    )
    assert triangle["volume_weighted_factors"]

    incremental = as_json(
        "actsim_build_loss_triangle", {"artifact_id": development["artifact_id"], "basis": "incremental"}
    )
    assert incremental["basis"] == "incremental"


def test_development_requires_dated_claims(claims_chain):
    message = fails(
        "actsim_simulate_claim_development",
        {"artifact_id": claims_chain["claims"]["artifact_id"], "base_ldfs": {"12": 1.0}},
    )
    assert "no occurrence dates" in message


def test_double_dating_is_refused(claims_chain):
    message = fails("actsim_assign_claim_dates", {"artifact_id": claims_chain["dated"]["artifact_id"]})
    assert "already has occurrence dates" in message


# --------------------------------------------------------------------------- #
# Analytics, artifacts and reporting
# --------------------------------------------------------------------------- #

def test_risk_metrics_on_a_simulation(simulation_id):
    metrics = as_json("actsim_risk_metrics", {"artifact_id": simulation_id, "return_period_years": [200]})
    assert "1-in-200" in metrics["return_periods"]


def test_percentage_quantiles_are_rejected(simulation_id):
    message = fails("actsim_risk_metrics", {"artifact_id": simulation_id, "quantiles": [95]})
    assert "not percentages" in message


def test_stress_test_scales_losses(simulation_id):
    stress = as_json(
        "actsim_stress_test",
        {
            "artifact_id": simulation_id,
            "scenarios": [
                {"name": "inflation", "severity_multiplier": 1.2},
                {"name": "cat load", "shift": 100000},
            ],
        },
    )
    assert stress["change_vs_base_pct"][0]["mean"] == pytest.approx(20.0, abs=0.5)


def test_stress_test_rejects_unknown_keys(simulation_id):
    message = fails(
        "actsim_stress_test",
        {"artifact_id": simulation_id, "scenarios": [{"name": "x", "multiplier": 2}]},
    )
    assert "unsupported key" in message


def test_compare_simulations(simulation_id):
    other = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [8.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.8],
            "num_simulations": 1000,
            "keep_events": False,
        },
    )
    comparison = as_json(
        "actsim_compare_simulations", {"artifact_ids": [simulation_id, other["artifact_id"]]}
    )
    assert comparison["baseline"] == simulation_id


def test_artifact_lifecycle(simulation_id, workspace):
    listing = as_json("actsim_list_artifacts", {"kind": "simulation"})
    assert any(a["artifact_id"] == simulation_id for a in listing["artifacts"])

    described = as_json("actsim_describe_artifact", {"artifact_id": simulation_id})
    assert described["rows"] > 0

    preview = as_json("actsim_preview_artifact", {"artifact_id": simulation_id, "limit": 5})
    assert preview["count"] == 5
    assert preview["has_more"] is True

    export = as_json(
        "actsim_export_artifact", {"artifact_id": simulation_id, "path": "out/sim.csv"}
    )
    assert (workspace / "out" / "sim.csv").exists()
    assert export["rows"] > 0

    scratch = as_json("actsim_load_dataset", {"values": [1.0, 2.0, 3.0]})
    deleted = as_json("actsim_delete_artifact", {"artifact_id": scratch["artifact_id"]})
    assert deleted["deleted"] is True


def test_preview_works_without_event_detail():
    """A keep_events=false simulation still previews and exports its aggregate."""
    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [2.0],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 0.5],
            "num_simulations": 300,
            "keep_events": False,
        },
    )
    preview = as_json("actsim_preview_artifact", {"artifact_id": result["artifact_id"], "limit": 5})
    assert preview["count"] == 5
    assert preview["total"] == 300


def test_layered_artifact_previews_annual_detail(simulation_id):
    layered = as_json(
        "actsim_apply_layer",
        {"artifact_id": simulation_id, "per_occurrence_deductible": 1000, "aggregate_limit": 1000000},
    )
    described = as_json("actsim_describe_artifact", {"artifact_id": layered["artifact_id"]})
    names = {c["name"] for c in described["columns"]}
    assert {"year", "ground_up", "ceded", "retained"} <= names


def test_unknown_artifact_error_is_actionable():
    message = fails("actsim_describe_artifact", {"artifact_id": "sim_deadbeef"})
    assert "actsim_list_artifacts" in message


def test_export_outside_workspace_is_refused(simulation_id):
    message = fails(
        "actsim_export_artifact", {"artifact_id": simulation_id, "path": "/etc/actsim_export.csv"}
    )
    assert "outside the server workspace" in message


def test_risk_report_includes_every_artifact(fit_id, simulation_id, claims_chain):
    report = ok(
        "actsim_generate_risk_report",
        {
            "artifact_ids": [
                fit_id,
                simulation_id,
                claims_chain["claims"]["artifact_id"],
                claims_chain["development"]["artifact_id"],
            ],
            "title": "Portfolio Review",
        },
    )
    assert "# Portfolio Review" in report
    assert fit_id in report and simulation_id in report


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #

def test_oep_counts_years_that_produced_no_event():
    """OEP is the largest event per YEAR, so claim-free years belong in the sample.

    Grouping the event table alone drops them, which inflated OEP by roughly 2x on
    a low-frequency book.
    """
    import numpy as np

    from actsim_mcp.analytics import largest_event_per_year
    from actsim_mcp.store import STORE

    result = as_json(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [0.5],
            "sev_dist": "lognormal",
            "sev_params": [9.0, 1.0],
            "num_simulations": 5000,
            "seed": 5,
            "keep_events": True,
            "quantiles": [0.9, 0.99],
        },
    )
    artifact = STORE.get(result["artifact_id"])
    events = artifact.payload["events"]
    aggregate = np.asarray(artifact.payload["aggregate"], dtype=float)

    claim_free = 5000 - events["year"].nunique()
    assert claim_free > 1000, "test needs a book where most years produce no claim"

    per_year = largest_event_per_year(aggregate, events)
    assert per_year.size == 5000
    assert (per_year == 0).sum() == claim_free

    reported = {row["quantile"]: row["OEP"] for row in result["risk_measures"]}
    assert reported["0.9"] == pytest.approx(float(np.quantile(per_year, 0.9)))
    # The naive computation over event-years only would sit well above this.
    assert reported["0.9"] < float(np.quantile(events.groupby("year")["amount"].max(), 0.9))


def test_log_likelihood_ranking_picks_the_best_not_the_worst(workspace):
    """actsim minimises whatever metric it is given; log-likelihood is maximised."""
    ranked = as_json(
        "actsim_fit_distributions",
        {
            "file_path": "losses.csv",
            "column": "loss",
            "distributions": ["lognormal", "gamma", "weibull"],
            "rank_by": "log_likelihood",
        },
    )
    best = ranked["candidates"][0]
    assert ranked["selected"]["distribution"] == best["name"]
    assert best["log_likelihood"] == max(c["log_likelihood"] for c in ranked["candidates"])
    assert "higher is better" in ranked["metric_note"]


def test_degenerate_fits_are_never_selected(workspace):
    """A failed optimiser can leave -inf log-likelihood rows; they must not win."""
    import numpy as np

    ranked = as_json(
        "actsim_fit_distributions",
        {
            "file_path": "losses.csv",
            "column": "loss",
            "distributions": ["lognormal", "gamma", "beta"],
            "rank_by": "log_likelihood",
        },
    )
    assert np.isfinite(ranked["selected"]["log_likelihood"])
    assert np.isfinite(ranked["selected"]["aic"])


def test_risk_table_keeps_its_quantile_labels_in_json(simulation_id):
    """JSON consumers need to know which quantile each row belongs to."""
    analysis = as_json("actsim_analyze_simulation", {"artifact_id": simulation_id, "quantiles": [0.9, 0.99]})
    labels = [row["quantile"] for row in analysis["risk_measures"]]
    assert labels == ["0.9", "0.99"]


def test_blank_file_path_is_treated_as_absent(simulation_id):
    """file_path='' used to pass the source guard and then fail with 'Empty path'."""
    metrics = as_json("actsim_risk_metrics", {"artifact_id": simulation_id, "file_path": ""})
    assert metrics["source"].endswith(simulation_id)


def test_conflicting_sources_are_rejected_not_ignored(simulation_id):
    message = fails("actsim_risk_metrics", {"artifact_id": simulation_id, "values": [1.0, 2.0, 3.0]})
    assert "together with values" in message


def test_integer_identifiers_render_without_thousands_separators(claims_chain):
    """An accident year is an identifier, not a quantity: 2024, never '2,024'."""
    triangle = ok(
        "actsim_build_loss_triangle", {"artifact_id": claims_chain["development"]["artifact_id"]}
    )
    assert "2,024" not in triangle
    assert "| 2024 |" in triangle


def test_http_transport_arguments_match_the_installed_sdk():
    """SDK 1.x run() takes no host/port; passing them unconditionally raises."""
    from actsim_mcp import compat

    class Settings:
        host = None
        port = None

    class OldStyleServer:
        """Mimics FastMCP 1.x: run() accepts only transport and mount_path."""

        def __init__(self):
            self.settings = Settings()
            self.called = None

        def run(self, transport="stdio", mount_path=None):
            self.called = {"transport": transport, "mount_path": mount_path}

    server = OldStyleServer()
    compat.run_server(server, "streamable-http", host="0.0.0.0", port=9111)
    assert server.called == {"transport": "streamable-http", "mount_path": None}
    assert server.settings.host == "0.0.0.0"
    assert server.settings.port == 9111

    class NewStyleServer:
        def __init__(self):
            self.called = None

        def run(self, transport="stdio", host=None, port=None):
            self.called = {"transport": transport, "host": host, "port": port}

    modern = NewStyleServer()
    compat.run_server(modern, "streamable-http", host="127.0.0.1", port=9112)
    assert modern.called == {"transport": "streamable-http", "host": "127.0.0.1", "port": 9112}


# --------------------------------------------------------------------------- #
# Transport safety
# --------------------------------------------------------------------------- #

def test_library_chatter_never_reaches_stdout(capsys, workspace):
    """actsim prints fit lines and timings; on stdio that would corrupt JSON-RPC."""
    capsys.readouterr()
    ok("actsim_fit_distributions", {"file_path": "losses.csv", "column": "loss", "distributions": ["gamma"]})
    ok(
        "actsim_simulate_aggregate",
        {
            "freq_dist": "poisson",
            "freq_params": [2.0],
            "sev_dist": "gamma",
            "sev_params": [2.0, 1000.0],
            "num_simulations": 200,
        },
    )
    captured = capsys.readouterr()
    assert captured.out == "", f"library wrote to stdout: {captured.out[:200]!r}"


def test_resources_read():
    for uri in ("actsim://distributions", "actsim://capabilities", "actsim://workflows", "actsim://examples", "actsim://config", "actsim://artifacts"):
        contents = asyncio.run(SERVER.read_resource(uri))
        body = list(contents)[0]
        json.loads(getattr(body, "content", getattr(body, "text", "")))


def test_prompts_render():
    result = asyncio.run(SERVER.get_prompt("price_excess_layer", {"data_source": "claims.csv"}))
    text = result.messages[0].content.text
    assert "actsim_apply_layer" in text

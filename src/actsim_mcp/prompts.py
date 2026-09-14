"""MCP prompts: reusable starting points for common actuarial engagements.

Each prompt names real tools with real arguments, so the model does not have to
guess the call sequence.
"""

from __future__ import annotations

from typing import Literal

from .compat import prompt_decorator
from .resources import WORKFLOWS


def _workflow_block(key: str) -> str:
    flow = WORKFLOWS[key]
    steps = "\n".join(f"{i}. `{step}`" for i, step in enumerate(flow["steps"], start=1))
    return f"{flow['goal']}\n\n{steps}"


def register(server) -> None:
    @prompt_decorator(
        server,
        name="price_excess_layer",
        description="Fit severity, simulate aggregate losses and price an excess-of-loss layer.",
    )
    def price_excess_layer(
        data_source: str = "claims.csv",
        attachment: str = "500000",
        limit: str = "5000000",
        claims_per_year: str = "12",
    ) -> str:
        return (
            f"Price an excess-of-loss layer attaching at {attachment} with a limit of {limit}, "
            f"using the historical claim amounts in {data_source} and an expected {claims_per_year} claims per year.\n\n"
            f"{_workflow_block('price_an_excess_layer')}\n\n"
            "Work through those steps with the actsim tools. Before simulating, check the fitted severity "
            "against the data with actsim_fit_statistics, and say explicitly which distribution you selected "
            "and why. Report the expected ceded loss, the share of ground-up loss it represents, how often the "
            "layer is hit and how often the limit is exhausted. Flag any assumption the numbers are sensitive to."
        )

    @prompt_decorator(
        server,
        name="portfolio_capital_review",
        description="Simulate correlated lines of business and quantify diversification and capital.",
    )
    def portfolio_capital_review(
        lines: str = "Property, Liability, Motor",
        capital_quantile: str = "0.995",
    ) -> str:
        return (
            f"Run a portfolio capital review across these lines of business: {lines}. "
            f"Report capital at the {capital_quantile} quantile.\n\n"
            f"{_workflow_block('capital_from_correlated_lines')}\n\n"
            "State the correlation assumptions you used and where they came from. Quantify the diversification "
            "benefit against the sum of standalone requirements, then stress the result for claims inflation and "
            "a frequency shock. Distinguish clearly between simulation output and any judgement you add."
        )

    @prompt_decorator(
        server,
        name="reserving_triangle_build",
        description="Simulate a policy book into a dated claim set and a development triangle.",
    )
    def reserving_triangle_build(
        policy_source: str = "policies.csv",
        development_months: str = "12, 24, 36, 48, 60",
    ) -> str:
        return (
            f"Build a loss development triangle from the policy book in {policy_source}, "
            f"with development ages at {development_months} months.\n\n"
            f"{_workflow_block('reserving_triangle_from_a_policy_book')}\n\n"
            "Choose loss development factors that are plausible for the line and say what you assumed. "
            "Present both the cumulative triangle and the implied age-to-age factors, and comment on how much "
            "the simulated volatility moves the factors between accident years."
        )

    @prompt_decorator(
        server,
        name="dependence_sensitivity",
        description="Measure how a frequency/severity copula assumption moves the tail.",
    )
    def dependence_sensitivity(
        copula: Literal["gaussian", "frank", "gumbel", "clayton"] = "gumbel",
        quantile: str = "0.995",
    ) -> str:
        return (
            f"Quantify how much the {copula} copula between claim frequency and severity moves the "
            f"{quantile} quantile of the aggregate loss, relative to an independent baseline.\n\n"
            f"{_workflow_block('check_frequency_severity_dependence')}\n\n"
            "Hold the marginals and the seed fixed so the comparison is clean. Show the base and dependent "
            "results side by side, verify with actsim_plot_frequency_severity that the dependence you asked for "
            "actually materialised, and state the tail impact in both absolute and percentage terms."
        )

    @prompt_decorator(
        server,
        name="distribution_selection_review",
        description="Fit, compare and defend the choice of a severity distribution.",
    )
    def distribution_selection_review(data_source: str = "losses.csv") -> str:
        return (
            f"Review which severity distribution best represents the losses in {data_source}.\n\n"
            "1. `actsim_load_dataset(file_path=...)`\n"
            "2. `actsim_fit_distributions(artifact_id=..., rank_by='aic')`\n"
            "3. `actsim_get_fit_summary(artifact_id=..., rank_by='ks')`\n"
            "4. `actsim_fit_statistics(artifact_id=...)`\n"
            "5. `actsim_plot_fit(artifact_id=...)`\n\n"
            "Compare the AIC, BIC, chi-square and KS rankings. Where they disagree, explain what that implies "
            "about body versus tail fit, and consider whether the data needs truncation (nil claims, a reporting "
            "threshold, outliers) before refitting. Finish with a recommended distribution and the parameters to "
            "carry into simulation."
        )

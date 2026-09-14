"""Policy-level claim simulation, occurrence dating and development triangles."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..analytics import moments
from ..catalog import FREQUENCY_DISTRIBUTIONS, SEVERITY_DISTRIBUTIONS, normalize, spec, validate_params
from ..compat import tool_decorator
from ..dataio import parse_params, read_json_file, read_table
from ..errors import ActsimToolError, unknown_value
from ..params import ArtifactId, Format, Limit, Seed
from ..render import ResponseFormat, bullets, document, next_steps, respond, table
from ..runtime import captured
from ..settings import COPULA_TYPES, SETTINGS
from ..store import STORE

POLICY_COLUMNS = (
    "policy_id",
    "freq_dist",
    "freq_params",
    "sev_dist",
    "sev_params",
    "start_date",
    "end_date",
)


def _normalise_policies(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Validate and coerce a policy table into the schema ClaimSimulator requires.

    Parameter cells are parsed with ``ast.literal_eval`` rather than ``eval`` so a
    hostile or malformed CSV cannot execute code, and distribution names are
    checked here to turn a downstream ``ValueError: Invalid distribution`` into a
    message naming the offending row.
    """
    missing = [c for c in POLICY_COLUMNS if c not in frame.columns]
    if missing:
        raise ActsimToolError(
            f"{source} is missing required policy column(s): {', '.join(missing)}.",
            hint=f"A policy table needs: {', '.join(POLICY_COLUMNS)}.",
        )
    if frame.empty:
        raise ActsimToolError(f"{source} contains no policies.", hint="Add at least one policy row.")
    if len(frame) > SETTINGS.max_policies:
        raise ActsimToolError(
            f"{len(frame):,} policies exceeds the limit of {SETTINGS.max_policies:,}.",
            hint="Split the book, or raise ACTSIM_MCP_MAX_POLICIES.",
        )

    out = frame.loc[:, list(POLICY_COLUMNS)].copy()
    for index, row in out.iterrows():
        for kind, column, allowed in (
            ("frequency", "freq_dist", FREQUENCY_DISTRIBUTIONS),
            ("severity", "sev_dist", SEVERITY_DISTRIBUTIONS),
        ):
            name = normalize(str(row[column]))
            if name not in allowed:
                raise ActsimToolError(
                    f"Policy {row['policy_id']!r} uses {name!r} as its {kind} distribution.",
                    hint=f"Valid {kind} distributions: {', '.join(allowed)}.",
                )
            out.at[index, column] = name

    for column, name_column in (("freq_params", "freq_dist"), ("sev_params", "sev_dist")):
        parsed = []
        for index, value in out[column].items():
            params = parse_params(value)
            dist = str(out.at[index, name_column])
            try:
                validate_params(dist, list(params))
            except ActsimToolError as exc:
                raise ActsimToolError(
                    f"Policy {out.at[index, 'policy_id']!r}: {exc.message}", hint=exc.hint
                ) from exc
            parsed.append(tuple(params))
        out[column] = parsed  # ClaimSimulator asserts these are tuples

    for column in ("start_date", "end_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
        if out[column].isna().any():
            bad = out.loc[out[column].isna(), "policy_id"].tolist()[:5]
            raise ActsimToolError(
                f"Unparseable {column} for policies: {bad}.",
                hint="Use ISO dates, e.g. 2024-01-01.",
            )
    invalid = out[out["end_date"] <= out["start_date"]]
    if not invalid.empty:
        raise ActsimToolError(
            f"end_date must be after start_date for policies: {invalid['policy_id'].tolist()[:5]}.",
            hint="Check for transposed dates or a one-day policy term.",
        )
    return out.reset_index(drop=True)


def _policy_preview(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    preview = frame.head(limit).copy()
    for column in ("freq_params", "sev_params"):
        preview[column] = preview[column].apply(lambda t: list(t))
    for column in ("start_date", "end_date"):
        preview[column] = preview[column].dt.strftime("%Y-%m-%d")
    return preview


def register(server) -> None:
    @tool_decorator(server, name="actsim_build_policy_book", title="Build Policy Book")
    async def actsim_build_policy_book(
        policies: Annotated[
            list[dict[str, Any]] | None,
            Field(
                default=None,
                description=(
                    "Explicit policies: [{'policy_id':1,'freq_dist':'poisson','freq_params':[0.8],"
                    "'sev_dist':'lognormal','sev_params':[9,0.5],'start_date':'2024-01-01','end_date':'2024-12-31'}]."
                ),
                max_length=2000,
            ),
        ] = None,
        file_path: Annotated[
            str | None,
            Field(default=None, description="CSV/JSON policy file in the workspace with the same seven columns."),
        ] = None,
        synthesize: Annotated[
            int | None,
            Field(default=None, description="Generate this many synthetic policies instead of supplying them.", ge=1, le=SETTINGS.max_policies),
        ] = None,
        freq_dist: Annotated[str, Field(default="poisson", description="Frequency distribution for synthetic policies.")] = "poisson",
        freq_params: Annotated[
            list[float] | None,
            Field(default=None, description="Fixed frequency parameters for synthetic policies. Omit to randomise the Poisson mean over [0.5, 2.0].", max_length=3),
        ] = None,
        sev_dist: Annotated[str, Field(default="lognormal", description="Severity distribution for synthetic policies.")] = "lognormal",
        sev_params: Annotated[
            list[float] | None,
            Field(default=None, description="Fixed severity parameters for synthetic policies. Omit to randomise lognormal mu over [8,12] and sigma over [0.3,1.0].", max_length=4),
        ] = None,
        start_date: Annotated[str, Field(default="2024-01-01", description="Inception date for synthetic policies.")] = "2024-01-01",
        end_date: Annotated[str, Field(default="2024-12-31", description="Expiry date for synthetic policies.")] = "2024-12-31",
        seed: Seed = SETTINGS.default_seed,
        preview_rows: Limit = 10,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Assemble a validated policy book from explicit rows, a file, or synthetic generation.

        Every route produces the same seven-column schema that actsim_simulate_claims
        needs, with distribution names and parameter arity checked per policy so an
        error names the offending policy rather than failing deep in the simulator.

        Returns: policy book artifact id, per-distribution counts and a row preview.
        """
        sources = [name for name, value in (("policies", policies), ("file_path", file_path), ("synthesize", synthesize)) if value]
        if len(sources) != 1:
            raise ActsimToolError(
                f"Provide exactly one of policies, file_path or synthesize; got {len(sources)}.",
                hint="Use synthesize=100 for a quick test book.",
            )

        if policies:
            frame = pd.DataFrame(policies)
            source = f"{len(policies)} inline policies"
        elif file_path:
            frame = read_table(file_path) if not str(file_path).endswith(".json") else pd.DataFrame(read_json_file(file_path))
            source = str(file_path)
        else:
            count = int(synthesize or 0)
            freq_name = normalize(freq_dist)
            sev_name = normalize(sev_dist)
            if freq_name not in FREQUENCY_DISTRIBUTIONS:
                raise unknown_value("frequency distribution", freq_dist, FREQUENCY_DISTRIBUTIONS)
            if sev_name not in SEVERITY_DISTRIBUTIONS:
                raise unknown_value("severity distribution", sev_dist, SEVERITY_DISTRIBUTIONS)
            rng = np.random.default_rng(seed)
            rows = []
            for i in range(1, count + 1):
                if freq_params:
                    fp = tuple(freq_params)
                elif freq_name == "poisson":
                    fp = (float(rng.uniform(0.5, 2.0)),)
                else:
                    fp = (float(rng.integers(1, 10)), float(rng.uniform(0.3, 0.8)))
                if sev_params:
                    sp = tuple(sev_params)
                elif sev_name == "lognormal":
                    sp = (float(rng.uniform(8, 12)), float(rng.uniform(0.3, 1.0)))
                elif sev_name == "gamma":
                    sp = (float(rng.uniform(1, 5)), float(rng.uniform(500, 2000)))
                else:
                    sp = tuple(float(v) for v in spec(sev_name).defaults)
                rows.append(
                    {
                        "policy_id": i,
                        "freq_dist": freq_name,
                        "freq_params": fp,
                        "sev_dist": sev_name,
                        "sev_params": sp,
                        "start_date": start_date,
                        "end_date": end_date,
                    }
                )
            frame = pd.DataFrame(rows)
            source = f"{count} synthetic policies (seed {seed})"

        book = _normalise_policies(frame, source=source)
        exposure_years = float(((book["end_date"] - book["start_date"]).dt.days / 365.25).sum())
        artifact = STORE.put(
            "policies",
            source,
            book,
            {
                "policies": int(len(book)),
                "source": source,
                "exposure_years": round(exposure_years, 2),
                "period": f"{book['start_date'].min():%Y-%m-%d} to {book['end_date'].max():%Y-%m-%d}",
            },
        )
        mix = (
            book.groupby(["freq_dist", "sev_dist"], as_index=False)
            .size()
            .rename(columns={"size": "policies"})
        )
        payload = {
            "artifact_id": artifact.id,
            "source": source,
            "policies": int(len(book)),
            "exposure_years": exposure_years,
            "period": artifact.meta["period"],
            "distribution_mix": mix,
            "preview": _policy_preview(book, preview_rows),
        }
        markdown = document(
            f"Policy book `{artifact.id}`",
            [
                ("Summary", bullets({
                    "source": source,
                    "policies": len(book),
                    "total exposure (years)": exposure_years,
                    "period": artifact.meta["period"],
                })),
                ("Distribution mix", table(mix, index=False)),
                ("Preview", table(payload["preview"], index=False)),
                ("", next_steps(f"actsim_simulate_claims(artifact_id='{artifact.id}')")),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_simulate_claims", title="Simulate Claims")
    async def actsim_simulate_claims(
        artifact_id: ArtifactId,
        correlation: Annotated[
            float | None,
            Field(default=None, description="Frequency/severity correlation applied within each policy group.", ge=-1.0, le=1.0),
        ] = None,
        copula: Annotated[
            Literal["gaussian", "frank", "gumbel", "clayton"] | None,
            Field(default=None, description="Copula for frequency/severity dependence."),
        ] = None,
        copula_param: Annotated[
            float, Field(default=0.0, description="Archimedean copula theta (frank/gumbel/clayton).")
        ] = 0.0,
        seed: Seed = 42,
        preview_rows: Limit = 10,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Simulate individual claims for every policy in a stored policy book.

        Policies sharing a frequency/severity specification are simulated as a
        group, so a book of similar risks runs in one pass per group rather than one
        per policy. Each claim is tagged with its originating policy.

        Returns: claims artifact id, claim counts, severity statistics, per-policy
        aggregates and a row preview.
        """
        if copula and copula.lower() not in COPULA_TYPES:
            raise unknown_value("copula", copula, COPULA_TYPES)
        if copula and correlation is None:
            correlation = 0.0  # actsim enters the copula branch only when correlation is set

        artifact = STORE.get(artifact_id, kind="policies")
        book = artifact.payload

        from actsim import ClaimSimulator

        with captured():
            np.random.seed(seed)
            simulator = ClaimSimulator(
                policies_df=book,
                random_seed=seed,
                correlation=correlation,
                copula_type=copula.lower() if copula else None,
                copula_param=copula_param,
            )
            simulator.simulate_claims()

        claims = simulator.claim_data
        if claims is None or claims.empty:
            payload = {"artifact_id": None, "claims": 0, "policies": int(len(book))}
            return respond(
                response_format,
                payload,
                "No claims were generated. With low Poisson means this is a legitimate outcome - "
                "increase the frequency parameters, extend the policy periods, or add more policies.",
            )

        amounts = claims["amount"].to_numpy(float)
        per_policy = (
            claims.groupby("policy_id")["amount"].agg(["count", "sum", "mean", "max"]).reset_index()
        )
        stored = STORE.put(
            "claims",
            f"claims from {artifact_id}",
            claims,
            {
                "policy_book": artifact_id,
                "claims": int(len(claims)),
                "policies_with_claims": int(claims["policy_id"].nunique()),
                "total_loss": round(float(amounts.sum()), 2),
                "seed": seed,
                "dated": False,
            },
        )
        payload = {
            "artifact_id": stored.id,
            "policy_book": artifact_id,
            "policies": int(len(book)),
            "policies_with_claims": int(claims["policy_id"].nunique()),
            "claims": int(len(claims)),
            "severity_statistics": moments(amounts),
            "top_policies": per_policy.nlargest(min(preview_rows, len(per_policy)), "sum"),
            "preview": claims.head(preview_rows),
        }
        markdown = document(
            f"Claims `{stored.id}`",
            [
                ("Summary", bullets({
                    "policy book": artifact_id,
                    "policies": len(book),
                    "policies with claims": payload["policies_with_claims"],
                    "claims": len(claims),
                    "total loss": float(amounts.sum()),
                    "dependence": f"{copula} copula" if copula else ("gaussian correlation" if correlation is not None else "independent"),
                })),
                ("Severity statistics", bullets(payload["severity_statistics"])),
                ("Largest policies by loss", table(payload["top_policies"], index=False)),
                ("", next_steps(
                    f"actsim_assign_claim_dates(artifact_id='{stored.id}') to give each claim an occurrence date",
                    f"actsim_export_artifact(artifact_id='{stored.id}', path='claims.csv')",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_assign_claim_dates", title="Assign Claim Occurrence Dates")
    async def actsim_assign_claim_dates(
        artifact_id: ArtifactId,
        lambda0: Annotated[float, Field(default=10.0, description="Baseline NHPP intensity.", gt=0.0)] = 10.0,
        alpha: Annotated[
            float,
            Field(default=0.5, description="Seasonality amplitude: rate(t) = lambda0 * (1 + alpha*sin(2*pi*t + phase)). 0 gives a homogeneous process.", ge=-1.0, le=1.0),
        ] = 0.5,
        phase: Annotated[float, Field(default=0.0, description="Seasonal phase shift in radians; shifts where the peak falls in the year.")] = 0.0,
        horizon: Annotated[float, Field(default=1.0, description="Length of the NHPP interval in years.", gt=0.0)] = 1.0,
        seed: Seed = 42,
        preview_rows: Limit = 10,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Assign each claim an occurrence date from a non-homogeneous Poisson process.

        The seasonal intensity concentrates claims in part of the year (weather,
        driving patterns), and each date is constrained to its own policy's active
        window. This also renames `amount` to `ultimate_loss`, which is what the
        development tool consumes.

        Returns: an updated claims artifact id, the monthly claim profile and a preview.
        """
        artifact = STORE.get(artifact_id, kind="claims")
        claims = artifact.payload
        if "incurred_date" in claims.columns:
            raise ActsimToolError(
                f"Claims artifact {artifact_id!r} already has occurrence dates.",
                hint="Re-run actsim_simulate_claims to produce an undated set, or go straight to actsim_simulate_claim_development.",
            )

        from actsim import ClaimSimulator

        book_id = artifact.meta.get("policy_book")
        book = STORE.get(str(book_id), kind="policies").payload

        with captured():
            np.random.seed(seed)
            simulator = ClaimSimulator(policies_df=book, random_seed=seed)
            simulator.claim_data = claims.copy()
            simulator.simulate_dates_nhpp(lambda0=lambda0, alpha=alpha, phase=phase, T=horizon)

        dated = simulator.claim_data
        monthly = (
            dated.assign(month=pd.to_datetime(dated["incurred_date"]).dt.to_period("M").astype(str))
            .groupby("month", as_index=False)
            .agg(claims=("ultimate_loss", "size"), loss=("ultimate_loss", "sum"))
        )
        stored = STORE.put(
            "claims",
            f"dated claims from {artifact_id}",
            dated,
            {
                "policy_book": book_id,
                "claims": int(len(dated)),
                "total_loss": round(float(dated["ultimate_loss"].sum()), 2),
                "dated": True,
                "nhpp": f"lambda0={lambda0}, alpha={alpha}, phase={phase}, T={horizon}",
                "seed": seed,
            },
        )
        payload = {
            "artifact_id": stored.id,
            "source_claims": artifact_id,
            "nhpp_parameters": {"lambda0": lambda0, "alpha": alpha, "phase": phase, "horizon_years": horizon},
            "claims": int(len(dated)),
            "date_range": [str(dated["incurred_date"].min()), str(dated["incurred_date"].max())],
            "monthly_profile": monthly,
            "preview": dated.head(preview_rows),
        }
        markdown = document(
            f"Dated claims `{stored.id}`",
            [
                ("NHPP", bullets(payload["nhpp_parameters"])),
                ("Coverage", bullets({"claims": len(dated), "from": payload["date_range"][0], "to": payload["date_range"][1]})),
                ("Monthly profile", table(monthly, index=False)),
                ("", next_steps(
                    f"actsim_simulate_claim_development(artifact_id='{stored.id}', base_ldfs={{'12':2.5,'24':1.4}})"
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_simulate_claim_development", title="Simulate Claim Development")
    async def actsim_simulate_claim_development(
        artifact_id: ArtifactId,
        base_ldfs: Annotated[
            dict[str, float],
            Field(
                description=(
                    "Age-to-age loss development factors keyed by development month, "
                    "e.g. {'12': 2.5, '24': 1.4, '36': 1.1, '48': 1.0}."
                ),
            ),
        ],
        volatility: Annotated[
            float,
            Field(default=0.1, description="Per-claim lognormal-style noise on each LDF; 0 makes development deterministic.", ge=0.0, le=2.0),
        ] = 0.1,
        tail_factor: Annotated[
            float, Field(default=1.0, description="Cumulative tail factor beyond the last development month.", gt=0.0)
        ] = 1.0,
        seed: Seed = 42,
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Develop dated claims backwards from ultimate into a reporting history.

        Each claim's ultimate loss is divided by a randomly perturbed cumulative
        development factor at every requested age, producing the incurred amount
        that would have been on the books at that point. Requires claims that have
        already been dated by actsim_assign_claim_dates.

        Returns: a development artifact id and the accident-year x development-month
        summary.
        """
        artifact = STORE.get(artifact_id, kind="claims")
        claims = artifact.payload
        if "ultimate_loss" not in claims.columns or "incurred_date" not in claims.columns:
            raise ActsimToolError(
                f"Claims artifact {artifact_id!r} has no occurrence dates.",
                hint="Run actsim_assign_claim_dates first; development is indexed by accident year.",
            )
        try:
            ldfs = {int(k): float(v) for k, v in base_ldfs.items()}
        except (TypeError, ValueError) as exc:
            raise ActsimToolError(
                f"base_ldfs keys must be development months and values must be numbers: {exc}",
                hint="For example {'12': 2.5, '24': 1.4, '36': 1.0}.",
            ) from exc
        if not ldfs:
            raise ActsimToolError("base_ldfs is empty.", hint="Supply at least one development month, e.g. {'12': 1.0}.")
        if any(v <= 0 for v in ldfs.values()):
            raise ActsimToolError(
                "Loss development factors must be positive.",
                hint="An LDF of 1.0 means no further development at that age.",
            )

        from actsim import ClaimSimulator

        book_id = artifact.meta.get("policy_book")
        book = STORE.get(str(book_id), kind="policies").payload

        with captured():
            np.random.seed(seed)
            simulator = ClaimSimulator(policies_df=book, random_seed=seed)
            simulator.claim_data = claims.copy()
            simulator.simulate_claim_development(
                base_LDFs=ldfs, volatility=volatility, cumulative_factor=tail_factor
            )

        development = simulator.claim_development
        if development is None or development.empty:
            raise ActsimToolError(
                "Development produced no rows.",
                hint="Check that the claims carry valid incurred_date values.",
            )
        stored = STORE.put(
            "triangle",
            f"development of {artifact_id}",
            development,
            {
                "source_claims": artifact_id,
                "rows": int(len(development)),
                "accident_years": int(development["accident_year"].nunique()),
                "development_months": ", ".join(str(m) for m in sorted(development["development_month"].unique())),
                "volatility": volatility,
                "tail_factor": tail_factor,
            },
        )
        summary = (
            development.groupby(["accident_year", "development_month"], as_index=False)["incurred_loss"]
            .sum()
            .pivot(index="accident_year", columns="development_month", values="incurred_loss")
            .round(2)
        )
        payload = {
            "artifact_id": stored.id,
            "source_claims": artifact_id,
            "base_ldfs": ldfs,
            "volatility": volatility,
            "tail_factor": tail_factor,
            "rows": int(len(development)),
            "triangle": summary,
        }
        markdown = document(
            f"Claim development `{stored.id}`",
            [
                ("Assumptions", bullets({"base LDFs": ldfs, "volatility": volatility, "tail factor": tail_factor})),
                ("Cumulative incurred by accident year", table(summary)),
                ("", next_steps(
                    f"actsim_build_loss_triangle(artifact_id='{stored.id}', basis='incremental') for the incremental view",
                    f"actsim_export_artifact(artifact_id='{stored.id}', path='development.csv')",
                )),
            ],
        )
        return respond(response_format, payload, markdown)

    @tool_decorator(server, name="actsim_build_loss_triangle", title="Build Loss Triangle")
    async def actsim_build_loss_triangle(
        artifact_id: ArtifactId,
        basis: Annotated[
            Literal["cumulative", "incremental"],
            Field(default="cumulative", description="'cumulative' reports incurred to date; 'incremental' reports the movement in each period."),
        ] = "cumulative",
        value: Annotated[
            Literal["incurred_loss", "claim_count"],
            Field(default="incurred_loss", description="Triangle contents: loss amounts or claim counts."),
        ] = "incurred_loss",
        response_format: Format = ResponseFormat.MARKDOWN,
    ) -> str:
        """Pivot a stored development artifact into an accident-year triangle.

        Also reports the implied age-to-age development factors, which is what a
        reserving review actually reads.

        Returns: the triangle, its age-to-age factors and the volume-weighted average
        factor per development period.
        """
        artifact = STORE.get(artifact_id, kind="triangle")
        development = artifact.payload
        aggfunc = "sum" if value == "incurred_loss" else "count"
        column = "incurred_loss"
        triangle = development.pivot_table(
            index="accident_year", columns="development_month", values=column, aggfunc=aggfunc
        ).sort_index(axis=1)

        if basis == "incremental" and triangle.shape[1]:
            shown = triangle.diff(axis=1)
            # diff() leaves the first development period empty; it is the opening amount.
            shown.iloc[:, 0] = triangle.iloc[:, 0]
            shown = shown.round(2)
        else:
            shown = triangle.round(2)

        ages = list(triangle.columns)
        factors = pd.DataFrame(index=triangle.index)
        weighted: dict[str, float] = {}
        for earlier, later in zip(ages, ages[1:]):
            ratio = triangle[later] / triangle[earlier].replace(0, np.nan)
            factors[f"{earlier}-{later}"] = ratio.round(4)
            denominator = triangle[earlier].sum()
            weighted[f"{earlier}-{later}"] = (
                float(triangle[later].sum() / denominator) if denominator else float("nan")
            )

        payload = {
            "artifact_id": artifact_id,
            "basis": basis,
            "value": value,
            "triangle": shown,
            "age_to_age_factors": factors,
            "volume_weighted_factors": weighted,
        }
        markdown = document(
            f"{basis.title()} {value.replace('_', ' ')} triangle (`{artifact_id}`)",
            [
                ("Triangle", table(shown)),
                ("Age-to-age factors", table(factors, digits=4)),
                ("Volume-weighted averages", bullets(weighted, digits=4)),
            ],
        )
        return respond(response_format, payload, markdown)

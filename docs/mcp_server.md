# actsim MCP Server

`actsim_mcp` exposes the whole actsim toolkit — distribution fitting, aggregate
loss simulation, correlated portfolios, policy-level claim simulation and loss
development triangles — to any MCP client (Claude Desktop, Claude Code, or your
own agent).

---

## Install and run

```bash
pip install "actsim[mcp]"      # or: pip install -e ".[mcp]" from a checkout
actsim-mcp                     # stdio transport, the default for local clients
```

Other entry points:

```bash
python -m actsim_mcp --list-tools           # print the registered surface and exit
python -m actsim_mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

The MCP extra requires Python 3.10 or newer (the MCP SDK's own floor), even
though the core `actsim` package supports older versions.

### Client configuration

```json
{
  "mcpServers": {
    "actsim": {
      "command": "actsim-mcp",
      "env": { "ACTSIM_MCP_WORKSPACE": "/path/to/your/analysis/folder" }
    }
  }
}
```

`mcp/actsim_server.py` remains as a shim for configurations that point at the
old path; it forwards to this package.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `ACTSIM_MCP_WORKSPACE` | process CWD | Only directory the server reads data from or writes exports to. |
| `ACTSIM_MCP_DEFAULT_SIMULATIONS` | 10000 | Default trial count. |
| `ACTSIM_MCP_MAX_SIMULATIONS` | 1000000 | Upper bound on `num_simulations`. |
| `ACTSIM_MCP_MAX_INLINE_VALUES` | 100000 | Cap on inline numeric arrays. |
| `ACTSIM_MCP_MAX_POLICIES` | 100000 | Cap on a policy book. |
| `ACTSIM_MCP_MAX_ARTIFACTS` | 64 | LRU capacity of the artifact store. |
| `ACTSIM_MCP_MAX_ARTIFACT_CELLS` | 25000000 | Total stored cells before eviction. |
| `ACTSIM_MCP_MAX_PREVIEW_ROWS` | 200 | Upper bound on paginated previews. |
| `ACTSIM_MCP_DEFAULT_SEED` | 1 | Default random seed. |

---

## Design

### Artifact handles instead of data in context

Simulations and claim sets run to tens of thousands of rows. Returning them
through the model's context is slow, expensive and lossy. Every producing tool
stores its result server-side and returns a short handle (`sim_7f3a1c2b`) plus a
compact summary; downstream tools take the handle.

```
actsim_load_dataset      -> data_…
actsim_fit_distributions -> fit_…
actsim_simulate_aggregate-> sim_…   (+ events, when keep_events=true)
actsim_apply_layer       -> sim_…   (layered annual detail)
actsim_build_policy_book -> book_…
actsim_simulate_claims   -> clm_…
actsim_simulate_claim_development -> tri_…
```

The store is a bounded LRU keyed on cryptographically random ids, capped on both
entry count and total cells. Artifacts do not survive a restart — every
"artifact not found" error says so and names the live handles.

Inspect, page through and export artifacts with `actsim_list_artifacts`,
`actsim_describe_artifact`, `actsim_preview_artifact` and
`actsim_export_artifact`.

### Every tool answers in markdown or JSON

`response_format` defaults to `markdown`, which costs far fewer tokens than
pretty-printed JSON for the same table, and switches to `json` for callers that
post-process. Distribution shapes come back as text histograms rather than raw
draws.

### Errors are instructions

Failures raise the SDK's `ToolError` carrying a hint, so the message reaches the
model rather than being replaced with a generic crash notice:

```
Unknown distribution: 'loggnormal'. Did you mean: lognormal, normal?
Valid distributions: beta, exponential, gamma, logistic, lognormal, …

poisson takes 1 parameter(s) but 2 were given.
Expected poisson(lambda_) - for example poisson params=[1].
```

### Transport safety

actsim prints while it works: `DistributionFitter.fit` logs each candidate and
`@timing_decorator` prints a timing line from every simulation entry point. On a
stdio transport that text lands inside the JSON-RPC stream and breaks the
session. `actsim_mcp.runtime` redirects library stdout to stderr around every
call, pins matplotlib to the headless Agg backend before actsim is imported, and
moves logging off stdout. A regression test asserts nothing reaches stdout.

### Sandboxed file access

Every path is resolved inside `ACTSIM_MCP_WORKSPACE`; anything outside is
refused. Policy parameter cells are parsed with `ast.literal_eval` and a numeric
fallback, never `eval`, so a crafted CSV cannot execute code.

### Plots are images

`actsim_plot_fit`, `actsim_plot_simulation` and
`actsim_plot_frequency_severity` return real PNG `ImageContent`. They reuse
actsim's own plotting code by capturing the figure that `plt.show()` would have
displayed.

---

## Tool reference

### Data and artifacts
| Tool | Purpose |
| --- | --- |
| `actsim_load_dataset` | Load a numeric loss sample into a dataset artifact. |
| `actsim_inspect_file` | Preview a workspace file's columns and rows. |
| `actsim_list_artifacts` | List stored artifacts, optionally by kind. |
| `actsim_describe_artifact` | Metadata, columns and summary statistics. |
| `actsim_preview_artifact` | Paginated rows. |
| `actsim_export_artifact` | Write to CSV/JSON/Parquet in the workspace. |
| `actsim_delete_artifact` | Free memory. |

### Distributions
| Tool | Purpose |
| --- | --- |
| `actsim_list_distributions` | Catalog with actuarial parameter names and support. |
| `actsim_distribution_stats` | Moments, percentiles, pdf/cdf at points. |
| `actsim_distribution_quantile` | Analytic quantiles and return periods. |
| `actsim_sample_distribution` | Draw a sample into an artifact. |
| `actsim_mix_distributions` | Two-component mixture sampling. |

### Fitting
| Tool | Purpose |
| --- | --- |
| `actsim_fit_distributions` | Fit and rank candidates, with optional truncation. |
| `actsim_get_fit_summary` | Re-rank a stored fit by another metric. |
| `actsim_select_fit` | Override the selected distribution. |
| `actsim_sample_from_fit` | Sample, with optional zero/one inflation. |
| `actsim_fit_statistics` | Data versus fitted comparison. |
| `actsim_plot_fit` | Fitted densities over the data histogram (PNG). |

### Simulation
| Tool | Purpose |
| --- | --- |
| `actsim_simulate_aggregate` | Frequency/severity Monte Carlo, with copulas. |
| `actsim_analyze_simulation` | VaR / TVaR / AEP / OEP at chosen quantiles. |
| `actsim_apply_layer` | Per-occurrence and aggregate deductibles and limits. |
| `actsim_simulate_correlated_portfolio` | Correlated lines of business. |
| `actsim_plot_simulation` | Aggregate loss histogram (PNG). |
| `actsim_plot_frequency_severity` | Dependence scatter with realised correlation (PNG). |

### Claims and reserving
| Tool | Purpose |
| --- | --- |
| `actsim_build_policy_book` | Validated policy book from rows, a file, or synthesis. |
| `actsim_simulate_claims` | Individual claims per policy. |
| `actsim_assign_claim_dates` | NHPP occurrence dates within each policy window. |
| `actsim_simulate_claim_development` | Develop ultimates backwards using LDFs. |
| `actsim_build_loss_triangle` | Cumulative/incremental triangle and age-to-age factors. |

### Analytics and reporting
| Tool | Purpose |
| --- | --- |
| `actsim_risk_metrics` | VaR, TVaR, moments and return periods for any loss vector. |
| `actsim_stress_test` | Severity, frequency and additive stresses versus base. |
| `actsim_compare_simulations` | Side-by-side risk measures. |
| `actsim_generate_risk_report` | Markdown report assembled from artifacts. |
| `actsim_server_info` | Versions, workspace and limits. |

### Resources
`actsim://distributions`, `actsim://config`, `actsim://capabilities`,
`actsim://workflows`, `actsim://examples`, `actsim://artifacts`

### Prompts
`price_excess_layer`, `portfolio_capital_review`, `reserving_triangle_build`,
`dependence_sensitivity`, `distribution_selection_review`

---

## Parameterisation

actsim follows actuarial conventions, not scipy's. This is the most common
source of silently wrong answers, so `actsim_list_distributions` is worth
calling before choosing parameters.

| Distribution | Parameters | Note |
| --- | --- | --- |
| `lognormal` | `mu, sigma` | Mean and sd of the **log** scale. |
| `gamma` | `alpha, theta` | `theta` is a **scale**, not a rate. |
| `pareto` | `alpha, beta` | Bahnemann; maps to scipy `lomax`. |
| `weibull` | `delta, beta` | Bahnemann shape and scale. |
| `exponential` | `beta` | `beta` is the mean. |
| `uniform` | `a, b` | Bounds, not loc/scale. |
| `poisson` | `lambda_` | |
| `negative binomial` | `r, p` | Failures and success probability. |
| `normal`, `logistic`, `beta` | `mu, sigma` / `mu, s` / `alpha, beta` | |

---

## Worked example: pricing an excess layer

```
actsim_load_dataset(file_path="claims.csv", column="gross_loss")
  -> data_9c2f1a4b (842 observations)

actsim_fit_distributions(artifact_id="data_9c2f1a4b",
                         distributions=["lognormal","gamma","weibull","pareto"],
                         remove_values=[0], rank_by="aic")
  -> fit_5e81d0c7, selected lognormal(9.42, 1.13)

actsim_simulate_aggregate(freq_dist="poisson", freq_params=[18],
                          sev_dist="lognormal", sev_params=[9.42, 1.13],
                          num_simulations=50000, keep_events=True)
  -> sim_7f3a1c2b, mean 251,880, VaR(99.5%) 892,400

actsim_apply_layer(artifact_id="sim_7f3a1c2b",
                   per_occurrence_deductible=250000,
                   per_occurrence_limit=1000000,
                   aggregate_limit=3000000)
  -> expected ceded loss, share of ground-up, hit and exhaustion frequency

actsim_generate_risk_report(artifact_ids=["fit_5e81d0c7","sim_7f3a1c2b"])
```

---

## Known actsim library behaviour the server works around

These are characteristics of the installed `actsim` package, not of the server.
The server routes around each one rather than surfacing the failure:

1. `StochasticSimulator.analyze_results()` raises `UnboundLocalError` when the
   simulator was built with `keep_all=False`, because `quantile_values` and
   `tvar_values` are only bound on the `keep_all=True` branch. Risk measures are
   computed in `actsim_mcp.analytics` instead, which handles both modes and does
   not run its OEP index off the end of the array at q=1.
2. `DistributionFitter.select_best_fit` indexes each result by metric name, so a
   metric not present on the result rows (notably `ks_statistic`) raises a bare
   `KeyError`. Tools validate metric names against `aic`, `bic`,
   `log_likelihood`, `chisquare`, `ks` first.
3. `StochasticSimulator.results` raises `AttributeError` rather than a helpful
   error before a simulation has been run, since `_results` is not initialised in
   `__init__`. Tools never expose an un-run simulator.
4. `gen_agg_simulations` loops in Python, so runtime grows linearly with
   `num_simulations`; 10,000 trials is a good default and large runs should be
   asked for explicitly.
5. Event-level data only has rows for years that produced a claim. A plain
   groupby over it therefore drops claim-free years, and every per-year statistic
   built that way is biased. `actsim_apply_layer` reinstates those years as
   zeros, and OEP is computed over all simulated years rather than only the ones
   with events - on a poisson(0.5) book that difference nearly doubles OEP(90%).
6. `DistributionFitter.select_best_fit` takes the *minimum* of whatever metric it
   is given, which selects the worst model when ranking by log-likelihood. The
   server ranks and selects on its own, maximising log-likelihood and minimising
   the rest, and never selects a candidate whose metric came back non-finite.

---

## Tests

```bash
pip install -e ".[mcp,dev]"
pytest tests/test_actsim_mcp.py
```

The suite drives every tool through the SDK's own dispatcher, covering the happy
path, the error path, path-traversal refusal, parameter parsing safety, and the
guarantee that library output never reaches stdout.

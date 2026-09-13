# ---------------------------------------------
# Import required modules
# ---------------------------------------------
from actsim import load_config, DistributionFitter
from actstats import actuarial as act

# ---------------------------------------------
# 1. Generate Example Data
# ---------------------------------------------
# Severity data: Using lognormal distribution with mu=0.5 and sigma=0.2
sev_data = act.lognormal(0.5, 0.2).rvs(size=10000)

# Frequency data: Using Poisson distribution with λ=10
freq_data = act.poisson.rvs(10, 10000)

# ---------------------------------------------
# 2. Load Configuration
# ---------------------------------------------
# This loads distribution lists and metrics from the actsim config file
config = load_config()

# ---------------------------------------------
# 3. Fit Severity Distributions
# ---------------------------------------------
# Get severity distributions and metrics from config
distribution_names = config.distributions['severity']
metrics = config.metrics

# Initialize severity fitter
sev_fitter = DistributionFitter(sev_data, distributions=distribution_names, metrics=metrics)

# Perform fitting
sev_fitter.fit()

# View best fits and selected distribution
print("Best fits:", sev_fitter.best_fits)
print("Selected fit:", sev_fitter.selected_fit)
print("Selected distribution object:", sev_fitter.get_selected_dist())

# Manually selecting a distribution (example: 'uniform')
sev_fitter.select_distribution('uniform')
selected_fit = sev_fitter.selected_fit

# Print details of the selected fit
print("Selected fitting distribution:", selected_fit['name'])
print("Parameters:", selected_fit['params'])
print("AIC:", selected_fit['aic'])
print("BIC:", selected_fit['bic'])

# Calculate statistics for severity
sev_fitter.calculate_statistics()

# Plot predictions
sev_fitter.plot_predictions()

# Print summary report
sev_fitter.summary()

# ---------------------------------------------
# 4. Generate Samples from Severity Fit
# ---------------------------------------------
samples = sev_fitter.sample(size=10)
print("Generated samples:", samples)

# Generate mixed samples (e.g., weighted combinations)
samples = sev_fitter.sample_mixed(0.1, 0.1, size=10)
print("Generated samples:", samples)

# ---------------------------------------------
# 5. Fit Frequency Distributions
# ---------------------------------------------
distribution_names = config.distributions['frequency']
metrics = config.metrics

# Initialize frequency fitter
freq_fitter = DistributionFitter(freq_data, distributions=distribution_names, metrics=metrics)

# Show available frequency distributions
print("Frequency distributions:", freq_fitter.distributions)

# Perform fitting
freq_fitter.fit()

# View best fits and summary
print("Frequency best fits:", freq_fitter.best_fits)
print("Frequency selected fit:", freq_fitter.selected_fit)
freq_fitter.summary()

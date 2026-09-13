import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.copula.api import (
    GaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula)
from actsim.utils.utils import timing_decorator
from actstats import actuarial as act
# from functools import cached_property # consider adding cached property

class StochasticSimulator:
    _available_distributions = {
            'uniform': act.uniform,
            'normal': act.normal,
            'logistic': act.logistic,
            'exponential': act.exponential,
            'gamma': act.gamma,
            'beta': act.beta,
            'pareto': act.pareto,
            'poisson': act.poisson,
            'weibull': act.weibull,
            'lognormal': act.lognormal,
            'negative binomial': act.negative_binomial,
        }
    def __init__(self, freq_dist, freq_params, sev_dist, sev_params,
                 num_sim=10000, keep_all=False, seed=1, correlation=None, copula_type=None, theta = 0):

        self.frequency_dist = self._validate_distribution(freq_dist)
        self.frequency_params = freq_params
        self.severity_dist = self._validate_distribution(sev_dist)
        self.severity_params = sev_params
        self.num_simulations = num_sim
        self.seed = seed
        self.correlation = correlation
        self.copula_type = copula_type
        self.theta = theta
        self._keep_all = keep_all
        rng = np.random.default_rng(seed)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def _validate_distribution(cls, dist_name):
        if dist_name in cls._available_distributions:
            return cls._available_distributions[dist_name]
        raise ValueError(f"Invalid distribution: {dist_name}")
    
    def _read_corr_matrix(self, file_path):
        """
        Reads a correlation matrix from a CSV file and converts it into a NumPy array.
        :param file_path: Path to the correlation matrix CSV file
        :return: NumPy correlation matrix
        """
        df = pd.read_csv(file_path, index_col=0)  # Read CSV and set index to first column
        return df.to_numpy()  # Convert DataFrame to NumPy array

    def _read_dist_list(self, file_path):
        """
        Reads a JSON file containing distributions and returns a list of dictionaries.
        :param file_path: Path to the JSON file
        :return: List of distribution dictionaries
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def _gen_copula(self):
        ### Generate copula for frequency and severity
        if self.copula_type == 'gaussian':
            corr_matrix = np.array([[1, self.correlation], [self.correlation, 1]])
            copula = GaussianCopula(corr_matrix)
        elif self.copula_type == 'frank':
            copula = FrankCopula(theta=self.theta)
        elif self.copula_type == 'gumbel':
            copula = GumbelCopula(theta=self.theta)
        elif self.copula_type == 'clayton':
            copula = ClaytonCopula(theta=self.theta)

        u = copula.rvs(self.num_simulations)

        return u[:, 0], u[:, 1]

    def _gen_cd_corr_percentile(self, corr_matrix, n):
        self.corr_matrix = corr_matrix
        random_var = []
        C = self.corr_matrix

        # Cholesky decomposition
        L = np.linalg.cholesky(C)

        # Generate uncorrelated standard normal variables
        Z = np.random.randn(n, self.num_simulations)

        # Introduce correlation
        correlated_normals = L @ Z

        # Create marginal distribution percentiles, currently only the frozen class behave the same as scipy distributions
        for i in range(n):
            random_var.append(act.normal(0, 1).cdf(correlated_normals[i,:]))
        
        return random_var
    
    @timing_decorator
    def gen_multivariate_corr_simulations(self, corr_matrix_file, dist_list_file, gen_marginal=False):
        """
        Generate correlated random variables for multiple distributions using _gen_cd_corr_percentile.
        """
        # Read inputs from files
        corr_matrix = self._read_corr_matrix(corr_matrix_file)
        dist_list = self._read_dist_list(dist_list_file)

        num_lobs = len(dist_list)
        lob_names = [dist["dist_name"] for dist in dist_list]

        # Validate correlation matrix shape
        if corr_matrix.shape != (num_lobs, num_lobs):
            raise ValueError(
                f"Dimension mismatch: correlation matrix shape is {corr_matrix.shape}, "
                f"but {num_lobs} distributions were provided in the LoB file. "
                f"Expected shape: ({num_lobs}, {num_lobs})"
            )
        
        # Generate correlated percentiles
        correlated_percentiles = self._gen_cd_corr_percentile(corr_matrix, num_lobs)

        # Generate marginal losses for each LoB
        simulation_results = np.zeros((num_lobs, self.num_simulations))

        for i, dist in enumerate(dist_list):
            dist_type = dist["dist_type"].lower()
            params = dist["dist_param"]
            
            dist = self._validate_distribution(dist_type)
            try:
                # Transform percentiles to target distribution values
                simulation_results[i, :] = dist.ppf(correlated_percentiles[i], *params)
            except Exception as e:
                self.logger.error(f"Error processing {dist['dist_name']}: {e}")
                raise

        self._results = np.sum(simulation_results, axis=0)

        if gen_marginal:
            self._all_simulations_data = simulation_results

        return self._results
    
    @timing_decorator
    def gen_agg_simulations(self):
        results = []
        all_simulations_data = []  # Store data for the DataFrame
        event_id = 0  # Overall event counter

        def simulate_annual_losses(results, i, num_events, severity_samples):
            """ Helper function to record losses for a given year """
            # self.logger.info(f"Simulation {i+1}/{self.num_simulations}")
            nonlocal event_id
            if num_events > 0:
                total_loss = np.sum(severity_samples)
                if self._keep_all:
                    for yearly_event_id, severity in enumerate(severity_samples, start=1):
                        event_id += 1
                        all_simulations_data.append({
                            'year': i + 1,
                            'event_id': event_id,
                            'yearly_event_id': yearly_event_id,
                            'amount': severity
                        })
            else:
                total_loss = 0
            results.append(total_loss)
        
        # If correlation is introduced via copula
        if self.correlation is not None and self.copula_type is not None:
            u_freq, u_sev = self._gen_copula()

            num_events_array = self.frequency_dist.ppf(u_freq, *self.frequency_params).astype(int)

            for i in range(self.num_simulations):
                num_events = num_events_array[i]
                severity_samples = (self.severity_dist.ppf(np.random.uniform(size=num_events, low=u_sev[i], high=1), *self.severity_params)
                                    if num_events > 0 else []
                                    )
                simulate_annual_losses(results, i, num_events, severity_samples)

        if self.correlation is not None and self.copula_type is None:
            # Correlation matrix
            C = np.array([[1, self.correlation],
                        [self.correlation, 1]])
            random_var = self._gen_cd_corr_percentile(C, 2)

            freq_random_var, sev_random_var = random_var

            num_events_array = self.frequency_dist.ppf(freq_random_var, *self.frequency_params).astype(int)

            for i in range(self.num_simulations):
            # Get number of events from frequency distribution
                num_events = num_events_array[i]
                severity_samples = (
                    self.severity_dist.ppf(np.random.uniform(low=sev_random_var[i], high=1, size=num_events), *self.severity_params)
                    if num_events > 0 else []
                )
                simulate_annual_losses(results, i, num_events, severity_samples)
        
        # If no correlation is introduced
        if self.correlation is None and self.copula_type is None:

            num_events_array = self.frequency_dist(*self.frequency_params).np_rvs(size=self.num_simulations)

            for i in range(self.num_simulations):
                # Get number of events from frequency distribution
                num_events = num_events_array[i]
                severity_samples = (self.severity_dist(*self.severity_params).np_rvs(size=num_events)
                        if num_events > 0 else []
                        )
                simulate_annual_losses(results, i, num_events, severity_samples)

        self._results = results
        self._all_simulations_data = all_simulations_data
        return self._results
    
    @property
    def results(self):
        """Returns simulation results as a Pandas Series."""
        if self._results is None:
            raise ValueError("Simulation results not found. Please run gen_agg_simulations() first.")
        return pd.Series(self._results)
    
    @property
    def all_simulations(self):
        """Returns simulation results as a Pandas dataframe."""
        if self._all_simulations_data is None:
            raise ValueError("Simulation results not found. Please run gen_agg_simulations() first.")
        return pd.DataFrame(self._all_simulations_data)
    
    def calc_agg_percentile(self, pct = 95):
        if hasattr(self, 'results'):
            return np.percentile(self._results, pct)
        else:
            raise ValueError('simulation results not found')
        
    def plot_distribution(self, bins = None, log_option=False):
        if hasattr(self, 'results'):
            if bins is None:
                bins = int(np.sqrt(len(self._results)))

            plt.hist(self._results, bins=bins, density=True, alpha=0.5, color='g', log=log_option)
            plt.title('Distribution of Simulated Aggregate Losses')
            plt.xlabel('Aggregate Loss')
            plt.ylabel('Density')
            plt.show()
        else:
            raise ValueError('Simulation results not found')
        
    def plot_correlated_variables(self):
        if not hasattr(self, '_all_simulations_data') or self._all_simulations_data is None:
            raise ValueError("Simulation data not found. Please run gen_agg_simulations() with _keep_all=True first.")

        df = self.all_simulations

        # Aggregate severity data by year to get total severity per year
        yearly_severity = df.groupby('year')['amount'].mean().reset_index()

        # Aggregate frequency data by year to get number of events per year
        yearly_frequency = df.groupby('year')['yearly_event_id'].max().reset_index()

        # Merge severity and frequency data
        yearly_data = pd.merge(yearly_frequency, yearly_severity, on='year', how='inner')
        yearly_data.rename(columns={'yearly_event_id': 'event_count', 'amount': 'mean_severity'}, inplace=True)

        # Replace inf with NaN to avoid seaborn/pandas warning
        yearly_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        yearly_data.dropna(subset=['event_count', 'mean_severity'], inplace=True)

        # Plot the correlation between frequency and severity
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=yearly_data, 
                x='event_count', 
                y='mean_severity', 
                alpha=0.6, 
                color='blue', 
                label='Yearly Data')
        try:
            sns.kdeplot(
                data=yearly_data,
                x='event_count',
                y='mean_severity',
                levels=10,
                fill=True,
                alpha=0.2,
                color='red',
                warn_singular=False  # suppress warning for small datasets
            )
        except Exception as e:
            print(f"Warning during KDE plot: {e}")
        # Add labels and title
        plt.title('Correlation Between Frequency and Severity')
        plt.xlabel('Number of Events (Frequency)')
        plt.ylabel('Mean Severity')
        plt.grid(True)

        # Calculate and display the correlation coefficient
        correlation = yearly_data['event_count'].corr(yearly_data['mean_severity'])
        plt.text(0.95, 0.95, f'Correlation: {correlation:.2f}', 
                transform=plt.gca().transAxes, ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.8))

        plt.legend()
        plt.show()

    def analyze_results(self, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], **kwargs):
        """Compute quantiles, OEP, and AEP from simulation results."""
        all_simulations = kwargs.get('all_simulations', None)
        results = all_simulations.groupby('year')['amount'].sum().reset_index()['amount'] if all_simulations is not None else self.results
        
        # Compute Quantile (VaR), TVaR, and OEP (Occurrence Exceedance Probability)
        if self._keep_all:
            if all_simulations is not None:
                df = all_simulations
            else:
                df = self.all_simulations
            simulations_array = df['amount'].values
            quantile_values = {q: np.percentile(simulations_array, q * 100) for q in quantiles}
            tvar_values = {
                q: simulations_array[simulations_array > quantile_values[q]].mean() 
                for q in quantiles
            }            
            max_losses_per_year = df.groupby('year')['amount'].max().values
            max_losses_per_year.sort()
            oep_values = {q: max_losses_per_year[int((q) * len(max_losses_per_year))] for q in quantiles}
        else:
            oep_values = None  # OEP cannot be computed without individual event data
        
        # Compute AEP (Aggregate Exceedance Probability)
        aep_values = {q: np.percentile(results, q * 100) for q in quantiles}
        
        return pd.DataFrame({
            'VaR': quantile_values,
            'TVaR': tvar_values,
            'oep': oep_values,
            'aep': aep_values
        }).round(2)
    
    def apply_deductible_and_limit(self, per_occurrence_ded, per_occurrence_limit, agg_ded, agg_limit):
        """Apply per-occurrence and aggregate deductibles and limits to the simulated losses."""
        if not self._keep_all:
            raise ValueError("Simulation must be run with _keep_all=True to calculate gross losses.")
        
        df = self.all_simulations.copy()
        
        # Apply per-occurrence deductible and limit
        df['gross_loss'] = df['amount'].clip(lower=per_occurrence_ded) - per_occurrence_ded
        df['gross_loss'] = df['gross_loss'].clip(upper=per_occurrence_limit)
        
        # Aggregate per-year deductible and limit
        annual_gross = df.groupby('year')['gross_loss'].sum().reset_index()
        annual_gross['gross_loss'] = annual_gross['gross_loss'].clip(lower=agg_ded) - agg_ded
        annual_gross['gross_loss'] = annual_gross['gross_loss'].clip(upper=agg_limit)
        
        return annual_gross

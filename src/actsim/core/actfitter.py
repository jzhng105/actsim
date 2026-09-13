import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import wraps
import pandas as pd
from actstats import actuarial as act
from typing import Any, Iterable, Sequence
from numpy.typing import ArrayLike, NDArray


# Decorator to check if a distribution has been selected
def check_selected_dist(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.selected_fit is None:
            raise ValueError("No distribution has been selected yet. Use 'select_distribution' method first.")
        return func(self, *args, **kwargs)
    return wrapper

class DistributionFitter:
    def __init__(self, data: ArrayLike, distributions: Iterable[str] | None = None, metrics: Sequence[str] | None = None):
        self.data = self._to_array(data)
        self._length = len(data)
        self.available_distributions = {
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

        # Filter available distributions based on user inputs
        if distributions:
            self.distributions = {name: self.available_distributions[name] for name in distributions if name in self.available_distributions}
        else:
            self.distributions = self.available_distributions

        self.metrics = metrics if metrics else ['aic', 'bic']

        self.results = []
        self.best_fits = {} 
        self.statistics = {}
        self.selected_fit = None
    
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_array(data):
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError(...)
            data = data.iloc[:, 0]
        arr = pd.to_numeric(pd.Series(np.asarray(data).ravel()) if not isinstance(data, pd.Series) else data,
                            errors="coerce").to_numpy(dtype=float)
        return arr

    @staticmethod
    def _frozen(distribution: Any, params: Sequence[float]) -> Any:
        """Freeze a distribution with its fitted parameters."""
        return distribution(*params)
 
    @staticmethod
    def _is_discrete(frozen: Any) -> bool:
        return hasattr(frozen, "pmf")
 
    @classmethod
    def _density(cls, frozen: Any, x: ArrayLike) -> NDArray[np.float64]:
        """pmf for discrete distributions, pdf otherwise."""
        return np.asarray(frozen.pmf(x) if cls._is_discrete(frozen) else frozen.pdf(x), dtype=float)

    def truncate_data(self, remove_values=None, lower=None, upper=None, q_low=None, q_high=None, dropna=True, inplace=True):
        """
        Truncate / clean data before distribution fitting.

        Parameters
        ----------
        remove_values : scalar or list-like, optional
            Values to remove, e.g. 0, -999, or [0, -999].
        lower : float, optional
            Keep observations >= lower.
        upper : float, optional
            Keep observations <= upper.
        q_low : float, optional
            Lower quantile cutoff, e.g. 0.01.
        q_high : float, optional
            Upper quantile cutoff, e.g. 0.99.
        dropna : bool, default True
            Whether to remove NaN values.
        inplace : bool, default True
            If True, update self.data. Otherwise return truncated data.

        Returns
        -------
        data or self
            Returns self if inplace=True, otherwise returns truncated data.
        """
        s = pd.Series(self.data)

        mask = pd.Series(True, index=s.index)

        if dropna:
            mask &= s.notna()

        if remove_values is not None:
            if not isinstance(remove_values, (list, tuple, set)):
                remove_values = [remove_values]
            mask &= ~s.isin(remove_values)

        if q_low is not None:
            lower = s[mask].quantile(q_low)

        if q_high is not None:
            upper = s[mask].quantile(q_high)

        if lower is not None:
            mask &= s >= lower

        if upper is not None:
            mask &= s <= upper

        truncated = s.loc[mask].reset_index(drop=True)

        if inplace:
            self.data = truncated
            self._length = len(truncated)
            return self

        return truncated

    def fit(self):
        if self.data is None:
            raise ValueError("No data has been loaded. Use 'load_data' method first.")
        
        for name, distribution in self.distributions.items():
            try:
                params = distribution.fit(self.data)
                print(f"{name}: {params}")
                log_likelihood = self.compute_log_likelihood(distribution, params, self.data)
                aic = self.compute_aic(log_likelihood, len(params))
                bic = self.compute_bic(log_likelihood, len(params), len(self.data))
                chi_square = self.compute_chi_square(distribution, params, self.data)
                ks_statistic = self.compute_ks_statistic(distribution, params, self.data)

                result = {
                    'name': name,
                    'distribution': distribution,
                    'params': params,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'chisquare': chi_square,
                    'ks': ks_statistic
                }

                self.results.append(result)
            except Exception as e:
                print(f"Could not fit {name} distribution: {e}")

        self.select_best_fit()

    def select_best_fit(self):
        if not self.results:
            raise ValueError("No distributions have been fitted yet. Call the 'fit' method first.")

        best_fit = None
        for metric in self.metrics:
            best_fit = min(self.results, key=lambda x: x[metric])
            self.best_fits[metric] = best_fit
        
        self.selected_fit = self.best_fits['aic'] if 'aic' in self.metrics else self.best_fits[self.metrics[0]]  # Default selected fit best fit under AIC else select first metric 
    
    def get_best_fit(self, metric):
        """Get the best-fitting distribution for a specific metric."""
        return self.best_fits.get(metric, None)

    def select_distribution(self, name):
        # Next () retrieves the first result that meets the condition.
        match = next((result for result in self.results if result['name'] == name), None)
        if match is None:
            raise ValueError(f"No distribution named '{name}' found in the fitted results.")
        self.selected_fit = match

    @check_selected_dist
    def get_selected_dist(self):
        return self.selected_fit['distribution']
    
    @check_selected_dist
    def get_selected_params(self):
        return self.selected_fit['params']

    @check_selected_dist
    def predict(self, x):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        return distribution.pdf(x, *params)

    @check_selected_dist
    def sample(self, size=1):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        return distribution.rvs(*params, size=size)
    
    @check_selected_dist
    def sample_mixed(self, zero_prop=0, one_prop=0, size=1):
        distribution = self.selected_fit['distribution']
        params = self.selected_fit['params']
        num_0 = int(zero_prop * size)
        num_1 = int(one_prop * size)
        num_sample = size - num_0 - num_1
        mixed_sample = distribution.rvs(*params, size=num_sample)
        sample = np.concatenate((np.zeros(num_0), np.ones(num_1), mixed_sample))
        np.random.shuffle(sample)
        return pd.Series(sample)

    @check_selected_dist
    def calculate_statistics(self):
        # Data statistics
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        data_percentiles = np.percentile(self.data, [5, 25, 50, 75, 95])

        # Predicted statistics
        x_values = np.linspace(min(self.data), max(self.data), len(self.data))
        predicted_pdf = self.predict(x_values)
        predicted_mean = np.sum(x_values * predicted_pdf) / np.sum(predicted_pdf)
        predicted_std = np.sqrt(np.sum((x_values - predicted_mean)**2 * predicted_pdf) / np.sum(predicted_pdf))
        predicted_percentiles = np.percentile(predicted_pdf, [5, 25, 50, 75, 95])

        self.statistics =  {
            'data': {
                'mean': data_mean,
                'std': data_std,
                'percentiles': data_percentiles
            },
            'predicted': {
                'mean': predicted_mean,
                'std': predicted_std,
                'percentiles': predicted_percentiles
            }
        }

        return pd.DataFrame(self.statistics)

    def plot_predictions(self, distribution_names=None):
        """Plot the data and the PDFs of selected distributions."""
        if distribution_names is None:
            distribution_names = [result['name'] for result in self.results]

        # Get colors for the distributions
        colors = mpl.colormaps.get_cmap('tab10')
        
        x_values = np.linspace(min(self.data), max(self.data), 100)
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of the data
        plt.hist(self.data, bins=30, density=True, alpha=0.6, color='gray', label='Actual Data')

        # Plot the PDFs of the selected distributions
        for idx, name in enumerate(distribution_names):
            result = next((result for result in self.results if result['name'] == name), None)
            if result:
                pdf_values = result['distribution'].pdf(x_values, *result['params'])
                plt.plot(x_values, pdf_values, lw=2, label=f'{name} PDF', color=colors(idx))

        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('Fitted Distributions vs Actual Data')
        plt.legend()
        plt.show()


    def summary(self):
        if not self.results:
            raise ValueError("No distributions have been fitted yet. Call the 'fit' method first.")
        return pd.DataFrame(self.results)
    
    @staticmethod
    def _prepare(distribution: Any, params: Sequence[float], data: ArrayLike) -> tuple[Any, NDArray[np.float64]]:
        x = np.asarray(data, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("Data contains no valid value.")
        return distribution(*params), x
    
    @classmethod
    def compute_log_likelihood(cls, distribution: Any, params: Sequence[float], data: ArrayLike) -> float:
        frozen, x = cls._prepare(distribution, params, data)
        try:
            return float(np.sum(frozen.logpmf(x) if cls._is_discrete(frozen) else frozen.logpdf(x)))
        except (TypeError, AttributeError, FloatingPointError) as e:
            raise RuntimeError(f"Error computing log-likelihood for {distribution}: {e}") from e

    @staticmethod
    def compute_aic(log_likelihood, num_params) -> float:
        return 2 * num_params - 2 * log_likelihood

    @staticmethod
    def compute_bic(log_likelihood, num_params, n_samples) -> float:
        return np.log(n_samples) * num_params - 2 * log_likelihood
    
    @classmethod
    def compute_chi_square(cls, distribution: Any, params: Sequence[float], data: ArrayLike, 
                           bins: int = 10) -> float:
        """Chi-Square test"""
        if bins < 2:
            raise ValueError("bins must be at least 2.")
        frozen, x = cls._prepare(distribution, params, data)
        n = x.size
        interior = np.asarray(frozen.ppf(np.linspace(0, 1, bins + 1)[1:-1]), dtype=float)
        edges = np.unique(np.concatenate(([-np.inf], interior[np.isfinite(interior)], [np.inf])))
        if edges.size < 3:
            raise ValueError("Fitted distribution does not yield at least two usable bins.")

        if cls._is_discrete(frozen):
            # Bins are (edge_i, edge_{i+1}] on integer support: count values <= edge.
            observed = np.diff(np.searchsorted(np.sort(x), edges, side="right"))
        else:
            observed = np.histogram(x, bins=edges)[0]
        expected = n * np.diff(np.asarray(frozen.cdf(edges), dtype=float))

        if not np.all(np.isfinite(expected)) or np.any(expected <= 0):
            raise ValueError("Fitted distribution produced non-positive expected counts.")
        return float(np.sum((observed - expected) ** 2 / expected))
        
    @classmethod
    def compute_ks_statistic(cls, distribution: Any, params: Sequence[float], data: ArrayLike) -> float:
        """two-sided KS statistic"""
        frozen, x = cls._prepare(distribution, params, data)
        n = x.size
        values, counts = np.unique(x, return_counts=True)
        ecdf_upper = np.cumsum(counts) / n
        ecdf_lower = ecdf_upper - counts / n
        cdf = np.asarray(frozen.cdf(values), dtype=float)
        if not np.all(np.isfinite(cdf)):
            raise ValueError("Fitted distribution returned invalid CDF values.")
        return float(max(np.max(ecdf_upper - cdf), np.max(cdf - ecdf_lower)))


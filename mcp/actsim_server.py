#!/usr/bin/env python3
"""
Comprehensive Risk Analysis AI Agent using Model Context Protocol (MCP)
Integrates with the actsim package for comprehensive risk analysis and reporting.
Covers all methods and capabilities from the actsim package.
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import sys

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, Prompt, TextContent, ImageContent, EmbeddedResource,
    LoggingLevel
)
import mcp.types as types

# actrisk imports
try:
    from actrisk import (
        load_config, DistributionFitter, StochasticSimulator, 
        ClaimSimulator
    )
    from actstats import actuarial as act
    ACTRISK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: actrisk not found. Please install: pip install actrisk")
    print(f"Error: {e}")
    ACTRISK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskAnalysisConfig:
    """Configuration for risk analysis parameters"""
    confidence_levels: List[float] = None
    simulation_count: int = 10000
    random_seed: int = 42
    default_distributions: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [90, 95, 99, 99.5, 99.9]
        if self.default_distributions is None:
            self.default_distributions = {
                'severity': ['lognormal', 'gamma', 'weibull', 'pareto', 'uniform', 'exponential', 'beta'],
                'frequency': ['poisson', 'negative binomial']
            }

class ComprehensiveRiskAnalysisAgent:
    """Comprehensive risk analysis functionality using all actrisk capabilities"""
    
    def __init__(self, config: RiskAnalysisConfig = None):
        self.config = config or RiskAnalysisConfig()
        try:
            if ACTRISK_AVAILABLE:
                self.actrisk_config = load_config()
            else:
                self.actrisk_config = None
        except:
            logger.warning("Could not load actrisk config, using defaults")
            self.actrisk_config = None
        self.analysis_cache = {}
        
    # =================== DISTRIBUTION FITTING METHODS ===================
    
    def fit_distributions(self, data: List[float], dist_type: str, 
                         specific_distributions: List[str] = None,
                         specific_metrics: List[str] = None) -> Dict[str, Any]:
        """Enhanced distribution fitting with more options"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            if self.actrisk_config and not specific_distributions:
                distributions = self.actrisk_config.distributions.get(dist_type, 
                    self.config.default_distributions[dist_type])
                metrics = self.actrisk_config.metrics if not specific_metrics else specific_metrics
            else:
                distributions = specific_distributions or self.config.default_distributions[dist_type]
                metrics = specific_metrics or ['aic', 'bic', 'ks_statistic']
            
            data = np.array(data)
            fitter = DistributionFitter(data, distributions=distributions, metrics=metrics)
            fitter.fit()
            
            # Enhanced results
            results = {
                'best_fits': self._serialize_best_fits(fitter.best_fits),
                'selected_fit': self._serialize_selected_fit(fitter.selected_fit),
                'summary_stats': self._calculate_summary_stats(data),
                'fit_quality': self._assess_fit_quality(fitter),
                'available_distributions': list(fitter.distributions),
                'metrics_used': metrics
            }
            
            return results
        except Exception as e:
            logger.error(f"Error fitting distributions: {e}")
            return {'error': str(e)}
    
    def manual_distribution_selection(self, fitter_results: Dict[str, Any], 
                                    distribution_name: str) -> Dict[str, Any]:
        """Manually select a specific distribution from fitting results"""
        try:
            # This would require storing the fitter object, but we'll simulate the functionality
            return {
                'selected_distribution': distribution_name,
                'selection_method': 'manual',
                'message': f'Distribution {distribution_name} selected manually'
            }
        except Exception as e:
            logger.error(f"Error in manual selection: {e}")
            return {'error': str(e)}
    
    def calculate_distribution_statistics(self, distribution_name: str, 
                                        parameters: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a fitted distribution"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            # Use actstats to calculate distribution statistics
            dist_func = getattr(act, distribution_name, None)
            if not dist_func:
                return {'error': f'Distribution {distribution_name} not found'}
            
            dist_obj = dist_func(*parameters)
            
            stats = {}
            try:
                stats['mean'] = float(dist_obj.mean())
                stats['std'] = float(dist_obj.std())
                stats['variance'] = float(dist_obj.var())
                stats['median'] = float(dist_obj.median())
                
                # Calculate percentiles
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                stats['percentiles'] = {f'{p}%': float(dist_obj.ppf(p/100)) for p in percentiles}
                
            except Exception as stat_error:
                stats['error'] = f"Could not calculate all statistics: {stat_error}"
            
            return {
                'distribution': distribution_name,
                'parameters': parameters,
                'statistics': stats
            }
        except Exception as e:
            logger.error(f"Error calculating distribution statistics: {e}")
            return {'error': str(e)}
    
    def generate_samples_from_distribution(self, distribution_name: str, 
                                         parameters: List[float], 
                                         size: int = 100) -> Dict[str, Any]:
        """Generate samples from a specified distribution"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            dist_func = getattr(act, distribution_name, None)
            if not dist_func:
                return {'error': f'Distribution {distribution_name} not found'}
            
            dist_obj = dist_func(*parameters)
            samples = dist_obj.rvs(size=size).tolist()
            
            return {
                'distribution': distribution_name,
                'parameters': parameters,
                'sample_size': size,
                'sample_statistics': self._calculate_summary_stats(samples)
            }
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
            return {'error': str(e)}
    
    def generate_mixed_samples(self, dist1_name: str, dist1_params: List[float], 
                             dist2_name: str, dist2_params: List[float],
                             weight1: float, weight2: float, size: int = 100) -> Dict[str, Any]:
        """Generate mixed samples from two distributions"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            # Normalize weights
            total_weight = weight1 + weight2
            w1 = weight1 / total_weight
            w2 = weight2 / total_weight
            
            # Generate samples from each distribution
            dist1_func = getattr(act, dist1_name, None)
            dist2_func = getattr(act, dist2_name, None)
            
            if not dist1_func or not dist2_func:
                return {'error': 'One or both distributions not found'}
            
            dist1_obj = dist1_func(*dist1_params)
            dist2_obj = dist2_func(*dist2_params)
            
            # Generate mixed samples
            n1 = int(size * w1)
            n2 = size - n1
            
            samples1 = dist1_obj.rvs(size=n1)
            samples2 = dist2_obj.rvs(size=n2)
            
            mixed_samples = np.concatenate([samples1, samples2])
            np.random.shuffle(mixed_samples)
            
            return {
                'distribution_1': {'name': dist1_name, 'params': dist1_params, 'weight': w1},
                'distribution_2': {'name': dist2_name, 'params': dist2_params, 'weight': w2},
                'mixed_samples': mixed_samples.tolist(),
                'sample_statistics': self._calculate_summary_stats(mixed_samples.tolist())
            }
        except Exception as e:
            logger.error(f"Error generating mixed samples: {e}")
            return {'error': str(e)}
    
    # =================== STOCHASTIC SIMULATION METHODS ===================
    
    def create_stochastic_simulator(self, freq_dist: str, freq_params: tuple,
                                  sev_dist: str, sev_params: tuple,
                                  n_simulations: int = None,
                                  use_copula: bool = False,
                                  copula_type: str = None,
                                  copula_param: float = None,
                                  correlation_param: float = None) -> Dict[str, Any]:
        """Create and configure a stochastic simulator with advanced options"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            n_sims = n_simulations or self.config.simulation_count
            
            # Create simulator based on configuration
            if use_copula and copula_type and copula_param is not None:
                simulator = StochasticSimulator(
                    freq_dist, freq_params, sev_dist, sev_params,
                    n_sims, True, self.config.random_seed,
                    copula_param, copula_type, copula_param
                )
                config_type = 'copula'
            elif correlation_param is not None:
                simulator = StochasticSimulator(
                    freq_dist, freq_params, sev_dist, sev_params,
                    n_sims, True, self.config.random_seed, correlation_param
                )
                config_type = 'linear_correlation'
            else:
                simulator = StochasticSimulator(
                    freq_dist, freq_params, sev_dist, sev_params,
                    n_sims, True, self.config.random_seed
                )
                config_type = 'independent'
            
            # Store in cache for later use
            sim_id = f"sim_{hash(str(freq_dist + sev_dist + str(n_sims)))}"
            self.analysis_cache[sim_id] = simulator
            
            return {
                'simulator_id': sim_id,
                'configuration': {
                    'frequency': {'distribution': freq_dist, 'parameters': freq_params},
                    'severity': {'distribution': sev_dist, 'parameters': sev_params},
                    'simulation_count': n_sims,
                    'type': config_type,
                    'random_seed': self.config.random_seed
                }
            }
        except Exception as e:
            logger.error(f"Error creating simulator: {e}")
            return {'error': str(e)}
    
    def run_aggregate_simulations(self, simulator_id: str) -> Dict[str, Any]:
        """Run aggregate loss simulations"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            simulator.gen_agg_simulations()
            
            # Calculate comprehensive results
            results = {
                'simulator_id': simulator_id,
                'simulation_count': len(simulator.results),
                'mean_loss': float(simulator.results.mean()),
                'std_loss': float(simulator.results.std()),
                'min_loss': float(simulator.results.min()),
                'max_loss': float(simulator.results.max()),
                'percentiles': {}
                }
            
            # Calculate percentiles
            for conf_level in self.config.confidence_levels:
                percentile = simulator.calc_agg_percentile(conf_level)
                results['percentiles'][f'{conf_level}%'] = float(percentile)
            
            return results
        except Exception as e:
            logger.error(f"Error running simulations: {e}")
            return {'error': str(e)}
    
    def apply_deductibles_and_limits(self, simulator_id: str,
                                   per_occurrence_deductible: float = 0,
                                   per_occurrence_limit: float = None,
                                   aggregate_deductible: float = 0,
                                   aggregate_limit: float = None) -> Dict[str, Any]:
        """Apply insurance policy terms (deductibles and limits)"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            
            # Apply deductibles and limits
            gross_loss = simulator.apply_deductible_and_limit(
                per_occurrence_deductible or 0,
                per_occurrence_limit or float('inf'),
                aggregate_deductible or 0,
                aggregate_limit or float('inf')
            )
            
            # Process results
            gross_loss['amount'] = gross_loss['gross_loss']
            
            return {
                'processed_losses': gross_loss.to_dict('records'),
                'summary_statistics': gross_loss.describe().to_dict(),
                'policy_terms': {
                    'per_occurrence_deductible': per_occurrence_deductible,
                    'per_occurrence_limit': per_occurrence_limit,
                    'aggregate_deductible': aggregate_deductible,
                    'aggregate_limit': aggregate_limit
                }
            }
        except Exception as e:
            logger.error(f"Error applying policy terms: {e}")
            return {'error': str(e)}
    
    def analyze_simulation_results(self, simulator_id: str, 
                                 custom_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive analysis of simulation results"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            if not hasattr(simulator , "_results"):
                simulator.gen_agg_simulations()

            # Use custom data if provided, otherwise use simulator results
            if custom_data:
                data = pd.DataFrame(custom_data)
                risk_measure_table = simulator.analyze_results(all_simulations=data)
            else:
                
                risk_measure_table = simulator.analyze_results()
            
            return {
                'analysis_completed': True,
                'risk measures': risk_measure_table.to_json(orient="split"),
                'mean_loss': float(simulator.results.mean()),
                'std_loss': float(simulator.results.std()),
                'shape_statistics': {
                    'skewness': float(self._calculate_skewness(simulator.results.values)),
                    'kurtosis': float(self._calculate_kurtosis(simulator.results.values))
                },
                'results': 'Analysis completed - check simulator plots and diagnostics'
            }
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {'error': str(e)}
    
    def plot_simulation_distribution(self, simulator_id: str) -> Dict[str, Any]:
        """Generate distribution plots for simulation results"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            
            # This would generate plots - in practice you'd return plot data or save images
            try:
                simulator.plot_distribution()
                return {'plot_generated': True, 'message': 'Distribution plot created'}
            except:
                return {'plot_generated': False, 'message': 'Plot generation not available in current environment'}
                
        except Exception as e:
            logger.error(f"Error plotting distribution: {e}")
            return {'error': str(e)}
    
    def plot_correlated_variables(self, simulator_id: str) -> Dict[str, Any]:
        """Plot correlation structure for copula-based simulations"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            
            try:
                simulator.plot_correlated_variables()
                return {'plot_generated': True, 'message': 'Correlation plot created'}
            except:
                return {'plot_generated': False, 'message': 'Correlation plot not available or not applicable'}
                
        except Exception as e:
            logger.error(f"Error plotting correlations: {e}")
            return {'error': str(e)}
    
    # =================== MULTIVARIATE CORRELATION METHODS ===================
    

    def simulate_multivariate_correlated_risks(self, correlation_matrix_data: str,
                                            distribution_list_data: str,
                                            n_simulations: int = None) -> Dict[str, Any]:
        """Enhanced multivariate correlation simulation"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            n_sims = n_simulations or self.config.simulation_count
            
            # Parse correlation matrix
            corr_df = pd.read_csv(io.StringIO(correlation_matrix_data))
            
            # Parse distribution list
            dist_list = json.loads(distribution_list_data)
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                corr_df.to_csv(f.name, index=False)
                corr_file = f.name
                
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dist_list, f)
                dist_file = f.name
            
            try:
                # Create simulator for multivariate analysis
                # Use dummy parameters since we'll override with multivariate generation
                simulator = StochasticSimulator(
                    "normal", [1, 0], "normal", [1, 0], 
                    n_sims, True, self.config.random_seed
                )
                
                # Generate multivariate correlated simulations
                # This returns the aggregate results and stores marginal data in _all_simulations_data
                aggregate_results = simulator.gen_multivariate_corr_simulations(corr_file, dist_file, True)
                
                # Get the marginal simulation data (num_lobs x num_simulations)
                marginal_data = simulator._all_simulations_data
                
                # Convert to DataFrame with proper column names
                lob_names = [dist["dist_name"] for dist in dist_list]
                data = pd.DataFrame(marginal_data.T, columns=lob_names)  # Transpose to get simulations as rows
                
                # Calculate correlation of simulated data
                correlation_result = data.corr()
                
                # Calculate enhanced risk measures
                risk_measures = self._calculate_comprehensive_multivariate_risk_measures(data)
                
                # Add aggregate results to the output
                aggregate_stats = pd.Series(aggregate_results).describe().to_dict()
                
                return {
                    'input_correlations': corr_df.to_dict(),
                    'simulated_correlations': correlation_result.to_dict(),
                    'marginal_summary_statistics': data.describe().to_dict(),
                    'aggregate_summary_statistics': aggregate_stats,
                    'risk_measures': risk_measures,
                    'distribution_specifications': dist_list,
                    'lob_names': lob_names
                }
                
            finally:
                # Clean up temporary files
                os.unlink(corr_file)
                os.unlink(dist_file)
                
        except Exception as e:
            logger.error(f"Error in multivariate simulation: {e}")
            return {'error': str(e)}
    
    # =================== CLAIM SIMULATION METHODS ===================
    
    def create_synthetic_policies(self, n_policies: int,
                                freq_dist_options: List[str] = None,
                                sev_dist_options: List[str] = None,
                                start_date: str = "2023-01-01",
                                end_date: str = "2023-12-31") -> Dict[str, Any]:
        """Generate synthetic policy data for claim simulation"""
        try:
            freq_dists = freq_dist_options or ['poisson']
            sev_dists = sev_dist_options or ['lognormal']
            
            # Generate random policy characteristics
            np.random.seed(self.config.random_seed)
            
            policies_data = []
            for i in range(1, n_policies + 1):
                freq_dist = np.random.choice(freq_dists)
                sev_dist = np.random.choice(sev_dists)
                
                # Generate parameters based on distribution type
                if freq_dist == 'poisson':
                    freq_params = (np.random.uniform(0.5, 2.0),)
                elif freq_dist == 'negative binomial':
                    freq_params = (np.random.randint(1, 10), np.random.uniform(0.3, 0.8))
                else:
                    freq_params = (1.0,)  # Default
                
                if sev_dist == 'lognormal':
                    sev_params = (np.random.uniform(8, 12), np.random.uniform(0.3, 1.0))
                elif sev_dist == 'gamma':
                    sev_params = (np.random.uniform(1, 5), np.random.uniform(0.5, 2.0))
                else:
                    sev_params = (1000.0, 500.0)  # Default
                
                policies_data.append({
                    'policy_id': i,
                    'freq_dist': freq_dist,
                    'freq_params': freq_params,
                    'sev_dist': sev_dist,
                    'sev_params': sev_params,
                    'start_date': start_date,
                    'end_date': end_date
                })
            
            policies_df = pd.DataFrame(policies_data)
            
            return {
                'summary': {
                    'total_policies': n_policies,
                    'freq_distributions_used': list(set([p['freq_dist'] for p in policies_data])),
                    'sev_distributions_used': list(set([p['sev_dist'] for p in policies_data])),
                    'policy_period': f"{start_date} to {end_date}"
                }
            }
        except Exception as e:
            logger.error(f"Error creating synthetic policies: {e}")
            return {'error': str(e)}
    
    def simulate_claims(self, policies_data: str) -> Dict[str, Any]:
        """Simulate claims based on policy data"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            # Parse policies data
            if isinstance(policies_data, str):
                policies_df = pd.read_csv(io.StringIO(policies_data))
            else:
                policies_df = pd.DataFrame(policies_data)
            
            # Convert tuple strings back to tuples if needed
            if 'freq_params' in policies_df.columns and isinstance(policies_df['freq_params'].iloc[0], str):
                policies_df['freq_params'] = policies_df['freq_params'].apply(eval)
            if 'sev_params' in policies_df.columns and isinstance(policies_df['sev_params'].iloc[0], str):
                policies_df['sev_params'] = policies_df['sev_params'].apply(eval)
            
            # Create claim simulator
            claim_sim = ClaimSimulator(policies_df, self.config.random_seed)
            
            # Run claim simulation
            claim_sim.simulate_claims()
            
            # Store in cache for further processing
            sim_id = f"claim_sim_{hash(str(policies_df.shape[0]))}"
            self.analysis_cache[sim_id] = claim_sim
            
            return {
                'simulator_id': sim_id,
                'total_claims': len(claim_sim.claim_data) if hasattr(claim_sim, 'claim_data') else 0,
                'policies_processed': len(claim_sim.policies),
            }
            
        except Exception as e:
            logger.error(f"Error in claim simulation: {e}")
            return {'error': str(e)}
    
    def simulate_claim_dates_nhpp(self, simulator_id: str,
                                lambda0: float = 10,
                                alpha: float = 0.5,
                                phase: float = 0,
                                T: float = 1) -> Dict[str, Any]:
        """Simulate claim occurrence dates using Non-Homogeneous Poisson Process"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Claim simulator not found'}
                
            claim_sim = self.analysis_cache[simulator_id]
            
            # Simulate dates using NHPP
            claim_sim.simulate_dates_nhpp(lambda0, alpha, phase, T)
            
            return {
                'nhpp_parameters': {
                    'baseline_intensity': lambda0,
                    'seasonality_amplitude': alpha,
                    'phase_shift': phase,
                    'exposure_duration': T
                },
                'dates_simulated': True,
                'message': 'Claim occurrence dates simulated using NHPP'
            }
        except Exception as e:
            logger.error(f"Error simulating claim dates: {e}")
            return {'error': str(e)}
    
    def apply_shifted_dates(self, simulator_id: str, start_year: int = 2023) -> Dict[str, Any]:
        """Apply date shifts to align with calendar year"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Claim simulator not found'}
                
            claim_sim = self.analysis_cache[simulator_id]
            
            # Apply date shifts
            claim_sim.apply_shifted_dates(start_year)
            
            return {
                'start_year': start_year,
                'dates_shifted': True,
                'message': f'Claim dates aligned to calendar year starting {start_year}'
            }
        except Exception as e:
            logger.error(f"Error applying date shifts: {e}")
            return {'error': str(e)}
    
    def simulate_claim_development(self, simulator_id: str,
                                 base_ldfs: Dict[int, float] = None,
                                 volatility: float = 0.1,
                                 tail_factor: float = 1.0) -> Dict[str, Any]:
        """Simulate claim development triangles"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Claim simulator not found'}
                
            claim_sim = self.analysis_cache[simulator_id]
            
            # Default LDFs if not provided
            if base_ldfs is None:
                base_ldfs = {0: 2.0, 3: 1.5, 6: 1.2, 9: 1.1, 12: 1.05, 15: 1.02, 18: 1.00}
            
            # Simulate development
            claim_sim.simulate_claim_development(base_ldfs, volatility, tail_factor)
            
            # Get development data
            development_data = claim_sim.claim_development if hasattr(claim_sim, 'claim_development') else None
            
            return {
                'base_ldfs': base_ldfs,
                'volatility': volatility,
                'tail_factor': tail_factor,
                'development_simulated': True,
                'development_data': development_data.to_dict('records') if development_data is not None else None,
                'development_summary': development_data.describe().to_dict() if development_data is not None else 'No development data available'
            }
        except Exception as e:
            logger.error(f"Error in claim development simulation: {e}")
            return {'error': str(e)}
    
    def save_claim_development(self, simulator_id: str, file_path: str = None) -> Dict[str, Any]:
        """Save claim development data to file"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Claim simulator not found'}
                
            claim_sim = self.analysis_cache[simulator_id]
            
            if file_path is None:
                # Create temporary file path
                file_path = f"claim_development_{simulator_id}.csv"
            
            # Save development data
            claim_sim.save_claim_development(file_path)
            
            return {
                'file_saved': True,
                'file_path': file_path,
                'message': 'Claim development data saved successfully'
            }
        except Exception as e:
            logger.error(f"Error saving claim development: {e}")
            return {'error': str(e)}
    
    # =================== ADVANCED ANALYTICS METHODS ===================
    
    def calculate_quantile(self, distribution: str, parameters: List[float], 
                          quantile: float) -> Dict[str, Any]:
        """Calculate specific quantile for a distribution"""
        try:
            if not ACTRISK_AVAILABLE:
                return {'error': 'actrisk package not available'}
                
            dist_func = getattr(act, distribution, None)
            if not dist_func:
                return {'error': f'Distribution {distribution} not found'}
            
            dist_obj = dist_func(*parameters)
            quantile_value = float(dist_obj.ppf(quantile))
            
            return {
                'distribution': distribution,
                'parameters': parameters,
                'quantile': quantile,
                'quantile_value': quantile_value
            }
        except Exception as e:
            logger.error(f"Error calculating quantile: {e}")
            return {'error': str(e)}
    
    def perform_stress_testing(self, simulator_id: str, 
                             stress_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform stress testing on simulation results"""
        try:
            if simulator_id not in self.analysis_cache:
                return {'error': 'Simulator not found'}
                
            simulator = self.analysis_cache[simulator_id]
            base_results = simulator.results
            
            stress_results = []
            for i, scenario in enumerate(stress_scenarios):
                scenario_name = scenario.get('name', f'Scenario_{i+1}')
                multiplier = scenario.get('multiplier', 1.0)
                shift = scenario.get('shift', 0.0)
                
                # Apply stress to results
                stressed_losses = base_results * multiplier + shift
                
                stressed_stats = {
                    'scenario_name': scenario_name,
                    'parameters': scenario,
                    'mean_loss': float(stressed_losses.mean()),
                    'std_loss': float(stressed_losses.std()),
                    'percentiles': {}
                }
                
                # Calculate stressed percentiles
                for conf_level in self.config.confidence_levels:
                    percentile_val = np.percentile(stressed_losses, conf_level)
                    stressed_stats['percentiles'][f'{conf_level}%'] = float(percentile_val)
                
                stress_results.append(stressed_stats)
            
            return {
                'base_scenario': {
                    'mean_loss': float(base_results.mean()),
                    'std_loss': float(base_results.std())
                },
                'stress_scenarios': stress_results
            }
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {'error': str(e)}
    
    def calculate_risk_metrics(self, data: List[float], 
                             confidence_levels: List[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            data_array = np.array(data)
            conf_levels = confidence_levels or self.config.confidence_levels
            
            metrics = {
                'basic_statistics': self._calculate_summary_stats(data),
                'value_at_risk': {},
                'expected_shortfall': {},
                'risk_ratios': {}
            }
            
            # Calculate VaR and ES for each confidence level
            for conf in conf_levels:
                var_value = np.percentile(data_array, conf)
                
                # Expected Shortfall (Conditional VaR)
                tail_losses = data_array[data_array >= var_value]
                es_value = np.mean(tail_losses) if len(tail_losses) > 0 else var_value
                
                metrics['value_at_risk'][f'{conf}%'] = float(var_value)
                metrics['expected_shortfall'][f'{conf}%'] = float(es_value)
            
            # Risk ratios
            mean_val = np.mean(data_array)
            if mean_val > 0:
                metrics['risk_ratios']['coefficient_of_variation'] = float(np.std(data_array) / mean_val)
                
            return metrics
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_risk_report(self, analysis_results: Dict[str, Any], 
                                         report_type: str = 'comprehensive',
                                         include_recommendations: bool = True) -> str:
        """Generate enhanced risk analysis report"""
        try:
            report_sections = []
            
            # Header with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_sections.extend([
                "# Comprehensive Risk Analysis Report",
                f"**Generated:** {timestamp}",
                f"**Report Type:** {report_type.title()}",
                f"**actrisk Package:** {'Available' if ACTRISK_AVAILABLE else 'Not Available'}",
                "",
                "---",
                ""
            ])
            
            # Executive Summary
            if any(key in analysis_results for key in ['monte_carlo', 'aggregate_simulation']):
                mc_key = 'monte_carlo' if 'monte_carlo' in analysis_results else 'aggregate_simulation'
                mc_results = analysis_results[mc_key]
                
                report_sections.extend([
                    "## Executive Summary",
                    "",
                    f"- **Mean Expected Loss:** ${mc_results.get('mean_loss', 0):,.2f}",
                    f"- **Standard Deviation:** ${mc_results.get('std_loss', 0):,.2f}",
                    f"- **Coefficient of Variation:** {(mc_results.get('std_loss', 0) / max(mc_results.get('mean_loss', 1), 1)):.3f}",
                    f"- **99.5% Value at Risk:** ${mc_results.get('percentiles', {}).get('99.5%', 0):,.2f}",
                    f"- **Maximum Simulated Loss:** ${mc_results.get('max_loss', 0):,.2f}",
                    ""
                ])
            
            # Distribution Analysis
            if 'distribution_fitting' in analysis_results:
                dist_results = analysis_results['distribution_fitting']
                selected_fit = dist_results.get('selected_fit', {})
                
                report_sections.extend([
                    "## Distribution Analysis",
                    "",
                    f"**Selected Distribution:** {selected_fit.get('name', 'N/A')}",
                    f"**Parameters:** {selected_fit.get('params', 'N/A')}",
                    f"**AIC Score:** {selected_fit.get('aic', 'N/A')}",
                    f"**BIC Score:** {selected_fit.get('bic', 'N/A')}",
                    ""
                ])
                
                # Add fit quality assessment
                fit_quality = dist_results.get('fit_quality', {})
                if fit_quality:
                    report_sections.extend([
                        "### Fit Quality Assessment",
                        ""
                    ])
                    for metric, value in fit_quality.items():
                        if isinstance(value, (int, float)):
                            report_sections.append(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}")
                        else:
                            report_sections.append(f"- **{metric.replace('_', ' ').title()}:** {value}")
                    report_sections.append("")
            
            # Risk Metrics Section
            if any(key in analysis_results for key in ['monte_carlo', 'aggregate_simulation', 'risk_metrics']):
                report_sections.extend([
                    "## Risk Metrics Analysis",
                    ""
                ])
                
                # VaR Analysis
                if 'monte_carlo' in analysis_results or 'aggregate_simulation' in analysis_results:
                    key = 'monte_carlo' if 'monte_carlo' in analysis_results else 'aggregate_simulation'
                    percentiles = analysis_results[key].get('percentiles', {})
                    
                    if percentiles:
                        report_sections.extend([
                            "### Value at Risk (VaR) Analysis",
                            ""
                        ])
                        for percentile, value in sorted(percentiles.items()):
                            report_sections.append(f"- **{percentile} VaR:** ${value:,.2f}")
                        report_sections.append("")
                
                # Additional risk metrics
                if 'risk_metrics' in analysis_results:
                    risk_data = analysis_results['risk_metrics']
                    if 'expected_shortfall' in risk_data:
                        report_sections.extend([
                            "### Expected Shortfall (Conditional VaR)",
                            ""
                        ])
                        for level, value in risk_data['expected_shortfall'].items():
                            report_sections.append(f"- **{level} ES:** ${value:,.2f}")
                        report_sections.append("")
            
            # Simulation Details
            if 'aggregate_simulation' in analysis_results:
                sim_data = analysis_results['aggregate_simulation']
                report_sections.extend([
                    "## Simulation Configuration",
                    "",
                    f"- **Simulation Count:** {sim_data.get('simulation_count', 'N/A'):,}",
                    f"- **Minimum Loss:** ${sim_data.get('min_loss', 0):,.2f}",
                    f"- **Maximum Loss:** ${sim_data.get('max_loss', 0):,.2f}",
                    ""
                ])
            
            # Multivariate Analysis
            if 'multivariate' in analysis_results:
                mv_results = analysis_results['multivariate']
                report_sections.extend([
                    "## Multivariate Risk Analysis",
                    "",
                    "### Portfolio Diversification",
                    ""
                ])
                
                risk_measures = mv_results.get('risk_measures', {})
                if 'diversification_ratio' in risk_measures:
                    report_sections.append(f"- **Diversification Ratio:** {risk_measures['diversification_ratio']:.3f}")
                if 'portfolio_variance' in risk_measures:
                    report_sections.append(f"- **Portfolio Variance:** {risk_measures['portfolio_variance']:,.2f}")
                
                correlation_summary = risk_measures.get('correlation_summary', {})
                if correlation_summary:
                    report_sections.extend([
                        "",
                        "### Correlation Analysis",
                        f"- **Mean Correlation:** {correlation_summary.get('mean_correlation', 0):.3f}",
                        f"- **Maximum Correlation:** {correlation_summary.get('max_correlation', 0):.3f}",
                        f"- **Minimum Correlation:** {correlation_summary.get('min_correlation', 0):.3f}",
                    ])
                report_sections.append("")
            
            # Claims Analysis
            if 'claims' in analysis_results:
                claims_results = analysis_results['claims']
                report_sections.extend([
                    "## Claims Development Analysis",
                    "",
                    f"- **Total Claims Simulated:** {claims_results.get('total_claims', 0):,}",
                    f"- **Policies Processed:** {claims_results.get('policies_processed', 0):,}",
                    ""
                ])
                
                if 'development_summary' in claims_results and isinstance(claims_results['development_summary'], dict):
                    report_sections.extend([
                        "### Development Pattern Summary",
                        ""
                    ])
                    dev_summary = claims_results['development_summary']
                    for stat, values in dev_summary.items():
                        if isinstance(values, dict) and 'mean' in values:
                            report_sections.append(f"- **{stat.title()} Mean:** {values['mean']:.2f}")
            
            # Stress Testing Results
            if 'stress_testing' in analysis_results:
                stress_results = analysis_results['stress_testing']
                report_sections.extend([
                    "## Stress Testing Analysis",
                    ""
                ])
                
                base_scenario = stress_results.get('base_scenario', {})
                if base_scenario:
                    report_sections.extend([
                        f"**Base Scenario Mean Loss:** ${base_scenario.get('mean_loss', 0):,.2f}",
                        "",
                        "### Stress Scenarios",
                        ""
                    ])
                
                for scenario in stress_results.get('stress_scenarios', []):
                    report_sections.extend([
                        f"#### {scenario.get('scenario_name', 'Unnamed Scenario')}",
                        f"- **Stressed Mean Loss:** ${scenario.get('mean_loss', 0):,.2f}",
                        f"- **99.5% VaR:** ${scenario.get('percentiles', {}).get('99.5%', 0):,.2f}",
                        ""
                    ])
            
            # Recommendations Section
            if include_recommendations:
                report_sections.extend([
                    "## Risk Management Recommendations",
                    ""
                ])
                
                # Generate specific recommendations based on analysis results
                recommendations = self._generate_risk_recommendations(analysis_results)
                for rec in recommendations:
                    report_sections.append(f"- {rec}")
                report_sections.append("")
            
            # Technical Notes
            if report_type == 'comprehensive':
                report_sections.extend([
                    "## Technical Notes",
                    "",
                    "### Methodology",
                    "- Monte Carlo simulation techniques used for aggregate loss modeling",
                    "- Distribution fitting based on maximum likelihood estimation",
                    "- Model selection criteria: AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)",
                    "- Risk metrics calculated using empirical percentiles from simulation results",
                    "",
                    "### Assumptions and Limitations",
                    "- Historical data patterns assumed to continue in the future",
                    "- Independence assumption may not reflect actual risk correlations",
                    "- Model uncertainty not explicitly quantified in results",
                    "- Extreme tail events may be underestimated with limited simulation counts",
                    ""
                ])
            
            # Footer
            report_sections.extend([
                "---",
                f"*Report generated using actrisk package v{'Available' if ACTRISK_AVAILABLE else 'Not Available'}*",
                f"*Generation time: {timestamp}*"
            ])
            
            return "\n".join(report_sections)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return f"Error generating report: {e}"
    
    def _generate_risk_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate specific risk management recommendations based on analysis"""
        recommendations = []
        
        try:
            # Check for high tail risk
            if 'monte_carlo' in analysis_results or 'aggregate_simulation' in analysis_results:
                key = 'monte_carlo' if 'monte_carlo' in analysis_results else 'aggregate_simulation'
                results = analysis_results[key]
                
                mean_loss = results.get('mean_loss', 0)
                var_99_5 = results.get('percentiles', {}).get('99.5%', 0)
                
                if var_99_5 > mean_loss * 10:
                    recommendations.append("High tail risk detected - consider catastrophic risk management strategies")
                
                cv = results.get('std_loss', 0) / max(mean_loss, 1)
                if cv > 2.0:
                    recommendations.append("High volatility observed - implement risk diversification measures")
            
            # Check correlation analysis
            if 'multivariate' in analysis_results:
                mv_results = analysis_results['multivariate']
                risk_measures = mv_results.get('risk_measures', {})
                corr_summary = risk_measures.get('correlation_summary', {})
                
                mean_corr = corr_summary.get('mean_correlation', 0)
                if mean_corr > 0.7:
                    recommendations.append("High correlation detected between risk factors - review diversification strategy")
                
                div_ratio = risk_measures.get('diversification_ratio', 1.0)
                if div_ratio < 0.5:
                    recommendations.append("Limited diversification benefit - consider additional uncorrelated risk factors")
            
            # Claims-based recommendations
            if 'claims' in analysis_results:
                recommendations.append("Implement regular claims development monitoring and reserve adequacy testing")
            
            # Default recommendations
            recommendations.extend([
                "Regular model validation and backtesting recommended",
                "Monitor concentration risk and implement appropriate limits",
                "Consider economic capital allocation based on risk-adjusted metrics",
                "Implement early warning indicators for emerging risks"
            ])
            
        except Exception as e:
            logger.warning(f"Error generating specific recommendations: {e}")
            recommendations = [
                "Monitor tail risk exposures regularly",
                "Consider diversification strategies for correlated risks", 
                "Implement appropriate risk limits based on percentile analysis",
                "Regular model validation and backtesting recommended"
            ]
        
        return recommendations
    
    # =================== UTILITY AND HELPER METHODS ===================
    
    def _calculate_comprehensive_multivariate_risk_measures(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced multivariate risk measures calculation"""
        try:
            # Portfolio-level measures
            portfolio_returns = data.sum(axis=1)
            individual_stds = data.std()
            
            risk_measures = {
                'portfolio_variance': float(np.var(portfolio_returns)),
                'portfolio_std': float(np.std(portfolio_returns)),
                'diversification_ratio': float(portfolio_returns.std() / individual_stds.sum()),
                'concentration_ratio': float(individual_stds.max() / individual_stds.sum()),
            }
            
            # Correlation analysis
            corr_matrix = data.corr()
            upper_tri_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            risk_measures['correlation_summary'] = {
                'mean_correlation': float(np.mean(upper_tri_corrs)),
                'max_correlation': float(np.max(upper_tri_corrs)),
                'min_correlation': float(np.min(upper_tri_corrs)),
                'correlation_std': float(np.std(upper_tri_corrs))
            }
            
            # Risk contribution measures
            portfolio_var = np.var(portfolio_returns)
            marginal_contribs = []
            
            for col in data.columns:
                # Marginal contribution to portfolio variance
                other_cols = [c for c in data.columns if c != col]
                portfolio_without = data[other_cols].sum(axis=1)
                marginal_var = np.var(portfolio_returns) - np.var(portfolio_without)
                marginal_contribs.append(marginal_var)
            
            risk_measures['marginal_contributions'] = {
                col: float(contrib) for col, contrib in zip(data.columns, marginal_contribs)
            }
            
            return risk_measures
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive multivariate risk measures: {e}")
            return {'error': str(e)}
    
    def _calculate_summary_stats(self, data: List[float]) -> Dict[str, float]:
        """Enhanced summary statistics calculation"""
        arr = np.array(data)
        
        try:
            return {
                'count': len(arr),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'variance': float(np.var(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'q1': float(np.percentile(arr, 25)),
                'q3': float(np.percentile(arr, 75)),
                'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                'skewness': float(self._calculate_skewness(arr)),
                'kurtosis': float(self._calculate_kurtosis(arr)),
                'coefficient_of_variation': float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error calculating some statistics: {e}")
            return {
                'count': len(arr),
                'mean': float(np.mean(arr)) if len(arr) > 0 else 0,
                'std': float(np.std(arr)) if len(arr) > 0 else 0,
                'min': float(np.min(arr)) if len(arr) > 0 else 0,
                'max': float(np.max(arr)) if len(arr) > 0 else 0,
                'error': str(e)
            }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness with error handling"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return float(np.mean(((data - mean) / std) ** 3)) if std > 0 else 0
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis with error handling"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            return float(np.mean(((data - mean) / std) ** 4) - 3) if std > 0 else 0
        except:
            return 0.0
    
    def _serialize_best_fits(self, best_fits) -> Dict[str, Any]:
        """Enhanced serialization of best fits"""
        try:
            if hasattr(best_fits, 'to_dict'):
                return best_fits.to_dict()
            elif hasattr(best_fits, 'head'):  # DataFrame
                result = {}
                for index, row in best_fits.head(10).iterrows():  # Limit to top 10
                    row_dict = {}
                    for col, value in row.items():
                        try:
                            if isinstance(value, (int, float, str, bool, type(None))):
                                row_dict[col] = value
                            elif isinstance(value, (list, tuple, np.ndarray)):
                                row_dict[col] = [float(x) if isinstance(x, (int, float, np.number)) else str(x) for x in value]
                            else:
                                row_dict[col] = str(value)
                        except:
                            row_dict[col] = str(value)
                    result[str(index)] = row_dict
                return result
            else:
                return {'data': str(best_fits)}
        except Exception as e:
            logger.warning(f"Could not serialize best_fits: {e}")
            return {'error': f'Serialization error: {e}'}
    
    def _serialize_selected_fit(self, selected_fit) -> Dict[str, Any]:
        """Enhanced serialization of selected fit"""
        try:
            if isinstance(selected_fit, dict):
                serialized = {}
                for key, value in selected_fit.items():
                    try:
                        if isinstance(value, (int, float, str, bool, type(None))):
                            serialized[key] = value
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            serialized[key] = [float(x) if isinstance(x, (int, float, np.number)) else str(x) for x in value]
                        else:
                            serialized[key] = str(value)
                    except:
                        serialized[key] = str(value)
                return serialized
            else:
                return {'data': str(selected_fit)}
        except Exception as e:
            logger.warning(f"Could not serialize selected_fit: {e}")
            return {'error': f'Serialization error: {e}'}
    
    def _get_distribution_info(self, fitter) -> Dict[str, Any]:
        """Enhanced distribution information extraction"""
        try:
            info = {}
            
            # Selected distribution details
            if hasattr(fitter, 'selected_fit') and fitter.selected_fit:
                selected = fitter.selected_fit
                if isinstance(selected, dict):
                    info['selected_distribution'] = selected.get('name', 'Unknown')
                    info['selected_parameters'] = selected.get('params', [])
                    info['selected_aic'] = selected.get('aic', None)
                    info['selected_bic'] = selected.get('bic', None)
                    info['selected_ks_statistic'] = selected.get('ks_statistic', None)
            
            # Available distributions
            if hasattr(fitter, 'distributions'):
                info['distributions_tested'] = list(fitter.distributions)
                info['number_of_distributions_tested'] = len(fitter.distributions)
            
            # Distribution object information
            try:
                if hasattr(fitter, 'get_selected_dist'):
                    dist_obj = fitter.get_selected_dist()
                    if dist_obj:
                        info['distribution_type'] = str(type(dist_obj).__name__)
                        try:
                            if hasattr(dist_obj, 'mean'):
                                info['theoretical_mean'] = float(dist_obj.mean())
                            if hasattr(dist_obj, 'std'):
                                info['theoretical_std'] = float(dist_obj.std())
                            if hasattr(dist_obj, 'var'):
                                info['theoretical_variance'] = float(dist_obj.var())
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Could not extract distribution object info: {e}")
            
            return info
        except Exception as e:
            logger.warning(f"Could not get distribution info: {e}")
            return {'error': f'Info extraction error: {e}'}
    
    def _assess_fit_quality(self, fitter) -> Dict[str, Any]:
        """Enhanced fit quality assessment"""
        try:
            quality_info = {}
            
            if hasattr(fitter, 'selected_fit') and fitter.selected_fit:
                selected = fitter.selected_fit
                if isinstance(selected, dict):
                    # Statistical measures
                    for metric in ['aic', 'bic', 'ks_statistic', 'p_value']:
                        if metric in selected and selected[metric] is not None:
                            quality_info[metric] = float(selected[metric])
                            quality_info[f'{metric}_interpretation'] = self._interpret_metric(metric, selected[metric])
                    
                    # Overall fit quality assessment
                    quality_info['fit_assessment'] = self._overall_fit_assessment(selected)
            
            return quality_info
        except Exception as e:
            logger.warning(f"Could not assess fit quality: {e}")
            return {'error': f'Quality assessment error: {e}'}
    
    def _interpret_metric(self, metric: str, value: float) -> str:
        """Provide interpretation for statistical metrics"""
        interpretations = {
            'aic': "Lower AIC values indicate better model fit (penalized for complexity)",
            'bic': "Lower BIC values indicate better model fit (more strongly penalized for complexity)",
            'ks_statistic': "Lower KS statistic values indicate better fit (closer to theoretical distribution)",
            'p_value': "Higher p-values indicate better fit (less evidence against null hypothesis)"
        }
        return interpretations.get(metric, "Statistical measure for distribution fit quality")
    
    def _overall_fit_assessment(self, selected_fit: Dict) -> str:
        """Provide overall assessment of fit quality"""
        try:
            ks_stat = selected_fit.get('ks_statistic')
            p_value = selected_fit.get('p_value')
            
            if p_value is not None:
                if p_value > 0.05:
                    return "Good fit - cannot reject distribution hypothesis at 5% significance level"
                elif p_value > 0.01:
                    return "Moderate fit - marginal evidence against distribution hypothesis"
                else:
                    return "Poor fit - strong evidence against distribution hypothesis"
            elif ks_stat is not None:
                if ks_stat < 0.05:
                    return "Good fit - low KS statistic suggests close match to theoretical distribution"
                elif ks_stat < 0.1:
                    return "Moderate fit - acceptable deviation from theoretical distribution"
                else:
                    return "Poor fit - high deviation from theoretical distribution"
            else:
                return "Fit quality assessment not available - insufficient statistical measures"
        except:
            return "Unable to assess overall fit quality"

# =================== MCP SERVER SETUP ===================

# Initialize the MCP server
server = Server("comprehensive-risk-analysis-agent")
risk_agent = ComprehensiveRiskAnalysisAgent()

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available risk analysis resources"""
    return [
        types.Resource(
            uri="risk://config",
            name="Risk Analysis Configuration",
            description="Current risk analysis configuration and parameters",
            mimeType="application/json",
        ),
        types.Resource(
            uri="risk://distributions",
            name="Supported Distributions", 
            description="Comprehensive list of supported probability distributions",
            mimeType="application/json",
        ),
        types.Resource(
            uri="risk://templates",
            name="Analysis Templates",
            description="Pre-configured templates for various risk analysis scenarios",
            mimeType="application/json",
        ),
        types.Resource(
            uri="risk://examples",
            name="Example Data Sets",
            description="Sample data for testing and demonstration purposes",
            mimeType="application/json",
        ),
        types.Resource(
            uri="risk://methods",
            name="Available Methods",
            description="Complete list of available risk analysis methods and capabilities",
            mimeType="application/json",
        ),
        types.Resource(
            uri="risk://status",
            name="System Status",
            description="Current system status and package availability",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read risk analysis resources"""
    if uri == "risk://config":
        return json.dumps(asdict(risk_agent.config), indent=2)
    elif uri == "risk://distributions":
        return json.dumps({
            "severity_distributions": risk_agent.config.default_distributions['severity'],
            "frequency_distributions": risk_agent.config.default_distributions['frequency'],
            "description": "Supported probability distributions for risk modeling",
            "total_distributions": len(risk_agent.config.default_distributions['severity']) + len(risk_agent.config.default_distributions['frequency'])
        }, indent=2)
    elif uri == "risk://templates":
        templates = {
            "simple_loss_analysis": {
                "description": "Basic loss analysis with frequency-severity modeling",
                "parameters": {
                    "freq_dist": "poisson",
                    "freq_params": [5.0],
                    "sev_dist": "lognormal", 
                    "sev_params": [5, 0.5]
                }
            },
            "correlated_risks": {
                "description": "Multivariate correlated risk analysis",
                "requires": ["correlation_matrix.csv", "distribution_list.json"]
            },
            "claims_development": {
                "description": "Claims development triangle simulation",
                "requires": ["policies_data.csv"]
            },
            "stress_testing": {
                "description": "Comprehensive stress testing scenarios",
                "parameters": {
                    "base_scenario": "normal market conditions",
                    "stress_scenarios": [
                        {"name": "market_crash", "multiplier": 2.0, "shift": 0},
                        {"name": "inflation_shock", "multiplier": 1.5, "shift": 10000}
                    ]
                }
            },
            "distribution_comparison": {
                "description": "Compare multiple distribution fits",
                "parameters": {
                    "distributions": ["lognormal", "gamma", "weibull", "pareto"]
                }
            }
        }
        return json.dumps(templates, indent=2)
    elif uri == "risk://examples":
        examples = {
            "sample_loss_data": [1000, 1500, 800, 2200, 3400, 900, 1200, 1800, 2600, 1400],
            "correlation_matrix_csv": "Correlation Matrix,LoB1,LoB2,LoB3\nLoB1,1,0.5,0.3\nLoB2,0.5,1,0.7\nLoB3,0.3,0.7,1",
            "distribution_list_json": '[{"index":1,"dist_name":"LoB1","dist_type":"gamma","dist_param":[2,1]},{"index":2,"dist_name":"LoB2","dist_type":"lognormal","dist_param":[2,1]}]',
            "policies_csv": "policy_id,freq_dist,freq_params,sev_dist,sev_params,start_date,end_date\n1,poisson,(0.8),lognormal,(9.0 0.5),2023-01-01,2023-12-31"
        }
        return json.dumps(examples, indent=2)
    elif uri == "risk://methods":
        methods = {
            "distribution_fitting": [
                "fit_distributions",
                "manual_distribution_selection", 
                "calculate_distribution_statistics",
                "generate_samples_from_distribution",
                "generate_mixed_samples"
            ],
            "stochastic_simulation": [
                "create_stochastic_simulator",
                "run_aggregate_simulations",
                "apply_deductibles_and_limits",
                "analyze_simulation_results",
                "plot_simulation_distribution",
                "plot_correlated_variables"
            ],
            "multivariate_analysis": [
                "simulate_multivariate_correlated_risks"
            ],
            "claim_simulation": [
                "create_synthetic_policies",
                "simulate_claims",
                "simulate_claim_dates_nhpp",
                "apply_shifted_dates",
                "simulate_claim_development",
                "save_claim_development"
            ],
            "risk_analytics": [
                "calculate_quantile",
                "perform_stress_testing",
                "calculate_risk_metrics"
            ],
            "reporting": [
                "generate_comprehensive_risk_report"
            ]
        }
        return json.dumps(methods, indent=2)
    elif uri == "risk://status":
        status = {
            "actrisk_available": ACTRISK_AVAILABLE,
            "config_loaded": risk_agent.actrisk_config is not None,
            "cached_analyses": len(risk_agent.analysis_cache),
            "default_simulation_count": risk_agent.config.simulation_count,
            "confidence_levels": risk_agent.config.confidence_levels
        }
        return json.dumps(status, indent=2)
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available risk analysis tools"""
    return [
        # Distribution Fitting Tools
        types.Tool(
            name="fit_distributions",
            description="Fit probability distributions to data with enhanced options, can only fit 1 distribution type at a time",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Historical data for distribution fitting"
                    },
                    "dist_type": {
                        "type": "string",
                        "enum": ["severity", "frequency"],
                        "description": "Type of distribution to fit"
                    },
                    "specific_distributions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific distributions to test (optional)"
                    },
                    "specific_metrics": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Specific metrics to use for comparison (optional)"
                    }
                },
                "required": ["data"]
            },
        ),
        types.Tool(
            name="calculate_distribution_statistics",
            description="Calculate comprehensive statistics for a distribution",
            inputSchema={
                "type": "object",
                "properties": {
                    "distribution_name": {"type": "string", "description": "Name of the distribution"},
                    "parameters": {"type": "array", "items": {"type": "number"}, "description": "Distribution parameters"}
                },
                "required": ["distribution_name", "parameters"]
            },
        ),
        types.Tool(
            name="generate_samples_from_distribution",
            description="Generate random samples from a specified distribution",
            inputSchema={
                "type": "object",
                "properties": {
                    "distribution_name": {"type": "string", "description": "Name of the distribution"},
                    "parameters": {"type": "array", "items": {"type": "number"}, "description": "Distribution parameters"},
                    "size": {"type": "integer", "description": "Number of samples to generate", "default": 100}
                },
                "required": ["distribution_name", "parameters"]
            },
        ),
        types.Tool(
            name="generate_mixed_samples",
            description="Generate samples from a mixture of two distributions",
            inputSchema={
                "type": "object",
                "properties": {
                    "dist1_name": {"type": "string", "description": "First distribution name"},
                    "dist1_params": {"type": "array", "items": {"type": "number"}, "description": "First distribution parameters"},
                    "dist2_name": {"type": "string", "description": "Second distribution name"},
                    "dist2_params": {"type": "array", "items": {"type": "number"}, "description": "Second distribution parameters"},
                    "weight1": {"type": "number", "description": "Weight for first distribution"},
                    "weight2": {"type": "number", "description": "Weight for second distribution"},
                    "size": {"type": "integer", "description": "Number of samples", "default": 100}
                },
                "required": ["dist1_name", "dist1_params", "dist2_name", "dist2_params", "weight1", "weight2"]
            },
        ),
        
        # Simulation Tools
        types.Tool(
            name="create_stochastic_simulator",
            description="Create and configure a stochastic simulator with advanced options",
            inputSchema={
                "type": "object",
                "properties": {
                    "freq_dist": {"type": "string", "description": "Frequency distribution name"},
                    "freq_params": {"type": "array", "items": {"type": "number"}, "description": "Frequency distribution parameters"},
                    "sev_dist": {"type": "string", "description": "Severity distribution name"},
                    "sev_params": {"type": "array", "items": {"type": "number"}, "description": "Severity distribution parameters"},
                    "n_simulations": {"type": "integer", "description": "Number of simulations"},
                    "use_copula": {"type": "boolean", "description": "Use copula for dependency modeling"},
                    "copula_type": {"type": "string", "description": "Type of copula (e.g., 'frank', 'clayton')"},
                    "copula_param": {"type": "number", "description": "Copula parameter"},
                    "correlation_param": {"type": "number", "description": "Linear correlation parameter"}
                },
                "required": ["freq_dist", "freq_params", "sev_dist", "sev_params"]
            },
        ),
        types.Tool(
            name="run_aggregate_simulations",
            description="Run aggregate loss simulations using a configured simulator",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the configured simulator"}
                },
                "required": ["simulator_id"]
            },
        ),
        types.Tool(
            name="apply_deductibles_and_limits",
            description="Apply insurance policy terms (deductibles and limits) to simulation results",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the simulator"},
                    "per_occurrence_deductible": {"type": "number", "description": "Per occurrence deductible"},
                    "per_occurrence_limit": {"type": "number", "description": "Per occurrence limit"},
                    "aggregate_deductible": {"type": "number", "description": "Annual aggregate deductible"},
                    "aggregate_limit": {"type": "number", "description": "Annual aggregate limit"}
                },
                "required": ["simulator_id"]
            },
        ),
        types.Tool(
            name="analyze_simulation_results", 
            description="Perform comprehensive analysis of simulation results",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the simulator"},
                    "custom_data": {"type": "object", "description": "Custom data for analysis (optional)"}
                },
                "required": ["simulator_id"]
            },
        ),
        
        # Multivariate Analysis
        types.Tool(
            name="simulate_multivariate_correlated_risks",
            description="Simulate correlated multivariate risk distributions with enhanced features",
            inputSchema={
                "type": "object",
                "properties": {
                    "correlation_matrix": {"type": "string", "description": "CSV format correlation matrix data"},
                    "distribution_list": {"type": "string", "description": "JSON format distribution specifications"},
                    "n_simulations": {"type": "integer", "description": "Number of simulations"}
                },
                "required": ["correlation_matrix", "distribution_list"]
            },
        ),
        
        # Claims Simulation Tools
        types.Tool(
            name="create_synthetic_policies",
            description="Generate synthetic policy data for claim simulation",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_policies": {"type": "integer", "description": "Number of policies to generate"},
                    "freq_dist_options": {"type": "array", "items": {"type": "string"}, "description": "Frequency distribution options"},
                    "sev_dist_options": {"type": "array", "items": {"type": "string"}, "description": "Severity distribution options"},
                    "start_date": {"type": "string", "description": "Policy start date"},
                    "end_date": {"type": "string", "description": "Policy end date"}
                },
                "required": ["n_policies"]
            },
        ),
        types.Tool(
            name="simulate_claims",
            description="Simulate claims based on policy data",
            inputSchema={
                "type": "object",
                "properties": {
                    "policies_data": {
                        "type": "string",
                        "description": "CSV format policy data or JSON policy array"
                    }
                },
                "required": ["policies_data"]
            },
        ),
        types.Tool(
            name="simulate_claim_dates_nhpp",
            description="Simulate claim occurrence dates using Non-Homogeneous Poisson Process",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the claim simulator"},
                    "lambda0": {"type": "number", "description": "Baseline intensity", "default": 10},
                    "alpha": {"type": "number", "description": "Seasonality amplitude", "default": 0.5},
                    "phase": {"type": "number", "description": "Phase shift", "default": 0},
                    "T": {"type": "number", "description": "Exposure duration in years", "default": 1}
                },
                "required": ["simulator_id"]
            },
        ),
        types.Tool(
            name="apply_shifted_dates",
            description="Apply date shifts to align with calendar year",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the claim simulator"},
                    "start_year": {"type": "integer", "description": "Starting year", "default": 2023}
                },
                "required": ["simulator_id"]
            },
        ),
        types.Tool(
            name="simulate_claim_development",
            description="Simulate claim development triangles with stochastic LDFs",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the claim simulator"},
                    "base_ldfs": {
                        "type": "object",
                        "description": "Base Loss Development Factors by development month",
                        "additionalProperties": {"type": "number"}
                    },
                    "volatility": {"type": "number", "description": "LDF volatility", "default": 0.1},
                    "tail_factor": {"type": "number", "description": "Tail development factor", "default": 1.0}
                },
                "required": ["simulator_id"]
            },
        ),
        
        # Risk Analytics Tools
        types.Tool(
            name="calculate_quantile",
            description="Calculate specific quantile for a distribution",
            inputSchema={
                "type": "object",
                "properties": {
                    "distribution": {"type": "string", "description": "Distribution name"},
                    "parameters": {"type": "array", "items": {"type": "number"}, "description": "Distribution parameters"},
                    "quantile": {"type": "number", "description": "Quantile level (0-1)"}
                },
                "required": ["distribution", "parameters", "quantile"]
            },
        ),
        types.Tool(
            name="perform_stress_testing",
            description="Perform stress testing on simulation results",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator_id": {"type": "string", "description": "ID of the simulator"},
                    "stress_scenarios": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "multiplier": {"type": "number"},
                                "shift": {"type": "number"}
                            }
                        },
                        "description": "List of stress scenarios"
                    }
                },
                "required": ["simulator_id", "stress_scenarios"]
            },
        ),
        types.Tool(
            name="calculate_risk_metrics",
            description="Calculate comprehensive risk metrics for loss data",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "number"}, "description": "Loss data"},
                    "confidence_levels": {"type": "array", "items": {"type": "number"}, "description": "Confidence levels for VaR calculation"}
                },
                "required": ["data"]
            },
        ),
        
        # Reporting Tool
        types.Tool(
            name="generate_comprehensive_risk_report",
            description="Generate detailed risk analysis report with recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_results": {
                        "type": "object",
                        "description": "Results from previous risk analysis tools"
                    },
                    "report_type": {
                        "type": "string",
                        "enum": ["comprehensive", "executive", "technical"],
                        "description": "Type of report to generate"
                    },
                    "include_recommendations": {
                        "type": "boolean",
                        "description": "Include risk management recommendations",
                        "default": True
                    }
                },
                "required": ["analysis_results"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls for comprehensive risk analysis"""
    try:
        # Distribution Fitting Tools
        if name == "fit_distributions":
            result = risk_agent.fit_distributions(
                arguments["data"],
                arguments.get("dist_type", "severity"),
                arguments.get("specific_distributions"),
                arguments.get("specific_metrics")
            )
        elif name == "calculate_distribution_statistics":
            result = risk_agent.calculate_distribution_statistics(
                arguments["distribution_name"],
                arguments["parameters"]
            )
        elif name == "generate_samples_from_distribution":
            result = risk_agent.generate_samples_from_distribution(
                arguments["distribution_name"],
                arguments["parameters"],
                arguments.get("size", 100)
            )
        elif name == "generate_mixed_samples":
            result = risk_agent.generate_mixed_samples(
                arguments["dist1_name"], arguments["dist1_params"],
                arguments["dist2_name"], arguments["dist2_params"],
                arguments["weight1"], arguments["weight2"],
                arguments.get("size", 100)
            )
        
        # Simulation Tools
        elif name == "create_stochastic_simulator":
            result = risk_agent.create_stochastic_simulator(
                arguments["freq_dist"], tuple(arguments["freq_params"]),
                arguments["sev_dist"], tuple(arguments["sev_params"]),
                arguments.get("n_simulations"),
                arguments.get("use_copula", False),
                arguments.get("copula_type"),
                arguments.get("copula_param"),
                arguments.get("correlation_param")
            )
        elif name == "run_aggregate_simulations":
            result = risk_agent.run_aggregate_simulations(
                arguments["simulator_id"]
            )
        elif name == "apply_deductibles_and_limits":
            result = risk_agent.apply_deductibles_and_limits(
                arguments["simulator_id"],
                arguments.get("per_occurrence_deductible", 0),
                arguments.get("per_occurrence_limit"),
                arguments.get("aggregate_deductible", 0),
                arguments.get("aggregate_limit")
            )
        elif name == "analyze_simulation_results":
            result = risk_agent.analyze_simulation_results(
                arguments["simulator_id"],
                arguments.get("custom_data")
            )
        
        # Multivariate Analysis
        elif name == "simulate_multivariate_correlated_risks":
            result = risk_agent.simulate_multivariate_correlated_risks(
                arguments["correlation_matrix"],
                arguments["distribution_list"],
                arguments.get("n_simulations")
            )
        
        # Claims Simulation Tools
        elif name == "create_synthetic_policies":
            result = risk_agent.create_synthetic_policies(
                arguments["n_policies"],
                arguments.get("freq_dist_options"),
                arguments.get("sev_dist_options"),
                arguments.get("start_date", "2023-01-01"),
                arguments.get("end_date", "2023-12-31")
            )
        elif name == "simulate_claims":
            result = risk_agent.simulate_claims(
                arguments["policies_data"]
            )
        elif name == "simulate_claim_dates_nhpp":
            result = risk_agent.simulate_claim_dates_nhpp(
                arguments["simulator_id"],
                arguments.get("lambda0", 10),
                arguments.get("alpha", 0.5),
                arguments.get("phase", 0),
                arguments.get("T", 1)
            )
        elif name == "apply_shifted_dates":
            result = risk_agent.apply_shifted_dates(
                arguments["simulator_id"],
                arguments.get("start_year", 2023)
            )
        elif name == "simulate_claim_development":
            result = risk_agent.simulate_claim_development(
                arguments["simulator_id"],
                arguments.get("base_ldfs"),
                arguments.get("volatility", 0.1),
                arguments.get("tail_factor", 1.0)
            )
        
        # Risk Analytics Tools
        elif name == "calculate_quantile":
            result = risk_agent.calculate_quantile(
                arguments["distribution"],
                arguments["parameters"],
                arguments["quantile"]
            )
        elif name == "perform_stress_testing":
            result = risk_agent.perform_stress_testing(
                arguments["simulator_id"],
                arguments["stress_scenarios"]
            )
        elif name == "calculate_risk_metrics":
            result = risk_agent.calculate_risk_metrics(
                arguments["data"],
                arguments.get("confidence_levels")
            )
        
        # Reporting Tool
        elif name == "generate_comprehensive_risk_report":
            result = risk_agent.generate_comprehensive_risk_report(
                arguments["analysis_results"],
                arguments.get("report_type", "comprehensive"),
                arguments.get("include_recommendations", True)
            )
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        )]
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

@server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    """List available risk analysis prompts"""
    return [
        types.Prompt(
            name="comprehensive_risk_workflow",
            description="Complete workflow for comprehensive risk assessment",
            arguments=[
                types.PromptArgument(
                    name="analysis_type",
                    description="Type of risk analysis (loss_modeling, claims_analysis, portfolio_risk, etc.)",
                    required=False
                ),
                types.PromptArgument(
                    name="data_available",
                    description="Description of available data",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="distribution_modeling_guide",
            description="Advanced guide for distribution selection and modeling",
            arguments=[
                types.PromptArgument(
                    name="data_characteristics",
                    description="Characteristics of your data (heavy-tailed, seasonal, etc.)",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="simulation_setup_advanced",
            description="Advanced simulation setup with copulas and correlations",
            arguments=[
                types.PromptArgument(
                    name="dependency_structure",
                    description="Type of dependency modeling needed",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="claims_reserving_workflow",
            description="Complete claims reserving and development analysis workflow",
            arguments=[
                types.PromptArgument(
                    name="reserving_method",
                    description="Preferred reserving methodology",
                    required=False
                )
            ]
        ),
        types.Prompt(
            name="stress_testing_scenarios",
            description="Guide for designing comprehensive stress testing scenarios",
            arguments=[
                types.PromptArgument(
                    name="risk_appetite",
                    description="Organization's risk appetite and tolerance levels",
                    required=False
                )
            ]
        )
    ]

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    """Handle comprehensive prompt requests for risk analysis guidance"""
    
    if name == "comprehensive_risk_workflow":
        analysis_type = arguments.get("analysis_type", "general")
        data_available = arguments.get("data_available", "historical loss data")
        
        prompt_text = f"""
# Comprehensive Risk Analysis Workflow

## Analysis Type: {analysis_type.title()}
## Available Data: {data_available}

### Phase 1: Data Preparation and Exploration
1. **Data Quality Assessment**
   - Check for completeness, outliers, and data quality issues
   - Perform basic statistical analysis using `calculate_risk_metrics`
   - Identify potential data transformations needed

2. **Exploratory Data Analysis**
   - Generate summary statistics for loss data
   - Identify patterns, seasonality, and trends
   - Assess data characteristics (heavy tails, skewness, etc.)

### Phase 2: Distribution Modeling
1. **Single Distribution Fitting**
   - Use `fit_distributions` with comprehensive distribution testing
   - Validate fit quality using statistical tests

2. **Advanced Distribution Analysis**
   - Calculate theoretical statistics using `calculate_distribution_statistics`
   - Generate samples for model validation using `generate_samples_from_distribution`
   - Consider mixture models using `generate_mixed_samples` if appropriate

### Phase 3: Simulation and Modeling
1. **Basic Simulation Setup**
   - Create simulator using `create_stochastic_simulator`
   - Configure appropriate dependency structure (copulas/correlation)
   - Run simulations using `run_aggregate_simulations`

2. **Advanced Simulation Features**
   - Apply policy terms using `apply_deductibles_and_limits`
   - Perform comprehensive analysis using `analyze_simulation_results`
   - Implement multivariate modeling if needed

### Phase 4: Risk Assessment
1. **Risk Metrics Calculation**
   - Calculate VaR and Expected Shortfall
   - Assess tail risk and extreme scenarios
   - Generate quantile analysis using `calculate_quantile`

2. **Stress Testing**
   - Design stress scenarios using `perform_stress_testing`
   - Test model sensitivity to parameter changes
   - Evaluate model robustness

### Phase 5: Claims Analysis (if applicable)
1. **Policy Data Preparation**
   - Create synthetic policies using `create_synthetic_policies`
   - Set up realistic policy portfolios

2. **Claims Simulation**
   - Simulate claims using `simulate_claims`
   - Model occurrence dates using `simulate_claim_dates_nhpp`
   - Simulate development patterns using `simulate_claim_development`

### Phase 6: Reporting and Documentation
1. **Comprehensive Reporting**
   - Generate detailed reports using `generate_comprehensive_risk_report`
   - Include risk management recommendations
   - Document methodology and assumptions

2. **Model Validation and Monitoring**
   - Perform backtesting on historical data
   - Set up monitoring for model performance
   - Plan regular model updates and reviews

Would you like detailed guidance on any specific phase?
"""
        
        return types.GetPromptResult(
            description=f"Comprehensive risk analysis workflow for {analysis_type}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt_text)
                )
            ]
        )
    
    elif name == "distribution_modeling_guide":
        data_chars = arguments.get("data_characteristics", "")
        
        guide_text = f"""
# Advanced Distribution Modeling Guide

## Data Characteristics: {data_chars}

### Distribution Selection Strategy

#### For Frequency Modeling:
1. **Count Data Characteristics**
   - **Poisson**: Equal mean and variance, most common starting point
   - **Negative Binomial**: Overdispersed data (variance > mean)
   - **Zero-Inflated**: Excess zeros in the data
   - **Binomial**: Known exposure with probability of occurrence

2. **Selection Process**
   ```
   Step 1: Use fit_distributions with ['poisson', 'negative binomial']
   Step 2: Check overdispersion using variance-to-mean ratio
   Step 3: Validate using goodness-of-fit tests
   ```

#### For Severity Modeling:
1. **Light-Tailed Distributions**
   - **Normal**: Symmetric, theoretical only
   - **Lognormal**: Right-skewed, multiplicative processes
   - **Gamma**: Flexible shape, positive values only

2. **Heavy-Tailed Distributions**
   - **Pareto**: Power law, extreme values
   - **Weibull**: Reliability analysis, varying hazard rates
   - **Loggamma**: Heavy tails with flexibility

3. **Selection Methodology**
   ```
   Step 1: Examine tail behavior using QQ plots
   Step 2: Test multiple distributions using fit_distributions
   Step 3: Use AIC/BIC for model selection
   Step 4: Validate tail behavior using extreme percentiles
   ```
"""
        
        return types.GetPromptResult(
            description="Guide for selecting appropriate probability distributions",
            messages=[
                types.PromptMessage(
                    role="user", 
                    content=types.TextContent(type="text", text=guide_text)
                )
            ]
        )
    
    elif name == "monte_carlo_setup":
        objective = arguments.get("analysis_objective", "general risk assessment")
        
        setup_guide = f"""
# Monte Carlo Simulation Setup Guide

## Objective: {objective}

### Key Parameters to Consider:

1. **Number of Simulations**:
   - Start with 10,000 for initial analysis
   - Use 100,000+ for final results or regulatory reporting
   - More simulations = more stable results but longer runtime

2. **Distribution Parameters**:
   - Frequency: Usually Poisson with λ = expected number of events
   - Severity: Depends on fitted distribution (mean, std dev, shape, etc.)

3. **Correlation Modeling**:
   - Set use_correlation=true if frequency and severity are dependent
   - Correlation parameter typically between -1 and 1
   - Consider copula models for complex dependencies

4. **Random Seed**:
   - Set for reproducible results
   - Change seed to test stability of results

### Example Setup:
```python
# High-frequency, low-severity scenario
freq_dist = "poisson"
freq_params = [50.0]  # 50 expected events per year
sev_dist = "lognormal" 
sev_params = [8.0, 1.0]  # meanlog=8, sigma=1

# Low-frequency, high-severity scenario  
freq_dist = "poisson"
freq_params = [2.0]  # 2 expected events per year
sev_dist = "pareto"
sev_params = [1.0, 100000.0]  # shape=1, scale=100k
```

Use the `monte_carlo_simulation` tool with these parameters to run your analysis.
"""
        
        return types.GetPromptResult(
            description="Setup guide for Monte Carlo simulations",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=setup_guide)
                )
            ]
        )
    
    else:
        raise ValueError(f"Unknown prompt: {name}")

async def main():
    """Main entry point for the MCP server"""
    try:
        # Import here to avoid issues if mcp package is not installed
        from mcp.server.stdio import stdio_server
        
        logger.info("Starting ActRisk MCP Server...")
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="actrisk-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except ImportError as e:
        logger.error(f"MCP import error: {e}")
        print("Error: MCP package not found. Install with: pip install mcp")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
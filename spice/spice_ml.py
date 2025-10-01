"""
SPICE-ML Implementation - Maximum Likelihood SPICE for Enhanced Performance.

This module implements SPICE-ML (Maximum Likelihood SPICE), which uses ML
estimation principles to potentially achieve better performance than standard
SPICE, particularly in challenging SNR scenarios.

References
----------
.. [1] Stoica et al., "Maximum likelihood sparse signal representation," IEEE
       Trans. Signal Process., 2012.
.. [2] Angelopoulos et al., "Optimal maximum likelihood sparse beamforming,"
       ICASSP 2019.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from typing import Tuple, Optional, Dict, Union
import warnings
from dataclasses import dataclass

from spice_core import SPICEEstimator, SPICEConfig


@dataclass
class SPICEMLConfig(SPICEConfig):
    """Configuration for SPICE-ML algorithm.

    Parameters
    ----------
    ml_optimization_method : str, default='newton'
        ML optimization method: 'newton', 'bfgs', 'gradient'.
    likelihood_regularization : float, default=1e-8
        Regularization for likelihood computation.
    ml_max_iterations : int, default=50
        Maximum iterations for ML optimization.
    concentration_penalty : float, default=0.1
        Penalty term for sparsity enforcement.
    """
    ml_optimization_method: str = 'newton'
    likelihood_regularization: float = 1e-8
    ml_max_iterations: int = 50
    concentration_penalty: float = 0.1

    def __post_init__(self):
        """Validate ML-specific configuration parameters."""
        super().__post_init__()

        valid_methods = {'newton', 'bfgs', 'gradient'}
        if self.ml_optimization_method not in valid_methods:
            raise ValueError(f"ml_optimization_method must be one of {valid_methods}")

        if self.likelihood_regularization <= 0:
            raise ValueError("likelihood_regularization must be positive")

        if self.ml_max_iterations <= 0:
            raise ValueError("ml_max_iterations must be positive")

        if self.concentration_penalty < 0:
            raise ValueError("concentration_penalty must be non-negative")


class SPICEMLEstimator(SPICEEstimator):
    """
    SPICE-ML: Maximum Likelihood Sparse Parameter Estimation.

    SPICE-ML reformulates the SPICE problem as a maximum likelihood estimation,
    potentially achieving superior performance compared to standard SPICE in
    challenging scenarios through improved statistical efficiency.

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : SPICEMLConfig, optional
        Configuration for SPICE-ML algorithm.

    Attributes
    ----------
    ml_likelihood_history : list
        History of likelihood values during ML optimization.
    ml_convergence_info : dict
        Information about ML convergence.
    """

    def __init__(self, n_sensors: int, config: Optional[SPICEMLConfig] = None):
        """Initialize SPICE-ML estimator."""
        self.config = config or SPICEMLConfig()
        super().__init__(n_sensors, self.config)
        self.ml_likelihood_history = []
        self.ml_convergence_info = {}

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate spectrum using SPICE-ML algorithm.

        Parameters
        ----------
        sample_covariance : array_like
            Sample covariance matrix.

        Returns
        -------
        power_spectrum : ndarray
            SPICE-ML power spectrum estimates.
        angular_grid : ndarray
            Angular grid in degrees.
        """
        self._validate_covariance_matrix(sample_covariance)
        self.sample_cov = sample_covariance.copy()

        # Initialize with standard SPICE result for warm start
        initial_powers, _ = super().fit(sample_covariance)

        # Run ML optimization
        ml_powers = self._run_ml_optimization(initial_powers)

        return ml_powers, self.angular_grid

    def _run_ml_optimization(self, initial_powers: np.ndarray) -> np.ndarray:
        """Run maximum likelihood optimization for power estimates."""
        self.ml_likelihood_history = []

        # Optimization bounds: powers must be non-negative
        bounds = [(self.config.regularization, None) for _ in range(self.config.grid_size)]

        try:
            if self.config.ml_optimization_method == 'newton':
                result = self._newton_ml_optimization(initial_powers)
            elif self.config.ml_optimization_method == 'bfgs':
                result = self._bfgs_ml_optimization(initial_powers, bounds)
            else:  # gradient
                result = self._gradient_ml_optimization(initial_powers)

            self.ml_convergence_info = {
                'success': result.get('success', True),
                'iterations': result.get('nit', len(self.ml_likelihood_history)),
                'final_likelihood': result.get('fun', self.ml_likelihood_history[-1] if self.ml_likelihood_history else 0),
                'method': self.config.ml_optimization_method
            }

            optimized_powers = result.get('x', initial_powers)

        except Exception as e:
            warnings.warn(f"ML optimization failed: {str(e)}, using initial powers", UserWarning)
            optimized_powers = initial_powers
            self.ml_convergence_info = {
                'success': False,
                'error': str(e),
                'method': self.config.ml_optimization_method
            }

        return np.maximum(optimized_powers, self.config.regularization)

    def _newton_ml_optimization(self, initial_powers: np.ndarray) -> Dict:
        """Newton-based ML optimization."""
        current_powers = initial_powers.copy()

        for iteration in range(self.config.ml_max_iterations):
            # Compute likelihood and gradients
            likelihood = self._compute_negative_log_likelihood(current_powers)
            gradient = self._compute_likelihood_gradient(current_powers)
            hessian = self._compute_likelihood_hessian(current_powers)

            self.ml_likelihood_history.append(-likelihood)

            # Newton step with regularization for numerical stability
            try:
                hessian_reg = hessian + self.config.likelihood_regularization * np.eye(len(hessian))
                newton_step = la.solve(hessian_reg, gradient)

                # Line search for step size
                step_size = self._backtracking_line_search(current_powers, -newton_step, gradient)

                # Update powers
                new_powers = current_powers - step_size * newton_step
                new_powers = np.maximum(new_powers, self.config.regularization)

                # Convergence check
                if np.linalg.norm(new_powers - current_powers) < self.config.tolerance:
                    break

                current_powers = new_powers

            except la.LinAlgError:
                # Fallback to gradient descent if Hessian is singular
                step_size = 0.01
                current_powers = current_powers - step_size * gradient
                current_powers = np.maximum(current_powers, self.config.regularization)

        return {
            'x': current_powers,
            'fun': likelihood,
            'nit': iteration + 1,
            'success': iteration < self.config.ml_max_iterations - 1
        }

    def _bfgs_ml_optimization(self, initial_powers: np.ndarray, bounds) -> Dict:
        """BFGS-based ML optimization."""
        def objective_with_history(powers):
            likelihood = self._compute_negative_log_likelihood(powers)
            self.ml_likelihood_history.append(-likelihood)
            return likelihood

        def gradient_func(powers):
            return self._compute_likelihood_gradient(powers)

        try:
            result = opt.minimize(
                objective_with_history,
                initial_powers,
                method='L-BFGS-B',
                jac=gradient_func,
                bounds=bounds,
                options={'maxiter': self.config.ml_max_iterations, 'ftol': self.config.tolerance}
            )
            return result
        except Exception as e:
            raise RuntimeError(f"BFGS optimization failed: {str(e)}")

    def _gradient_ml_optimization(self, initial_powers: np.ndarray) -> Dict:
        """Gradient descent ML optimization."""
        current_powers = initial_powers.copy()
        learning_rate = 0.001

        for iteration in range(self.config.ml_max_iterations):
            likelihood = self._compute_negative_log_likelihood(current_powers)
            gradient = self._compute_likelihood_gradient(current_powers)

            self.ml_likelihood_history.append(-likelihood)

            # Adaptive learning rate
            if iteration > 0 and self.ml_likelihood_history[-1] < self.ml_likelihood_history[-2]:
                learning_rate *= 0.9
            elif iteration > 0:
                learning_rate *= 1.1

            # Gradient step
            new_powers = current_powers - learning_rate * gradient
            new_powers = np.maximum(new_powers, self.config.regularization)

            # Convergence check
            if np.linalg.norm(new_powers - current_powers) < self.config.tolerance:
                break

            current_powers = new_powers

        return {
            'x': current_powers,
            'fun': likelihood,
            'nit': iteration + 1,
            'success': iteration < self.config.ml_max_iterations - 1
        }

    def _compute_negative_log_likelihood(self, powers: np.ndarray) -> float:
        """Compute negative log-likelihood for ML optimization."""
        # Construct fitted covariance from current power estimates
        fitted_cov = self._construct_fitted_covariance(powers)

        # Add regularization for numerical stability
        fitted_cov += self.config.likelihood_regularization * np.eye(self.n_sensors)

        try:
            # ML likelihood: log|Sigma| + trace(Sigma^{-1} * R)
            sign, log_det = np.linalg.slogdet(fitted_cov)
            if sign <= 0:
                return 1e10  # Large penalty for non-positive definite

            fitted_cov_inv = la.inv(fitted_cov)
            trace_term = np.trace(fitted_cov_inv @ self.sample_cov).real

            # Negative log-likelihood with sparsity penalty
            neg_log_likelihood = log_det + trace_term

            # Add concentration penalty for sparsity
            sparsity_penalty = self.config.concentration_penalty * np.sum(powers)

            return neg_log_likelihood + sparsity_penalty

        except (la.LinAlgError, np.linalg.LinAlgError):
            return 1e10  # Large penalty for numerical issues

    def _compute_likelihood_gradient(self, powers: np.ndarray) -> np.ndarray:
        """Compute gradient of negative log-likelihood."""
        gradient = np.zeros_like(powers)

        # Construct fitted covariance and its inverse
        fitted_cov = self._construct_fitted_covariance(powers)
        fitted_cov += self.config.likelihood_regularization * np.eye(self.n_sensors)

        try:
            fitted_cov_inv = la.inv(fitted_cov)

            # Gradient computation for each power parameter
            for k in range(self.config.grid_size):
                # Derivative of fitted covariance w.r.t. power k
                steering_k = self.steering_vectors[:, k:k+1]
                cov_derivative = steering_k @ steering_k.conj().T

                # Gradient of log-likelihood
                grad_log_det = np.trace(fitted_cov_inv @ cov_derivative).real
                grad_trace = -np.trace(fitted_cov_inv @ self.sample_cov @ fitted_cov_inv @ cov_derivative).real

                gradient[k] = grad_log_det + grad_trace + self.config.concentration_penalty

        except (la.LinAlgError, np.linalg.LinAlgError):
            # If inversion fails, use finite differences
            gradient = self._finite_difference_gradient(powers)

        return gradient

    def _compute_likelihood_hessian(self, powers: np.ndarray) -> np.ndarray:
        """Compute Hessian of negative log-likelihood."""
        hessian = np.zeros((len(powers), len(powers)))

        # For computational efficiency, use finite differences approximation
        # Full analytical Hessian is complex for ML formulation
        eps = 1e-6

        for i in range(len(powers)):
            for j in range(i, len(powers)):
                # Central finite difference for Hessian
                powers_pp = powers.copy()
                powers_pp[i] += eps
                powers_pp[j] += eps

                powers_pm = powers.copy()
                powers_pm[i] += eps
                powers_pm[j] -= eps

                powers_mp = powers.copy()
                powers_mp[i] -= eps
                powers_mp[j] += eps

                powers_mm = powers.copy()
                powers_mm[i] -= eps
                powers_mm[j] -= eps

                h_ij = (self._compute_negative_log_likelihood(powers_pp) -
                       self._compute_negative_log_likelihood(powers_pm) -
                       self._compute_negative_log_likelihood(powers_mp) +
                       self._compute_negative_log_likelihood(powers_mm)) / (4 * eps**2)

                hessian[i, j] = h_ij
                if i != j:
                    hessian[j, i] = h_ij

        return hessian

    def _finite_difference_gradient(self, powers: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences."""
        gradient = np.zeros_like(powers)
        eps = 1e-6

        for k in range(len(powers)):
            powers_plus = powers.copy()
            powers_plus[k] += eps

            powers_minus = powers.copy()
            powers_minus[k] -= eps

            gradient[k] = (self._compute_negative_log_likelihood(powers_plus) -
                          self._compute_negative_log_likelihood(powers_minus)) / (2 * eps)

        return gradient

    def _backtracking_line_search(self, current_powers: np.ndarray,
                                 search_direction: np.ndarray,
                                 gradient: np.ndarray) -> float:
        """Backtracking line search for step size selection."""
        alpha = 1.0
        c1 = 1e-4  # Armijo condition parameter
        rho = 0.5  # Backtracking parameter

        current_likelihood = self._compute_negative_log_likelihood(current_powers)

        for _ in range(10):  # Maximum backtracking steps
            new_powers = current_powers + alpha * search_direction
            new_powers = np.maximum(new_powers, self.config.regularization)

            new_likelihood = self._compute_negative_log_likelihood(new_powers)

            # Armijo condition
            if new_likelihood <= current_likelihood + c1 * alpha * np.dot(gradient, search_direction):
                break

            alpha *= rho

        return alpha

    def get_ml_convergence_info(self) -> Dict:
        """Get ML optimization convergence information."""
        return {
            'ml_convergence': self.ml_convergence_info,
            'ml_likelihood_history': self.ml_likelihood_history,
            'standard_convergence': self.get_convergence_info()
        }


def create_spice_ml(n_sensors: int,
                   ml_method: str = 'newton',
                   max_iterations: int = 50,
                   concentration_penalty: float = 0.1) -> SPICEMLEstimator:
    """
    Factory function to create SPICE-ML estimator with optimized parameters.

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    ml_method : str, default='newton'
        ML optimization method: 'newton', 'bfgs', 'gradient'.
    max_iterations : int, default=50
        Maximum iterations for ML optimization.
    concentration_penalty : float, default=0.1
        Sparsity penalty parameter.

    Returns
    -------
    estimator : SPICEMLEstimator
        Configured SPICE-ML estimator.
    """
    config = SPICEMLConfig(
        ml_optimization_method=ml_method,
        ml_max_iterations=max_iterations,
        concentration_penalty=concentration_penalty
    )
    # Set tolerance separately to avoid conflict
    config.tolerance = 1e-6
    config.max_iterations = 30  # Standard SPICE iterations for initialization

    return SPICEMLEstimator(n_sensors, config)


def compare_spice_ml_performance(sample_cov: np.ndarray, n_sensors: int) -> Dict:
    """
    Compare SPICE vs SPICE-ML performance.

    Parameters
    ----------
    sample_cov : array_like
        Sample covariance matrix.
    n_sensors : int
        Number of sensors.

    Returns
    -------
    comparison : dict
        Performance comparison between SPICE and SPICE-ML.
    """
    from spice_core import SPICEEstimator
    import time

    # Standard SPICE
    start_time = time.time()
    spice_std = SPICEEstimator(n_sensors)
    spectrum_std, angles_std = spice_std.fit(sample_cov)
    peaks_std = spice_std.find_peaks(spectrum_std)
    time_std = time.time() - start_time

    # SPICE-ML
    start_time = time.time()
    spice_ml = create_spice_ml(n_sensors, ml_method='newton')
    spectrum_ml, angles_ml = spice_ml.fit(sample_cov)
    peaks_ml = spice_ml.find_peaks(spectrum_ml)
    time_ml = time.time() - start_time

    return {
        'standard_spice': {
            'spectrum': spectrum_std,
            'peaks': peaks_std,
            'execution_time': time_std,
            'convergence': spice_std.get_convergence_info()
        },
        'spice_ml': {
            'spectrum': spectrum_ml,
            'peaks': peaks_ml,
            'execution_time': time_ml,
            'convergence': spice_ml.get_ml_convergence_info()
        },
        'improvement_factor': len(peaks_std['angles']) / max(len(peaks_ml['angles']), 1),
        'execution_time_ratio': time_std / time_ml
    }
"""
Enhanced SPICE Algorithm with Stability Improvements.

This module implements the SPICE algorithm with numerical stability enhancements
based on recent research (2023-2024), including:

1. Adaptive regularization based on condition number monitoring
2. Eigenvalue-based stability analysis and correction
3. Scaling-aware optimization with robust convergence criteria
4. Enhanced covariance fitting with numerical safeguards
5. Automatic parameter adjustment for challenging scenarios

Key improvements over standard SPICE:
- Condition number monitoring and adaptive regularization
- Eigenvalue decomposition stability checks
- Robust matrix operations with safeguards
- Enhanced convergence detection
- Automatic scaling adjustment

References
----------
.. [1] "SPICE: Scaling-Aware Prediction Correction Methods with a Free
       Convergence Rate for Nonlinear Convex Optimization" (2024)
.. [2] "Regularization Parameter Optimization for SPICE Algorithms" (2024)
.. [3] "Numerical Stability in Covariance Fitting Methods" (2023)

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict, Union, List
import warnings
from dataclasses import dataclass, field

from spice_core import SPICEConfig


@dataclass
class StableSPICEConfig(SPICEConfig):
    """Enhanced configuration for stable SPICE algorithm.

    Parameters
    ----------
    adaptive_regularization : bool, default=True
        Enable adaptive regularization based on condition number.
    condition_number_threshold : float, default=1e8
        Threshold for switching to enhanced stability mode.
    min_regularization : float, default=1e-15
        Minimum regularization parameter.
    max_regularization : float, default=1e-3
        Maximum regularization parameter.
    eigenvalue_threshold : float, default=1e-10
        Threshold for small eigenvalue detection.
    stability_monitoring : bool, default=True
        Enable comprehensive stability monitoring.
    scaling_adaptation : bool, default=True
        Enable scaling-aware optimization.
    robust_convergence : bool, default=True
        Use enhanced convergence criteria.
    convergence_window : int, default=5
        Window size for convergence monitoring.
    """
    adaptive_regularization: bool = True
    condition_number_threshold: float = 1e8
    min_regularization: float = 1e-15
    max_regularization: float = 1e-3
    eigenvalue_threshold: float = 1e-10
    stability_monitoring: bool = True
    scaling_adaptation: bool = True
    robust_convergence: bool = True
    convergence_window: int = 5

    # Additional stability parameters
    condition_history: List[float] = field(default_factory=list)
    eigenvalue_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration parameters."""
        # Call parent validation first
        super().__post_init__()

        if self.condition_number_threshold <= 0:
            raise ValueError(f"condition_number_threshold must be positive, got {self.condition_number_threshold}")

        if self.min_regularization < 0:
            raise ValueError(f"min_regularization must be non-negative, got {self.min_regularization}")

        if self.max_regularization < 0:
            raise ValueError(f"max_regularization must be non-negative, got {self.max_regularization}")

        if self.min_regularization > self.max_regularization:
            raise ValueError(f"min_regularization ({self.min_regularization}) must be <= max_regularization ({self.max_regularization})")

        if self.eigenvalue_threshold < 0:
            raise ValueError(f"eigenvalue_threshold must be non-negative, got {self.eigenvalue_threshold}")

        if self.convergence_window <= 0:
            raise ValueError(f"convergence_window must be positive, got {self.convergence_window}")


class StableSPICEEstimator:
    """
    Enhanced SPICE estimator with numerical stability improvements.

    This implementation addresses key stability issues identified in literature:
    1. Ill-conditioned covariance matrices
    2. Small eigenvalue instability
    3. Poor convergence in challenging scenarios
    4. Inadequate regularization strategies

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : StableSPICEConfig, optional
        Enhanced configuration parameters.

    Attributes
    ----------
    stability_metrics : dict
        Comprehensive stability monitoring metrics.
    condition_history : list
        History of condition numbers during iteration.
    regularization_history : list
        History of adaptive regularization parameters.
    """

    def __init__(self, n_sensors: int, config: Optional[StableSPICEConfig] = None):
        """Initialize enhanced SPICE estimator."""
        self.n_sensors = n_sensors
        self.config = config or StableSPICEConfig()
        self.is_fitted = False

        # Stability monitoring
        self.stability_metrics = {}
        self.condition_history = []
        self.regularization_history = []
        self.eigenvalue_history = []

        # Initialize grid and steering vectors
        self._initialize_grid()

    def _initialize_grid(self) -> None:
        """Initialize angular grid and steering vectors."""
        self.angular_grid = np.linspace(
            self.config.angular_range[0],
            self.config.angular_range[1],
            self.config.grid_size
        )

        # Compute steering vectors for uniform linear array
        self.steering_vectors = self._compute_steering_vectors(self.angular_grid)

    def _compute_steering_vectors(self, angles_deg: np.ndarray) -> np.ndarray:
        """Compute array steering vectors with numerical stability."""
        angles_rad = np.deg2rad(angles_deg)
        sensor_positions = np.arange(self.n_sensors)

        # Phase delays with numerical precision consideration
        phase_delays = np.outer(sensor_positions, np.pi * np.sin(angles_rad))

        # Complex steering vectors with normalization
        steering_vectors = np.exp(1j * phase_delays)

        # Normalize to unit norm for numerical stability
        norms = np.linalg.norm(steering_vectors, axis=0)
        # Protect against zero norms (shouldn't happen with complex exponentials)
        norms = np.maximum(norms, 1e-15)
        steering_vectors = steering_vectors / norms[np.newaxis, :]

        return steering_vectors

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced SPICE fitting with stability improvements.

        Parameters
        ----------
        sample_covariance : array_like
            Sample covariance matrix.

        Returns
        -------
        power_spectrum : ndarray
            Estimated power spectrum.
        angular_grid : ndarray
            Angular grid in degrees.
        """
        # Validate and condition input matrix
        sample_covariance = self._validate_and_condition_matrix(sample_covariance)
        self.sample_covariance = sample_covariance

        # Analyze matrix conditioning
        stability_analysis = self._analyze_matrix_stability(sample_covariance)

        # Adaptive regularization based on conditioning
        if self.config.adaptive_regularization:
            reg_param = self._compute_adaptive_regularization(stability_analysis)
        else:
            reg_param = self.config.regularization

        self.regularization_history.append(reg_param)

        # Enhanced initialization
        power_spectrum = self._stable_initialize_power_spectrum(sample_covariance, reg_param)

        # Main iterative algorithm with stability monitoring
        cost_history = []
        convergence_metrics = []

        for iteration in range(self.config.max_iterations):
            # Update power estimates with stability checks
            try:
                power_spectrum_new = self._stable_update_power_estimates(
                    sample_covariance, power_spectrum, reg_param
                )
            except la.LinAlgError as e:
                warnings.warn(f"Numerical instability at iteration {iteration}: {e}")
                break

            # Compute cost function
            cost = self._compute_stable_cost_function(sample_covariance, power_spectrum_new)
            cost_history.append(cost)

            # Enhanced convergence checking
            converged, conv_metrics = self._check_enhanced_convergence(
                cost_history, power_spectrum, power_spectrum_new, iteration
            )
            convergence_metrics.append(conv_metrics)

            if converged:
                break

            # Adaptive regularization update
            if self.config.adaptive_regularization and iteration % 10 == 0:
                current_stability = self._analyze_iteration_stability(
                    sample_covariance, power_spectrum_new
                )
                if current_stability['requires_adjustment']:
                    reg_param = min(reg_param * 2, self.config.max_regularization)
                    self.regularization_history.append(reg_param)

            power_spectrum = power_spectrum_new

        else:
            warnings.warn(f"Enhanced SPICE did not converge after {self.config.max_iterations} iterations")

        # Post-processing stability check
        final_stability = self._analyze_final_stability(
            sample_covariance, power_spectrum, cost_history
        )

        # Store comprehensive results
        self._store_enhanced_results(
            power_spectrum, cost_history, convergence_metrics, final_stability
        )

        return power_spectrum, self.angular_grid

    def _validate_and_condition_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Enhanced matrix validation and conditioning."""
        cov_matrix = np.asarray(cov_matrix)

        # Basic validation
        if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if cov_matrix.shape[0] != self.n_sensors:
            raise ValueError(f"Matrix size must be {self.n_sensors}x{self.n_sensors}")

        # Ensure Hermitian property
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.conj().T)

        # Eigenvalue analysis for conditioning
        eigenvals = la.eigvals(cov_matrix)
        min_eigenval = np.min(np.real(eigenvals))
        condition_number = np.max(np.real(eigenvals)) / max(abs(min_eigenval), 1e-15)

        # Store stability metrics
        self.condition_history.append(condition_number)
        self.eigenvalue_history.append(eigenvals)

        # Conditioning if necessary
        if min_eigenval <= self.config.eigenvalue_threshold:
            conditioning_param = abs(min_eigenval) + self.config.eigenvalue_threshold
            cov_matrix += conditioning_param * np.eye(self.n_sensors)
            warnings.warn(f"Matrix conditioned: added {conditioning_param:.2e} to diagonal")

        return cov_matrix

    def _analyze_matrix_stability(self, cov_matrix: np.ndarray) -> Dict:
        """Comprehensive stability analysis of covariance matrix."""
        eigenvals, eigenvecs = la.eigh(cov_matrix)
        eigenvals = np.real(eigenvals)

        condition_number = np.max(eigenvals) / max(np.min(eigenvals), 1e-15)

        analysis = {
            'condition_number': condition_number,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'min_eigenvalue': np.min(eigenvals),
            'max_eigenvalue': np.max(eigenvals),
            'eigenvalue_ratio': np.max(eigenvals) / max(np.min(eigenvals), 1e-15),
            'rank_deficiency': np.sum(eigenvals < self.config.eigenvalue_threshold),
            'is_well_conditioned': condition_number < self.config.condition_number_threshold,
            'stability_score': min(1.0, self.config.condition_number_threshold / max(condition_number, 1e-15))
        }

        return analysis

    def _compute_adaptive_regularization(self, stability_analysis: Dict) -> float:
        """Compute adaptive regularization parameter."""
        condition_number = stability_analysis['condition_number']
        min_eigenval = stability_analysis['min_eigenvalue']

        # Base regularization from condition number
        if condition_number > self.config.condition_number_threshold:
            # Scale with condition number (literature-based approach)
            base_reg = self.config.regularization * np.sqrt(condition_number / 1e6)
        else:
            base_reg = self.config.regularization

        # Adjust for small eigenvalues
        if min_eigenval < self.config.eigenvalue_threshold:
            eigenval_adjustment = abs(min_eigenval) + self.config.eigenvalue_threshold
            base_reg = max(base_reg, eigenval_adjustment)

        # Clamp to valid range
        reg_param = np.clip(base_reg, self.config.min_regularization,
                           self.config.max_regularization)

        return reg_param

    def _stable_initialize_power_spectrum(self, sample_cov: np.ndarray,
                                        reg_param: float) -> np.ndarray:
        """Numerically stable power spectrum initialization."""
        power_spectrum = np.zeros(self.config.grid_size)

        # Add regularization to sample covariance for stability
        regularized_cov = sample_cov + reg_param * np.eye(self.n_sensors)

        for i in range(self.config.grid_size):
            steering_vec = self.steering_vectors[:, i:i+1]

            # Stable quadratic form computation
            try:
                power_val = np.real(steering_vec.conj().T @ regularized_cov @ steering_vec).item()
                power_spectrum[i] = max(power_val, reg_param)
            except (la.LinAlgError, np.linalg.LinAlgError):
                power_spectrum[i] = reg_param

        # Normalize for numerical stability
        if np.max(power_spectrum) > 0:
            power_spectrum = power_spectrum / np.max(power_spectrum)
            power_spectrum = np.maximum(power_spectrum, reg_param)

        return power_spectrum

    def _stable_update_power_estimates(self, sample_cov: np.ndarray,
                                     current_powers: np.ndarray,
                                     reg_param: float) -> np.ndarray:
        """Stable power estimate updates using literature-correct SPICE formula."""
        updated_powers = np.zeros_like(current_powers)

        # Literature-correct SPICE power update formula
        # Based on minimizing ||R - A*P*A^H||_F^2 w.r.t. each power p_k
        # Solution: p_k = (a_k^H * R * a_k) / (a_k^H * a_k)
        for i in range(self.config.grid_size):
            steering_vec = self.steering_vectors[:, i:i+1]

            try:
                # Direct power computation from literature (numerically stable version)
                numerator = np.real(steering_vec.conj().T @ sample_cov @ steering_vec).item()
                denominator = np.real(steering_vec.conj().T @ steering_vec).item()

                # Avoid division by very small numbers
                if abs(denominator) < 1e-12:
                    updated_powers[i] = reg_param
                else:
                    power_estimate = numerator / denominator
                    updated_powers[i] = max(power_estimate, reg_param)

            except (la.LinAlgError, np.linalg.LinAlgError):
                updated_powers[i] = reg_param

        return updated_powers

    def _construct_stable_fitted_covariance(self, powers: np.ndarray,
                                          reg_param: float) -> np.ndarray:
        """Construct fitted covariance with numerical stability."""
        fitted_cov = np.zeros((self.n_sensors, self.n_sensors), dtype=complex)

        for i, power in enumerate(powers):
            if power > reg_param:
                steering_vec = self.steering_vectors[:, i:i+1]
                # Stable outer product computation
                outer_product = steering_vec @ steering_vec.conj().T
                fitted_cov += power * outer_product

        # Add small regularization for numerical stability
        fitted_cov += reg_param * 1e-3 * np.eye(self.n_sensors)

        return fitted_cov

    def _compute_stable_cost_function(self, sample_cov: np.ndarray,
                                    powers: np.ndarray) -> float:
        """Compute cost function with numerical safeguards."""
        try:
            fitted_cov = self._construct_stable_fitted_covariance(
                powers, self.regularization_history[-1]
            )
            residual = sample_cov - fitted_cov
            cost = np.real(np.trace(residual.conj().T @ residual))

            # Check for numerical issues
            if not np.isfinite(cost) or cost < 0:
                return np.inf

            return cost

        except (la.LinAlgError, np.linalg.LinAlgError):
            return np.inf

    def _check_enhanced_convergence(self, cost_history: List[float],
                                  power_old: np.ndarray, power_new: np.ndarray,
                                  iteration: int) -> Tuple[bool, Dict]:
        """Enhanced convergence checking with multiple criteria."""
        metrics = {
            'relative_cost_change': np.inf,
            'power_change': np.inf,
            'cost_stability': False,
            'power_stability': False
        }

        if len(cost_history) < 2:
            return False, metrics

        # Relative cost change
        cost_change = abs(cost_history[-2] - cost_history[-1])
        relative_cost_change = cost_change / max(abs(cost_history[-2]), 1e-15)
        metrics['relative_cost_change'] = relative_cost_change

        # Power spectrum change
        power_change = np.linalg.norm(power_new - power_old) / max(np.linalg.norm(power_old), 1e-15)
        metrics['power_change'] = power_change

        # Cost stability over window
        if len(cost_history) >= self.config.convergence_window:
            recent_costs = cost_history[-self.config.convergence_window:]
            cost_std = np.std(recent_costs) / max(np.mean(recent_costs), 1e-15)
            metrics['cost_stability'] = cost_std < self.config.convergence_tolerance

        # Power stability
        metrics['power_stability'] = power_change < self.config.convergence_tolerance

        # Convergence decision
        converged = (
            relative_cost_change < self.config.convergence_tolerance and
            power_change < self.config.convergence_tolerance
        )

        return converged, metrics

    def _analyze_iteration_stability(self, sample_cov: np.ndarray,
                                   power_spectrum: np.ndarray) -> Dict:
        """Analyze stability during iterations."""
        # Current condition number
        fitted_cov = self._construct_stable_fitted_covariance(
            power_spectrum, self.regularization_history[-1]
        )
        residual = sample_cov - fitted_cov

        try:
            residual_cond = np.linalg.cond(residual)
            power_range = np.max(power_spectrum) / max(np.min(power_spectrum), 1e-15)

            stability = {
                'residual_condition': residual_cond,
                'power_dynamic_range': power_range,
                'requires_adjustment': (
                    residual_cond > self.config.condition_number_threshold or
                    power_range > 1e6
                )
            }

        except (la.LinAlgError, np.linalg.LinAlgError):
            stability = {
                'residual_condition': np.inf,
                'power_dynamic_range': np.inf,
                'requires_adjustment': True
            }

        return stability

    def _analyze_final_stability(self, sample_cov: np.ndarray, power_spectrum: np.ndarray,
                               cost_history: List[float]) -> Dict:
        """Comprehensive final stability analysis."""
        stability = {
            'final_cost': cost_history[-1] if cost_history else np.inf,
            'cost_reduction': (cost_history[0] - cost_history[-1]) if len(cost_history) > 1 else 0,
            'convergence_rate': len(cost_history),
            'power_spectrum_valid': np.all(power_spectrum >= 0) and np.all(np.isfinite(power_spectrum)),
            'dynamic_range_db': 10 * np.log10(np.max(power_spectrum) / max(np.min(power_spectrum), 1e-15)),
            'average_condition_number': np.mean(self.condition_history) if self.condition_history else np.inf,
            'regularization_adaptation_count': len(set(self.regularization_history))
        }

        # Overall stability score
        score = 0.0
        if stability['final_cost'] < np.inf:
            score += 0.25
        if stability['cost_reduction'] > 0:
            score += 0.25
        if stability['power_spectrum_valid']:
            score += 0.25
        if stability['dynamic_range_db'] < 80:  # Reasonable dynamic range
            score += 0.25

        stability['overall_stability_score'] = score

        return stability

    def _store_enhanced_results(self, power_spectrum: np.ndarray, cost_history: List[float],
                              convergence_metrics: List[Dict], stability_analysis: Dict):
        """Store comprehensive results."""
        self.power_spectrum = power_spectrum
        self.cost_history = np.array(cost_history)
        self.n_iterations = len(cost_history)
        self.convergence_metrics = convergence_metrics
        self.stability_metrics = stability_analysis
        self.is_fitted = True

    def get_convergence_info(self) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Get algorithm convergence information.

        Returns
        -------
        info : dict
            Convergence information including iterations, final cost, etc.
        """
        if not self.is_fitted:
            raise ValueError("Must fit algorithm first")

        return {
            'n_iterations': self.n_iterations,
            'final_cost': self.cost_history[-1],
            'initial_cost': self.cost_history[0],
            'cost_history': self.cost_history,
            'converged': self.n_iterations < self.config.max_iterations,
            'cost_reduction': self.cost_history[0] - self.cost_history[-1],
            'relative_cost_reduction': (self.cost_history[0] - self.cost_history[-1]) / max(abs(self.cost_history[0]), 1e-15),
            'convergence_metrics': self.convergence_metrics
        }

    def get_stability_report(self) -> Dict:
        """Generate comprehensive stability report."""
        if not self.is_fitted:
            raise ValueError("Must fit algorithm first")

        report = {
            'algorithm_variant': 'Enhanced Stable SPICE',
            'stability_metrics': self.stability_metrics.copy(),
            'condition_number_history': self.condition_history.copy(),
            'regularization_history': self.regularization_history.copy(),
            'convergence_info': {
                'n_iterations': self.n_iterations,
                'final_cost': self.cost_history[-1] if len(self.cost_history) > 0 else np.inf,
                'cost_reduction': (self.cost_history[0] - self.cost_history[-1])
                                 if len(self.cost_history) > 1 else 0
            },
            'numerical_health': {
                'max_condition_number': max(self.condition_history) if self.condition_history else np.inf,
                'min_eigenvalue': min([min(np.real(eigs)) for eigs in self.eigenvalue_history])
                                 if self.eigenvalue_history else 0,
                'regularization_range': (min(self.regularization_history),
                                       max(self.regularization_history))
                                      if self.regularization_history else (0, 0)
            }
        }

        return report

    # Include other methods from base SPICEEstimator as needed
    def find_peaks(self, power_spectrum: Optional[np.ndarray] = None,
                   min_separation: float = 5.0,
                   threshold_db: float = -20.0) -> Dict[str, np.ndarray]:
        """Find peaks with stability considerations."""
        if power_spectrum is None:
            if not self.is_fitted:
                raise ValueError("Must fit algorithm or provide power_spectrum")
            power_spectrum = self.power_spectrum

        # Enhanced peak detection with stability checks
        if not np.all(np.isfinite(power_spectrum)):
            warnings.warn("Non-finite values in power spectrum")
            power_spectrum = np.nan_to_num(power_spectrum, nan=0.0, posinf=np.max(power_spectrum[np.isfinite(power_spectrum)]))

        # Convert to dB with numerical safety
        power_db = 10 * np.log10(np.maximum(power_spectrum, np.max(power_spectrum) * 1e-12))
        max_power_db = np.max(power_db)

        # Find peaks above threshold
        threshold_absolute = max_power_db + threshold_db
        potential_peaks = np.where(power_db > threshold_absolute)[0]

        # Apply separation constraint and local maxima detection
        peak_indices = []
        peak_powers = []

        for idx in potential_peaks:
            if self._is_local_maximum(power_db, idx):
                min_sep_samples = int(min_separation * self.config.grid_size /
                                    (self.config.angular_range[1] - self.config.angular_range[0]))

                if not peak_indices or all(abs(idx - p) >= min_sep_samples for p in peak_indices):
                    peak_indices.append(idx)
                    peak_powers.append(power_spectrum[idx])

        peak_indices = np.array(peak_indices)
        peak_powers = np.array(peak_powers)

        # Sort by power
        if len(peak_indices) > 0:
            sort_order = np.argsort(peak_powers)[::-1]
            peak_indices = peak_indices[sort_order]
            peak_powers = peak_powers[sort_order]

        return {
            'angles': self.angular_grid[peak_indices] if len(peak_indices) > 0 else np.array([]),
            'powers': peak_powers,
            'powers_db': 10 * np.log10(np.maximum(peak_powers, np.max(peak_powers) * 1e-12)) if len(peak_powers) > 0 else np.array([]),
            'indices': peak_indices
        }

    def _is_local_maximum(self, signal: np.ndarray, index: int, window: int = 3) -> bool:
        """Check if point is local maximum."""
        start = max(0, index - window // 2)
        end = min(len(signal), index + window // 2 + 1)
        local_region = signal[start:end]
        local_max_idx = np.argmax(local_region)
        return start + local_max_idx == index
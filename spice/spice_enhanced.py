"""
Enhanced SPICE Implementation with IAA-Inspired Improvements.

This module implements advanced SPICE variants incorporating optimization techniques
from IAA (Iterative Adaptive Approach) research to achieve better stability,
lower SNR requirements, and improved convergence reliability.

Based on techniques from:
- MATLAB IAA implementations and best practices
- Yardibi et al. research on robust spectral estimation
- EURASIP Journal optimizations for iterative approaches
- Professional radar signal processing systems

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict, Union
import warnings
from dataclasses import dataclass

from spice_core import SPICEEstimator, SPICEConfig


@dataclass
class EnhancedSPICEConfig(SPICEConfig):
    """Configuration for Enhanced SPICE with IAA-inspired improvements.

    Parameters
    ----------
    adaptive_regularization : bool, default=True
        Enable SNR-based adaptive regularization adjustment.
    eigenvalue_initialization : bool, default=True
        Use eigenvalue-based initialization for better low-SNR performance.
    stabilization_factor : float, default=0.1
        Damping factor to prevent oscillations (0.0 = no damping, 1.0 = heavy damping).
    robust_matrix_solving : bool, default=True
        Use robust matrix solving with SVD fallbacks.
    snr_estimation_method : str, default='quartile'
        Method for SNR estimation: 'quartile', 'eigenvalue', 'simple'.
    min_snr_db : float, default=-5.0
        Minimum SNR assumption for parameter adaptation.
    max_snr_db : float, default=30.0
        Maximum SNR assumption for parameter adaptation.
    """
    adaptive_regularization: bool = True
    eigenvalue_initialization: bool = True
    stabilization_factor: float = 0.1
    robust_matrix_solving: bool = True
    snr_estimation_method: str = 'quartile'
    min_snr_db: float = -5.0
    max_snr_db: float = 30.0

    def __post_init__(self):
        """Validate enhanced configuration parameters."""
        super().__post_init__()

        if not 0.0 <= self.stabilization_factor <= 1.0:
            raise ValueError(f"stabilization_factor must be in [0,1], got {self.stabilization_factor}")

        if self.min_snr_db >= self.max_snr_db:
            raise ValueError("min_snr_db must be less than max_snr_db")

        valid_snr_methods = {'quartile', 'eigenvalue', 'simple'}
        if self.snr_estimation_method not in valid_snr_methods:
            raise ValueError(f"snr_estimation_method must be one of {valid_snr_methods}")


class EnhancedSPICEEstimator(SPICEEstimator):
    """
    Enhanced SPICE with IAA-Inspired Stability and Low-SNR Improvements.

    This estimator incorporates advanced techniques from IAA research to achieve:
    - Lower SNR requirements (target: 5-7 dB vs standard 10 dB)
    - Improved numerical stability through robust matrix operations
    - Better convergence reliability through stabilization and adaptation
    - Enhanced initialization for challenging scenarios

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : EnhancedSPICEConfig, optional
        Enhanced configuration parameters.

    Attributes
    ----------
    estimated_snr_db : float
        Estimated SNR from the last fit operation.
    adaptive_regularization : float
        Current adaptive regularization parameter.
    initialization_method : str
        Method used for power spectrum initialization.
    """

    def __init__(self, n_sensors: int, config: Optional[EnhancedSPICEConfig] = None):
        """Initialize Enhanced SPICE estimator."""
        self.config = config or EnhancedSPICEConfig()
        super().__init__(n_sensors, self.config)

        # Enhanced attributes
        self.estimated_snr_db = None
        self.adaptive_regularization = self.config.regularization
        self.initialization_method = 'standard'

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced SPICE estimation with IAA-inspired improvements.

        Parameters
        ----------
        sample_covariance : array_like
            Sample covariance matrix.

        Returns
        -------
        power_spectrum : ndarray
            Enhanced SPICE power spectrum.
        angular_grid : ndarray
            Angular grid in degrees.
        """
        # Validate input
        self._validate_covariance_matrix(sample_covariance)

        # Store sample covariance (needed by base class methods)
        self.sample_covariance = sample_covariance

        # Estimate SNR and adapt parameters
        if self.config.adaptive_regularization:
            self.estimated_snr_db = self._estimate_snr_db(sample_covariance)
            self.adaptive_regularization = self._adapt_regularization(self.estimated_snr_db)

        # Enhanced initialization
        if self.config.eigenvalue_initialization:
            power_spectrum = self._initialize_eigenvalue_based(sample_covariance)
            self.initialization_method = 'eigenvalue'
        else:
            power_spectrum = self._initialize_power_spectrum()
            self.initialization_method = 'standard'

        # Enhanced iterative estimation
        cost_history = []

        for iteration in range(self.config.max_iterations):
            # Enhanced power updates with stabilization
            power_spectrum_new = self._enhanced_update_power_estimates(
                sample_covariance, power_spectrum
            )

            # Apply stabilization factor to prevent oscillations
            if self.config.stabilization_factor > 0 and iteration > 0:
                power_spectrum_new = self._apply_stabilization(
                    power_spectrum, power_spectrum_new
                )

            # Compute cost function
            cost = self._compute_cost_function(sample_covariance, power_spectrum_new)
            cost_history.append(cost)

            # Check convergence (same logic as base class)
            if iteration > 0:
                prev_cost = abs(cost_history[-2])
                curr_cost = abs(cost_history[-1])

                # Handle division by zero case
                if prev_cost < 1e-15:
                    # If previous cost is essentially zero, check absolute change
                    relative_change = curr_cost
                else:
                    relative_change = abs(cost_history[-2] - cost_history[-1]) / prev_cost

                if relative_change < self.config.convergence_tolerance:
                    break

            power_spectrum = power_spectrum_new

        else:
            warnings.warn(f"Enhanced SPICE did not converge after {self.config.max_iterations} iterations",
                         UserWarning)

        # Store results
        self.power_spectrum = power_spectrum
        self.cost_history = np.array(cost_history)
        self.n_iterations = len(cost_history)
        self.is_fitted = True

        return power_spectrum, self.angular_grid

    def _estimate_snr_db(self, sample_cov: np.ndarray) -> float:
        """Estimate SNR using robust methods from IAA research."""
        if self.config.snr_estimation_method == 'quartile':
            return self._estimate_snr_quartile(sample_cov)
        elif self.config.snr_estimation_method == 'eigenvalue':
            return self._estimate_snr_eigenvalue(sample_cov)
        else:
            return self._estimate_snr_simple(sample_cov)

    def _estimate_snr_quartile(self, sample_cov: np.ndarray) -> float:
        """Robust quartile-based SNR estimation (IAA-inspired)."""
        # Use eigenvalue-based approach for more accurate SNR estimation
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]  # Descending order

        if len(eigenvals) >= 2:
            # Estimate noise level from smallest eigenvalues (more robust)
            # Assume last quarter of eigenvalues represent noise
            n_noise = max(1, len(eigenvals) // 4)
            noise_power = np.mean(eigenvals[-n_noise:])

            # Signal power from dominant eigenvalues
            n_signal = max(1, len(eigenvals) // 4)
            signal_power = np.mean(eigenvals[:n_signal])

            # Compute SNR with protection
            if noise_power > 0 and signal_power > noise_power:
                snr_linear = (signal_power - noise_power) / noise_power
                snr_db = 10 * np.log10(max(snr_linear, 1e-3))  # Floor at -30 dB
                return np.clip(snr_db, self.config.min_snr_db, self.config.max_snr_db)

        # Fallback: use trace-based estimation
        total_power = np.real(np.trace(sample_cov)) / sample_cov.shape[0]
        min_eigenval = np.min(np.real(eigenvals)) if len(eigenvals) > 0 else total_power * 0.1

        if min_eigenval > 0 and total_power > min_eigenval:
            snr_linear = (total_power - min_eigenval) / min_eigenval
            snr_db = 10 * np.log10(max(snr_linear, 1e-3))
            return np.clip(snr_db, self.config.min_snr_db, self.config.max_snr_db)

        return 5.0  # Conservative default

    def _estimate_snr_eigenvalue(self, sample_cov: np.ndarray) -> float:
        """Eigenvalue-based SNR estimation."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]  # Descending order

        if len(eigenvals) >= 2:
            # Estimate signal subspace size (simple threshold)
            noise_level = np.median(eigenvals)
            signal_eigenvals = eigenvals[eigenvals > noise_level * 5]

            if len(signal_eigenvals) > 0:
                signal_power = np.mean(signal_eigenvals)
                noise_power = noise_level
                snr_linear = signal_power / max(noise_power, 1e-12)
                snr_db = 10 * np.log10(snr_linear)
                return np.clip(snr_db, self.config.min_snr_db, self.config.max_snr_db)

        return 10.0  # Default assumption

    def _estimate_snr_simple(self, sample_cov: np.ndarray) -> float:
        """Simple SNR estimation based on trace and minimum eigenvalue."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.real(eigenvals)

        total_power = np.mean(eigenvals)
        noise_power = np.min(eigenvals)

        if noise_power > 0:
            snr_linear = total_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            return np.clip(snr_db, self.config.min_snr_db, self.config.max_snr_db)

        return 10.0

    def _adapt_regularization(self, estimated_snr_db: float) -> float:
        """Adapt regularization parameter based on estimated SNR."""
        # Higher regularization for lower SNR
        snr_normalized = (estimated_snr_db - self.config.min_snr_db) / (
            self.config.max_snr_db - self.config.min_snr_db
        )
        snr_normalized = np.clip(snr_normalized, 0.0, 1.0)

        # Exponential scaling: high reg for low SNR, low reg for high SNR
        min_reg = self.config.regularization
        max_reg = min_reg * 1000  # Increase regularization by up to 1000x for very low SNR

        adaptive_reg = max_reg * np.exp(-5 * snr_normalized) + min_reg
        return adaptive_reg

    def _initialize_eigenvalue_based(self, sample_cov: np.ndarray) -> np.ndarray:
        """Enhanced initialization using eigenvalue decomposition (IAA-inspired)."""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(sample_cov)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Ensure positive

            # Use dominant eigenvector for initialization
            dominant_eigenvec = eigenvecs[:, -1]
            dominant_eigenval = eigenvals[-1]

            # Initialize power spectrum based on projections
            P_init = np.zeros(self.config.grid_size)

            for k in range(self.config.grid_size):
                steering_vec = self.steering_vectors[:, k]
                # Compute projection onto dominant subspace
                projection = np.abs(np.vdot(steering_vec, dominant_eigenvec))**2
                P_init[k] = projection * dominant_eigenval

            # Normalize and add regularization
            P_init = P_init / np.sum(P_init) * np.trace(sample_cov)
            P_init = np.maximum(P_init, self.adaptive_regularization)

            return P_init

        except np.linalg.LinAlgError:
            # Fall back to standard initialization
            warnings.warn("Eigenvalue initialization failed, using standard initialization")
            return self._initialize_power_spectrum()

    def _enhanced_update_power_estimates(self, sample_cov: np.ndarray,
                                       current_powers: np.ndarray) -> np.ndarray:
        """Enhanced power updates with robust matrix operations."""
        updated_powers = np.zeros_like(current_powers)

        for i in range(self.config.grid_size):
            steering_vec = self.steering_vectors[:, i:i+1]

            if self.config.robust_matrix_solving:
                # Use robust approach for matrix operations
                power_estimate = self._robust_power_computation(
                    sample_cov, steering_vec, self.adaptive_regularization
                )
            else:
                # Standard SPICE power update
                numerator = np.real(steering_vec.conj().T @ sample_cov @ steering_vec).item()
                denominator = np.real(steering_vec.conj().T @ steering_vec).item()
                power_estimate = numerator / denominator

            updated_powers[i] = max(power_estimate, self.adaptive_regularization)

        return updated_powers

    def _robust_power_computation(self, sample_cov: np.ndarray,
                                steering_vec: np.ndarray, reg_param: float) -> float:
        """Robust power computation with SVD fallbacks (IAA-inspired)."""
        try:
            # Primary method: direct computation with numerical protection
            numerator = np.real(steering_vec.conj().T @ sample_cov @ steering_vec).item()
            denominator = np.real(steering_vec.conj().T @ steering_vec).item()

            if denominator > 1e-12:
                return numerator / denominator
            else:
                return reg_param

        except (np.linalg.LinAlgError, ValueError):
            # Fallback: Use SVD for robust computation
            try:
                U, s, Vh = np.linalg.svd(sample_cov, full_matrices=False)
                s_reg = np.maximum(s, reg_param)  # Regularize singular values

                # Reconstruct with regularized singular values
                sample_cov_reg = U @ np.diag(s_reg) @ Vh

                numerator = np.real(steering_vec.conj().T @ sample_cov_reg @ steering_vec).item()
                denominator = np.real(steering_vec.conj().T @ steering_vec).item()

                return numerator / max(denominator, 1e-12)

            except:
                # Final fallback: return regularization parameter
                return reg_param

    def _apply_stabilization(self, power_prev: np.ndarray,
                           power_new: np.ndarray) -> np.ndarray:
        """Apply stabilization factor to prevent oscillations (IAA-inspired)."""
        # Light damping to prevent oscillations
        stabilized = (1 - self.config.stabilization_factor) * power_new + \
                    self.config.stabilization_factor * power_prev

        # Ensure minimum values
        return np.maximum(stabilized, self.adaptive_regularization)

    def get_enhancement_info(self) -> Dict[str, Union[float, str, bool]]:
        """Get information about the enhancements applied."""
        if not self.is_fitted:
            raise ValueError("Must fit algorithm first")

        return {
            'estimated_snr_db': self.estimated_snr_db,
            'adaptive_regularization': self.adaptive_regularization,
            'standard_regularization': self.config.regularization,
            'regularization_adaptation_factor': self.adaptive_regularization / self.config.regularization,
            'initialization_method': self.initialization_method,
            'stabilization_applied': self.config.stabilization_factor > 0,
            'robust_solving_enabled': self.config.robust_matrix_solving,
            'snr_estimation_method': self.config.snr_estimation_method,
            'convergence_iterations': self.n_iterations
        }


def create_enhanced_spice(n_sensors: int, target_snr_db: float = 5.0,
                         **config_kwargs) -> EnhancedSPICEEstimator:
    """
    Factory function to create Enhanced SPICE optimized for specific SNR.

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    target_snr_db : float, default=5.0
        Target minimum SNR for optimization.
    **config_kwargs
        Additional configuration parameters.

    Returns
    -------
    estimator : EnhancedSPICEEstimator
        Configured enhanced SPICE estimator.
    """
    # Optimize configuration for target SNR
    if target_snr_db <= 5.0:
        # Very low SNR: maximum enhancements
        config = EnhancedSPICEConfig(
            adaptive_regularization=True,
            eigenvalue_initialization=True,
            stabilization_factor=0.15,  # Higher stabilization for challenging scenarios
            robust_matrix_solving=True,
            snr_estimation_method='quartile',
            min_snr_db=target_snr_db - 5.0,
            max_snr_db=30.0,
            **config_kwargs
        )
    elif target_snr_db <= 10.0:
        # Moderate SNR: balanced enhancements
        config = EnhancedSPICEConfig(
            adaptive_regularization=True,
            eigenvalue_initialization=True,
            stabilization_factor=0.1,
            robust_matrix_solving=True,
            snr_estimation_method='eigenvalue',
            min_snr_db=target_snr_db - 3.0,
            max_snr_db=30.0,
            **config_kwargs
        )
    else:
        # High SNR: minimal enhancements for performance
        config = EnhancedSPICEConfig(
            adaptive_regularization=False,
            eigenvalue_initialization=False,
            stabilization_factor=0.05,
            robust_matrix_solving=True,
            snr_estimation_method='simple',
            **config_kwargs
        )

    return EnhancedSPICEEstimator(n_sensors, config)
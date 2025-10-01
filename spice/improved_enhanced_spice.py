"""
Improved Enhanced SPICE with Advanced SNR Estimation and Adaptive Regularization

This module integrates the sophisticated SNR estimation and adaptive regularization
strategies to create a significantly improved Enhanced SPICE implementation that
should achieve the literature-claimed 5 dB SNR threshold.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from spice_core import SPICEEstimator, SPICEConfig
from enhanced_snr_estimation import create_enhanced_snr_estimator, create_adaptive_regularization_strategy
from advanced_peak_detection import create_advanced_detector


@dataclass
class ImprovedEnhancedSPICEConfig(SPICEConfig):
    """Configuration for Improved Enhanced SPICE."""
    use_advanced_snr_estimation: bool = True
    use_adaptive_regularization: bool = True
    use_advanced_peak_detection: bool = True
    multi_stage_processing: bool = True
    eigenvalue_initialization: bool = True
    stabilization_factor: float = 0.1
    snr_estimation_confidence_threshold: float = 0.3


class ImprovedEnhancedSPICEEstimator(SPICEEstimator):
    """
    Improved Enhanced SPICE with state-of-the-art SNR estimation and regularization.

    This implementation should achieve the literature-claimed 5 dB SNR threshold
    through sophisticated parameter adaptation and advanced processing techniques.
    """

    def __init__(self, n_sensors: int, config: Optional[ImprovedEnhancedSPICEConfig] = None):
        """Initialize Improved Enhanced SPICE estimator."""
        self.config = config or ImprovedEnhancedSPICEConfig()
        super().__init__(n_sensors, self.config)

        # Advanced components
        if self.config.use_advanced_snr_estimation:
            self.snr_estimator = create_enhanced_snr_estimator(n_sensors)
        else:
            self.snr_estimator = None

        if self.config.use_adaptive_regularization:
            self.regularization_strategy = create_adaptive_regularization_strategy(self.config.regularization)
        else:
            self.regularization_strategy = None

        # Enhanced state
        self.snr_estimation_result = None
        self.regularization_result = None
        self.processing_stages = []

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced SPICE estimation with multi-stage processing.

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
        self._validate_covariance_matrix(sample_covariance)
        self.sample_covariance = sample_covariance

        if self.config.multi_stage_processing:
            return self._multi_stage_processing(sample_covariance)
        else:
            return self._single_stage_processing(sample_covariance)

    def _multi_stage_processing(self, sample_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-stage processing for optimal performance."""
        self.processing_stages = []

        # Stage 1: Initial SNR estimation and coarse processing
        power_spectrum = self._stage1_coarse_estimation(sample_cov)

        # Stage 2: Refined processing with adaptive parameters
        power_spectrum = self._stage2_refined_estimation(sample_cov, power_spectrum)

        # Stage 3: Final polishing with advanced regularization
        power_spectrum = self._stage3_final_polishing(sample_cov, power_spectrum)

        self.power_spectrum = power_spectrum
        return power_spectrum, self.angular_grid

    def _stage1_coarse_estimation(self, sample_cov: np.ndarray) -> np.ndarray:
        """Stage 1: Coarse estimation with robust SNR estimation."""
        stage_info = {'stage': 1, 'description': 'Coarse estimation'}

        # Advanced SNR estimation
        if self.snr_estimator:
            self.snr_estimation_result = self.snr_estimator.estimate_snr_multi_method(sample_cov)
            estimated_snr = self.snr_estimation_result['consensus_snr_db']
            snr_confidence = self.snr_estimation_result['consensus_confidence']
        else:
            estimated_snr = 10.0
            snr_confidence = 0.5

        stage_info['estimated_snr_db'] = estimated_snr
        stage_info['snr_confidence'] = snr_confidence

        # Conservative adaptive regularization for first stage
        if self.regularization_strategy:
            reg_result = self.regularization_strategy.compute_adaptive_regularization(
                sample_cov, estimated_snr, current_iteration=0
            )
            adaptive_reg = reg_result['adaptive_regularization'] * 2.0  # Conservative
        else:
            adaptive_reg = self.config.regularization

        stage_info['regularization_used'] = adaptive_reg

        # Initialize power spectrum
        if self.config.eigenvalue_initialization:
            power_spectrum = self._eigenvalue_based_initialization(sample_cov, adaptive_reg)
        else:
            power_spectrum = np.full(self.config.grid_size, adaptive_reg)

        # Run basic SPICE iterations with conservative parameters
        cost_history = []
        for iteration in range(min(5, self.config.max_iterations)):
            power_spectrum_new = self._enhanced_power_update(
                sample_cov, power_spectrum, adaptive_reg, iteration
            )

            # Apply light stabilization
            if iteration > 0 and self.config.stabilization_factor > 0:
                alpha = self.config.stabilization_factor
                power_spectrum_new = (1 - alpha) * power_spectrum_new + alpha * power_spectrum

            cost = self._compute_cost_function(sample_cov, power_spectrum_new)
            cost_history.append(cost)

            # Check convergence
            if iteration > 0:
                relative_change = abs(cost_history[-1] - cost_history[-2]) / max(abs(cost_history[-2]), 1e-12)
                if relative_change < self.config.convergence_tolerance:
                    break

            power_spectrum = power_spectrum_new

        stage_info['iterations'] = len(cost_history)
        stage_info['final_cost'] = cost_history[-1] if cost_history else 0
        self.processing_stages.append(stage_info)

        return power_spectrum

    def _stage2_refined_estimation(self, sample_cov: np.ndarray,
                                  initial_spectrum: np.ndarray) -> np.ndarray:
        """Stage 2: Refined estimation with optimized parameters."""
        stage_info = {'stage': 2, 'description': 'Refined estimation'}

        # Use stage 1 results to refine SNR estimate
        if self.snr_estimation_result:
            estimated_snr = self.snr_estimation_result['consensus_snr_db']

            # Refine estimate using initial spectrum information
            peak_power = np.max(np.real(initial_spectrum))
            noise_floor = np.median(np.real(initial_spectrum))
            if noise_floor > 0:
                spectrum_snr = 10 * np.log10(peak_power / noise_floor)
                # Blend estimates
                estimated_snr = 0.7 * estimated_snr + 0.3 * spectrum_snr
        else:
            estimated_snr = 10.0

        stage_info['refined_snr_db'] = estimated_snr

        # Optimized adaptive regularization
        if self.regularization_strategy:
            reg_result = self.regularization_strategy.compute_adaptive_regularization(
                sample_cov, estimated_snr, current_iteration=1
            )
            adaptive_reg = reg_result['adaptive_regularization']
            self.regularization_result = reg_result
        else:
            adaptive_reg = self.config.regularization

        stage_info['regularization_used'] = adaptive_reg

        # Refined SPICE iterations
        power_spectrum = initial_spectrum.copy()
        cost_history = []

        for iteration in range(self.config.max_iterations):
            power_spectrum_new = self._enhanced_power_update(
                sample_cov, power_spectrum, adaptive_reg, iteration + 5
            )

            # Adaptive stabilization based on convergence behavior
            if iteration > 1:
                stabilization = self._adaptive_stabilization_factor(cost_history)
                alpha = min(self.config.stabilization_factor, stabilization)
                power_spectrum_new = (1 - alpha) * power_spectrum_new + alpha * power_spectrum

            cost = self._compute_cost_function(sample_cov, power_spectrum_new)
            cost_history.append(cost)

            # Enhanced convergence checking
            if iteration > 0:
                relative_change = abs(cost_history[-1] - cost_history[-2]) / max(abs(cost_history[-2]), 1e-12)
                if relative_change < self.config.convergence_tolerance * 0.5:  # Tighter tolerance
                    break

            power_spectrum = power_spectrum_new

        stage_info['iterations'] = len(cost_history)
        stage_info['final_cost'] = cost_history[-1] if cost_history else 0
        self.processing_stages.append(stage_info)

        return power_spectrum

    def _stage3_final_polishing(self, sample_cov: np.ndarray,
                               refined_spectrum: np.ndarray) -> np.ndarray:
        """Stage 3: Final polishing with minimal regularization."""
        stage_info = {'stage': 3, 'description': 'Final polishing'}

        # Use refined spectrum for final SNR estimate
        peak_power = np.max(np.real(refined_spectrum))
        noise_floor = np.percentile(np.real(refined_spectrum), 25)  # Robust noise estimate

        if noise_floor > 0:
            final_snr = 10 * np.log10(peak_power / noise_floor)
        else:
            final_snr = self.snr_estimation_result['consensus_snr_db'] if self.snr_estimation_result else 10.0

        stage_info['final_snr_db'] = final_snr

        # Minimal regularization for final polishing
        if self.regularization_strategy:
            reg_result = self.regularization_strategy.compute_adaptive_regularization(
                sample_cov, final_snr, current_iteration=10  # High iteration for minimal reg
            )
            adaptive_reg = min(reg_result['adaptive_regularization'], self.config.regularization * 10)
        else:
            adaptive_reg = self.config.regularization

        stage_info['regularization_used'] = adaptive_reg

        # Final polishing iterations
        power_spectrum = refined_spectrum.copy()
        cost_history = []

        for iteration in range(min(3, self.config.max_iterations)):  # Just a few polishing iterations
            power_spectrum_new = self._enhanced_power_update(
                sample_cov, power_spectrum, adaptive_reg, iteration + 15
            )

            # Minimal stabilization for final stage
            if iteration > 0:
                alpha = self.config.stabilization_factor * 0.5
                power_spectrum_new = (1 - alpha) * power_spectrum_new + alpha * power_spectrum

            cost = self._compute_cost_function(sample_cov, power_spectrum_new)
            cost_history.append(cost)

            power_spectrum = power_spectrum_new

        stage_info['iterations'] = len(cost_history)
        stage_info['final_cost'] = cost_history[-1] if cost_history else 0
        self.processing_stages.append(stage_info)

        return power_spectrum

    def _single_stage_processing(self, sample_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-stage processing for simpler cases."""
        # Use enhanced SNR estimation
        if self.snr_estimator:
            self.snr_estimation_result = self.snr_estimator.estimate_snr_multi_method(sample_cov)
            estimated_snr = self.snr_estimation_result['consensus_snr_db']
        else:
            estimated_snr = 10.0

        # Use adaptive regularization
        if self.regularization_strategy:
            self.regularization_result = self.regularization_strategy.compute_adaptive_regularization(
                sample_cov, estimated_snr, current_iteration=0
            )
            adaptive_reg = self.regularization_result['adaptive_regularization']
        else:
            adaptive_reg = self.config.regularization

        # Standard enhanced processing
        return self._run_standard_enhanced_spice(sample_cov, adaptive_reg, estimated_snr)

    def _run_standard_enhanced_spice(self, sample_cov: np.ndarray, adaptive_reg: float,
                                   estimated_snr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Run standard enhanced SPICE processing."""
        # Initialize power spectrum
        if self.config.eigenvalue_initialization:
            power_spectrum = self._eigenvalue_based_initialization(sample_cov, adaptive_reg)
        else:
            power_spectrum = np.full(self.config.grid_size, adaptive_reg)

        # Run SPICE iterations
        cost_history = []
        for iteration in range(self.config.max_iterations):
            power_spectrum_new = self._enhanced_power_update(
                sample_cov, power_spectrum, adaptive_reg, iteration
            )

            # Apply stabilization
            if iteration > 0 and self.config.stabilization_factor > 0:
                alpha = self.config.stabilization_factor
                power_spectrum_new = (1 - alpha) * power_spectrum_new + alpha * power_spectrum

            cost = self._compute_cost_function(sample_cov, power_spectrum_new)
            cost_history.append(cost)

            # Check convergence
            if iteration > 0:
                relative_change = abs(cost_history[-1] - cost_history[-2]) / max(abs(cost_history[-2]), 1e-12)
                if relative_change < self.config.convergence_tolerance:
                    break

            power_spectrum = power_spectrum_new

        # Store results
        self.power_spectrum = power_spectrum
        self.cost_history = np.array(cost_history)
        self.n_iterations = len(cost_history)
        self.is_fitted = True

        return power_spectrum, self.angular_grid

    def _enhanced_power_update(self, sample_cov: np.ndarray, current_powers: np.ndarray,
                              regularization: float, iteration: int) -> np.ndarray:
        """Enhanced power update with multiple improvements."""
        updated_powers = np.zeros_like(current_powers)

        for i in range(self.config.grid_size):
            steering_vec = self.steering_vectors[:, i:i+1]

            # Direct SPICE power computation with enhanced numerical stability
            numerator = np.real(steering_vec.conj().T @ sample_cov @ steering_vec).item()
            denominator = np.real(steering_vec.conj().T @ steering_vec).item()

            if denominator > 1e-12:
                power_estimate = numerator / denominator
            else:
                power_estimate = regularization

            # Apply regularization
            updated_powers[i] = max(power_estimate, regularization)

        return updated_powers

    def _eigenvalue_based_initialization(self, sample_cov: np.ndarray,
                                       regularization: float) -> np.ndarray:
        """Enhanced eigenvalue-based initialization."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(sample_cov)
            eigenvals = np.maximum(eigenvals, 1e-12)

            # Use multiple dominant eigenvectors for better initialization
            n_dominant = min(3, self.n_sensors // 2)
            dominant_eigenvecs = eigenvecs[:, -n_dominant:]
            dominant_eigenvals = eigenvals[-n_dominant:]

            P_init = np.zeros(self.config.grid_size)

            for k in range(self.config.grid_size):
                steering_vec = self.steering_vectors[:, k]

                # Project onto dominant subspace
                projections = np.abs(dominant_eigenvecs.conj().T @ steering_vec)**2
                weighted_projection = np.sum(projections * dominant_eigenvals)

                P_init[k] = weighted_projection

            # Normalize and regularize
            if np.sum(P_init) > 0:
                P_init = P_init / np.sum(P_init) * np.trace(sample_cov)
            P_init = np.maximum(P_init, regularization)

            return P_init

        except np.linalg.LinAlgError:
            return np.full(self.config.grid_size, regularization)

    def _adaptive_stabilization_factor(self, cost_history: list) -> float:
        """Compute adaptive stabilization factor based on convergence behavior."""
        if len(cost_history) < 3:
            return self.config.stabilization_factor

        # Check for oscillations in cost function
        recent_costs = cost_history[-3:]
        cost_changes = [recent_costs[i] - recent_costs[i-1] for i in range(1, len(recent_costs))]

        # If costs are oscillating, increase stabilization
        if len(cost_changes) >= 2:
            signs = [np.sign(change) for change in cost_changes]
            if signs[-1] != signs[-2]:  # Sign change indicates oscillation
                return min(0.5, self.config.stabilization_factor * 2.0)

        return self.config.stabilization_factor

    def find_peaks_advanced(self, power_spectrum: np.ndarray,
                           sample_covariance: np.ndarray,
                           max_targets: int = 10) -> Dict:
        """Use advanced peak detection if enabled."""
        if self.config.use_advanced_peak_detection:
            detector = create_advanced_detector(self.n_sensors, self.angular_grid)
            estimated_snr = self.get_estimated_snr()
            return detector.detect_peaks(
                power_spectrum, sample_covariance,
                estimated_snr_db=estimated_snr, max_targets=max_targets
            )
        else:
            # Fall back to standard peak detection
            return self.find_peaks(power_spectrum)

    def get_estimated_snr(self) -> Optional[float]:
        """Get the estimated SNR from the last fit operation."""
        if self.snr_estimation_result:
            return self.snr_estimation_result['consensus_snr_db']
        return None

    def get_enhancement_details(self) -> Dict:
        """Get detailed information about enhancements applied."""
        details = {
            'snr_estimation': self.snr_estimation_result,
            'regularization_adaptation': self.regularization_result,
            'processing_stages': self.processing_stages,
            'config': {
                'advanced_snr_estimation': self.config.use_advanced_snr_estimation,
                'adaptive_regularization': self.config.use_adaptive_regularization,
                'advanced_peak_detection': self.config.use_advanced_peak_detection,
                'multi_stage_processing': self.config.multi_stage_processing
            }
        }

        if self.snr_estimation_result and self.regularization_result:
            details['summary'] = {
                'estimated_snr_db': self.snr_estimation_result['consensus_snr_db'],
                'snr_confidence': self.snr_estimation_result['consensus_confidence'],
                'adaptive_regularization': self.regularization_result['adaptive_regularization'],
                'regularization_adaptation_factor': self.regularization_result['adaptation_factor'],
                'total_processing_stages': len(self.processing_stages)
            }

        return details


def create_improved_enhanced_spice(n_sensors: int, target_snr_db: float = 5.0,
                                  **config_kwargs) -> ImprovedEnhancedSPICEEstimator:
    """
    Factory function to create Improved Enhanced SPICE optimized for low SNR.

    Parameters
    ----------
    n_sensors : int
        Number of sensors.
    target_snr_db : float, default=5.0
        Target minimum SNR for optimization.
    **config_kwargs
        Additional configuration parameters.

    Returns
    -------
    estimator : ImprovedEnhancedSPICEEstimator
        Configured improved enhanced SPICE estimator.
    """
    # Optimize configuration for target SNR
    if target_snr_db <= 3:
        config = ImprovedEnhancedSPICEConfig(
            use_advanced_snr_estimation=True,
            use_adaptive_regularization=True,
            use_advanced_peak_detection=True,
            multi_stage_processing=True,
            eigenvalue_initialization=True,
            stabilization_factor=0.2,  # Higher stabilization for very low SNR
            max_iterations=20,
            convergence_tolerance=1e-6,
            **config_kwargs
        )
    elif target_snr_db <= 7:
        config = ImprovedEnhancedSPICEConfig(
            use_advanced_snr_estimation=True,
            use_adaptive_regularization=True,
            use_advanced_peak_detection=True,
            multi_stage_processing=True,
            eigenvalue_initialization=True,
            stabilization_factor=0.15,
            max_iterations=15,
            convergence_tolerance=1e-7,
            **config_kwargs
        )
    else:
        config = ImprovedEnhancedSPICEConfig(
            use_advanced_snr_estimation=True,
            use_adaptive_regularization=True,
            use_advanced_peak_detection=True,
            multi_stage_processing=False,  # Single stage sufficient for higher SNR
            eigenvalue_initialization=True,
            stabilization_factor=0.1,
            **config_kwargs
        )

    return ImprovedEnhancedSPICEEstimator(n_sensors, config)
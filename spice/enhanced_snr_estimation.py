"""
Enhanced SNR Estimation and Adaptive Regularization for SPICE

This module implements sophisticated SNR estimation and adaptive regularization
strategies to improve Enhanced SPICE performance at low SNR levels.

Key improvements:
1. Multiple robust SNR estimation methods with validation
2. Condition number-aware regularization adaptation
3. Eigenvalue gap-based signal detection
4. Iterative refinement with parameter updates
5. Multi-stage processing pipeline
"""

import numpy as np
from scipy.stats import chi2
from typing import Tuple, Dict, Optional
import warnings


class EnhancedSNREstimator:
    """Advanced SNR estimation with multiple robust methods."""

    def __init__(self, n_sensors: int, min_snr_db: float = -10.0, max_snr_db: float = 40.0):
        """
        Initialize enhanced SNR estimator.

        Parameters
        ----------
        n_sensors : int
            Number of array sensors.
        min_snr_db : float, default=-10.0
            Minimum reasonable SNR estimate.
        max_snr_db : float, default=40.0
            Maximum reasonable SNR estimate.
        """
        self.n_sensors = n_sensors
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def estimate_snr_multi_method(self, sample_cov: np.ndarray) -> Dict:
        """
        Estimate SNR using multiple methods and return consensus estimate.

        Parameters
        ----------
        sample_cov : array_like
            Sample covariance matrix.

        Returns
        -------
        snr_result : dict
            SNR estimates from multiple methods with consensus.
        """
        methods = {
            'eigenvalue_gap': self._estimate_snr_eigenvalue_gap,
            'quartile_robust': self._estimate_snr_quartile_robust,
            'information_theoretic': self._estimate_snr_information_theoretic,
            'condition_number': self._estimate_snr_condition_number,
            'trace_based': self._estimate_snr_trace_based
        }

        estimates = {}
        confidences = {}

        for method_name, method_func in methods.items():
            try:
                snr_db, confidence = method_func(sample_cov)
                estimates[method_name] = snr_db
                confidences[method_name] = confidence
            except Exception as e:
                estimates[method_name] = None
                confidences[method_name] = 0.0

        # Compute consensus estimate
        consensus_snr, consensus_confidence = self._compute_consensus(estimates, confidences)

        return {
            'consensus_snr_db': consensus_snr,
            'consensus_confidence': consensus_confidence,
            'individual_estimates': estimates,
            'individual_confidences': confidences,
            'method_count': sum(1 for v in estimates.values() if v is not None)
        }

    def _estimate_snr_eigenvalue_gap(self, sample_cov: np.ndarray) -> Tuple[float, float]:
        """Estimate SNR using eigenvalue gap analysis."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]  # Descending order

        if len(eigenvals) < 2:
            return 10.0, 0.1

        # Find largest eigenvalue gap (indicates signal/noise separation)
        gaps = eigenvals[:-1] - eigenvals[1:]
        max_gap_idx = np.argmax(gaps)

        # Estimate signal and noise powers
        signal_eigenvals = eigenvals[:max_gap_idx+1]
        noise_eigenvals = eigenvals[max_gap_idx+1:]

        if len(noise_eigenvals) == 0:
            noise_power = eigenvals[-1]
        else:
            noise_power = np.mean(noise_eigenvals)

        signal_power = np.mean(signal_eigenvals)

        # Compute SNR
        if noise_power > 0 and signal_power > noise_power:
            snr_linear = (signal_power - noise_power) / noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-3))

            # Confidence based on gap magnitude and consistency
            gap_strength = gaps[max_gap_idx] / np.mean(eigenvals)
            noise_consistency = 1.0 / (1.0 + np.std(noise_eigenvals) / np.mean(noise_eigenvals)) if len(noise_eigenvals) > 1 else 0.5
            confidence = min(1.0, gap_strength * noise_consistency * 2.0)

            return np.clip(snr_db, self.min_snr_db, self.max_snr_db), confidence
        else:
            return 5.0, 0.2

    def _estimate_snr_quartile_robust(self, sample_cov: np.ndarray) -> Tuple[float, float]:
        """Robust quartile-based SNR estimation."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]

        n = len(eigenvals)
        if n < 4:
            return 8.0, 0.3

        # Use quartiles for robust estimation
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        # Signal power from top quartile
        signal_power = np.mean(eigenvals[:q1_idx]) if q1_idx > 0 else eigenvals[0]

        # Noise power from bottom quartile
        noise_power = np.mean(eigenvals[q3_idx:])

        if noise_power > 0 and signal_power > noise_power:
            snr_linear = (signal_power - noise_power) / noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-3))

            # Confidence based on separation and consistency
            separation = (signal_power - noise_power) / signal_power
            noise_var = np.var(eigenvals[q3_idx:]) / (noise_power**2) if noise_power > 0 else 1.0
            confidence = min(1.0, separation / (1.0 + noise_var))

            return np.clip(snr_db, self.min_snr_db, self.max_snr_db), confidence
        else:
            return 6.0, 0.2

    def _estimate_snr_information_theoretic(self, sample_cov: np.ndarray) -> Tuple[float, float]:
        """Information theoretic SNR estimation using MDL/AIC."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]

        n = len(eigenvals)
        best_snr = 10.0
        best_confidence = 0.1

        # Try different signal subspace dimensions
        for k in range(1, min(n, n//2 + 1)):
            signal_eigenvals = eigenvals[:k]
            noise_eigenvals = eigenvals[k:]

            if len(noise_eigenvals) == 0:
                continue

            signal_power = np.mean(signal_eigenvals)
            noise_power = np.mean(noise_eigenvals)

            # MDL criterion for model selection
            geo_mean = np.exp(np.mean(np.log(noise_eigenvals + 1e-12)))
            arith_mean = np.mean(noise_eigenvals)

            if geo_mean > 0 and arith_mean > 0:
                mdl_penalty = (n - k) * np.log(arith_mean / geo_mean)

                # SNR estimate
                if noise_power > 0 and signal_power > noise_power:
                    snr_linear = (signal_power - noise_power) / noise_power
                    snr_db = 10 * np.log10(max(snr_linear, 1e-3))

                    # Confidence based on MDL criterion
                    confidence = min(1.0, 1.0 / (1.0 + mdl_penalty / 10.0))

                    if confidence > best_confidence:
                        best_snr = snr_db
                        best_confidence = confidence

        return np.clip(best_snr, self.min_snr_db, self.max_snr_db), best_confidence

    def _estimate_snr_condition_number(self, sample_cov: np.ndarray) -> Tuple[float, float]:
        """SNR estimation based on condition number analysis."""
        condition_number = np.linalg.cond(sample_cov)

        # Empirical relationship between condition number and SNR
        if condition_number > 1e12:
            snr_db = -5.0
            confidence = 0.8
        elif condition_number > 1e6:
            snr_db = 0.0
            confidence = 0.7
        elif condition_number > 1e3:
            snr_db = 5.0 + 5.0 * np.log10(condition_number / 1e3)
            confidence = 0.6
        else:
            snr_db = 15.0
            confidence = 0.5

        return np.clip(snr_db, self.min_snr_db, self.max_snr_db), confidence

    def _estimate_snr_trace_based(self, sample_cov: np.ndarray) -> Tuple[float, float]:
        """Simple trace-based SNR estimation."""
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.real(eigenvals)

        total_power = np.mean(eigenvals)
        min_power = np.min(eigenvals)

        if min_power > 0 and total_power > min_power:
            snr_linear = (total_power - min_power) / min_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-3))

            # Confidence based on dynamic range
            dynamic_range = total_power / min_power if min_power > 0 else 1.0
            confidence = min(1.0, np.log10(dynamic_range) / 3.0)

            return np.clip(snr_db, self.min_snr_db, self.max_snr_db), confidence
        else:
            return 8.0, 0.3

    def _compute_consensus(self, estimates: Dict, confidences: Dict) -> Tuple[float, float]:
        """Compute consensus SNR estimate from multiple methods."""
        valid_estimates = [(snr, conf) for snr, conf in zip(estimates.values(), confidences.values())
                          if snr is not None and conf > 0]

        if not valid_estimates:
            return 10.0, 0.1

        # Weighted average based on confidence
        total_weight = sum(conf for _, conf in valid_estimates)
        if total_weight == 0:
            return 10.0, 0.1

        weighted_snr = sum(snr * conf for snr, conf in valid_estimates) / total_weight
        consensus_confidence = min(1.0, total_weight / len(valid_estimates))

        return weighted_snr, consensus_confidence


class AdaptiveRegularizationStrategy:
    """Advanced adaptive regularization based on multiple criteria."""

    def __init__(self, base_regularization: float = 1e-12):
        """
        Initialize adaptive regularization strategy.

        Parameters
        ----------
        base_regularization : float, default=1e-12
            Base regularization parameter.
        """
        self.base_regularization = base_regularization

    def compute_adaptive_regularization(self, sample_cov: np.ndarray,
                                      estimated_snr_db: float,
                                      current_iteration: int = 0) -> Dict:
        """
        Compute adaptive regularization parameter.

        Parameters
        ----------
        sample_cov : array_like
            Sample covariance matrix.
        estimated_snr_db : float
            Estimated SNR in dB.
        current_iteration : int, default=0
            Current iteration number for adaptive schemes.

        Returns
        -------
        regularization_result : dict
            Adaptive regularization parameters and diagnostics.
        """
        # Multiple regularization strategies
        strategies = {
            'snr_based': self._snr_based_regularization,
            'condition_based': self._condition_based_regularization,
            'eigenvalue_based': self._eigenvalue_based_regularization,
            'iterative_adaptive': self._iterative_adaptive_regularization
        }

        regularizations = {}
        weights = {}

        for strategy_name, strategy_func in strategies.items():
            try:
                reg_value, weight = strategy_func(sample_cov, estimated_snr_db, current_iteration)
                regularizations[strategy_name] = reg_value
                weights[strategy_name] = weight
            except Exception:
                regularizations[strategy_name] = self.base_regularization
                weights[strategy_name] = 0.1

        # Compute weighted combination
        total_weight = sum(weights.values())
        if total_weight > 0:
            adaptive_reg = sum(reg * weight for reg, weight in zip(regularizations.values(), weights.values())) / total_weight
        else:
            adaptive_reg = self.base_regularization

        # Ensure reasonable bounds
        adaptive_reg = max(adaptive_reg, self.base_regularization)
        adaptive_reg = min(adaptive_reg, self.base_regularization * 10000)

        return {
            'adaptive_regularization': adaptive_reg,
            'base_regularization': self.base_regularization,
            'adaptation_factor': adaptive_reg / self.base_regularization,
            'individual_regularizations': regularizations,
            'strategy_weights': weights
        }

    def _snr_based_regularization(self, sample_cov: np.ndarray,
                                 snr_db: float, iteration: int) -> Tuple[float, float]:
        """SNR-based regularization adaptation."""
        # Higher regularization for lower SNR
        if snr_db < 0:
            reg_factor = 1000.0
            weight = 1.0
        elif snr_db < 5:
            reg_factor = 100.0 * np.exp(-(snr_db + 5) / 5.0)
            weight = 0.9
        elif snr_db < 10:
            reg_factor = 10.0 * np.exp(-(snr_db - 5) / 5.0)
            weight = 0.7
        else:
            reg_factor = 1.0
            weight = 0.5

        regularization = self.base_regularization * reg_factor
        return regularization, weight

    def _condition_based_regularization(self, sample_cov: np.ndarray,
                                      snr_db: float, iteration: int) -> Tuple[float, float]:
        """Condition number-based regularization."""
        condition_num = np.linalg.cond(sample_cov)

        if condition_num > 1e12:
            reg_factor = 1000.0
            weight = 1.0
        elif condition_num > 1e8:
            reg_factor = 100.0
            weight = 0.8
        elif condition_num > 1e4:
            reg_factor = 10.0
            weight = 0.6
        else:
            reg_factor = 1.0
            weight = 0.4

        regularization = self.base_regularization * reg_factor
        return regularization, weight

    def _eigenvalue_based_regularization(self, sample_cov: np.ndarray,
                                       snr_db: float, iteration: int) -> Tuple[float, float]:
        """Eigenvalue-based regularization using smallest eigenvalue."""
        eigenvals = np.linalg.eigvals(sample_cov)
        min_eigenval = np.min(np.real(eigenvals))
        max_eigenval = np.max(np.real(eigenvals))

        if min_eigenval <= 0 or max_eigenval / min_eigenval > 1e6:
            reg_factor = min_eigenval * 0.01 if min_eigenval > 0 else 1000.0
            weight = 0.9
        else:
            reg_factor = min_eigenval * 0.001
            weight = 0.6

        regularization = max(self.base_regularization, abs(reg_factor))
        return regularization, weight

    def _iterative_adaptive_regularization(self, sample_cov: np.ndarray,
                                         snr_db: float, iteration: int) -> Tuple[float, float]:
        """Iterative adaptive regularization that decreases with iterations."""
        # Start with higher regularization, decrease as algorithm converges
        if iteration == 0:
            reg_factor = 100.0
            weight = 0.8
        elif iteration == 1:
            reg_factor = 10.0
            weight = 0.6
        else:
            reg_factor = 1.0
            weight = 0.4

        # Adjust based on SNR
        snr_adjustment = max(0.1, 1.0 - max(0, snr_db - 5) / 20.0)
        reg_factor *= snr_adjustment

        regularization = self.base_regularization * reg_factor
        return regularization, weight


def create_enhanced_snr_estimator(n_sensors: int) -> EnhancedSNREstimator:
    """Factory function to create enhanced SNR estimator."""
    return EnhancedSNREstimator(n_sensors)


def create_adaptive_regularization_strategy(base_reg: float = 1e-12) -> AdaptiveRegularizationStrategy:
    """Factory function to create adaptive regularization strategy."""
    return AdaptiveRegularizationStrategy(base_reg)
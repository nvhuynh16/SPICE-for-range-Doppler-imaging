"""
Simple but Effective SPICE Optimization.

This module provides practical optimizations for SPICE algorithms that
actually improve performance without adding excessive overhead.

Focus on:
- Efficient vectorized operations
- Memory layout optimization
- Reduced function call overhead
- Smart initialization

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict
from spice_core import SPICEEstimator, SPICEConfig


class EfficientSPICEEstimator(SPICEEstimator):
    """
    Efficient SPICE Implementation with Practical Optimizations.

    This class provides optimizations that actually improve performance:
    - Vectorized batch operations
    - Efficient memory access patterns
    - Reduced overhead in core loops
    - Smart convergence detection

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : SPICEConfig, optional
        SPICE algorithm configuration.
    """

    def __init__(self, n_sensors: int, config: Optional[SPICEConfig] = None):
        """Initialize efficient SPICE estimator."""
        super().__init__(n_sensors, config)

        # Pre-compute frequently used values
        self._steering_hermitian = self.steering_vectors.conj().T  # (grid_size, n_sensors)
        self._norm_factors = np.ones(self.config.grid_size) * self.n_sensors
        self._regularization_array = np.full(self.config.grid_size, self.config.regularization)

    def _batch_power_update(self, sample_cov: np.ndarray) -> np.ndarray:
        """Efficient batch power update using vectorized operations."""
        # Compute a_k^H * R * a_k for all k simultaneously
        # Shape: (grid_size, n_sensors) @ (n_sensors, n_sensors) = (grid_size, n_sensors)
        intermediate = self._steering_hermitian @ sample_cov

        # Element-wise multiply and sum: diag(intermediate @ A)
        # Shape: (grid_size, n_sensors) * (grid_size, n_sensors) -> (grid_size,)
        numerators = np.sum(intermediate * self._steering_hermitian.conj(), axis=1).real

        # Normalize by array response: p_k = numerator_k / |a_k|^2
        powers = numerators / self._norm_factors

        # Apply regularization
        return np.maximum(powers, self._regularization_array)

    def _efficient_cost_computation(self, sample_cov: np.ndarray, powers: np.ndarray) -> float:
        """Efficient cost computation without explicit covariance construction."""
        # Cost = ||R - A*P*A^H||_F^2
        # = trace(R^H*R) - 2*Re(trace(R^H*A*P*A^H)) + trace((A*P*A^H)^H*(A*P*A^H))

        # Term 1: trace(R^H*R) = trace(R*R^H) for Hermitian R
        trace_R_squared = np.trace(sample_cov @ sample_cov.conj().T).real

        # Term 2: trace(R^H * fitted_cov) = sum_k p_k * a_k^H * R * a_k
        active_mask = powers > self.config.regularization
        if not np.any(active_mask):
            return trace_R_squared

        active_powers = powers[active_mask]
        active_steering = self._steering_hermitian[active_mask]  # (n_active, n_sensors)

        # Vectorized computation of a_k^H * R * a_k for active elements
        active_traces = np.sum((active_steering @ sample_cov) * active_steering.conj(), axis=1).real
        trace_R_fitted = np.sum(active_powers * active_traces)

        # Term 3: trace(fitted_cov^H * fitted_cov)
        # = sum_i sum_j p_i * p_j * |a_i^H * a_j|^2
        cross_terms = np.abs(active_steering @ active_steering.conj().T) ** 2
        trace_fitted_squared = np.sum(active_powers[:, None] * active_powers[None, :] * cross_terms)

        return trace_R_squared - 2 * trace_R_fitted + trace_fitted_squared

    def _smart_initialization(self, sample_cov: np.ndarray) -> np.ndarray:
        """Smart initialization based on matched filter response."""
        # Initialize with normalized matched filter response
        matched_response = np.sum(
            (self._steering_hermitian @ sample_cov) * self._steering_hermitian.conj(),
            axis=1
        ).real

        # Normalize and add small regularization
        matched_response = np.maximum(matched_response / self.n_sensors,
                                    self._regularization_array)

        return matched_response

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient SPICE fitting with optimized operations."""
        start_time = time.time()

        # Validate input
        self._validate_covariance_matrix(sample_covariance)

        # Smart initialization
        powers = self._smart_initialization(sample_covariance)

        # Optimization loop with efficient operations
        cost_history = []
        converged = False

        for iteration in range(self.config.max_iterations):
            # Efficient power update
            new_powers = self._batch_power_update(sample_covariance)

            # Efficient cost computation
            cost = self._efficient_cost_computation(sample_covariance, new_powers)
            cost_history.append(cost)

            # Early convergence detection
            if iteration > 0:
                relative_change = abs(cost_history[-1] - cost_history[-2]) / abs(cost_history[-2])
                if relative_change < self.config.convergence_tolerance:
                    converged = True
                    break

            powers = new_powers

        # Store convergence information
        self.convergence_info = {
            'converged': converged,
            'iterations': iteration + 1,
            'final_cost': cost_history[-1] if cost_history else 0,
            'cost_history': cost_history,
            'execution_time': time.time() - start_time
        }

        # Set compatibility attributes
        self.n_iterations = iteration + 1
        self.cost_history = cost_history
        self.final_cost = cost_history[-1] if cost_history else 0
        self.is_fitted = True

        return powers, self.angular_grid


def performance_comparison(n_sensors_list: list, grid_sizes: list, n_trials: int = 3) -> Dict:
    """Compare standard vs efficient SPICE performance."""
    results = {}

    print("SPICE Performance Comparison")
    print("=" * 40)

    for n_sensors in n_sensors_list:
        for grid_size in grid_sizes:
            print(f"Testing {n_sensors} sensors, {grid_size} grid points: ", end="", flush=True)

            # Generate test data
            np.random.seed(42)
            test_cov = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            test_cov = test_cov @ test_cov.conj().T
            test_cov /= np.trace(test_cov)

            # Test configurations
            config = SPICEConfig(grid_size=grid_size, max_iterations=20)

            # Standard SPICE timing
            std_times = []
            for _ in range(n_trials):
                estimator_std = SPICEEstimator(n_sensors, config)
                start_time = time.time()
                estimator_std.fit(test_cov)
                std_times.append(time.time() - start_time)

            # Efficient SPICE timing
            eff_times = []
            for _ in range(n_trials):
                estimator_eff = EfficientSPICEEstimator(n_sensors, config)
                start_time = time.time()
                estimator_eff.fit(test_cov)
                eff_times.append(time.time() - start_time)

            avg_std_time = np.mean(std_times)
            avg_eff_time = np.mean(eff_times)
            speedup = avg_std_time / avg_eff_time

            key = f"{n_sensors}_{grid_size}"
            results[key] = {
                'standard_time': avg_std_time,
                'efficient_time': avg_eff_time,
                'speedup': speedup
            }

            print(f"Speedup: {speedup:.2f}x ({avg_std_time:.3f}s -> {avg_eff_time:.3f}s)")

    return results


def validate_algorithmic_correctness(n_sensors: int = 8) -> bool:
    """Validate that optimizations preserve algorithmic correctness."""
    print("\nAlgorithmic Correctness Validation")
    print("-" * 35)

    # Generate test scenario
    np.random.seed(42)
    true_angles = [-10, 15]
    n_snapshots = 100
    snr_db = 10

    # Create test covariance
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    source_signals = (np.random.randn(len(true_angles), n_snapshots) +
                     1j * np.random.randn(len(true_angles), n_snapshots))
    received_signals = steering_matrix @ source_signals
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) +
        1j * np.random.randn(n_sensors, n_snapshots)
    )

    array_data = received_signals + noise
    sample_cov = array_data @ array_data.conj().T / n_snapshots

    # Compare algorithms
    config = SPICEConfig(grid_size=180, max_iterations=30)

    estimator_std = SPICEEstimator(n_sensors, config)
    spectrum_std, angles_std = estimator_std.fit(sample_cov)
    peaks_std = estimator_std.find_peaks(spectrum_std)

    estimator_eff = EfficientSPICEEstimator(n_sensors, config)
    spectrum_eff, angles_eff = estimator_eff.fit(sample_cov)
    peaks_eff = estimator_eff.find_peaks(spectrum_eff)

    # Check differences
    spectrum_diff = np.max(np.abs(spectrum_std - spectrum_eff))
    angle_diff = np.max(np.abs(angles_std - angles_eff))

    print(f"Max spectrum difference: {spectrum_diff:.2e}")
    print(f"Max angle difference: {angle_diff:.2e}")
    print(f"Standard peaks: {len(peaks_std['angles'])} detected")
    print(f"Efficient peaks: {len(peaks_eff['angles'])} detected")

    # Convergence comparison
    std_conv = estimator_std.get_convergence_info()
    eff_conv = estimator_eff.get_convergence_info()

    print(f"Standard: {std_conv.get('n_iterations', 'N/A')} iterations, {std_conv.get('execution_time', 0):.3f}s")
    print(f"Efficient: {eff_conv.get('iterations', 'N/A')} iterations, {eff_conv.get('execution_time', 0):.3f}s")

    std_time = std_conv.get('execution_time', 1)
    eff_time = eff_conv.get('execution_time', 1)
    speedup = std_time / eff_time if eff_time > 0 else 1
    print(f"Speedup: {speedup:.2f}x")

    # Validation criteria
    correctness_ok = (spectrum_diff < 1e-10 and angle_diff < 1e-10 and
                     len(peaks_std['angles']) == len(peaks_eff['angles']))

    if correctness_ok:
        print("[SUCCESS] Algorithmic correctness PRESERVED")
    else:
        print("[WARNING] Algorithmic correctness may be affected")

    return correctness_ok, speedup


def main():
    """Main computational optimization validation."""
    print("Simple SPICE Optimization Validation")
    print("=" * 45)

    # Validate correctness
    correctness_ok, basic_speedup = validate_algorithmic_correctness()

    # Performance comparison
    array_sizes = [8, 16, 32]
    grid_sizes = [90, 180]

    perf_results = performance_comparison(array_sizes, grid_sizes, n_trials=3)

    # Analysis
    speedups = [result['speedup'] for result in perf_results.values()]
    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)

    print(f"\nPerformance Analysis:")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Maximum speedup: {max_speedup:.2f}x")
    print(f"  Algorithmic correctness: {'PRESERVED' if correctness_ok else 'AFFECTED'}")

    # Literature validation
    print(f"\n[COMPUTATIONAL EFFICIENCY VALIDATION]")
    if avg_speedup >= 2.0:
        print(f"[SUCCESS] Significant optimization achieved: {avg_speedup:.2f}x average speedup")
    elif avg_speedup >= 1.5:
        print(f"[GOOD] Solid optimization achieved: {avg_speedup:.2f}x average speedup")
    elif avg_speedup >= 1.2:
        print(f"[MODEST] Modest optimization achieved: {avg_speedup:.2f}x average speedup")
    else:
        print(f"[LIMITED] Limited optimization: {avg_speedup:.2f}x average speedup")

    return {
        'avg_speedup': avg_speedup,
        'max_speedup': max_speedup,
        'correctness_preserved': correctness_ok,
        'detailed_results': perf_results
    }


if __name__ == "__main__":
    results = main()
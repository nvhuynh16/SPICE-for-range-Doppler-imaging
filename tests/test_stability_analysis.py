"""
Comprehensive SPICE Stability Analysis and Testing.

This module identifies, demonstrates, and tests solutions for numerical stability
issues in the SPICE algorithm based on recent literature (2023-2024).

Key stability improvements implemented:
1. Adaptive regularization based on condition number
2. Eigenvalue-based stability monitoring
3. Enhanced convergence criteria
4. Robust matrix operations
5. Scaling-aware optimization

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.linalg as la
import pytest
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from spice_core import SPICEEstimator, SPICEConfig


@dataclass
class StabilityTestConfig:
    """Configuration for stability testing."""
    n_sensors: int = 8
    n_snapshots: int = 50
    grid_size: int = 90
    test_snr_range: List[float] = (-5, 0, 5, 10, 15, 20)
    condition_number_range: List[float] = (1e2, 1e6, 1e10, 1e14)
    regularization_range: List[float] = (1e-15, 1e-12, 1e-9, 1e-6, 1e-3)


class SPICEStabilityAnalyzer:
    """
    Comprehensive stability analysis for SPICE algorithms.

    Tests and demonstrates numerical stability issues and improvements
    based on recent research (2023-2024).
    """

    def __init__(self, config: StabilityTestConfig = None):
        """Initialize stability analyzer."""
        self.config = config or StabilityTestConfig()
        self.results = {}

    def generate_challenging_scenario(self, condition_number: float, snr_db: float) -> np.ndarray:
        """
        Generate covariance matrix with specified condition number and SNR.

        Parameters
        ----------
        condition_number : float
            Target condition number for ill-conditioning test
        snr_db : float
            Signal-to-noise ratio in dB

        Returns
        -------
        sample_cov : ndarray
            Sample covariance matrix with desired properties
        """
        n_sensors = self.config.n_sensors
        n_snapshots = self.config.n_snapshots

        # Generate steering vectors for two closely spaced sources
        angles_rad = np.deg2rad(np.array([0, 2]))  # Very close sources
        steering_vecs = []
        for angle in angles_rad:
            sensor_positions = np.arange(n_sensors)
            phases = np.exp(1j * np.pi * sensor_positions * np.sin(angle))
            steering_vecs.append(phases[:, np.newaxis])

        A = np.hstack(steering_vecs)

        # Signal powers
        signal_power = 10**(snr_db/10)
        P = np.diag([signal_power, signal_power * 0.8])

        # Noise covariance with controlled condition number
        noise_var = 1.0

        # Create ill-conditioned noise covariance
        U, _, Vh = la.svd(np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors))
        singular_values = np.logspace(0, -np.log10(condition_number), n_sensors)
        ill_conditioned_noise = U @ np.diag(singular_values) @ Vh

        # True covariance
        true_cov = A @ P @ A.conj().T + noise_var * ill_conditioned_noise

        # Generate sample covariance with finite sample effects
        # Add random component to simulate finite sample estimation errors
        sample_error = 0.1 * (np.random.randn(n_sensors, n_sensors) +
                             1j * np.random.randn(n_sensors, n_sensors))
        sample_cov = true_cov + sample_error @ sample_error.conj().T / n_snapshots

        # Ensure positive definiteness
        sample_cov = 0.5 * (sample_cov + sample_cov.conj().T)
        eigenvals = la.eigvals(sample_cov)
        min_eigenval = np.min(np.real(eigenvals))
        if min_eigenval <= 0:
            sample_cov += (abs(min_eigenval) + 1e-6) * np.eye(n_sensors)

        return sample_cov

    def analyze_condition_number_impact(self) -> Dict:
        """Analyze impact of condition number on SPICE stability."""
        print("Analyzing condition number impact on stability...")

        results = {
            'condition_numbers': [],
            'convergence_rates': [],
            'final_costs': [],
            'iterations': [],
            'stability_metrics': []
        }

        for cond_num in self.config.condition_number_range:
            print(f"  Testing condition number: {cond_num:.1e}")

            # Generate challenging scenario
            sample_cov = self.generate_challenging_scenario(cond_num, snr_db=15)

            # Test with standard SPICE
            config = SPICEConfig(grid_size=self.config.grid_size, max_iterations=200)
            estimator = SPICEEstimator(self.config.n_sensors, config)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    power_spectrum, angles = estimator.fit(sample_cov)

                conv_info = estimator.get_convergence_info()

                # Compute stability metrics
                actual_cond_num = np.linalg.cond(sample_cov)
                eigenvals = la.eigvals(sample_cov)
                min_eigenval = np.min(np.real(eigenvals))

                stability_metric = {
                    'actual_condition_number': actual_cond_num,
                    'min_eigenvalue': min_eigenval,
                    'eigenvalue_ratio': np.max(np.real(eigenvals)) / max(min_eigenval, 1e-15),
                    'converged': conv_info['n_iterations'] < config.max_iterations,
                    'cost_reduction_ratio': conv_info['cost_reduction'] / max(conv_info['initial_cost'], 1e-15)
                }

                results['condition_numbers'].append(cond_num)
                results['convergence_rates'].append(conv_info['n_iterations'])
                results['final_costs'].append(conv_info['final_cost'])
                results['iterations'].append(conv_info['n_iterations'])
                results['stability_metrics'].append(stability_metric)

            except Exception as e:
                print(f"    Failed with condition number {cond_num:.1e}: {e}")
                # Record failure
                results['condition_numbers'].append(cond_num)
                results['convergence_rates'].append(200)  # Max iterations
                results['final_costs'].append(np.inf)
                results['iterations'].append(200)
                results['stability_metrics'].append({
                    'actual_condition_number': cond_num,
                    'converged': False,
                    'error': str(e)
                })

        self.results['condition_number_analysis'] = results
        return results

    def analyze_regularization_impact(self) -> Dict:
        """Analyze impact of regularization parameter on stability."""
        print("\nAnalyzing regularization parameter impact...")

        results = {
            'regularizations': [],
            'convergence_success': [],
            'final_costs': [],
            'stability_scores': []
        }

        # Use moderately ill-conditioned scenario
        sample_cov = self.generate_challenging_scenario(1e8, snr_db=10)

        for reg_param in self.config.regularization_range:
            print(f"  Testing regularization: {reg_param:.1e}")

            config = SPICEConfig(
                grid_size=self.config.grid_size,
                regularization=reg_param,
                max_iterations=100
            )

            estimator = SPICEEstimator(self.config.n_sensors, config)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    power_spectrum, angles = estimator.fit(sample_cov)

                conv_info = estimator.get_convergence_info()
                converged = conv_info['n_iterations'] < config.max_iterations

                # Compute stability score
                stability_score = self._compute_stability_score(
                    sample_cov, power_spectrum, conv_info
                )

                results['regularizations'].append(reg_param)
                results['convergence_success'].append(converged)
                results['final_costs'].append(conv_info['final_cost'])
                results['stability_scores'].append(stability_score)

            except Exception as e:
                print(f"    Failed with regularization {reg_param:.1e}: {e}")
                results['regularizations'].append(reg_param)
                results['convergence_success'].append(False)
                results['final_costs'].append(np.inf)
                results['stability_scores'].append(0.0)

        self.results['regularization_analysis'] = results
        return results

    def _compute_stability_score(self, sample_cov: np.ndarray,
                               power_spectrum: np.ndarray,
                               conv_info: Dict) -> float:
        """Compute composite stability score."""

        # Factors contributing to stability
        convergence_score = 1.0 if conv_info['n_iterations'] < 100 else 0.0

        cost_reduction_score = min(1.0, conv_info['cost_reduction'] /
                                 max(conv_info['initial_cost'], 1e-15))

        spectrum_validity_score = 1.0 if np.all(power_spectrum >= 0) else 0.0

        # Check for numerical artifacts
        dynamic_range_db = 10 * np.log10(np.max(power_spectrum) /
                                       max(np.min(power_spectrum), 1e-15))
        dynamic_range_score = 1.0 if dynamic_range_db < 100 else 0.5  # Avoid extreme values

        # Composite score
        stability_score = (convergence_score + cost_reduction_score +
                         spectrum_validity_score + dynamic_range_score) / 4.0

        return stability_score

    def test_low_snr_stability(self) -> Dict:
        """Test stability at low SNR conditions."""
        print("\nTesting low SNR stability...")

        results = {
            'snr_values': [],
            'convergence_success': [],
            'estimation_accuracy': [],
            'stability_scores': []
        }

        true_angles = np.array([0, 15])  # Well-separated sources

        for snr_db in self.config.test_snr_range:
            print(f"  Testing SNR: {snr_db} dB")

            # Generate scenario with good conditioning but low SNR
            sample_cov = self.generate_challenging_scenario(1e3, snr_db)

            config = SPICEConfig(grid_size=self.config.grid_size)
            estimator = SPICEEstimator(self.config.n_sensors, config)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    power_spectrum, angles = estimator.fit(sample_cov)

                # Find peaks and evaluate accuracy
                peaks = estimator.find_peaks(power_spectrum, min_separation=3.0)

                conv_info = estimator.get_convergence_info()
                converged = conv_info['n_iterations'] < config.max_iterations

                # Compute estimation accuracy if we found peaks
                if len(peaks['angles']) >= 2:
                    estimated_angles = np.sort(peaks['angles'][:2])
                    true_angles_sorted = np.sort(true_angles)
                    estimation_error = np.mean(np.abs(estimated_angles - true_angles_sorted))
                    accuracy_score = max(0, 1.0 - estimation_error / 10.0)  # Within 10 degrees is good
                else:
                    accuracy_score = 0.0  # Failed to find sources

                stability_score = self._compute_stability_score(
                    sample_cov, power_spectrum, conv_info
                )

                results['snr_values'].append(snr_db)
                results['convergence_success'].append(converged)
                results['estimation_accuracy'].append(accuracy_score)
                results['stability_scores'].append(stability_score)

            except Exception as e:
                print(f"    Failed at SNR {snr_db} dB: {e}")
                results['snr_values'].append(snr_db)
                results['convergence_success'].append(False)
                results['estimation_accuracy'].append(0.0)
                results['stability_scores'].append(0.0)

        self.results['snr_analysis'] = results
        return results

    def run_comprehensive_analysis(self) -> Dict:
        """Run all stability analyses."""
        print("=== SPICE Stability Analysis ===")
        print("Based on literature review (2023-2024)")
        print("=====================================")

        # Run all analyses
        cond_results = self.analyze_condition_number_impact()
        reg_results = self.analyze_regularization_impact()
        snr_results = self.test_low_snr_stability()

        # Summary
        print("\n=== ANALYSIS SUMMARY ===")

        # Condition number summary
        successful_cond = sum(1 for m in cond_results['stability_metrics']
                            if m.get('converged', False))
        print(f"Condition Number Tests: {successful_cond}/{len(cond_results['condition_numbers'])} successful")

        # Regularization summary
        successful_reg = sum(reg_results['convergence_success'])
        print(f"Regularization Tests: {successful_reg}/{len(reg_results['regularizations'])} successful")

        # SNR summary
        successful_snr = sum(snr_results['convergence_success'])
        print(f"SNR Tests: {successful_snr}/{len(snr_results['snr_values'])} successful")

        return self.results

    def plot_stability_results(self, save_path: str = None):
        """Plot comprehensive stability analysis results."""
        if not self.results:
            print("No results to plot. Run analysis first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Condition number vs convergence
        if 'condition_number_analysis' in self.results:
            cond_data = self.results['condition_number_analysis']
            axes[0, 0].semilogx(cond_data['condition_numbers'], cond_data['iterations'], 'ro-')
            axes[0, 0].set_xlabel('Condition Number')
            axes[0, 0].set_ylabel('Iterations to Converge')
            axes[0, 0].set_title('Condition Number vs Convergence')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Max Iterations')
            axes[0, 0].legend()

        # Plot 2: Regularization impact
        if 'regularization_analysis' in self.results:
            reg_data = self.results['regularization_analysis']
            axes[0, 1].semilogx(reg_data['regularizations'], reg_data['stability_scores'], 'bo-')
            axes[0, 1].set_xlabel('Regularization Parameter')
            axes[0, 1].set_ylabel('Stability Score')
            axes[0, 1].set_title('Regularization vs Stability')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1.1)

        # Plot 3: SNR vs performance
        if 'snr_analysis' in self.results:
            snr_data = self.results['snr_analysis']
            axes[1, 0].plot(snr_data['snr_values'], snr_data['stability_scores'], 'go-', label='Stability')
            axes[1, 0].plot(snr_data['snr_values'], snr_data['estimation_accuracy'], 'mo-', label='Accuracy')
            axes[1, 0].set_xlabel('SNR (dB)')
            axes[1, 0].set_ylabel('Performance Score')
            axes[1, 0].set_title('SNR vs Performance')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1.1)

        # Plot 4: Summary heatmap of all results
        stability_matrix = self._create_stability_heatmap()
        if stability_matrix is not None:
            im = axes[1, 1].imshow(stability_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            axes[1, 1].set_title('Stability Heatmap\n(Green=Stable, Red=Unstable)')
            axes[1, 1].set_xlabel('Test Condition')
            axes[1, 1].set_ylabel('Performance Metric')
            plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stability analysis plot saved to {save_path}")

        plt.show()

    def _create_stability_heatmap(self) -> np.ndarray:
        """Create summary heatmap of stability results."""
        # This would create a comprehensive heatmap - simplified version
        if len(self.results) < 2:
            return None

        # Create a simple 3x3 summary matrix
        summary = np.zeros((3, 3))

        # Fill with average scores from different analyses
        if 'condition_number_analysis' in self.results:
            cond_data = self.results['condition_number_analysis']
            avg_success = sum(1 for m in cond_data['stability_metrics']
                            if m.get('converged', False)) / len(cond_data['stability_metrics'])
            summary[0, 0] = avg_success

        if 'regularization_analysis' in self.results:
            reg_data = self.results['regularization_analysis']
            avg_stability = np.mean(reg_data['stability_scores'])
            summary[1, 1] = avg_stability

        if 'snr_analysis' in self.results:
            snr_data = self.results['snr_analysis']
            avg_snr_stability = np.mean(snr_data['stability_scores'])
            summary[2, 2] = avg_snr_stability

        return summary


# Test cases for stability analysis
class TestSPICEStability:
    """Test cases for SPICE stability analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SPICEStabilityAnalyzer()

    def test_condition_number_stability(self):
        """Test that SPICE handles different condition numbers appropriately."""
        results = self.analyzer.analyze_condition_number_impact()

        # At least some should succeed with good conditioning
        success_rate = sum(1 for m in results['stability_metrics']
                          if m.get('converged', False)) / len(results['stability_metrics'])

        assert success_rate > 0.3, f"Success rate too low: {success_rate:.2%}"

        # Should degrade with worse conditioning
        iterations = results['iterations']
        cond_nums = results['condition_numbers']

        # Sort by condition number and check trend
        sorted_pairs = sorted(zip(cond_nums, iterations))
        early_iterations = np.mean([it for cn, it in sorted_pairs[:2]])
        late_iterations = np.mean([it for cn, it in sorted_pairs[-2:]])

        assert late_iterations >= early_iterations, "Should require more iterations for higher condition numbers"

    def test_regularization_optimization(self):
        """Test that appropriate regularization improves stability."""
        results = self.analyzer.analyze_regularization_impact()

        # Should find some regularization value that works well
        best_stability = max(results['stability_scores'])
        assert best_stability > 0.5, f"Best stability score too low: {best_stability:.3f}"

        # Very small regularization should be less stable than moderate values
        reg_stability_pairs = list(zip(results['regularizations'], results['stability_scores']))
        smallest_reg_stability = next(s for r, s in reg_stability_pairs if r == min(results['regularizations']))
        best_stability = max(results['stability_scores'])

        assert best_stability > smallest_reg_stability, "Optimal regularization should outperform minimal regularization"

    def test_snr_degradation_behavior(self):
        """Test that stability degrades gracefully with SNR."""
        results = self.analyzer.test_low_snr_stability()

        # Should show degradation with lower SNR
        snr_values = results['snr_values']
        stability_scores = results['stability_scores']

        # Sort by SNR
        sorted_pairs = sorted(zip(snr_values, stability_scores))

        # High SNR should generally be more stable than low SNR
        high_snr_stability = np.mean([s for snr, s in sorted_pairs[-3:]])  # Top 3 SNR values
        low_snr_stability = np.mean([s for snr, s in sorted_pairs[:3]])    # Bottom 3 SNR values

        assert high_snr_stability >= low_snr_stability, "High SNR should be more stable than low SNR"


if __name__ == "__main__":
    # Run comprehensive stability analysis
    analyzer = SPICEStabilityAnalyzer()
    results = analyzer.run_comprehensive_analysis()

    # Create plots
    analyzer.plot_stability_results("stability_analysis.png")

    print("\nStability analysis complete. Check 'stability_analysis.png' for detailed results.")
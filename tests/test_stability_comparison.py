"""
Comprehensive comparison between original and enhanced stable SPICE.

This test suite validates the stability improvements by comparing:
1. Original SPICE vs Enhanced Stable SPICE
2. Performance across challenging scenarios
3. Numerical stability metrics
4. Convergence behavior

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import pytest
import warnings
import time
from typing import Dict, Tuple

from spice_core import SPICEEstimator, SPICEConfig
from spice_stable import StableSPICEEstimator, StableSPICEConfig


class SPICEStabilityComparison:
    """Comprehensive comparison of SPICE implementations."""

    def __init__(self):
        """Initialize comparison framework."""
        self.n_sensors = 8
        self.results = {}

    def generate_challenging_scenarios(self) -> Dict[str, np.ndarray]:
        """Generate various challenging covariance matrices."""
        scenarios = {}

        # Scenario 1: Well-conditioned (baseline)
        scenarios['well_conditioned'] = self._create_covariance_matrix(
            condition_number=1e3, snr_db=15, n_sources=2
        )

        # Scenario 2: Moderately ill-conditioned
        scenarios['moderate_ill_conditioned'] = self._create_covariance_matrix(
            condition_number=1e8, snr_db=10, n_sources=2
        )

        # Scenario 3: Severely ill-conditioned
        scenarios['severe_ill_conditioned'] = self._create_covariance_matrix(
            condition_number=1e12, snr_db=5, n_sources=2
        )

        # Scenario 4: Low SNR with good conditioning
        scenarios['low_snr'] = self._create_covariance_matrix(
            condition_number=1e4, snr_db=0, n_sources=2
        )

        # Scenario 5: Close sources (challenging for resolution)
        scenarios['close_sources'] = self._create_covariance_matrix(
            condition_number=1e6, snr_db=12, n_sources=2, source_separation=1.0
        )

        return scenarios

    def _create_covariance_matrix(self, condition_number: float, snr_db: float,
                                n_sources: int, source_separation: float = 10.0) -> np.ndarray:
        """Create covariance matrix with specified properties."""
        # Source angles
        if n_sources == 2:
            angles = np.array([0, source_separation])  # degrees
        else:
            angles = np.linspace(-20, 20, n_sources)

        # Steering vectors
        angles_rad = np.deg2rad(angles)
        steering_vecs = []
        for angle in angles_rad:
            sensor_positions = np.arange(self.n_sensors)
            phases = np.exp(1j * np.pi * sensor_positions * np.sin(angle))
            steering_vecs.append(phases[:, np.newaxis])

        A = np.hstack(steering_vecs)

        # Signal powers
        signal_power = 10**(snr_db/10)
        powers = signal_power * np.ones(n_sources)
        if n_sources > 1:
            powers[1:] *= np.linspace(0.8, 0.6, n_sources-1)
        P = np.diag(powers)

        # Create noise covariance with desired condition number
        # Generate random Hermitian matrix
        noise_base = (np.random.randn(self.n_sensors, self.n_sensors) +
                     1j * np.random.randn(self.n_sensors, self.n_sensors))
        noise_base = 0.5 * (noise_base + noise_base.conj().T)

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(noise_base)

        # Scale eigenvalues to achieve desired condition number
        min_eigenval = 1.0 / condition_number
        max_eigenval = 1.0
        scaled_eigenvals = np.linspace(min_eigenval, max_eigenval, self.n_sensors)

        # Reconstruct with desired conditioning
        noise_cov = eigenvecs @ np.diag(scaled_eigenvals) @ eigenvecs.conj().T

        # Total covariance
        signal_cov = A @ P @ A.conj().T
        total_cov = signal_cov + noise_cov

        # Add small random perturbation for realism
        perturbation = 0.01 * (np.random.randn(self.n_sensors, self.n_sensors) +
                              1j * np.random.randn(self.n_sensors, self.n_sensors))
        perturbation = 0.5 * (perturbation + perturbation.conj().T)
        total_cov += perturbation

        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(total_cov)
        min_eig = np.min(np.real(eigenvals))
        if min_eig <= 1e-12:
            total_cov += (abs(min_eig) + 1e-10) * np.eye(self.n_sensors)

        return total_cov

    def run_comparison(self) -> Dict:
        """Run comprehensive comparison between implementations."""
        print("=== SPICE Stability Comparison ===")
        print("Comparing Original vs Enhanced Stable SPICE")
        print("==========================================")

        scenarios = self.generate_challenging_scenarios()
        comparison_results = {}

        for scenario_name, sample_cov in scenarios.items():
            print(f"\nTesting scenario: {scenario_name}")
            print(f"  Condition number: {np.linalg.cond(sample_cov):.2e}")

            # Test original SPICE
            original_result = self._test_original_spice(sample_cov, scenario_name)

            # Test enhanced stable SPICE
            stable_result = self._test_stable_spice(sample_cov, scenario_name)

            # Compare results
            comparison = self._compare_results(original_result, stable_result, scenario_name)

            comparison_results[scenario_name] = {
                'original': original_result,
                'stable': stable_result,
                'comparison': comparison
            }

        # Generate summary
        summary = self._generate_comparison_summary(comparison_results)
        comparison_results['summary'] = summary

        self.results = comparison_results
        return comparison_results

    def _test_original_spice(self, sample_cov: np.ndarray, scenario_name: str) -> Dict:
        """Test original SPICE implementation."""
        print("    Testing original SPICE...")

        result = {
            'scenario': scenario_name,
            'implementation': 'Original SPICE',
            'success': False,
            'error': None,
            'execution_time': 0,
            'convergence_info': {},
            'stability_metrics': {}
        }

        try:
            start_time = time.time()

            config = SPICEConfig(max_iterations=100, grid_size=90)
            estimator = SPICEEstimator(self.n_sensors, config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(sample_cov)

            execution_time = time.time() - start_time

            conv_info = estimator.get_convergence_info()

            result.update({
                'success': True,
                'execution_time': execution_time,
                'convergence_info': conv_info,
                'power_spectrum': power_spectrum,
                'peaks': estimator.find_peaks(power_spectrum),
                'stability_metrics': {
                    'converged': conv_info['n_iterations'] < config.max_iterations,
                    'final_cost': conv_info['final_cost'],
                    'cost_reduction_ratio': conv_info['cost_reduction'] / max(conv_info['initial_cost'], 1e-15),
                    'power_spectrum_valid': np.all(power_spectrum >= 0) and np.all(np.isfinite(power_spectrum))
                }
            })

            print(f"      Converged: {result['stability_metrics']['converged']}")
            print(f"      Iterations: {conv_info['n_iterations']}")
            print(f"      Execution time: {execution_time:.3f}s")

        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            print(f"      FAILED: {e}")

        return result

    def _test_stable_spice(self, sample_cov: np.ndarray, scenario_name: str) -> Dict:
        """Test enhanced stable SPICE implementation."""
        print("    Testing enhanced stable SPICE...")

        result = {
            'scenario': scenario_name,
            'implementation': 'Enhanced Stable SPICE',
            'success': False,
            'error': None,
            'execution_time': 0,
            'convergence_info': {},
            'stability_metrics': {}
        }

        try:
            start_time = time.time()

            config = StableSPICEConfig(max_iterations=100, grid_size=90)
            estimator = StableSPICEEstimator(self.n_sensors, config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(sample_cov)

            execution_time = time.time() - start_time

            stability_report = estimator.get_stability_report()

            result.update({
                'success': True,
                'execution_time': execution_time,
                'convergence_info': stability_report['convergence_info'],
                'stability_report': stability_report,
                'power_spectrum': power_spectrum,
                'peaks': estimator.find_peaks(power_spectrum),
                'stability_metrics': stability_report['stability_metrics']
            })

            print(f"      Converged: {stability_report['convergence_info']['n_iterations'] < 100}")
            print(f"      Iterations: {stability_report['convergence_info']['n_iterations']}")
            print(f"      Execution time: {execution_time:.3f}s")
            print(f"      Stability score: {stability_report['stability_metrics'].get('overall_stability_score', 'N/A'):.3f}")

        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            print(f"      FAILED: {e}")

        return result

    def _compare_results(self, original: Dict, stable: Dict, scenario_name: str) -> Dict:
        """Compare results between implementations."""
        comparison = {
            'scenario': scenario_name,
            'both_succeeded': original['success'] and stable['success'],
            'improvement_metrics': {}
        }

        if comparison['both_succeeded']:
            # Convergence comparison
            orig_conv = original['stability_metrics']['converged']
            stable_conv = stable['convergence_info']['n_iterations'] < 100

            # Performance comparison
            orig_iterations = original['convergence_info']['n_iterations']
            stable_iterations = stable['convergence_info']['n_iterations']

            # Execution time comparison
            time_ratio = stable['execution_time'] / max(original['execution_time'], 1e-6)

            # Stability comparison
            orig_valid = original['stability_metrics']['power_spectrum_valid']
            stable_valid = (np.all(stable['power_spectrum'] >= 0) and
                           np.all(np.isfinite(stable['power_spectrum'])))

            # Peak detection comparison
            orig_peaks = len(original['peaks']['angles'])
            stable_peaks = len(stable['peaks']['angles'])

            comparison['improvement_metrics'] = {
                'convergence_improvement': stable_conv and not orig_conv,
                'iteration_reduction': max(0, orig_iterations - stable_iterations),
                'execution_time_ratio': time_ratio,
                'stability_improvement': stable_valid and not orig_valid,
                'peak_detection_improvement': stable_peaks >= orig_peaks,
                'overall_better': (
                    (stable_conv >= orig_conv) and
                    (stable_valid >= orig_valid) and
                    (stable_iterations <= orig_iterations * 1.2)  # Allow 20% more iterations
                )
            }

        elif stable['success'] and not original['success']:
            comparison['improvement_metrics'] = {
                'stability_rescue': True,
                'original_failed': True,
                'stable_succeeded': True
            }

        elif original['success'] and not stable['success']:
            comparison['improvement_metrics'] = {
                'regression': True,
                'original_succeeded': True,
                'stable_failed': True
            }

        else:
            comparison['improvement_metrics'] = {
                'both_failed': True
            }

        return comparison

    def _generate_comparison_summary(self, results: Dict) -> Dict:
        """Generate comprehensive comparison summary."""
        scenarios = [k for k in results.keys() if k != 'summary']
        n_scenarios = len(scenarios)

        # Count successes
        original_successes = sum(1 for s in scenarios if results[s]['original']['success'])
        stable_successes = sum(1 for s in scenarios if results[s]['stable']['success'])

        # Count improvements
        convergence_improvements = sum(1 for s in scenarios
                                     if results[s]['comparison']['improvement_metrics'].get('convergence_improvement', False))

        stability_rescues = sum(1 for s in scenarios
                              if results[s]['comparison']['improvement_metrics'].get('stability_rescue', False))

        overall_improvements = sum(1 for s in scenarios
                                 if results[s]['comparison']['improvement_metrics'].get('overall_better', False))

        # Average execution time ratios
        time_ratios = [results[s]['comparison']['improvement_metrics'].get('execution_time_ratio', 1.0)
                      for s in scenarios if results[s]['comparison']['both_succeeded']]
        avg_time_ratio = np.mean(time_ratios) if time_ratios else float('inf')

        summary = {
            'total_scenarios': n_scenarios,
            'original_success_rate': original_successes / n_scenarios,
            'stable_success_rate': stable_successes / n_scenarios,
            'success_rate_improvement': (stable_successes - original_successes) / n_scenarios,
            'convergence_improvements': convergence_improvements,
            'stability_rescues': stability_rescues,
            'overall_improvements': overall_improvements,
            'average_execution_time_ratio': avg_time_ratio,
            'recommendation': self._generate_recommendation(
                stable_successes, original_successes, overall_improvements,
                stability_rescues, avg_time_ratio
            )
        }

        return summary

    def _generate_recommendation(self, stable_success: int, orig_success: int,
                               improvements: int, rescues: int, time_ratio: float) -> str:
        """Generate recommendation based on results."""
        if stable_success > orig_success + 1:
            return "Strong recommendation: Enhanced Stable SPICE shows significant stability improvements"
        elif stable_success > orig_success:
            return "Moderate recommendation: Enhanced Stable SPICE shows some stability improvements"
        elif improvements + rescues > 2:
            return "Qualified recommendation: Enhanced Stable SPICE helps in challenging scenarios"
        elif time_ratio < 2.0:
            return "Neutral: Enhanced Stable SPICE maintains performance with better stability monitoring"
        else:
            return "Caution: Enhanced Stable SPICE may have performance overhead"

    def print_detailed_summary(self):
        """Print detailed comparison summary."""
        if not self.results:
            print("No results available. Run comparison first.")
            return

        summary = self.results['summary']

        print("\n" + "="*60)
        print("DETAILED COMPARISON SUMMARY")
        print("="*60)

        print(f"Total scenarios tested: {summary['total_scenarios']}")
        print(f"Original SPICE success rate: {summary['original_success_rate']:.1%}")
        print(f"Enhanced Stable SPICE success rate: {summary['stable_success_rate']:.1%}")
        print(f"Success rate improvement: {summary['success_rate_improvement']:.1%}")

        print(f"\nPerformance Improvements:")
        print(f"  Convergence improvements: {summary['convergence_improvements']}")
        print(f"  Stability rescues: {summary['stability_rescues']}")
        print(f"  Overall improvements: {summary['overall_improvements']}")

        print(f"\nExecution time ratio (Enhanced/Original): {summary['average_execution_time_ratio']:.2f}x")

        print(f"\nRecommendation: {summary['recommendation']}")

        # Detailed scenario results
        print(f"\nDetailed Scenario Results:")
        print(f"{'Scenario':<25} {'Original':<10} {'Enhanced':<10} {'Improvement':<12}")
        print(f"{'-'*60}")

        scenarios = [k for k in self.results.keys() if k != 'summary']
        for scenario in scenarios:
            orig = "PASS" if self.results[scenario]['original']['success'] else "FAIL"
            stable = "PASS" if self.results[scenario]['stable']['success'] else "FAIL"
            improvement = "YES" if self.results[scenario]['comparison']['improvement_metrics'].get('overall_better', False) else "NO"

            print(f"{scenario:<25} {orig:<10} {stable:<10} {improvement:<12}")


def test_stability_improvements():
    """Pytest test for stability improvements."""
    comparator = SPICEStabilityComparison()
    results = comparator.run_comparison()

    # Test that enhanced version performs at least as well as original
    summary = results['summary']

    # Should not reduce success rate significantly
    assert summary['success_rate_improvement'] >= -0.2, "Enhanced SPICE should not significantly reduce success rate"

    # Should show some improvements in challenging scenarios
    assert summary['stability_rescues'] + summary['overall_improvements'] > 0, "Should show some stability improvements"

    # Execution time should not be excessive
    assert summary['average_execution_time_ratio'] < 5.0, "Execution time overhead should be reasonable"

    return results


if __name__ == "__main__":
    # Run comprehensive comparison
    comparator = SPICEStabilityComparison()
    results = comparator.run_comparison()

    # Print detailed summary
    comparator.print_detailed_summary()

    print("\nStability comparison complete!")
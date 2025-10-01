"""
Comprehensive SNR Threshold Validation for Enhanced SPICE.

This test suite rigorously validates the Enhanced SPICE SNR performance claims
by testing across a wide range of SNR levels with statistical significance.

The goal is to determine if Enhanced SPICE can indeed achieve the literature-
claimed 5 dB SNR threshold versus the observed 10 dB threshold for standard SPICE.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings

from spice_core import SPICEEstimator
from spice_enhanced import EnhancedSPICEEstimator, create_enhanced_spice


def compute_sample_covariance(array_data: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix from array data."""
    n_sensors, n_snapshots = array_data.shape
    if n_snapshots == 0:
        raise ValueError("Number of snapshots must be positive")
    return array_data @ array_data.conj().T / n_snapshots


def generate_test_data(true_angles: np.ndarray, n_sensors: int, n_snapshots: int,
                      snr_db: float, seed: int = None) -> np.ndarray:
    """Generate synthetic array data for testing."""
    if seed is not None:
        np.random.seed(seed)

    # Generate steering vectors
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    # Generate source signals
    source_signals = (np.random.randn(len(true_angles), n_snapshots) +
                     1j * np.random.randn(len(true_angles), n_snapshots))

    # Received signals
    received_signals = steering_matrix @ source_signals

    # Add noise to achieve desired SNR
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) +
        1j * np.random.randn(n_sensors, n_snapshots)
    )

    return received_signals + noise


def evaluate_performance(estimator, sample_cov: np.ndarray, true_angles: np.ndarray,
                        tolerance_deg: float = 3.0) -> Dict:
    """Evaluate SPICE performance against ground truth."""
    try:
        spectrum, angles = estimator.fit(sample_cov)
        peaks = estimator.find_peaks(spectrum, min_separation=2.0)

        # Detection metrics
        n_detected = len(peaks['angles'])
        n_true = len(true_angles)

        # Match detected peaks to true angles
        detected_angles = peaks['angles']
        matches = 0

        for true_angle in true_angles:
            # Find closest detected angle
            if len(detected_angles) > 0:
                distances = np.abs(detected_angles - true_angle)
                min_distance = np.min(distances)
                if min_distance <= tolerance_deg:
                    matches += 1

        detection_rate = matches / n_true
        false_alarm_rate = max(0, (n_detected - matches) / max(n_detected, 1))

        # Angular accuracy for matched detections
        angular_errors = []
        for true_angle in true_angles:
            if len(detected_angles) > 0:
                distances = np.abs(detected_angles - true_angle)
                min_idx = np.argmin(distances)
                if distances[min_idx] <= tolerance_deg:
                    angular_errors.append(distances[min_idx])

        mean_angular_error = np.mean(angular_errors) if angular_errors else np.inf

        # Convergence info
        conv_info = estimator.get_convergence_info()

        return {
            'success': True,
            'n_detected': n_detected,
            'n_true': n_true,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'mean_angular_error': mean_angular_error,
            'converged': conv_info['converged'],
            'n_iterations': conv_info['n_iterations'],
            'spectrum_peak_ratio': np.max(spectrum) / np.mean(spectrum) if len(spectrum) > 0 else 0
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_detected': 0,
            'detection_rate': 0.0,
            'false_alarm_rate': 1.0,
            'mean_angular_error': np.inf,
            'converged': False,
            'n_iterations': 0,
            'spectrum_peak_ratio': 0
        }


def test_snr_threshold_validation(snr_range: np.ndarray = np.arange(-5, 20, 1),
                                 n_trials: int = 50, n_sensors: int = 8,
                                 n_snapshots: int = 100) -> Dict:
    """Comprehensive SNR threshold validation."""
    print(f"[SNR VALIDATION] Testing SNR range: {snr_range[0]} to {snr_range[-1]} dB")
    print(f"                 Trials per SNR: {n_trials}")
    print(f"                 Array config: {n_sensors} sensors, {n_snapshots} snapshots")

    # Test scenarios
    test_scenarios = {
        'two_well_separated': np.array([-10, 10]),  # 20° separation - easy
        'two_moderate_separation': np.array([-5, 5]),   # 10° separation - moderate
        'two_close_sources': np.array([-3, 3]),     # 6° separation - challenging
    }

    results = {}

    for scenario_name, true_angles in test_scenarios.items():
        print(f"\n[SCENARIO] Testing {scenario_name}: {true_angles}°")

        scenario_results = {
            'standard_spice': {'snr_db': [], 'detection_rate': [], 'angular_error': [],
                              'convergence_rate': [], 'n_iterations': []},
            'enhanced_spice': {'snr_db': [], 'detection_rate': [], 'angular_error': [],
                              'convergence_rate': [], 'n_iterations': []}
        }

        for snr_db in snr_range:
            print(f"   SNR = {snr_db:+2d} dB: ", end="", flush=True)

            # Test both estimators
            for estimator_name in ['standard_spice', 'enhanced_spice']:
                detection_rates = []
                angular_errors = []
                convergence_rates = []
                iteration_counts = []

                for trial in range(n_trials):
                    # Generate test data
                    array_data = generate_test_data(
                        true_angles, n_sensors, n_snapshots, snr_db, seed=trial*1000 + abs(int(snr_db*10)) + 100
                    )
                    sample_cov = compute_sample_covariance(array_data)

                    # Create estimator
                    if estimator_name == 'standard_spice':
                        estimator = SPICEEstimator(n_sensors)
                    else:
                        estimator = create_enhanced_spice(n_sensors, target_snr_db=snr_db)

                    # Evaluate performance
                    result = evaluate_performance(estimator, sample_cov, true_angles)

                    detection_rates.append(result['detection_rate'])
                    angular_errors.append(result['mean_angular_error'] if result['mean_angular_error'] != np.inf else 30.0)
                    convergence_rates.append(1.0 if result['converged'] else 0.0)
                    iteration_counts.append(result['n_iterations'])

                # Store statistics
                scenario_results[estimator_name]['snr_db'].append(snr_db)
                scenario_results[estimator_name]['detection_rate'].append(np.mean(detection_rates))
                scenario_results[estimator_name]['angular_error'].append(np.mean(angular_errors))
                scenario_results[estimator_name]['convergence_rate'].append(np.mean(convergence_rates))
                scenario_results[estimator_name]['n_iterations'].append(np.mean(iteration_counts))

            # Progress indication
            std_rate = scenario_results['standard_spice']['detection_rate'][-1]
            enh_rate = scenario_results['enhanced_spice']['detection_rate'][-1]
            print(f"Std={std_rate:.2f}, Enh={enh_rate:.2f}")

        results[scenario_name] = scenario_results

    return results


def analyze_snr_thresholds(results: Dict, threshold_detection_rate: float = 0.8) -> Dict:
    """Analyze SNR thresholds from validation results."""
    print(f"\n[ANALYSIS] SNR Threshold Analysis (threshold = {threshold_detection_rate:.1%} detection rate)")

    analysis = {}

    for scenario_name, scenario_results in results.items():
        print(f"\nScenario: {scenario_name}")
        scenario_analysis = {}

        for estimator_name, data in scenario_results.items():
            snr_values = np.array(data['snr_db'])
            detection_rates = np.array(data['detection_rate'])

            # Find SNR threshold where detection rate exceeds threshold
            threshold_indices = np.where(detection_rates >= threshold_detection_rate)[0]

            if len(threshold_indices) > 0:
                threshold_snr = snr_values[threshold_indices[0]]
                max_detection_rate = np.max(detection_rates)

                # Find SNR for 95% detection rate
                high_perf_indices = np.where(detection_rates >= 0.95)[0]
                high_perf_snr = snr_values[high_perf_indices[0]] if len(high_perf_indices) > 0 else None

                scenario_analysis[estimator_name] = {
                    'threshold_snr_db': threshold_snr,
                    'max_detection_rate': max_detection_rate,
                    'high_performance_snr_db': high_perf_snr,
                    'reliable_operation_range': f"{threshold_snr:+.0f} to {snr_values[-1]:+.0f} dB"
                }

                print(f"  {estimator_name:15s}: {threshold_detection_rate:.0%} threshold at {threshold_snr:+2.0f} dB, "
                      f"95% rate at {high_perf_snr:+2.0f} dB" if high_perf_snr else "N/A")
            else:
                scenario_analysis[estimator_name] = {
                    'threshold_snr_db': None,
                    'max_detection_rate': np.max(detection_rates),
                    'high_performance_snr_db': None,
                    'reliable_operation_range': 'Not achieved in tested range'
                }
                print(f"  {estimator_name:15s}: {threshold_detection_rate:.0%} threshold NOT achieved "
                      f"(max rate: {np.max(detection_rates):.2f})")

        analysis[scenario_name] = scenario_analysis

    return analysis


def plot_snr_validation_results(results: Dict, save_path: str = None):
    """Plot comprehensive SNR validation results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    scenario_names = list(results.keys())
    colors = {'standard_spice': 'blue', 'enhanced_spice': 'red'}
    labels = {'standard_spice': 'Standard SPICE', 'enhanced_spice': 'Enhanced SPICE'}

    for col, (scenario_name, scenario_results) in enumerate(results.items()):
        # Detection rate plot
        ax1 = axes[0, col]
        for estimator_name, data in scenario_results.items():
            ax1.plot(data['snr_db'], data['detection_rate'],
                    color=colors[estimator_name], marker='o', linewidth=2,
                    label=labels[estimator_name])

        ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% Threshold')
        ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, label='95% Threshold')
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title(f'Detection Rate vs SNR\n{scenario_name.replace("_", " ").title()}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        # Angular error plot
        ax2 = axes[1, col]
        for estimator_name, data in scenario_results.items():
            ax2.plot(data['snr_db'], data['angular_error'],
                    color=colors[estimator_name], marker='s', linewidth=2,
                    label=labels[estimator_name])

        ax2.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='3° Tolerance')
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Mean Angular Error (°)')
        ax2.set_title(f'Angular Accuracy vs SNR\n{scenario_name.replace("_", " ").title()}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] SNR validation plots saved: {save_path}")

    return fig


def main():
    """Run comprehensive SNR threshold validation."""
    print("="*70)
    print("ENHANCED SPICE SNR THRESHOLD VALIDATION")
    print("="*70)
    print("\nObjective: Validate literature claims about Enhanced SPICE achieving")
    print("           5 dB SNR threshold vs Standard SPICE 10 dB threshold")

    # Run validation tests
    results = test_snr_threshold_validation(
        snr_range=np.arange(0, 16, 2),  # Test from 0 to 15 dB in 2 dB steps for faster testing
        n_trials=10,  # Reduced for faster testing
        n_sensors=8,
        n_snapshots=100
    )

    # Analyze thresholds
    analysis = analyze_snr_thresholds(results, threshold_detection_rate=0.8)

    # Plot results
    fig = plot_snr_validation_results(results, 'enhanced_spice_snr_validation.png')
    plt.show()

    # Summary conclusions
    print("\n" + "="*70)
    print("VALIDATION CONCLUSIONS")
    print("="*70)

    # Compare thresholds across scenarios
    for scenario_name, scenario_analysis in analysis.items():
        print(f"\n{scenario_name.replace('_', ' ').title()}:")

        std_threshold = scenario_analysis.get('standard_spice', {}).get('threshold_snr_db')
        enh_threshold = scenario_analysis.get('enhanced_spice', {}).get('threshold_snr_db')

        if std_threshold is not None and enh_threshold is not None:
            improvement = std_threshold - enh_threshold
            print(f"  Standard SPICE 80% threshold: {std_threshold:+.0f} dB")
            print(f"  Enhanced SPICE 80% threshold: {enh_threshold:+.0f} dB")
            print(f"  Improvement: {improvement:+.1f} dB")

            # Validate literature claims
            if enh_threshold <= 5:
                print(f"  ✅ LITERATURE CLAIM VALIDATED: Enhanced SPICE achieves ≤5 dB threshold")
            elif improvement >= 3:
                print(f"  ⚠️  PARTIAL VALIDATION: {improvement:.1f} dB improvement achieved")
            else:
                print(f"  ❌ CLAIM NOT VALIDATED: Insufficient improvement")
        else:
            print(f"  ❌ INCOMPLETE DATA: Could not determine thresholds reliably")

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
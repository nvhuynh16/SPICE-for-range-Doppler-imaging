"""
Test Improved Enhanced SPICE for 5 dB SNR Threshold Validation

This test validates whether the sophisticated SNR estimation and adaptive
regularization strategies enable Enhanced SPICE to achieve the literature-
claimed 5 dB SNR threshold for reliable operation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from spice_core import SPICEEstimator
from spice_enhanced import create_enhanced_spice
from improved_enhanced_spice import create_improved_enhanced_spice
from advanced_peak_detection import create_advanced_detector


def compute_sample_covariance(array_data: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix from array data."""
    n_sensors, n_snapshots = array_data.shape
    return array_data @ array_data.conj().T / n_snapshots


def generate_test_data(true_angles: np.ndarray, n_sensors: int, n_snapshots: int, snr_db: float) -> np.ndarray:
    """Generate test data with specified SNR."""
    np.random.seed(42)

    # Steering vectors
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    # Source signals
    source_signals = (np.random.randn(len(true_angles), n_snapshots) +
                     1j * np.random.randn(len(true_angles), n_snapshots))

    # Received signals + noise
    received_signals = steering_matrix @ source_signals
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) +
        1j * np.random.randn(n_sensors, n_snapshots)
    )

    return received_signals + noise


def evaluate_detection_performance(detected_angles: np.ndarray, true_angles: np.ndarray) -> Dict:
    """Evaluate detection performance."""
    n_true = len(true_angles)
    n_detected = len(detected_angles)

    if n_detected == 0:
        return {'detection_rate': 0.0, 'false_alarm_rate': 0.0, 'angular_error': np.inf}

    matches = 0
    angular_errors = []

    for true_angle in true_angles:
        if len(detected_angles) > 0:
            distances = np.abs(detected_angles - true_angle)
            min_distance = np.min(distances)
            if min_distance <= 3.0:  # 3° tolerance
                matches += 1
                angular_errors.append(min_distance)

    detection_rate = matches / n_true if n_true > 0 else 0
    false_alarms = max(0, n_detected - matches)
    false_alarm_rate = false_alarms / max(n_detected, 1)
    mean_angular_error = np.mean(angular_errors) if angular_errors else np.inf

    return {
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'angular_error': mean_angular_error,
        'n_detected': n_detected,
        'matches': matches
    }


def test_snr_threshold_validation():
    """Test Enhanced SPICE variants across SNR range to find thresholds."""
    print("="*70)
    print("IMPROVED ENHANCED SPICE SNR THRESHOLD VALIDATION")
    print("="*70)
    print("Goal: Validate literature claim of 5 dB SNR threshold for Enhanced SPICE")

    # Test parameters
    true_angles = np.array([-8, 8])  # Well-separated targets for best performance
    n_sensors = 8
    n_snapshots = 200  # More snapshots for better estimation
    snr_range = np.arange(1, 12, 1)  # Focus on critical low SNR range
    n_trials = 5  # Multiple trials for statistical significance

    print(f"Test configuration:")
    print(f"  True angles: {true_angles}")
    print(f"  Array: {n_sensors} sensors, {n_snapshots} snapshots")
    print(f"  SNR range: {snr_range[0]} to {snr_range[-1]} dB")
    print(f"  Trials per SNR: {n_trials}")

    # Store results for all methods
    results = {
        'standard_spice': [],
        'enhanced_spice': [],
        'improved_enhanced_spice': []
    }

    print(f"\n[TESTING ACROSS SNR RANGE]")

    for snr_db in snr_range:
        print(f"\\nSNR = {snr_db:2d} dB:")

        # Run multiple trials for statistical significance
        trial_results = {method: [] for method in results.keys()}

        for trial in range(n_trials):
            # Generate test data
            array_data = generate_test_data(true_angles, n_sensors, n_snapshots, snr_db)
            sample_cov = compute_sample_covariance(array_data)

            # Test Standard SPICE + Advanced Detection
            estimator_std = SPICEEstimator(n_sensors)
            spectrum_std, angles_std = estimator_std.fit(sample_cov)
            detector_std = create_advanced_detector(n_sensors, angles_std)
            peaks_std = detector_std.detect_peaks(spectrum_std, sample_cov, snr_db)
            perf_std = evaluate_detection_performance(peaks_std['angles'], true_angles)
            trial_results['standard_spice'].append(perf_std)

            # Test Original Enhanced SPICE + Advanced Detection
            estimator_enh = create_enhanced_spice(n_sensors, target_snr_db=snr_db)
            spectrum_enh, angles_enh = estimator_enh.fit(sample_cov)
            detector_enh = create_advanced_detector(n_sensors, angles_enh)
            peaks_enh = detector_enh.detect_peaks(spectrum_enh, sample_cov, snr_db)
            perf_enh = evaluate_detection_performance(peaks_enh['angles'], true_angles)
            trial_results['enhanced_spice'].append(perf_enh)

            # Test Improved Enhanced SPICE + Advanced Detection
            estimator_imp = create_improved_enhanced_spice(n_sensors, target_snr_db=snr_db)
            spectrum_imp, angles_imp = estimator_imp.fit(sample_cov)
            peaks_imp = estimator_imp.find_peaks_advanced(spectrum_imp, sample_cov)
            perf_imp = evaluate_detection_performance(peaks_imp['angles'], true_angles)
            trial_results['improved_enhanced_spice'].append(perf_imp)

        # Compute average performance for this SNR
        for method_name, trial_data in trial_results.items():
            avg_detection_rate = np.mean([trial['detection_rate'] for trial in trial_data])
            avg_false_alarm_rate = np.mean([trial['false_alarm_rate'] for trial in trial_data])
            avg_angular_error = np.mean([trial['angular_error'] for trial in trial_data if trial['angular_error'] != np.inf])

            results[method_name].append({
                'snr_db': snr_db,
                'detection_rate': avg_detection_rate,
                'false_alarm_rate': avg_false_alarm_rate,
                'angular_error': avg_angular_error if not np.isnan(avg_angular_error) else np.inf
            })

        # Print results for this SNR
        std_rate = results['standard_spice'][-1]['detection_rate']
        enh_rate = results['enhanced_spice'][-1]['detection_rate']
        imp_rate = results['improved_enhanced_spice'][-1]['detection_rate']

        print(f"  Standard: {std_rate:.2f}")
        print(f"  Enhanced: {enh_rate:.2f}")
        print(f"  Improved: {imp_rate:.2f}")

    return results


def analyze_snr_thresholds(results: Dict) -> Dict:
    """Analyze SNR thresholds from results."""
    print(f"\\n[SNR THRESHOLD ANALYSIS]")

    thresholds = {}
    threshold_detection_rate = 0.8  # 80% detection rate threshold

    for method_name, method_results in results.items():
        # Find SNR where detection rate first exceeds threshold
        threshold_snr = None
        for result in method_results:
            if result['detection_rate'] >= threshold_detection_rate:
                threshold_snr = result['snr_db']
                break

        thresholds[method_name] = threshold_snr
        print(f"{method_name:25s}: {threshold_snr} dB" if threshold_snr else f"{method_name:25s}: No 80% threshold found")

    # Analyze improvements
    std_threshold = thresholds.get('standard_spice')
    enh_threshold = thresholds.get('enhanced_spice')
    imp_threshold = thresholds.get('improved_enhanced_spice')

    print(f"\\n[IMPROVEMENT ANALYSIS]")

    if std_threshold and enh_threshold:
        enh_improvement = std_threshold - enh_threshold
        print(f"Enhanced SPICE improvement over Standard: {enh_improvement} dB")

    if std_threshold and imp_threshold:
        imp_improvement = std_threshold - imp_threshold
        print(f"Improved Enhanced SPICE improvement over Standard: {imp_improvement} dB")

    if enh_threshold and imp_threshold:
        imp_over_enh = enh_threshold - imp_threshold
        print(f"Improved over Enhanced SPICE: {imp_over_enh} dB")

    print(f"\\n[LITERATURE CLAIM VALIDATION]")

    if imp_threshold and imp_threshold <= 5:
        print(f"[VALIDATED] Improved Enhanced SPICE achieves 5 dB threshold: {imp_threshold} dB")
    elif enh_threshold and enh_threshold <= 5:
        print(f"[VALIDATED] Enhanced SPICE achieves 5 dB threshold: {enh_threshold} dB")
    elif imp_threshold and imp_threshold <= 7:
        print(f"[PARTIAL] Improved Enhanced SPICE close to claim: {imp_threshold} dB (within 2 dB)")
    else:
        print(f"[NOT VALIDATED] Neither variant achieves 5 dB threshold")
        if imp_threshold:
            print(f"  Best achieved: {imp_threshold} dB (Improved Enhanced SPICE)")

    return thresholds


def plot_snr_performance_comparison(results: Dict):
    """Plot SNR performance comparison."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Detection rate vs SNR
    plt.subplot(2, 2, 1)
    colors = {'standard_spice': 'blue', 'enhanced_spice': 'red', 'improved_enhanced_spice': 'green'}
    labels = {'standard_spice': 'Standard SPICE', 'enhanced_spice': 'Enhanced SPICE', 'improved_enhanced_spice': 'Improved Enhanced SPICE'}

    for method_name, method_results in results.items():
        snr_values = [r['snr_db'] for r in method_results]
        detection_rates = [r['detection_rate'] for r in method_results]

        plt.plot(snr_values, detection_rates, 'o-', color=colors[method_name],
                linewidth=2, markersize=6, label=labels[method_name])

    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% Threshold')
    plt.axvline(x=5, color='orange', linestyle=':', alpha=0.7, label='Literature Claim (5 dB)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)

    # Plot 2: False alarm rate vs SNR
    plt.subplot(2, 2, 2)
    for method_name, method_results in results.items():
        snr_values = [r['snr_db'] for r in method_results]
        false_alarm_rates = [r['false_alarm_rate'] for r in method_results]

        plt.plot(snr_values, false_alarm_rates, 'o-', color=colors[method_name],
                linewidth=2, markersize=6, label=labels[method_name])

    plt.xlabel('SNR (dB)')
    plt.ylabel('False Alarm Rate')
    plt.title('False Alarm Rate vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)

    # Plot 3: Angular error vs SNR
    plt.subplot(2, 2, 3)
    for method_name, method_results in results.items():
        snr_values = [r['snr_db'] for r in method_results]
        angular_errors = [r['angular_error'] if r['angular_error'] != np.inf else 10 for r in method_results]

        plt.plot(snr_values, angular_errors, 'o-', color=colors[method_name],
                linewidth=2, markersize=6, label=labels[method_name])

    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Angular Error (degrees)')
    plt.title('Angular Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')

    # Plot 4: Performance improvement factors
    plt.subplot(2, 2, 4)
    snr_values = [r['snr_db'] for r in results['standard_spice']]
    enh_improvement = []
    imp_improvement = []

    for i, snr in enumerate(snr_values):
        std_rate = results['standard_spice'][i]['detection_rate']
        enh_rate = results['enhanced_spice'][i]['detection_rate']
        imp_rate = results['improved_enhanced_spice'][i]['detection_rate']

        enh_factor = enh_rate / max(std_rate, 0.01)
        imp_factor = imp_rate / max(std_rate, 0.01)

        enh_improvement.append(enh_factor)
        imp_improvement.append(imp_factor)

    plt.plot(snr_values, enh_improvement, 'o-', color='red', linewidth=2, label='Enhanced vs Standard')
    plt.plot(snr_values, imp_improvement, 'o-', color='green', linewidth=2, label='Improved vs Standard')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No improvement')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Improvement Factor')
    plt.title('Detection Rate Improvement Factor')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('improved_enhanced_spice_validation.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main validation function."""
    print("Testing Improved Enhanced SPICE for Literature Claim Validation...")

    # Run SNR threshold validation
    results = test_snr_threshold_validation()

    # Analyze thresholds
    thresholds = analyze_snr_thresholds(results)

    # Plot results
    plot_snr_performance_comparison(results)

    print(f"\\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return results, thresholds


if __name__ == "__main__":
    results, thresholds = main()
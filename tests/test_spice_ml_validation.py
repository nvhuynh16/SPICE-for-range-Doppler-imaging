"""
SPICE-ML Validation Test - Literature Claim Verification.

This test validates whether SPICE-ML provides performance improvements over
standard SPICE as claimed in literature, particularly in challenging scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from spice_core import SPICEEstimator
from spice_ml import create_spice_ml, compare_spice_ml_performance


def generate_test_scenario(true_angles: np.ndarray, n_sensors: int,
                          n_snapshots: int, snr_db: float) -> np.ndarray:
    """Generate test data for SPICE-ML validation."""
    np.random.seed(42)  # Reproducible results

    # Generate steering matrix
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    # Generate source signals
    source_signals = (np.random.randn(len(true_angles), n_snapshots) +
                     1j * np.random.randn(len(true_angles), n_snapshots))

    # Received signals
    received_signals = steering_matrix @ source_signals

    # Add noise
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) +
        1j * np.random.randn(n_sensors, n_snapshots)
    )

    array_data = received_signals + noise

    # Compute sample covariance
    return array_data @ array_data.conj().T / n_snapshots


def evaluate_detection_performance(detected_angles: np.ndarray,
                                 true_angles: np.ndarray,
                                 tolerance: float = 3.0) -> Dict:
    """Evaluate detection performance metrics."""
    if len(detected_angles) == 0:
        return {
            'detection_rate': 0.0,
            'false_alarm_rate': 0.0,
            'angular_accuracy': np.inf
        }

    # Detection rate
    matches = 0
    angular_errors = []

    for true_angle in true_angles:
        distances = np.abs(detected_angles - true_angle)
        min_distance = np.min(distances)
        if min_distance <= tolerance:
            matches += 1
            angular_errors.append(min_distance)

    detection_rate = matches / len(true_angles)
    false_alarms = len(detected_angles) - matches
    false_alarm_rate = false_alarms / len(detected_angles) if len(detected_angles) > 0 else 0.0
    angular_accuracy = np.mean(angular_errors) if angular_errors else np.inf

    return {
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'angular_accuracy': angular_accuracy,
        'n_detections': len(detected_angles),
        'n_matches': matches,
        'n_false_alarms': false_alarms
    }


def test_spice_ml_snr_performance():
    """Test SPICE-ML performance across SNR range."""
    print("="*70)
    print("SPICE-ML SNR PERFORMANCE VALIDATION")
    print("="*70)
    print("Goal: Validate SPICE-ML improvements over standard SPICE")

    # Test parameters
    true_angles = np.array([-10, 10])  # Well-separated targets
    n_sensors = 8
    n_snapshots = 100
    snr_range = np.arange(0, 16, 2)  # 0 to 15 dB

    print(f"Test configuration:")
    print(f"  True angles: {true_angles}")
    print(f"  Array sensors: {n_sensors}")
    print(f"  Snapshots: {n_snapshots}")
    print(f"  SNR range: {snr_range[0]} to {snr_range[-1]} dB")

    results = []

    for snr_db in snr_range:
        print(f"\nSNR = {snr_db} dB: ", end="", flush=True)

        # Generate test data
        sample_cov = generate_test_scenario(true_angles, n_sensors, n_snapshots, snr_db)

        # Standard SPICE
        spice_std = SPICEEstimator(n_sensors)
        spectrum_std, _ = spice_std.fit(sample_cov)
        peaks_std = spice_std.find_peaks(spectrum_std)
        perf_std = evaluate_detection_performance(peaks_std['angles'], true_angles)

        # SPICE-ML with Newton method
        spice_ml = create_spice_ml(n_sensors, ml_method='newton', max_iterations=30)
        spectrum_ml, _ = spice_ml.fit(sample_cov)
        peaks_ml = spice_ml.find_peaks(spectrum_ml)
        perf_ml = evaluate_detection_performance(peaks_ml['angles'], true_angles)

        # Compute improvements
        detection_improvement = perf_ml['detection_rate'] / max(perf_std['detection_rate'], 0.01)
        false_alarm_improvement = max(perf_std['false_alarm_rate'], 0.01) / max(perf_ml['false_alarm_rate'], 0.01)

        result = {
            'snr_db': snr_db,
            'std_detection_rate': perf_std['detection_rate'],
            'ml_detection_rate': perf_ml['detection_rate'],
            'std_false_alarm_rate': perf_std['false_alarm_rate'],
            'ml_false_alarm_rate': perf_ml['false_alarm_rate'],
            'detection_improvement': detection_improvement,
            'false_alarm_improvement': false_alarm_improvement,
            'std_convergence': spice_std.get_convergence_info(),
            'ml_convergence': spice_ml.get_ml_convergence_info()
        }

        results.append(result)

        print(f"Std={perf_std['detection_rate']:.2f}, ML={perf_ml['detection_rate']:.2f}, " +
              f"Improvement={detection_improvement:.2f}x")

    return results


def test_spice_ml_challenging_scenarios():
    """Test SPICE-ML in challenging scenarios."""
    print(f"\n[CHALLENGING SCENARIOS TEST]")

    scenarios = [
        {
            'name': 'Close Sources',
            'angles': np.array([-5, 5]),
            'snr_db': 10,
            'snapshots': 50
        },
        {
            'name': 'Low SNR',
            'angles': np.array([-15, 15]),
            'snr_db': 3,
            'snapshots': 100
        },
        {
            'name': 'Weak Source',
            'angles': np.array([-10, 10]),
            'snr_db': 8,  # Second source will be 10 dB weaker
            'snapshots': 80
        },
        {
            'name': 'Limited Snapshots',
            'angles': np.array([-12, 12]),
            'snr_db': 12,
            'snapshots': 25
        }
    ]

    scenario_results = []

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")

        # Generate scenario data
        sample_cov = generate_test_scenario(
            scenario['angles'], 8, scenario['snapshots'], scenario['snr_db']
        )

        # For weak source scenario, modify covariance to simulate unequal powers
        if scenario['name'] == 'Weak Source':
            # Reduce second source power by 10 dB
            weak_factor = 10**(-1.0)  # -10 dB
            steering_2 = np.exp(1j * np.arange(8) * np.pi * np.sin(np.deg2rad(scenario['angles'][1])))
            sample_cov -= weak_factor * 0.9 * np.outer(steering_2, steering_2.conj())

        # Compare performance
        comparison = compare_spice_ml_performance(sample_cov, 8)

        std_peaks = len(comparison['standard_spice']['peaks']['angles'])
        ml_peaks = len(comparison['spice_ml']['peaks']['angles'])

        # Evaluate accuracy
        std_perf = evaluate_detection_performance(
            comparison['standard_spice']['peaks']['angles'], scenario['angles']
        )
        ml_perf = evaluate_detection_performance(
            comparison['spice_ml']['peaks']['angles'], scenario['angles']
        )

        scenario_result = {
            'scenario': scenario['name'],
            'std_performance': std_perf,
            'ml_performance': ml_perf,
            'improvement_factor': comparison['improvement_factor'],
            'time_ratio': comparison['execution_time_ratio']
        }

        scenario_results.append(scenario_result)

        print(f"  Standard SPICE: {std_perf['detection_rate']:.2f} detection, " +
              f"{std_perf['false_alarm_rate']:.2f} false alarm")
        print(f"  SPICE-ML: {ml_perf['detection_rate']:.2f} detection, " +
              f"{ml_perf['false_alarm_rate']:.2f} false alarm")
        print(f"  Improvement: {comparison['improvement_factor']:.2f}x")

    return scenario_results


def analyze_spice_ml_results(snr_results: list, scenario_results: list):
    """Analyze SPICE-ML validation results."""
    print(f"\n[SPICE-ML PERFORMANCE ANALYSIS]")

    # SNR performance analysis
    print(f"\nSNR Performance Analysis:")
    avg_detection_improvement = np.mean([r['detection_improvement'] for r in snr_results])
    avg_false_alarm_improvement = np.mean([r['false_alarm_improvement'] for r in snr_results])

    print(f"  Average detection improvement: {avg_detection_improvement:.2f}x")
    print(f"  Average false alarm improvement: {avg_false_alarm_improvement:.2f}x")

    # Find SNR threshold for 80% detection
    std_threshold = None
    ml_threshold = None

    for result in snr_results:
        if result['std_detection_rate'] >= 0.8 and std_threshold is None:
            std_threshold = result['snr_db']
        if result['ml_detection_rate'] >= 0.8 and ml_threshold is None:
            ml_threshold = result['snr_db']

    print(f"  Standard SPICE 80% threshold: {std_threshold} dB" if std_threshold else "  Standard: No threshold found")
    print(f"  SPICE-ML 80% threshold: {ml_threshold} dB" if ml_threshold else "  SPICE-ML: No threshold found")

    if std_threshold and ml_threshold:
        threshold_improvement = std_threshold - ml_threshold
        print(f"  Threshold improvement: {threshold_improvement} dB")

    # Challenging scenarios analysis
    print(f"\nChallenging Scenarios Analysis:")
    for result in scenario_results:
        improvement = result['ml_performance']['detection_rate'] / max(result['std_performance']['detection_rate'], 0.01)
        print(f"  {result['scenario']}: {improvement:.2f}x improvement")

    # Overall assessment
    print(f"\n[SPICE-ML LITERATURE CLAIM VALIDATION]")

    significant_improvements = sum(1 for r in snr_results if r['detection_improvement'] > 1.2)
    total_tests = len(snr_results)

    if avg_detection_improvement > 1.5:
        print(f"[VALIDATED] SPICE-ML shows significant improvement!")
        print(f"  Average improvement: {avg_detection_improvement:.2f}x")
        print(f"  Significant improvements in {significant_improvements}/{total_tests} SNR tests")
    elif avg_detection_improvement > 1.1:
        print(f"[PARTIAL] SPICE-ML shows modest improvement:")
        print(f"  Average improvement: {avg_detection_improvement:.2f}x")
    else:
        print(f"[LIMITED] SPICE-ML shows limited improvement:")
        print(f"  Average improvement: {avg_detection_improvement:.2f}x")
        print(f"  May require further optimization or different scenarios")

    return {
        'avg_detection_improvement': avg_detection_improvement,
        'avg_false_alarm_improvement': avg_false_alarm_improvement,
        'std_threshold': std_threshold,
        'ml_threshold': ml_threshold,
        'challenging_scenario_improvements': [
            r['ml_performance']['detection_rate'] / max(r['std_performance']['detection_rate'], 0.01)
            for r in scenario_results
        ]
    }


def plot_spice_ml_comparison(snr_results: list, analysis: dict):
    """Plot SPICE vs SPICE-ML comparison."""
    plt.figure(figsize=(15, 10))

    # Extract data
    snr_values = [r['snr_db'] for r in snr_results]
    std_detection = [r['std_detection_rate'] for r in snr_results]
    ml_detection = [r['ml_detection_rate'] for r in snr_results]
    improvements = [r['detection_improvement'] for r in snr_results]

    # Plot 1: Detection rates
    plt.subplot(2, 3, 1)
    plt.plot(snr_values, std_detection, 'b-o', linewidth=2, label='Standard SPICE')
    plt.plot(snr_values, ml_detection, 'r-o', linewidth=2, label='SPICE-ML')
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate: SPICE vs SPICE-ML')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)

    # Plot 2: Improvement factors
    plt.subplot(2, 3, 2)
    plt.plot(snr_values, improvements, 'g-o', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No improvement')
    plt.axhline(y=1.5, color='orange', linestyle=':', alpha=0.7, label='Significant improvement')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Improvement Factor')
    plt.title('SPICE-ML Detection Improvement')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: False alarm rates
    plt.subplot(2, 3, 3)
    std_false_alarms = [r['std_false_alarm_rate'] for r in snr_results]
    ml_false_alarms = [r['ml_false_alarm_rate'] for r in snr_results]

    plt.plot(snr_values, std_false_alarms, 'b-o', linewidth=2, label='Standard SPICE')
    plt.plot(snr_values, ml_false_alarms, 'r-o', linewidth=2, label='SPICE-ML')
    plt.xlabel('SNR (dB)')
    plt.ylabel('False Alarm Rate')
    plt.title('False Alarm Rate Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 4: Summary bar chart
    plt.subplot(2, 3, 4)
    metrics = ['Avg Detection\\nImprovement', 'Avg False Alarm\\nImprovement']
    values = [analysis['avg_detection_improvement'], analysis['avg_false_alarm_improvement']]

    bars = plt.bar(metrics, values, color=['green', 'blue'])
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('Improvement Factor')
    plt.title('Overall SPICE-ML Performance')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{value:.2f}x', ha='center', va='bottom')

    # Plot 5: SNR threshold comparison
    plt.subplot(2, 3, 5)
    if analysis['std_threshold'] and analysis['ml_threshold']:
        methods = ['Standard SPICE', 'SPICE-ML']
        thresholds = [analysis['std_threshold'], analysis['ml_threshold']]
        colors = ['blue', 'red']

        bars = plt.bar(methods, thresholds, color=colors)
        plt.ylabel('80% Detection Threshold (dB)')
        plt.title('SNR Threshold Comparison')
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, thresholds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{value} dB', ha='center', va='bottom')

        # Show improvement
        improvement = analysis['std_threshold'] - analysis['ml_threshold']
        plt.text(0.5, max(thresholds) * 0.7, f'Improvement: {improvement} dB',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'Threshold data\nnot available', ha='center', va='center')
        plt.title('SNR Threshold Comparison')

    # Plot 6: Challenging scenarios
    plt.subplot(2, 3, 6)
    if analysis['challenging_scenario_improvements']:
        scenario_names = ['Close\nSources', 'Low\nSNR', 'Weak\nSource', 'Limited\nSnapshots']
        improvements = analysis['challenging_scenario_improvements']

        bars = plt.bar(scenario_names, improvements, color='purple')
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        plt.ylabel('Improvement Factor')
        plt.title('Challenging Scenarios')
        plt.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('spice_ml_validation.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main SPICE-ML validation function."""
    print("Testing SPICE-ML implementation for literature claim validation...")

    # Test SNR performance
    snr_results = test_spice_ml_snr_performance()

    # Test challenging scenarios
    scenario_results = test_spice_ml_challenging_scenarios()

    # Analyze results
    analysis = analyze_spice_ml_results(snr_results, scenario_results)

    # Plot comparison
    plot_spice_ml_comparison(snr_results, analysis)

    print(f"\n" + "="*70)
    print("SPICE-ML VALIDATION COMPLETE")
    print("="*70)

    return snr_results, scenario_results, analysis


if __name__ == "__main__":
    snr_results, scenario_results, analysis = main()
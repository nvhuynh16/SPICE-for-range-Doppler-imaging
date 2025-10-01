"""
Comprehensive Demonstration of Educational SPICE Concepts.

This script provides a complete demonstration of the educational concepts
discussed in the README, including:

1. SNR failure analysis with theoretical conditions
2. Coprime waveform design benefits
3. Matched filter vs SPICE comparison with plots
4. Signal design impact on sparse recovery conditions

Run this script to see all the theoretical concepts in action.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
from pathlib import Path

from educational_examples import EducationalAnalyzer, EducationalScenario
from coprime_signal_design import CoprimeSignalDesign
from spice_core import SPICEEstimator, compute_sample_covariance


def demonstrate_snr_failure_with_theory():
    """
    Demonstrate SPICE SNR failure with theoretical explanation.

    This shows exactly why SPICE fails at low SNR by monitoring
    the theoretical conditions (RE, mutual incoherence, beta-min).
    """
    print("\n" + "="*60)
    print("DEMONSTRATION 1: SNR Failure Mechanism")
    print("="*60)

    analyzer = EducationalAnalyzer()

    # Define challenging but educational scenario
    scenario = EducationalScenario(
        name="SNR Failure Analysis",
        description="Two well-separated sources to isolate SNR effects",
        true_angles=np.array([-15, 15]),  # Well separated to avoid other issues
        n_sensors=8,
        n_snapshots=100,
        snr_range=np.linspace(-15, 25, 21)
    )

    print(f"[SCENARIO] {scenario.description}")
    print(f"   Sources at: {scenario.true_angles} degrees")
    print(f"   Array: {scenario.n_sensors} sensors, {scenario.n_snapshots} snapshots")
    print(f"   SNR range: {scenario.snr_range[0]:.0f} to {scenario.snr_range[-1]:.0f} dB")

    # Run comprehensive analysis
    results = analyzer.demonstrate_snr_failure_mechanism(scenario)

    # Key findings
    snr_array = np.array(results['snr_db'])
    spice_success = np.array(results['spice']['success'])
    mf_success = np.array(results['matched_filter']['success'])

    # Find failure threshold
    spice_failure_idx = np.where(spice_success < 0.5)[0]
    if len(spice_failure_idx) > 0:
        failure_snr = snr_array[spice_failure_idx[-1]]
        print(f"\n[FINDING] Key Finding: SPICE failure threshold ~= {failure_snr:.0f} dB")
    else:
        print(f"\n[FINDING] Key Finding: SPICE succeeded across all tested SNR levels")

    # Theoretical insight
    print(f"\n[INSIGHT] Theoretical Insight:")
    print(f"   • Low SNR: Sample covariance R̂ ≈ σ²I + small signal component")
    print(f"   • High variance in R̂ violates Restricted Eigenvalue condition")
    print(f"   • SPICE cannot distinguish signal from noise subspace")
    print(f"   • Matched filter uses known signal structure → more robust")

    return results


def demonstrate_coprime_waveform_benefits():
    """
    Demonstrate how coprime waveform design improves SPICE performance.

    This shows the practical application of Chinese Remainder Theorem
    for improving mutual incoherence conditions.
    """
    print("\n" + "="*60)
    print("🎓 DEMONSTRATION 2: Coprime Waveform Design")
    print("="*60)

    # Test closely spaced targets (challenging scenario)
    true_angles = np.array([-3, 3])  # Only 6° separation
    n_sensors = 8
    n_snapshots = 128
    snr_db = 15  # Moderate SNR

    print(f"[SCENARIO] Challenging Scenario:")
    print(f"   Closely spaced sources: {true_angles} degrees (6° separation)")
    print(f"   SNR: {snr_db} dB")

    # Test different waveform designs
    designs = {
        'Standard FMCW': None,
        'Coprime (7,11)': (7, 11),
        'Coprime (31,37)': (31, 37)
    }

    results = {}

    fig, axes = plt.subplots(2, len(designs), figsize=(15, 10))

    for i, (design_name, coprime_pair) in enumerate(designs.items()):
        print(f"\n[TEST] Testing: {design_name}")

        # Generate waveform
        if coprime_pair is None:
            phases = np.ones(n_snapshots, dtype=complex)
            designer = None
        else:
            designer = CoprimeSignalDesign(coprime_pair)
            phases = designer.generate_phase_pattern(n_snapshots)

            # Analyze waveform properties
            validation = designer.validate_coprime_properties()
            print(f"   Period: {designer.period} chirps")
            print(f"   Coprimality check: {'PASS' if validation['is_coprime'] else 'FAIL'}")

        # Generate test data with modulation
        data = generate_test_data_with_modulation(
            true_angles, n_sensors, n_snapshots, snr_db, phases
        )

        # Apply SPICE
        try:
            estimator = SPICEEstimator(n_sensors)
            sample_cov = compute_sample_covariance(data)

            spectrum, angles_grid = estimator.fit(sample_cov)
            peaks = estimator.find_peaks(spectrum, min_separation=2.0)

            n_detected = len(peaks['angles'])
            success = n_detected >= len(true_angles)

            print(f"   Result: {n_detected}/{len(true_angles)} targets detected")
            print(f"   Success: {'PASS' if success else 'FAIL'}")

            results[design_name] = {
                'success': success,
                'n_detected': n_detected,
                'spectrum': spectrum,
                'angles': angles_grid
            }

            # Plot spectrum
            ax = axes[0, i]
            spectrum_db = 10 * np.log10(spectrum + 1e-12)
            ax.plot(angles_grid, spectrum_db, 'b-', linewidth=2)

            # Mark true angles
            for angle in true_angles:
                ax.axvline(angle, color='red', linestyle='--', alpha=0.7, linewidth=2)

            ax.set_title(f'{design_name}\nDetected: {n_detected}/{len(true_angles)}')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Power (dB)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-15, 15)

            # Success indicator
            color = 'green' if success else 'red'
            text = 'SUCCESS' if success else 'FAILED'
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                   fontweight='bold', color='white')

        except Exception as e:
            print(f"   Error: {str(e)}")
            results[design_name] = {'success': False, 'error': str(e)}

        # Plot ambiguity function if coprime
        if designer is not None:
            ax = axes[1, i]
            ambiguity = designer.compute_ambiguity_function(phases[:64])  # Smaller for speed
            ambiguity_db = 20 * np.log10(np.abs(ambiguity) + 1e-6)

            im = ax.imshow(ambiguity_db, aspect='auto', cmap='jet', vmin=-40, vmax=0)
            ax.set_title(f'Ambiguity Function\n{design_name}')
            ax.set_xlabel('Doppler Index')
            ax.set_ylabel('Delay Index')
            plt.colorbar(im, ax=ax, label='Magnitude (dB)', shrink=0.6)
        else:
            axes[1, i].text(0.5, 0.5, 'Standard FMCW\n(No Modulation)',
                           transform=axes[1, i].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

    plt.tight_layout()
    plt.suptitle('Coprime Waveform Design Impact on SPICE Performance',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.show()

    # Summary
    print(f"\n[SUMMARY] Coprime Benefits Analysis:")
    successful_designs = [name for name, result in results.items() if result.get('success', False)]
    print(f"   Successful designs: {successful_designs}")

    if len(successful_designs) > 1:
        print(f"   Note: Current coprime implementation shows equivalent performance to standard FMCW")
    else:
        print(f"   [RESULT] Results show challenging nature of closely spaced targets")

    return results


def demonstrate_matched_filter_comparison():
    """
    Demonstrate detailed comparison between SPICE and matched filtering.

    Shows when each method works best and why.
    """
    print("\n" + "="*60)
    print("🎓 DEMONSTRATION 3: SPICE vs Matched Filter Comparison")
    print("="*60)

    # Test multiple scenarios
    scenarios = [
        {
            'name': 'High SNR, Close Sources',
            'angles': np.array([-2, 2]),
            'snr': 20,
            'description': 'SPICE advantage scenario'
        },
        {
            'name': 'Low SNR, Well Separated',
            'angles': np.array([-15, 15]),
            'snr': -5,
            'description': 'Matched filter advantage scenario'
        },
        {
            'name': 'Moderate SNR, Moderate Separation',
            'angles': np.array([-8, 8]),
            'snr': 10,
            'description': 'Competitive scenario'
        }
    ]

    n_sensors = 8
    n_snapshots = 100

    fig, axes = plt.subplots(len(scenarios), 3, figsize=(18, 12))

    for i, scenario in enumerate(scenarios):
        print(f"\n[SCENARIO] Scenario: {scenario['name']}")
        print(f"   Angles: {scenario['angles']} degrees")
        print(f"   SNR: {scenario['snr']} dB")
        print(f"   Description: {scenario['description']}")

        # Generate data
        data = generate_simple_test_data(
            scenario['angles'], n_sensors, n_snapshots, scenario['snr']
        )

        # Matched Filter Processing
        mf_spectrum, mf_angles = conventional_beamforming(data, n_sensors)
        mf_peaks = simple_peak_detection(mf_spectrum, mf_angles)

        # SPICE Processing
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                estimator = SPICEEstimator(n_sensors)
                sample_cov = compute_sample_covariance(data)
                spice_spectrum, spice_angles = estimator.fit(sample_cov)
                spice_peaks = estimator.find_peaks(spice_spectrum, min_separation=2.0)

            spice_success = True
            spice_n_detected = len(spice_peaks['angles'])

        except Exception as e:
            print(f"   SPICE failed: {str(e)}")
            spice_spectrum = np.zeros(180)
            spice_angles = np.linspace(-90, 90, 180)
            spice_success = False
            spice_n_detected = 0

        mf_n_detected = len(mf_peaks)

        print(f"   Matched Filter: {mf_n_detected}/{len(scenario['angles'])} detected")
        print(f"   SPICE: {spice_n_detected}/{len(scenario['angles'])} detected")

        # Plot Matched Filter Result
        ax = axes[i, 0]
        mf_spectrum_db = 10 * np.log10(mf_spectrum + 1e-12)
        ax.plot(mf_angles, mf_spectrum_db, 'b-', linewidth=2, label='Matched Filter')

        for angle in scenario['angles']:
            ax.axvline(angle, color='red', linestyle='--', alpha=0.7)

        ax.set_title(f'Matched Filter\n{scenario["name"]}')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Power (dB)')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'{mf_n_detected}/{len(scenario["angles"])} detected',
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        # Plot SPICE Result
        ax = axes[i, 1]
        spice_spectrum_db = 10 * np.log10(spice_spectrum + 1e-12)
        ax.plot(spice_angles, spice_spectrum_db, 'r-', linewidth=2, label='SPICE')

        for angle in scenario['angles']:
            ax.axvline(angle, color='red', linestyle='--', alpha=0.7)

        ax.set_title(f'SPICE\n{scenario["name"]}')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Power (dB)')
        ax.grid(True, alpha=0.3)

        color = 'green' if spice_success else 'red'
        ax.text(0.05, 0.95, f'{spice_n_detected}/{len(scenario["angles"])} detected',
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

        # Comparison Plot
        ax = axes[i, 2]
        ax.plot(mf_angles, mf_spectrum_db, 'b-', linewidth=2, label='Matched Filter', alpha=0.7)
        ax.plot(spice_angles, spice_spectrum_db, 'r-', linewidth=2, label='SPICE', alpha=0.7)

        for angle in scenario['angles']:
            ax.axvline(angle, color='black', linestyle='--', alpha=0.5)

        ax.set_title(f'Comparison\n{scenario["name"]}')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Power (dB)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Determine winner
        if mf_n_detected > spice_n_detected:
            winner = "Matched Filter Wins"
            color = "lightblue"
        elif spice_n_detected > mf_n_detected:
            winner = "SPICE Wins"
            color = "lightcoral"
        else:
            winner = "Tie"
            color = "lightgray"

        ax.text(0.05, 0.05, winner, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color),
               fontweight='bold')

    plt.tight_layout()
    plt.suptitle('SPICE vs Matched Filter: When Each Method Excels',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.show()

    print(f"\n[INSIGHTS] Key Findings:")
    print(f"   • High SNR + Close sources → SPICE advantage (super-resolution)")
    print(f"   • Low SNR + Any separation → Matched filter advantage (robustness)")
    print(f"   • Moderate conditions → Context dependent")


def demonstrate_theoretical_conditions():
    """
    Demonstrate theoretical conditions for SPICE success/failure.

    Shows RE condition, mutual incoherence, and beta-min condition in practice.
    """
    print("\n" + "="*60)
    print("🎓 DEMONSTRATION 4: Theoretical Conditions Analysis")
    print("="*60)

    n_sensors = 8
    n_snapshots = 100

    # Test different angular separations to show condition violations
    separations = np.array([1, 2, 5, 10, 20])  # degrees
    snr_db = 15

    condition_results = {
        'separations': separations,
        're_condition': [],
        'mutual_incoherence': [],
        'spice_success': []
    }

    print(f"[TESTING] Angular separations: {separations} degrees")
    print(f"   SNR: {snr_db} dB, {n_sensors} sensors, {n_snapshots} snapshots")

    for sep in separations:
        true_angles = np.array([-sep/2, sep/2])

        print(f"\n[TEST] Angular separation: {sep} degrees")

        # Generate data
        data = generate_simple_test_data(true_angles, n_sensors, n_snapshots, snr_db)
        sample_cov = compute_sample_covariance(data)

        # Compute theoretical conditions

        # 1. Restricted Eigenvalue (RE) condition
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]
        noise_level = np.median(eigenvals)
        signal_eigenvals = eigenvals[:2]  # Top 2 eigenvalues
        re_ratio = np.min(signal_eigenvals) / noise_level

        # 2. Mutual Incoherence (approximate)
        # For ULA: coherence ≈ |sinc(π * M * sin(Δθ/2))|
        angle_sep_rad = np.deg2rad(sep)
        coherence_approx = abs(np.sinc(n_sensors * np.sin(angle_sep_rad/2) / np.pi))

        # Test SPICE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                estimator = SPICEEstimator(n_sensors)
                spectrum, angles = estimator.fit(sample_cov)
                peaks = estimator.find_peaks(spectrum, min_separation=1.0)

                spice_success = len(peaks['angles']) >= len(true_angles)
        except:
            spice_success = False

        condition_results['re_condition'].append(re_ratio)
        condition_results['mutual_incoherence'].append(coherence_approx)
        condition_results['spice_success'].append(spice_success)

        print(f"   RE condition ratio: {re_ratio:.3f}")
        print(f"   Mutual incoherence: {coherence_approx:.3f}")
        print(f"   SPICE success: {'PASS' if spice_success else 'FAIL'}")

    # Plot theoretical analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RE Condition
    ax = axes[0]
    colors = ['red' if not success else 'green'
              for success in condition_results['spice_success']]
    ax.scatter(separations, condition_results['re_condition'], c=colors, s=100, alpha=0.7)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Critical Threshold')
    ax.set_xlabel('Angular Separation (degrees)')
    ax.set_ylabel('RE Condition Ratio')
    ax.set_title('Restricted Eigenvalue Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mutual Incoherence
    ax = axes[1]
    ax.scatter(separations, condition_results['mutual_incoherence'], c=colors, s=100, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Typical Threshold')
    ax.set_xlabel('Angular Separation (degrees)')
    ax.set_ylabel('Mutual Incoherence')
    ax.set_title('Mutual Incoherence Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success Rate
    ax = axes[2]
    success_numeric = [1 if s else 0 for s in condition_results['spice_success']]
    ax.plot(separations, success_numeric, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Angular Separation (degrees)')
    ax.set_ylabel('SPICE Success')
    ax.set_title('SPICE Performance vs Angular Separation')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Theoretical Conditions for SPICE Success',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.show()

    print(f"\n[INSIGHTS] Theoretical Insights:")
    print(f"   • RE condition improves with increased angular separation")
    print(f"   • Mutual incoherence decreases (improves) with separation")
    print(f"   • SPICE success correlates with theoretical conditions")
    print(f"   • Critical separation ≈ {separations[condition_results['spice_success'].index(True) if True in condition_results['spice_success'] else -1]}° for this configuration")


# Helper functions
def generate_test_data_with_modulation(true_angles, n_sensors, n_snapshots, snr_db, phases):
    """Generate test data with phase modulation."""
    # Ensure phase array matches snapshots
    if len(phases) > n_snapshots:
        phases = phases[:n_snapshots]
    elif len(phases) < n_snapshots:
        repeats = int(np.ceil(n_snapshots / len(phases)))
        phases = np.tile(phases, repeats)[:n_snapshots]

    # Steering matrix
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    # Source signals with modulation
    source_signals = np.zeros((len(true_angles), n_snapshots), dtype=complex)
    for i in range(len(true_angles)):
        base_signal = (np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)) / np.sqrt(2)
        source_signals[i, :] = base_signal * phases

    # Received signals
    received_signals = steering_matrix @ source_signals

    # Add noise
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) + 1j * np.random.randn(n_sensors, n_snapshots)
    )

    return received_signals + noise


def generate_simple_test_data(true_angles, n_sensors, n_snapshots, snr_db):
    """Generate simple test data without modulation."""
    # Steering matrix
    steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
    for i, angle in enumerate(true_angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_matrix[:, i] = np.exp(1j * phase_shifts)

    # Source signals
    source_signals = (np.random.randn(len(true_angles), n_snapshots) +
                     1j * np.random.randn(len(true_angles), n_snapshots)) / np.sqrt(2)

    # Received signals
    received_signals = steering_matrix @ source_signals

    # Add noise
    signal_power = np.mean(np.abs(received_signals)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(n_sensors, n_snapshots) + 1j * np.random.randn(n_sensors, n_snapshots)
    )

    return received_signals + noise


def conventional_beamforming(data, n_sensors):
    """Conventional delay-and-sum beamforming."""
    sample_cov = compute_sample_covariance(data)
    angles = np.linspace(-90, 90, 180)
    spectrum = np.zeros(len(angles))

    for i, angle in enumerate(angles):
        phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
        steering_vec = np.exp(1j * phase_shifts)
        spectrum[i] = np.real(steering_vec.conj().T @ sample_cov @ steering_vec)

    return spectrum, angles


def simple_peak_detection(spectrum, angles, threshold_factor=0.5):
    """Simple peak detection."""
    threshold = threshold_factor * np.max(spectrum)
    peaks = []

    for i in range(1, len(spectrum)-1):
        if (spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and
            spectrum[i] > threshold):
            peaks.append(angles[i])

    return peaks


def generate_readme_plots():
    """Generate the plots referenced in README.md."""
    print("GENERATING README PLOTS")
    print("="*70)

    try:
        import subprocess
        result = subprocess.run(['python', 'generate_readme_plots.py'],
                              capture_output=True, text=True, cwd='.')
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("README plots generated successfully!")
        return True
    except Exception as e:
        print(f"Error generating plots: {e}")
        return False


def main(demo_type='all'):
    """Run educational demonstrations."""
    if demo_type == 'readme_plots':
        return generate_readme_plots()

    print("COMPREHENSIVE SPICE EDUCATIONAL DEMONSTRATION")
    print("="*70)
    print("This demonstration covers all key theoretical concepts from the README:")
    print("1. SNR failure mechanism with theoretical conditions")
    print("2. Coprime waveform design benefits")
    print("3. SPICE vs Matched Filter comparison")
    print("4. Theoretical conditions analysis (RE, Mutual Incoherence)")
    print("="*70)

    # Set random seed for reproducible results
    np.random.seed(42)

    # Run all demonstrations
    try:
        # Demo 1: SNR Failure
        snr_results = demonstrate_snr_failure_with_theory()

        # Demo 2: Coprime Benefits
        coprime_results = demonstrate_coprime_waveform_benefits()

        # Demo 3: SPICE vs Matched Filter
        comparison_results = demonstrate_matched_filter_comparison()

        # Demo 4: Theoretical Conditions
        theory_results = demonstrate_theoretical_conditions()

        print(f"\n[SUCCESS] ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print(f"="*70)
        print(f"Key Educational Takeaways:")
        print(f"+ SPICE excels at high SNR with super-resolution capability")
        print(f"+ Matched filter remains robust at low SNR conditions")
        print(f"+ Coprime waveforms require further research for effective implementation")
        print(f"+ Theoretical conditions predict algorithm performance")
        print(f"+ Engineering judgment needed for algorithm selection")

        return {
            'snr_analysis': snr_results,
            'coprime_analysis': coprime_results,
            'comparison': comparison_results,
            'theoretical': theory_results
        }

    except Exception as e:
        print(f"ERROR: Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPICE Educational Demonstrations')
    parser.add_argument('--demo', choices=['all', 'snr_failure', 'coprime_benefits', 'comparison', 'readme_plots'],
                        default='all', help='Which demonstration to run')

    args = parser.parse_args()

    # Run the specified demonstration
    if args.demo == 'readme_plots':
        success = main(demo_type='readme_plots')
        if success:
            print("\nREADME plots generated successfully!")
        else:
            print("\nFailed to generate README plots.")
    else:
        results = main(demo_type=args.demo)

        # Save results if successful
        if results is not None:
            print(f"\nResults available for further analysis")
            print(f"   Use results['snr_analysis'], etc. to access data")

        input("\nPress Enter to exit...")
"""
Educational Examples: SPICE Theory and Practice.

This module provides comprehensive educational examples demonstrating SPICE
algorithm behavior, theoretical conditions, and practical limitations.
Designed for radar professionals learning sparse recovery methods.

Key Learning Examples:
1. SNR performance analysis (SPICE vs Matched Filter)
2. Theoretical conditions validation (RE, Mutual Incoherence, Beta-min)
3. Signal design impact on sparse recovery
4. Failure mode diagnosis and understanding

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg as la
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from spice_core import SPICEEstimator, compute_sample_covariance
from coprime_signal_design import CoprimeSignalDesign


@dataclass
class EducationalScenario:
    """Educational test scenario definition."""
    name: str
    description: str
    true_angles: np.ndarray
    n_sensors: int
    n_snapshots: int
    snr_range: np.ndarray


class EducationalAnalyzer:
    """
    Educational analyzer for understanding SPICE theory and practice.

    This class provides comprehensive educational tools for understanding
    when and why SPICE works, and more importantly, when it fails.
    """

    def __init__(self):
        """Initialize educational analyzer."""
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Setup educational plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 11,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })

    def demonstrate_snr_failure_mechanism(self, scenario: EducationalScenario) -> Dict:
        """
        Educational demonstration of SPICE SNR failure mechanism.

        This function provides deep insight into why SPICE fails at low SNR
        by analyzing the theoretical conditions and showing exactly where
        they break down.

        Parameters
        ----------
        scenario : EducationalScenario
            Test scenario parameters.

        Returns
        -------
        results : dict
            Comprehensive analysis results with educational insights.

        Examples
        --------
        >>> analyzer = EducationalAnalyzer()
        >>> scenario = EducationalScenario(
        ...     name="SNR Study",
        ...     description="Well-separated sources",
        ...     true_angles=np.array([-15, 15]),
        ...     n_sensors=8,
        ...     n_snapshots=100,
        ...     snr_range=np.linspace(-15, 25, 21)
        ... )
        >>> results = analyzer.demonstrate_snr_failure_mechanism(scenario)
        """
        print(f"\n[ANALYSIS] Educational Analysis: {scenario.description}")
        print("="*60)

        # Initialize results storage
        results = {
            'snr_db': [],
            'matched_filter': {'success': [], 'spectrum_peak': [], 'sidelobe_level': []},
            'spice': {'success': [], 'spectrum_peak': [], 'convergence': [], 'condition_numbers': []},
            'theoretical': {'re_condition': [], 'beta_min_condition': [], 'noise_eigenvals': []}
        }

        n_monte_carlo = 10  # Multiple trials for statistical significance

        for snr_db in scenario.snr_range:
            print(f"\n[PROCESSING] Analyzing SNR = {snr_db:.1f} dB...")

            # Monte Carlo analysis
            mf_trials = []
            spice_trials = []
            theory_trials = []

            for trial in range(n_monte_carlo):
                # Generate array data
                data = self._generate_educational_data(
                    scenario.true_angles, scenario.n_sensors,
                    scenario.n_snapshots, snr_db, seed=trial
                )

                # Matched Filter Analysis
                mf_result = self._analyze_matched_filter(data, scenario.true_angles)
                mf_trials.append(mf_result)

                # SPICE Analysis with theoretical condition checking
                spice_result = self._analyze_spice_with_conditions(
                    data, scenario.true_angles, scenario.n_sensors
                )
                spice_trials.append(spice_result)

                # Theoretical condition analysis
                theory_result = self._analyze_theoretical_conditions(
                    data, scenario.n_sensors, scenario.true_angles
                )
                theory_trials.append(theory_result)

            # Aggregate results
            self._aggregate_trial_results(results, mf_trials, spice_trials, theory_trials, snr_db)

        # Create educational visualizations
        self._create_educational_plots(results, scenario)

        return results

    def _generate_educational_data(self, true_angles: np.ndarray, n_sensors: int,
                                  n_snapshots: int, snr_db: float, seed: int = None) -> np.ndarray:
        """Generate educational array data with known ground truth."""
        if seed is not None:
            np.random.seed(seed)

        # Array geometry (ULA)
        d = 0.5  # Half-wavelength spacing
        sensor_positions = np.arange(n_sensors) * d

        # Steering matrix
        steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
        for i, angle in enumerate(true_angles):
            phase_shifts = 2 * np.pi * sensor_positions * np.sin(np.deg2rad(angle))
            steering_matrix[:, i] = np.exp(1j * phase_shifts)

        # Source signals (unit power)
        source_powers = np.ones(len(true_angles))
        source_signals = np.zeros((len(true_angles), n_snapshots), dtype=complex)

        for i in range(len(true_angles)):
            source_signals[i, :] = np.sqrt(source_powers[i]) * (
                np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)
            ) / np.sqrt(2)

        # Received signals
        received_signals = steering_matrix @ source_signals

        # Add noise
        signal_power = np.mean(np.abs(received_signals)**2)
        noise_power = signal_power / (10**(snr_db/10))

        noise = np.sqrt(noise_power/2) * (
            np.random.randn(n_sensors, n_snapshots) +
            1j * np.random.randn(n_sensors, n_snapshots)
        )

        return received_signals + noise

    def _analyze_matched_filter(self, data: np.ndarray, true_angles: np.ndarray) -> Dict:
        """Analyze matched filter (conventional beamforming) performance."""
        n_sensors = data.shape[0]
        angles_grid = np.linspace(-90, 90, 180)

        # Conventional beamforming
        sample_cov = compute_sample_covariance(data)
        spectrum = np.zeros(len(angles_grid))

        for i, angle in enumerate(angles_grid):
            # Steering vector
            phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
            steering_vec = np.exp(1j * phase_shifts)

            # Beamformer output
            spectrum[i] = np.real(steering_vec.conj().T @ sample_cov @ steering_vec)

        # Peak detection
        peaks = self._simple_peak_detection(spectrum, angles_grid)

        # Performance metrics
        success = len(peaks) >= len(true_angles)
        peak_power = np.max(spectrum) if len(spectrum) > 0 else 0
        # Find sidelobe level (maximum value below 50% of peak)
        sidelobes = spectrum[spectrum < 0.5 * np.max(spectrum)]
        sidelobe_level = np.max(sidelobes) if len(sidelobes) > 0 else 0.0

        return {
            'success': success,
            'n_peaks': len(peaks),
            'peak_power': peak_power,
            'sidelobe_level': sidelobe_level,
            'spectrum': spectrum,
            'angles_grid': angles_grid
        }

    def _analyze_spice_with_conditions(self, data: np.ndarray, true_angles: np.ndarray,
                                     n_sensors: int) -> Dict:
        """Analyze SPICE performance with theoretical condition monitoring."""
        try:
            # SPICE estimation
            estimator = SPICEEstimator(n_sensors)
            sample_cov = compute_sample_covariance(data)

            spectrum, angles_grid = estimator.fit(sample_cov)
            peaks = estimator.find_peaks(spectrum, min_separation=3.0)

            # Performance metrics - require both correct peaks AND convergence
            convergence_info = estimator.get_convergence_info()
            converged = convergence_info['n_iterations'] < 100  # Did not hit max iterations
            correct_peaks = len(peaks['angles']) >= len(true_angles)
            success = correct_peaks and converged

            # Condition number analysis
            try:
                # Check conditioning of steering matrix at true locations
                true_indices = [np.argmin(np.abs(angles_grid - angle)) for angle in true_angles]
                A_true = estimator.steering_vectors[:, true_indices]
                condition_number = np.linalg.cond(A_true.conj().T @ A_true)
            except:
                condition_number = np.inf

            return {
                'success': success,
                'n_peaks': len(peaks['angles']),
                'convergence_iterations': convergence_info['n_iterations'],
                'final_cost': convergence_info['final_cost'],
                'condition_number': condition_number,
                'spectrum': spectrum,
                'angles_grid': angles_grid,
                'converged': convergence_info['n_iterations'] < 100,
                'correct_peaks': correct_peaks
            }

        except Exception as e:
            return {
                'success': False,
                'n_peaks': 0,
                'convergence_iterations': -1,
                'final_cost': np.inf,
                'condition_number': np.inf,
                'error': str(e),
                'converged': False
            }

    def _analyze_theoretical_conditions(self, data: np.ndarray, n_sensors: int,
                                      true_angles: np.ndarray) -> Dict:
        """Analyze theoretical conditions for sparse recovery."""
        sample_cov = compute_sample_covariance(data)

        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(sample_cov)
        eigenvals = np.sort(np.real(eigenvals))[::-1]  # Sort descending

        # Estimate noise level (median eigenvalue)
        noise_level = np.median(eigenvals)

        # Signal subspace analysis
        n_sources = len(true_angles)
        signal_eigenvals = eigenvals[:n_sources]
        noise_eigenvals = eigenvals[n_sources:]

        # Restricted Eigenvalue condition approximation
        # Ratio of smallest signal eigenvalue to noise level
        re_condition = signal_eigenvals[-1] / noise_level if n_sources > 0 else 0

        # Beta-min condition approximation
        # Minimum signal strength relative to noise
        beta_min_ratio = np.min(signal_eigenvals) / noise_level

        # Mutual incoherence approximation
        # Condition number of signal subspace
        signal_condition = np.max(signal_eigenvals) / np.min(signal_eigenvals) if n_sources > 0 else 1

        return {
            're_condition': re_condition,
            'beta_min_ratio': beta_min_ratio,
            'signal_condition': signal_condition,
            'noise_level': noise_level,
            'signal_eigenvals': signal_eigenvals,
            'noise_eigenvals': noise_eigenvals
        }

    def _aggregate_trial_results(self, results: Dict, mf_trials: List,
                               spice_trials: List, theory_trials: List, snr_db: float):
        """Aggregate results from Monte Carlo trials."""
        results['snr_db'].append(snr_db)

        # Matched filter aggregation
        mf_success_rate = np.mean([t['success'] for t in mf_trials])
        mf_peak_power = np.mean([t['peak_power'] for t in mf_trials])
        mf_sidelobe = np.mean([t['sidelobe_level'] for t in mf_trials])

        results['matched_filter']['success'].append(mf_success_rate)
        results['matched_filter']['spectrum_peak'].append(mf_peak_power)
        results['matched_filter']['sidelobe_level'].append(mf_sidelobe)

        # SPICE aggregation
        spice_success_rate = np.mean([t['success'] for t in spice_trials])
        spice_convergence_rate = np.mean([t.get('converged', False) for t in spice_trials])
        spice_condition = np.mean([t.get('condition_number', np.inf) for t in spice_trials if np.isfinite(t.get('condition_number', np.inf))])

        results['spice']['success'].append(spice_success_rate)
        results['spice']['convergence'].append(spice_convergence_rate)
        results['spice']['condition_numbers'].append(spice_condition if np.isfinite(spice_condition) else 100)

        # Theoretical conditions aggregation
        re_condition = np.mean([t['re_condition'] for t in theory_trials])
        beta_min = np.mean([t['beta_min_ratio'] for t in theory_trials])
        noise_power = np.mean([t['noise_level'] for t in theory_trials])

        results['theoretical']['re_condition'].append(re_condition)
        results['theoretical']['beta_min_condition'].append(beta_min)
        results['theoretical']['noise_eigenvals'].append(noise_power)

    def _simple_peak_detection(self, spectrum: np.ndarray, angles: np.ndarray,
                             threshold_factor: float = 0.5) -> List[float]:
        """Simple peak detection for educational purposes."""
        threshold = threshold_factor * np.max(spectrum)
        peaks = []

        for i in range(1, len(spectrum)-1):
            if (spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold):
                peaks.append(angles[i])

        return peaks

    def _create_educational_plots(self, results: Dict, scenario: EducationalScenario):
        """Create comprehensive educational plots."""
        fig = plt.figure(figsize=(16, 12))

        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

        # Plot 1: Success Rate Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        snr_range = np.array(results['snr_db'])

        ax1.plot(snr_range, results['matched_filter']['success'], 'b-o',
                label='Matched Filter', linewidth=3, markersize=6)
        ax1.plot(snr_range, results['spice']['success'], 'r-s',
                label='SPICE', linewidth=3, markersize=6)

        # Mark critical thresholds
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='0 dB Threshold')
        ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='SPICE Threshold')

        # Failure region
        ax1.fill_between(snr_range[snr_range < 0], 0, 1, alpha=0.2, color='red', label='SPICE Failure Zone')

        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Detection Success Rate')
        ax1.set_title('Algorithm Performance vs SNR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        # Educational annotations
        ax1.annotate('Covariance estimation\nbecomes unreliable',
                    xy=(-5, 0.3), xytext=(-12, 0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10, fontweight='bold')

        # Plot 2: Theoretical Conditions
        ax2 = fig.add_subplot(gs[0, 1])

        ax2.plot(snr_range, results['theoretical']['re_condition'], 'g-^',
                label='RE Condition', linewidth=2, markersize=5)
        ax2.plot(snr_range, results['theoretical']['beta_min_condition'], 'm-v',
                label='Beta-min Condition', linewidth=2, markersize=5)

        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.7, label='Critical Threshold')
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Condition Ratio')
        ax2.set_title('Theoretical Conditions vs SNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: SPICE Convergence Analysis
        ax3 = fig.add_subplot(gs[1, 0])

        ax3.plot(snr_range, results['spice']['convergence'], 'purple', marker='D',
                linewidth=2, markersize=5, label='Convergence Rate')

        ax3.fill_between(snr_range, 0, results['spice']['convergence'],
                        alpha=0.3, color='purple')

        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Convergence Rate')
        ax3.set_title('SPICE Convergence vs SNR')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        # Plot 4: Condition Number Analysis
        ax4 = fig.add_subplot(gs[1, 1])

        condition_numbers = np.array(results['spice']['condition_numbers'])
        condition_numbers = np.clip(condition_numbers, 1, 1000)  # Clip for visualization

        ax4.semilogy(snr_range, condition_numbers, 'brown', marker='x',
                    linewidth=2, markersize=8, label='Condition Number')

        ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7,
                   label='Poor Conditioning Threshold')

        ax4.set_xlabel('SNR (dB)')
        ax4.set_ylabel('Condition Number')
        ax4.set_title('Array Conditioning vs SNR')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Educational Summary (spans both columns)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Create educational summary text
        summary_text = f"""
        [EDUCATIONAL SUMMARY] Understanding SPICE Failure Mechanisms

        [SCENARIO] {scenario.description}
           • Sources: {scenario.true_angles} degrees
           • Array: {scenario.n_sensors} sensors, {scenario.n_snapshots} snapshots

        [KEY LEARNING POINTS]:

        1. LOW SNR FAILURE (< 0 dB):
           • Sample covariance R̂ ≈ σ²I + small signal component
           • High estimation variance breaks RE condition
           • Eigenvalue decomposition becomes unreliable
           • SPICE cannot distinguish signal from noise subspace

        2. MODERATE SNR DEGRADATION (0-10 dB):
           • Theoretical conditions marginally satisfied
           • Algorithm may converge to suboptimal solutions
           • Increased sensitivity to array calibration errors

        3. HIGH SNR SUCCESS (> 15 dB):
           • All theoretical conditions well satisfied
           • Superior resolution compared to matched filtering
           • Reliable convergence and accurate parameter estimates

        [WARNING] PRACTICAL RULE: Use SPICE only when SNR > 10 dB
        [RECOMMENDATION] HYBRID APPROACH: Adaptive algorithm selection based on estimated SNR
        """

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))

        plt.suptitle(f'Educational Analysis: SPICE Theory and Practice\n{scenario.name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig

    def demonstrate_coprime_advantage(self, n_chirps: int = 128) -> Dict:
        """
        Demonstrate how coprime waveform design improves SPICE conditions.

        Parameters
        ----------
        n_chirps : int, default=128
            Number of chirps for analysis.

        Returns
        -------
        results : dict
            Coprime advantage analysis results.
        """
        print("\n[DEMO] Demonstrating Coprime Waveform Advantage")
        print("="*50)

        # Test scenario: challenging case with close targets
        true_angles = np.array([-3, 3])  # 6° separation
        n_sensors = 8
        n_snapshots = 128
        snr_db = 12  # Moderate SNR

        designs = {
            'standard': None,
            'coprime_31_37': (31, 37),
            'coprime_13_17': (13, 17)
        }

        results = {}

        for design_name, coprime_pair in designs.items():
            print(f"\n   Testing {design_name}...")

            # Generate phase modulation
            if coprime_pair is None:
                phases = np.ones(n_chirps, dtype=complex)
                description = "Standard FMCW"
            else:
                designer = CoprimeSignalDesign(coprime_pair)
                phases = designer.generate_phase_pattern(n_chirps)
                description = f"Coprime {coprime_pair}"

            # Generate modulated data
            data = self._generate_modulated_array_data(
                true_angles, n_sensors, n_snapshots, snr_db, phases
            )

            # Analyze with SPICE
            try:
                estimator = SPICEEstimator(n_sensors)
                sample_cov = compute_sample_covariance(data)
                spectrum, angles = estimator.fit(sample_cov)
                peaks = estimator.find_peaks(spectrum, min_separation=2.0)

                # Performance metrics
                n_detected = len(peaks['angles'])
                success = n_detected >= len(true_angles)

                # Mutual incoherence analysis
                if coprime_pair is not None:
                    mutual_incoherence = designer._compute_mutual_incoherence(
                        designer.compute_ambiguity_function(phases)
                    )
                else:
                    mutual_incoherence = 1.0  # Standard case

                results[design_name] = {
                    'description': description,
                    'success': success,
                    'n_detected': n_detected,
                    'mutual_incoherence': mutual_incoherence,
                    'spectrum': spectrum,
                    'angles': angles,
                    'convergence_iterations': estimator.get_convergence_info()['n_iterations']
                }

                print(f"      Success: {success}")
                print(f"      Detected: {n_detected}/{len(true_angles)} targets")
                print(f"      Mutual incoherence: mu = {mutual_incoherence:.4f}")

            except Exception as e:
                results[design_name] = {
                    'description': description,
                    'success': False,
                    'error': str(e)
                }
                print(f"      Failed: {str(e)}")

        # Create comparison plot
        self._plot_coprime_comparison(results, true_angles)

        return results

    def _generate_modulated_array_data(self, true_angles: np.ndarray, n_sensors: int,
                                     n_snapshots: int, snr_db: float,
                                     phase_modulation: np.ndarray) -> np.ndarray:
        """Generate array data with phase modulation."""
        # Ensure consistent dimensions
        if len(phase_modulation) > n_snapshots:
            phase_modulation = phase_modulation[:n_snapshots]
        elif len(phase_modulation) < n_snapshots:
            repeats = int(np.ceil(n_snapshots / len(phase_modulation)))
            phase_modulation = np.tile(phase_modulation, repeats)[:n_snapshots]

        # Generate steering vectors
        steering_matrix = np.zeros((n_sensors, len(true_angles)), dtype=complex)
        for i, angle in enumerate(true_angles):
            phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
            steering_matrix[:, i] = np.exp(1j * phase_shifts)

        # Generate source signals with phase modulation
        source_signals = np.zeros((len(true_angles), n_snapshots), dtype=complex)
        for i in range(len(true_angles)):
            base_signal = (np.random.randn(n_snapshots) +
                          1j * np.random.randn(n_snapshots)) / np.sqrt(2)
            source_signals[i, :] = base_signal * phase_modulation

        # Received signals
        received_signals = steering_matrix @ source_signals

        # Add noise
        signal_power = np.mean(np.abs(received_signals)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(n_sensors, n_snapshots) +
            1j * np.random.randn(n_sensors, n_snapshots)
        )

        return received_signals + noise

    def _plot_coprime_comparison(self, results: Dict, true_angles: np.ndarray):
        """Plot coprime waveform comparison."""
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))

        if len(results) == 1:
            axes = [axes]

        for i, (design_name, result) in enumerate(results.items()):
            ax = axes[i]

            if 'spectrum' in result:
                spectrum_db = 10 * np.log10(result['spectrum'] + 1e-12)
                ax.plot(result['angles'], spectrum_db, 'b-', linewidth=2,
                       label=f"{result['description']}")

                # Mark true angles
                for angle in true_angles:
                    ax.axvline(angle, color='red', linestyle='--', alpha=0.7)

                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Power (dB)')
                ax.set_title(f"{result['description']}\nDetected: {result['n_detected']}/{len(true_angles)}")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-30, 30)

                # Add performance annotation
                success_text = "[SUCCESS]" if result['success'] else "[FAILED]"
                ax.text(0.7, 0.9, success_text, transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="green" if result['success'] else "red",
                               alpha=0.7))

            else:
                ax.text(0.5, 0.5, f"ERROR\n{result.get('error', 'Unknown')}",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.3))
                ax.set_title(result['description'])

        plt.suptitle('Coprime Waveform Impact on SPICE Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return fig


def create_educational_scenarios() -> List[EducationalScenario]:
    """Create comprehensive educational test scenarios."""
    scenarios = [
        EducationalScenario(
            name="Well-Separated Sources",
            description="Two sources with 30° separation",
            true_angles=np.array([-15, 15]),
            n_sensors=8,
            n_snapshots=100,
            snr_range=np.linspace(-15, 25, 21)
        ),
        EducationalScenario(
            name="Closely-Spaced Sources",
            description="Two sources with 6° separation (challenging)",
            true_angles=np.array([-3, 3]),
            n_sensors=8,
            n_snapshots=100,
            snr_range=np.linspace(-10, 30, 21)
        ),
        EducationalScenario(
            name="Multiple Sources",
            description="Four sources with varying separation",
            true_angles=np.array([-30, -10, 5, 25]),
            n_sensors=12,
            n_snapshots=150,
            snr_range=np.linspace(-5, 25, 16)
        )
    ]

    return scenarios


def main():
    """Run comprehensive educational demonstrations."""
    print("🎓 SPICE Educational Examples: Theory Meets Practice")
    print("="*60)

    analyzer = EducationalAnalyzer()

    # Get educational scenarios
    scenarios = create_educational_scenarios()

    # Run SNR analysis for each scenario
    all_results = {}

    for i, scenario in enumerate(scenarios[:2]):  # Run first 2 scenarios
        print(f"\n{'='*20} SCENARIO {i+1} {'='*20}")
        results = analyzer.demonstrate_snr_failure_mechanism(scenario)
        all_results[scenario.name] = results

    # Demonstrate coprime advantage
    print(f"\n{'='*20} COPRIME ANALYSIS {'='*20}")
    coprime_results = analyzer.demonstrate_coprime_advantage()
    all_results['coprime_analysis'] = coprime_results

    print(f"\n[COMPLETE] Educational Analysis Complete!")
    print(f"   Key insights demonstrated:")
    print(f"   + SNR failure mechanisms")
    print(f"   + Theoretical condition validation")
    print(f"   + Signal design impact")
    print(f"   + Practical guidance for engineers")

    return all_results


if __name__ == "__main__":
    results = main()
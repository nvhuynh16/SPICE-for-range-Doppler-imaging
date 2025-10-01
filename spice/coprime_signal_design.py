"""
Coprime Signal Design for Enhanced SPICE Performance.

This module implements coprime phase modulation based on the Chinese Remainder
Theorem to improve mutual incoherence conditions for sparse recovery in radar
applications.

References
----------
.. [1] N. Levanon & E. Mozeson, "Radar Signals," Wiley-IEEE Press, 2004.
.. [2] M. Soltanalian, "Signal Design for Active Sensing," Springer, 2014.
.. [3] S. Sen & A. Nehorai, "OFDM MIMO Radar with Mutual-Information Waveform
       Design for Low-Grazing Angle Tracking," IEEE TSP, 2010.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings


class CoprimeSignalDesign:
    """
    Coprime signal design for improved SPICE performance.

    This class implements coprime phase modulation patterns based on the
    Chinese Remainder Theorem to reduce cross-correlations and improve
    mutual incoherence conditions for sparse recovery.

    Parameters
    ----------
    coprime_pair : tuple of int, default=(31, 37)
        Coprime integers for phase modulation. Must satisfy gcd(p1, p2) = 1.

    Attributes
    ----------
    p1, p2 : int
        Coprime moduli.
    period : int
        Full period = p1 * p2 (Chinese Remainder Theorem).

    Examples
    --------
    >>> designer = CoprimeSignalDesign(coprime_pair=(31, 37))
    >>> phases = designer.generate_phase_pattern(128)
    >>> ambiguity = designer.compute_ambiguity_function(phases)
    >>> improvement = designer.analyze_performance_improvement()
    """

    def __init__(self, coprime_pair: Tuple[int, int] = (31, 37)):
        """Initialize coprime signal designer."""
        self.p1, self.p2 = coprime_pair

        if np.gcd(self.p1, self.p2) != 1:
            raise ValueError(f"Numbers {self.p1} and {self.p2} are not coprime")

        self.period = self.p1 * self.p2

        print(f"Coprime design initialized: ({self.p1}, {self.p2})")
        print(f"Full period: {self.period} chirps")

    def generate_phase_pattern(self, n_chirps: int) -> np.ndarray:
        """
        Generate coprime phase modulation pattern.

        Parameters
        ----------
        n_chirps : int
            Number of chirps to generate.

        Returns
        -------
        phases : ndarray, shape (n_chirps,)
            Complex phase pattern with coprime modulation.

        Notes
        -----
        The phase pattern is generated using:

        φ(k) = 2π[(k mod p₁)/p₁ + (k mod p₂)/p₂]

        This creates a bijective mapping for k ∈ [0, p₁p₂-1] due to the
        Chinese Remainder Theorem, ensuring maximum period and balanced
        cross-correlation properties.
        """
        phases = np.zeros(n_chirps, dtype=complex)

        for k in range(n_chirps):
            # CRT-based phase assignment
            phase_component_1 = 2 * np.pi * (k % self.p1) / self.p1
            phase_component_2 = 2 * np.pi * (k % self.p2) / self.p2

            # Combined phase (modulo 2π)
            total_phase = (phase_component_1 + phase_component_2) % (2 * np.pi)
            phases[k] = np.exp(1j * total_phase)

        return phases

    def compute_ambiguity_function(self, phases: np.ndarray) -> np.ndarray:
        """
        Compute waveform ambiguity function.

        Parameters
        ----------
        phases : array_like
            Complex phase pattern.

        Returns
        -------
        ambiguity : ndarray, shape (n_delay, n_doppler)
            2D ambiguity function |χ(τ, f_d)|.

        Notes
        -----
        The ambiguity function is defined as:

        χ(τ, f_d) = Σₖ s*(k)s(k+τ)exp(j2πf_d k/N)

        where s(k) are the complex phase samples.
        """
        n = len(phases)
        ambiguity = np.zeros((n, n), dtype=complex)

        for tau in range(n):
            for fd_idx in range(n):
                # Normalized Doppler frequency
                fd_norm = fd_idx / n

                correlation = 0
                for k in range(n - tau):
                    correlation += (phases[k].conj() * phases[k + tau] *
                                  np.exp(1j * 2 * np.pi * fd_norm * k))

                ambiguity[tau, fd_idx] = correlation / n

        return ambiguity

    def compute_range_doppler_xcorr(self, phases: np.ndarray) -> np.ndarray:
        """
        Compute range-Doppler cross-correlation matrix.

        Parameters
        ----------
        phases : array_like
            Complex phase pattern.

        Returns
        -------
        xcorr_matrix : ndarray
            Cross-correlation matrix between all range-Doppler cells.
        """
        n = len(phases)

        # Create range-Doppler grid
        range_profiles = []
        doppler_profiles = []

        # Generate profiles for different range-Doppler cells
        for r_idx in range(n//4):  # Sample range cells
            for d_idx in range(n//4):  # Sample Doppler cells
                # Range delay
                range_delayed = np.roll(phases, r_idx)

                # Doppler shift
                doppler_shifted = range_delayed * np.exp(1j * 2 * np.pi * d_idx * np.arange(n) / n)

                range_profiles.append(range_delayed)
                doppler_profiles.append(doppler_shifted)

        # Compute cross-correlation matrix
        n_profiles = len(range_profiles)
        xcorr_matrix = np.zeros((n_profiles, n_profiles), dtype=complex)

        for i in range(n_profiles):
            for j in range(n_profiles):
                xcorr = np.abs(np.vdot(range_profiles[i], range_profiles[j])) / n
                xcorr_matrix[i, j] = xcorr

        return xcorr_matrix

    def _compute_mutual_incoherence(self, ambiguity: np.ndarray) -> float:
        """
        Compute mutual incoherence parameter from ambiguity function.

        Parameters
        ----------
        ambiguity : ndarray
            2D ambiguity function.

        Returns
        -------
        mu : float
            Mutual incoherence parameter (0 <= mu <= 1).
        """
        # Get off-peak values (exclude main peak at origin)
        ambiguity_magnitude = np.abs(ambiguity)

        # Exclude main peak at (0,0)
        ambiguity_magnitude[0, 0] = 0

        # Maximum off-peak correlation
        max_sidelobe = np.max(ambiguity_magnitude)

        # Mutual incoherence is max off-peak correlation
        return max_sidelobe

    def analyze_performance_improvement(self, n_chirps: int = 128) -> Dict:
        """
        Quantify SPICE performance improvement with coprime design.

        Parameters
        ----------
        n_chirps : int, default=128
            Number of chirps to analyze.

        Returns
        -------
        analysis : dict
            Performance improvement metrics.

        Examples
        --------
        >>> designer = CoprimeSignalDesign()
        >>> results = designer.analyze_performance_improvement()
        >>> print(f"Coherence reduction: {results['coherence_reduction']:.2f}x")
        """
        # Standard FMCW (no phase modulation)
        standard_phases = np.ones(n_chirps, dtype=complex)

        # Coprime FMCW
        coprime_phases = self.generate_phase_pattern(n_chirps)

        # Compute ambiguity functions
        amb_standard = self.compute_ambiguity_function(standard_phases)
        amb_coprime = self.compute_ambiguity_function(coprime_phases)

        # Analyze mutual incoherence
        mu_standard = self._compute_mutual_incoherence(amb_standard)
        mu_coprime = self._compute_mutual_incoherence(amb_coprime)

        # Compute range-Doppler cross-correlations
        xcorr_standard = self.compute_range_doppler_xcorr(standard_phases)
        xcorr_coprime = self.compute_range_doppler_xcorr(coprime_phases)

        # Analysis metrics
        coherence_reduction = mu_standard / mu_coprime if mu_coprime > 0 else float('inf')

        # Find maximum off-diagonal cross-correlations (exclude near-unity values)
        off_diag_std = xcorr_standard[np.abs(xcorr_standard) < 0.99]
        off_diag_cop = xcorr_coprime[np.abs(xcorr_coprime) < 0.99]

        xcorr_std_max = np.max(np.abs(off_diag_std)) if len(off_diag_std) > 0 else 0.0
        xcorr_cop_max = np.max(np.abs(off_diag_cop)) if len(off_diag_cop) > 0 else 0.0

        xcorr_reduction = xcorr_std_max / xcorr_cop_max if xcorr_cop_max > 0 else float('inf')

        results = {
            'standard_coherence': mu_standard,
            'coprime_coherence': mu_coprime,
            'coherence_reduction': coherence_reduction,
            'standard_max_xcorr': xcorr_std_max,
            'coprime_max_xcorr': xcorr_cop_max,
            'xcorr_reduction': xcorr_reduction,
            'period_utilization': n_chirps / self.period,
            'theoretical_improvement': self.period / max(self.p1, self.p2)
        }

        print("\n[ANALYSIS] Coprime Performance Analysis:")
        print(f"   Standard coherence: mu = {mu_standard:.4f}")
        print(f"   Coprime coherence:  mu = {mu_coprime:.4f}")
        print(f"   Improvement factor: {coherence_reduction:.2f}x")
        print(f"   Cross-corr reduction: {xcorr_reduction:.2f}x")
        print(f"   Period utilization: {results['period_utilization']:.1%}")

        return results

    def plot_ambiguity_comparison(self, n_chirps: int = 128,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ambiguity function comparison.

        Parameters
        ----------
        n_chirps : int, default=128
            Number of chirps.
        save_path : str, optional
            Path to save figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Comparison figure.
        """
        # Generate both waveforms
        standard_phases = np.ones(n_chirps, dtype=complex)
        coprime_phases = self.generate_phase_pattern(n_chirps)

        # Compute ambiguity functions
        amb_standard = self.compute_ambiguity_function(standard_phases)
        amb_coprime = self.compute_ambiguity_function(coprime_phases)

        # Convert to dB
        amb_standard_db = 20 * np.log10(np.abs(amb_standard) + 1e-6)
        amb_coprime_db = 20 * np.log10(np.abs(amb_coprime) + 1e-6)

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Standard FMCW ambiguity
        im1 = axes[0, 0].imshow(amb_standard_db, aspect='auto', cmap='jet',
                               vmin=-60, vmax=0, extent=[-0.5, 0.5, -0.5, 0.5])
        axes[0, 0].set_title('Standard FMCW Ambiguity Function', fontweight='bold')
        axes[0, 0].set_xlabel('Normalized Doppler')
        axes[0, 0].set_ylabel('Normalized Delay')
        plt.colorbar(im1, ax=axes[0, 0], label='Magnitude (dB)')

        # Coprime FMCW ambiguity
        im2 = axes[0, 1].imshow(amb_coprime_db, aspect='auto', cmap='jet',
                               vmin=-60, vmax=0, extent=[-0.5, 0.5, -0.5, 0.5])
        axes[0, 1].set_title(f'Coprime FMCW Ambiguity ({self.p1},{self.p2})', fontweight='bold')
        axes[0, 1].set_xlabel('Normalized Doppler')
        axes[0, 1].set_ylabel('Normalized Delay')
        plt.colorbar(im2, ax=axes[0, 1], label='Magnitude (dB)')

        # Zero-Doppler cuts
        zero_doppler_std = amb_standard_db[:, n_chirps//2]
        zero_doppler_cop = amb_coprime_db[:, n_chirps//2]

        delay_axis = np.arange(n_chirps) - n_chirps//2

        axes[1, 0].plot(delay_axis, zero_doppler_std, 'b-', linewidth=2, label='Standard FMCW')
        axes[1, 0].plot(delay_axis, zero_doppler_cop, 'r-', linewidth=2, label=f'Coprime ({self.p1},{self.p2})')
        axes[1, 0].set_xlabel('Delay (samples)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].set_title('Zero-Doppler Cut Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(-60, 5)

        # Zero-delay cuts
        zero_delay_std = amb_standard_db[0, :]
        zero_delay_cop = amb_coprime_db[0, :]

        doppler_axis = np.arange(n_chirps) - n_chirps//2

        axes[1, 1].plot(doppler_axis, zero_delay_std, 'b-', linewidth=2, label='Standard FMCW')
        axes[1, 1].plot(doppler_axis, zero_delay_cop, 'r-', linewidth=2, label=f'Coprime ({self.p1},{self.p2})')
        axes[1, 1].set_xlabel('Doppler (samples)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].set_title('Zero-Delay Cut Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-60, 5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Ambiguity comparison saved: {save_path}")

        return fig

    def validate_coprime_properties(self) -> Dict:
        """
        Validate mathematical properties of coprime design.

        Returns
        -------
        validation : dict
            Validation results and theoretical checks.
        """
        validation = {}

        # Check coprimality
        gcd_value = np.gcd(self.p1, self.p2)
        validation['is_coprime'] = gcd_value == 1
        validation['gcd'] = gcd_value

        # Check period property
        expected_period = self.p1 * self.p2
        validation['expected_period'] = expected_period
        validation['actual_period'] = self.period
        validation['period_correct'] = expected_period == self.period

        # Generate phase pattern and check uniqueness over one period
        if self.period <= 10000:  # Avoid memory issues for very large periods
            phases = self.generate_phase_pattern(self.period)

            # Check that all phase values are unique (up to numerical precision)
            unique_phases = np.unique(np.round(np.angle(phases), decimals=6))
            validation['unique_phases'] = len(unique_phases)
            validation['phase_uniqueness'] = len(unique_phases) == self.period
        else:
            validation['unique_phases'] = 'Not computed (period too large)'
            validation['phase_uniqueness'] = 'Not computed (period too large)'

        # Theoretical mutual incoherence bound
        # For coprime design, expect mu ~ 1/sqrt(p1*p2)
        theoretical_mu = 1.0 / np.sqrt(self.period)
        validation['theoretical_coherence'] = theoretical_mu

        print("\n[VALIDATION] Coprime Validation Results:")
        print(f"   Coprime check: {validation['is_coprime']} (gcd={gcd_value})")
        print(f"   Period: {self.period} (expected: {expected_period})")
        print(f"   Theoretical mu bound: {theoretical_mu:.4f}")

        return validation


def compare_signal_designs(n_chirps: int = 128, snr_db: float = 15) -> Dict:
    """
    Compare different signal designs for SPICE performance.

    Parameters
    ----------
    n_chirps : int, default=128
        Number of chirps to generate.
    snr_db : float, default=15
        Signal-to-noise ratio for simulation.

    Returns
    -------
    comparison : dict
        Performance comparison results.
    """
    from spice_core import SPICEEstimator
    from range_doppler_imaging import compute_sample_covariance

    print(f"\n[COMPARISON] Comparing Signal Designs (SNR={snr_db} dB)")

    # Signal design configurations
    designs = {
        'standard': {'description': 'Standard FMCW (no modulation)'},
        'coprime_31_37': {'description': 'Coprime (31,37)', 'coprime_pair': (31, 37)},
        'coprime_13_17': {'description': 'Coprime (13,17)', 'coprime_pair': (13, 17)},
        'coprime_7_11': {'description': 'Coprime (7,11)', 'coprime_pair': (7, 11)}
    }

    # Test scenario: closely spaced sources
    n_sensors = 8
    n_snapshots = 100
    true_angles = np.array([-5, 5])  # 10° separation

    results = {}

    for design_name, config in designs.items():
        print(f"\n   Testing {config['description']}...")

        try:
            # Generate signal
            if design_name == 'standard':
                phases = np.ones(n_chirps, dtype=complex)
            else:
                designer = CoprimeSignalDesign(config['coprime_pair'])
                phases = designer.generate_phase_pattern(n_chirps)

            # Generate synthetic array data with phase modulation
            data = generate_modulated_array_data(
                true_angles, n_sensors, n_snapshots, snr_db, phases
            )

            # Apply SPICE
            estimator = SPICEEstimator(n_sensors)
            sample_cov = compute_sample_covariance(data)

            spectrum, angles = estimator.fit(sample_cov)
            peaks = estimator.find_peaks(spectrum, min_separation=2.0)

            # Evaluate performance
            n_detected = len(peaks['angles'])
            detection_rate = min(n_detected / len(true_angles), 1.0)

            # Compute resolution metric
            if n_detected >= 2:
                detected_separation = np.max(peaks['angles']) - np.min(peaks['angles'])
                resolution_error = abs(detected_separation - 10.0) / 10.0
            else:
                resolution_error = 1.0  # Failed to resolve

            results[design_name] = {
                'description': config['description'],
                'n_detected': n_detected,
                'detection_rate': detection_rate,
                'resolution_error': resolution_error,
                'convergence_iterations': estimator.get_convergence_info()['n_iterations'],
                'spectrum': spectrum,
                'angles_grid': angles
            }

            print(f"      Detected: {n_detected}/{len(true_angles)} targets")
            print(f"      Detection rate: {detection_rate:.2f}")
            print(f"      Resolution error: {resolution_error:.3f}")

        except Exception as e:
            print(f"      ERROR: Failed: {str(e)}")
            results[design_name] = {
                'description': config['description'],
                'n_detected': 0,
                'detection_rate': 0.0,
                'resolution_error': 1.0,
                'error': str(e)
            }

    return results


def generate_modulated_array_data(true_angles: np.ndarray, n_sensors: int,
                                 n_snapshots: int, snr_db: float,
                                 phase_modulation: np.ndarray) -> np.ndarray:
    """
    Generate synthetic array data with phase modulation.

    Parameters
    ----------
    true_angles : array_like
        True source angles in degrees.
    n_sensors : int
        Number of array sensors.
    n_snapshots : int
        Number of data snapshots.
    snr_db : float
        Signal-to-noise ratio in dB.
    phase_modulation : array_like
        Phase modulation pattern to apply.

    Returns
    -------
    data : ndarray, shape (n_sensors, n_snapshots)
        Complex array data with modulation and noise.
    """
    # Ensure we have enough snapshots for modulation
    if len(phase_modulation) > n_snapshots:
        phase_modulation = phase_modulation[:n_snapshots]
    else:
        # Repeat pattern if needed
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
        # Base signal
        base_signal = np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)
        # Apply phase modulation
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


def main():
    """Demonstrate coprime signal design capabilities."""
    print("[DEMO] Coprime Signal Design for SPICE Enhancement")
    print("="*60)

    # Create coprime designer
    designer = CoprimeSignalDesign(coprime_pair=(31, 37))

    # Analyze performance improvement
    improvement = designer.analyze_performance_improvement(n_chirps=128)

    # Validate coprime properties
    validation = designer.validate_coprime_properties()

    # Plot ambiguity functions
    fig = designer.plot_ambiguity_comparison(n_chirps=128)
    plt.show()

    # Compare different signal designs
    comparison = compare_signal_designs(n_chirps=128, snr_db=15)

    print(f"\n[RESULT] Best performing design:")
    best_design = max(comparison.keys(),
                     key=lambda k: comparison[k].get('detection_rate', 0))
    print(f"   {comparison[best_design]['description']}")
    print(f"   Detection rate: {comparison[best_design]['detection_rate']:.2f}")

    return improvement, validation, comparison


if __name__ == "__main__":
    results = main()
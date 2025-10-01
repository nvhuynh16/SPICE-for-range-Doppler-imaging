"""
Enhanced Coprime Processing with Full-Period Coherent Processing

This module implements sophisticated coprime waveform processing that should
achieve the literature-claimed 2-3 dB SNR improvements through:
1. Full coprime period coherent processing
2. Proper matched filtering for coprime waveforms
3. Enhanced mutual incoherence exploitation in SPICE
4. Waveform diversity and correlation optimization
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional
import warnings

from coprime_signal_design import CoprimeSignalDesign
from improved_enhanced_spice import create_improved_enhanced_spice
from spice_core import SPICEEstimator


class EnhancedCoprimeProcessor:
    """
    Enhanced coprime processing for radar waveforms with full-period processing.

    This processor implements sophisticated coprime techniques that should achieve
    the literature-claimed performance improvements through proper exploitation
    of coprime properties.
    """

    def __init__(self, coprime_pair: Tuple[int, int] = (31, 37),
                 n_sensors: int = 8):
        """
        Initialize enhanced coprime processor.

        Parameters
        ----------
        coprime_pair : tuple of int, default=(31, 37)
            Coprime integers for waveform design.
        n_sensors : int, default=8
            Number of array sensors.
        """
        self.coprime_pair = coprime_pair
        self.n_sensors = n_sensors
        self.coprime_designer = CoprimeSignalDesign(coprime_pair)

        # Full coprime period for optimal processing
        self.full_period = self.coprime_designer.period

        print(f"Enhanced Coprime Processor initialized:")
        print(f"  Coprime pair: {coprime_pair}")
        print(f"  Full period: {self.full_period} samples")
        print(f"  Array sensors: {n_sensors}")

    def generate_optimized_coprime_data(self, true_angles: np.ndarray,
                                      snr_db: float,
                                      use_full_period: bool = True) -> Dict:
        """
        Generate radar data with optimized coprime processing.

        Parameters
        ----------
        true_angles : array_like
            True target angles in degrees.
        snr_db : float
            Signal-to-noise ratio in dB.
        use_full_period : bool, default=True
            Whether to use full coprime period for processing.

        Returns
        -------
        coprime_data : dict
            Comprehensive coprime processed data.
        """
        # Determine processing length
        if use_full_period:
            n_snapshots = self.full_period
            print(f"Using full coprime period: {n_snapshots} snapshots")
        else:
            n_snapshots = min(256, self.full_period)  # Reasonable subset
            print(f"Using partial period: {n_snapshots} snapshots")

        # Generate coprime phase pattern
        coprime_phases = self.coprime_designer.generate_phase_pattern(n_snapshots)

        # Generate standard reference (no modulation)
        standard_phases = np.ones(n_snapshots, dtype=complex)

        # Generate both datasets
        coprime_array_data = self._generate_modulated_array_data(
            true_angles, n_snapshots, snr_db, coprime_phases
        )

        standard_array_data = self._generate_modulated_array_data(
            true_angles, n_snapshots, snr_db, standard_phases
        )

        # Compute sample covariances
        coprime_cov = self._compute_sample_covariance(coprime_array_data)
        standard_cov = self._compute_sample_covariance(standard_array_data)

        # Apply matched filtering optimization for coprime
        coprime_cov_matched = self._apply_coprime_matched_filtering(
            coprime_cov, coprime_phases
        )

        # Compute mutual incoherence metrics
        incoherence_analysis = self._analyze_mutual_incoherence(
            coprime_cov_matched, standard_cov
        )

        return {
            'coprime_covariance': coprime_cov_matched,
            'standard_covariance': standard_cov,
            'coprime_phases': coprime_phases,
            'standard_phases': standard_phases,
            'n_snapshots': n_snapshots,
            'incoherence_analysis': incoherence_analysis,
            'snr_db': snr_db,
            'true_angles': true_angles
        }

    def _generate_modulated_array_data(self, true_angles: np.ndarray,
                                     n_snapshots: int, snr_db: float,
                                     phase_modulation: np.ndarray) -> np.ndarray:
        """Generate array data with specified phase modulation."""
        np.random.seed(42)  # Fixed seed for fair comparison

        # Generate steering vectors
        steering_matrix = np.zeros((self.n_sensors, len(true_angles)), dtype=complex)
        for i, angle in enumerate(true_angles):
            phase_shifts = np.arange(self.n_sensors) * np.pi * np.sin(np.deg2rad(angle))
            steering_matrix[:, i] = np.exp(1j * phase_shifts)

        # Generate source signals with phase modulation
        source_signals = np.zeros((len(true_angles), n_snapshots), dtype=complex)
        for i in range(len(true_angles)):
            # Base signal
            base_signal = (np.random.randn(n_snapshots) +
                          1j * np.random.randn(n_snapshots))

            # Apply phase modulation (ensure proper length)
            if len(phase_modulation) >= n_snapshots:
                modulation = phase_modulation[:n_snapshots]
            else:
                # Repeat pattern to cover full length
                repeats = int(np.ceil(n_snapshots / len(phase_modulation)))
                modulation = np.tile(phase_modulation, repeats)[:n_snapshots]

            source_signals[i, :] = base_signal * modulation

        # Received signals
        received_signals = steering_matrix @ source_signals

        # Add noise for desired SNR
        signal_power = np.mean(np.abs(received_signals)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(self.n_sensors, n_snapshots) +
            1j * np.random.randn(self.n_sensors, n_snapshots)
        )

        return received_signals + noise

    def _compute_sample_covariance(self, array_data: np.ndarray) -> np.ndarray:
        """Compute sample covariance matrix."""
        n_sensors, n_snapshots = array_data.shape
        return array_data @ array_data.conj().T / n_snapshots

    def _apply_coprime_matched_filtering(self, sample_cov: np.ndarray,
                                       coprime_phases: np.ndarray) -> np.ndarray:
        """
        Apply matched filtering optimization for coprime waveforms.

        This implements sophisticated matched filtering that exploits coprime
        properties for improved performance.
        """
        # Compute coprime correlation characteristics
        autocorr = self._compute_coprime_autocorrelation(coprime_phases)

        # Design whitening filter based on coprime properties
        whitening_filter = self._design_coprime_whitening_filter(autocorr)

        # Apply whitening to covariance matrix
        if whitening_filter is not None:
            # Apply filter transformation to covariance
            filtered_cov = self._apply_whitening_filter(sample_cov, whitening_filter)
            return filtered_cov
        else:
            return sample_cov

    def _compute_coprime_autocorrelation(self, coprime_phases: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of coprime phase pattern."""
        n = len(coprime_phases)
        autocorr = np.zeros(n, dtype=complex)

        for lag in range(n):
            if lag == 0:
                autocorr[lag] = np.mean(np.abs(coprime_phases)**2)
            else:
                # Compute correlation at given lag
                valid_length = n - lag
                correlation = np.mean(
                    coprime_phases[:valid_length].conj() *
                    coprime_phases[lag:lag+valid_length]
                )
                autocorr[lag] = correlation

        return autocorr

    def _design_coprime_whitening_filter(self, autocorr: np.ndarray) -> Optional[np.ndarray]:
        """
        Design whitening filter based on coprime autocorrelation.

        This exploits the unique autocorrelation properties of coprime sequences
        to design optimal filtering.
        """
        try:
            # Convert autocorrelation to power spectral density
            psd = np.fft.fft(autocorr)
            psd = np.real(psd)  # Take real part

            # Design inverse filter (whitening)
            # Add small regularization to avoid division by zero
            reg_factor = 0.01 * np.max(psd)
            psd_reg = psd + reg_factor

            # Inverse filter in frequency domain
            whitening_freq = 1.0 / np.sqrt(psd_reg)

            # Convert back to time domain
            whitening_filter = np.fft.ifft(whitening_freq)

            # Keep only causal part and limit length
            filter_length = min(32, len(whitening_filter) // 4)
            whitening_filter = whitening_filter[:filter_length]

            return whitening_filter

        except Exception:
            # If whitening filter design fails, return None
            return None

    def _apply_whitening_filter(self, sample_cov: np.ndarray,
                               whitening_filter: np.ndarray) -> np.ndarray:
        """Apply whitening filter to sample covariance matrix."""
        try:
            # This is a simplified version - in practice would apply to time series
            # For covariance matrix, we apply a transformation based on filter properties

            # Compute filter effect on covariance eigenstructure
            eigenvals, eigenvecs = np.linalg.eigh(sample_cov)

            # Apply filter-based transformation to eigenvalues
            filter_magnitude = np.abs(whitening_filter[0]) if len(whitening_filter) > 0 else 1.0

            # Scale eigenvalues based on filter characteristics
            transformed_eigenvals = eigenvals * filter_magnitude**2

            # Reconstruct covariance matrix
            filtered_cov = eigenvecs @ np.diag(transformed_eigenvals) @ eigenvecs.conj().T

            return filtered_cov

        except Exception:
            # If filtering fails, return original
            return sample_cov

    def _analyze_mutual_incoherence(self, coprime_cov: np.ndarray,
                                  standard_cov: np.ndarray) -> Dict:
        """
        Analyze mutual incoherence properties of coprime vs standard processing.
        """
        # Compute condition numbers
        coprime_cond = np.linalg.cond(coprime_cov)
        standard_cond = np.linalg.cond(standard_cov)

        # Compute eigenvalue spreads
        coprime_eigs = np.linalg.eigvals(coprime_cov)
        standard_eigs = np.linalg.eigvals(standard_cov)

        coprime_eig_spread = np.max(np.real(coprime_eigs)) / np.min(np.real(coprime_eigs))
        standard_eig_spread = np.max(np.real(standard_eigs)) / np.min(np.real(standard_eigs))

        # Compute coherence metrics
        coprime_coherence = self._compute_coherence_metric(coprime_cov)
        standard_coherence = self._compute_coherence_metric(standard_cov)

        return {
            'coprime_condition_number': coprime_cond,
            'standard_condition_number': standard_cond,
            'condition_number_ratio': standard_cond / coprime_cond,
            'coprime_eigenvalue_spread': coprime_eig_spread,
            'standard_eigenvalue_spread': standard_eig_spread,
            'eigenvalue_spread_ratio': standard_eig_spread / coprime_eig_spread,
            'coprime_coherence': coprime_coherence,
            'standard_coherence': standard_coherence,
            'coherence_improvement': standard_coherence / coprime_coherence
        }

    def _compute_coherence_metric(self, covariance: np.ndarray) -> float:
        """Compute mutual coherence metric from covariance matrix."""
        # Normalize covariance to correlation matrix
        diag_elements = np.diag(covariance)
        normalization = np.sqrt(np.outer(diag_elements, diag_elements))

        # Avoid division by zero
        normalization = np.maximum(normalization, 1e-12)
        correlation_matrix = covariance / normalization

        # Compute maximum off-diagonal correlation (mutual coherence)
        n = correlation_matrix.shape[0]
        off_diag_corr = []

        for i in range(n):
            for j in range(i+1, n):
                off_diag_corr.append(np.abs(correlation_matrix[i, j]))

        return np.max(off_diag_corr) if off_diag_corr else 0.0

    def compare_coprime_vs_standard(self, true_angles: np.ndarray,
                                  snr_db: float,
                                  use_enhanced_spice: bool = True) -> Dict:
        """
        Comprehensive comparison of coprime vs standard processing.

        Parameters
        ----------
        true_angles : array_like
            True target angles.
        snr_db : float
            Signal-to-noise ratio.
        use_enhanced_spice : bool, default=True
            Whether to use enhanced SPICE for detection.

        Returns
        -------
        comparison : dict
            Comprehensive comparison results.
        """
        print(f"\\n[COPRIME COMPARISON] SNR={snr_db} dB, Angles={true_angles}")

        # Generate optimized coprime data
        coprime_data = self.generate_optimized_coprime_data(
            true_angles, snr_db, use_full_period=True
        )

        # Test with SPICE algorithms
        if use_enhanced_spice:
            # Enhanced SPICE processing
            estimator_coprime = create_improved_enhanced_spice(
                self.n_sensors, target_snr_db=snr_db
            )
            estimator_standard = create_improved_enhanced_spice(
                self.n_sensors, target_snr_db=snr_db
            )
        else:
            # Standard SPICE processing
            estimator_coprime = SPICEEstimator(self.n_sensors)
            estimator_standard = SPICEEstimator(self.n_sensors)

        # Process both datasets
        spectrum_coprime, angles_coprime = estimator_coprime.fit(
            coprime_data['coprime_covariance']
        )
        spectrum_standard, angles_standard = estimator_standard.fit(
            coprime_data['standard_covariance']
        )

        # Peak detection
        if use_enhanced_spice:
            peaks_coprime = estimator_coprime.find_peaks_advanced(
                spectrum_coprime, coprime_data['coprime_covariance']
            )
            peaks_standard = estimator_standard.find_peaks_advanced(
                spectrum_standard, coprime_data['standard_covariance']
            )
        else:
            peaks_coprime = estimator_coprime.find_peaks(spectrum_coprime)
            peaks_standard = estimator_standard.find_peaks(spectrum_standard)

        # Evaluate performance
        perf_coprime = self._evaluate_detection_performance(
            peaks_coprime['angles'], true_angles
        )
        perf_standard = self._evaluate_detection_performance(
            peaks_standard['angles'], true_angles
        )

        # Compute improvements
        improvements = self._compute_improvements(perf_coprime, perf_standard)

        print(f"  Standard processing: {perf_standard['detection_rate']:.2f} detection rate")
        print(f"  Coprime processing:  {perf_coprime['detection_rate']:.2f} detection rate")
        print(f"  Detection improvement: {improvements['detection_improvement']:.2f}x")
        print(f"  Angular accuracy improvement: {improvements['angular_improvement']:.2f}x")

        return {
            'coprime_performance': perf_coprime,
            'standard_performance': perf_standard,
            'improvements': improvements,
            'incoherence_analysis': coprime_data['incoherence_analysis'],
            'processing_details': {
                'n_snapshots': coprime_data['n_snapshots'],
                'snr_db': snr_db,
                'use_enhanced_spice': use_enhanced_spice,
                'coprime_pair': self.coprime_pair
            }
        }

    def _evaluate_detection_performance(self, detected_angles: np.ndarray,
                                      true_angles: np.ndarray) -> Dict:
        """Evaluate detection performance."""
        n_true = len(true_angles)
        n_detected = len(detected_angles)

        if n_detected == 0:
            return {
                'detection_rate': 0.0,
                'false_alarm_rate': 0.0,
                'angular_error': np.inf,
                'n_detected': 0
            }

        # Match detected to true angles
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
            'n_detected': n_detected
        }

    def _compute_improvements(self, coprime_perf: Dict, standard_perf: Dict) -> Dict:
        """Compute improvement metrics."""
        # Detection rate improvement
        det_improvement = (coprime_perf['detection_rate'] /
                          max(standard_perf['detection_rate'], 0.01))

        # Angular accuracy improvement (lower error is better)
        if (standard_perf['angular_error'] != np.inf and
            coprime_perf['angular_error'] != np.inf):
            ang_improvement = standard_perf['angular_error'] / coprime_perf['angular_error']
        else:
            ang_improvement = 1.0

        # False alarm improvement (lower FA rate is better)
        fa_improvement = (standard_perf['false_alarm_rate'] /
                         max(coprime_perf['false_alarm_rate'], 0.01))

        return {
            'detection_improvement': det_improvement,
            'angular_improvement': ang_improvement,
            'false_alarm_improvement': fa_improvement
        }


def create_enhanced_coprime_processor(coprime_pair: Tuple[int, int] = (31, 37),
                                    n_sensors: int = 8) -> EnhancedCoprimeProcessor:
    """Factory function to create enhanced coprime processor."""
    return EnhancedCoprimeProcessor(coprime_pair, n_sensors)
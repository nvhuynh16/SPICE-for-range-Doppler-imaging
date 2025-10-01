"""
Advanced Peak Detection for SPICE Algorithms

This module implements sophisticated peak detection that validates detected peaks
against physical and mathematical constraints to eliminate spurious detections
and improve true target identification.

Key innovations:
1. Physical constraint validation using steering vector projections
2. SNR-aware adaptive thresholding
3. Peak persistence across multiple criteria
4. Eigenstructure coherence validation
5. Multi-criteria peak quality assessment
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional
import warnings


class AdvancedPeakDetector:
    """
    Advanced peak detection for SPICE power spectra with intelligent validation.

    This detector addresses the fundamental issue that SPICE spectra contain many
    spurious local maxima that don't correspond to true targets. It implements
    multiple validation criteria to identify genuine peaks.
    """

    def __init__(self, n_sensors: int, angular_grid: np.ndarray):
        """
        Initialize advanced peak detector.

        Parameters
        ----------
        n_sensors : int
            Number of array sensors.
        angular_grid : array_like
            Angular grid in degrees.
        """
        self.n_sensors = n_sensors
        self.angular_grid = angular_grid
        self.grid_resolution = angular_grid[1] - angular_grid[0]

        # Precompute steering vectors for validation
        self.steering_vectors = self._compute_steering_vectors()

    def _compute_steering_vectors(self) -> np.ndarray:
        """Compute steering vectors for angular grid."""
        angles_rad = np.deg2rad(self.angular_grid)
        positions = np.arange(self.n_sensors)

        steering_matrix = np.exp(1j * np.pi * positions[:, np.newaxis] *
                                np.sin(angles_rad[np.newaxis, :]))
        return steering_matrix

    def detect_peaks(self, power_spectrum: np.ndarray,
                    sample_covariance: np.ndarray,
                    estimated_snr_db: Optional[float] = None,
                    max_targets: int = 10,
                    confidence_level: float = 0.95) -> Dict:
        """
        Detect peaks using advanced validation criteria.

        Parameters
        ----------
        power_spectrum : array_like
            SPICE power spectrum.
        sample_covariance : array_like
            Sample covariance matrix used for validation.
        estimated_snr_db : float, optional
            Estimated SNR for adaptive thresholding.
        max_targets : int, default=10
            Maximum number of targets to detect.
        confidence_level : float, default=0.95
            Statistical confidence level for validation.

        Returns
        -------
        detection_result : dict
            Comprehensive detection results with quality metrics.
        """
        # Step 1: Initial peak candidates using multiple criteria
        candidates = self._find_peak_candidates(power_spectrum)

        if len(candidates) == 0:
            return self._empty_detection_result()

        # Step 2: Physical constraint validation
        candidates = self._validate_physical_constraints(
            candidates, power_spectrum, sample_covariance
        )

        # Step 3: SNR-aware thresholding
        if estimated_snr_db is not None:
            candidates = self._apply_snr_thresholding(
                candidates, power_spectrum, estimated_snr_db
            )

        # Step 4: Eigenstructure coherence validation
        candidates = self._validate_eigenstructure_coherence(
            candidates, power_spectrum, sample_covariance, confidence_level
        )

        # Step 5: Multi-criteria quality assessment and ranking
        candidates = self._assess_peak_quality(
            candidates, power_spectrum, sample_covariance
        )

        # Step 6: Final selection with spatial separation constraints
        final_peaks = self._select_final_peaks(candidates, max_targets)

        return self._format_detection_result(final_peaks, power_spectrum)

    def _find_peak_candidates(self, power_spectrum: np.ndarray) -> List[Dict]:
        """Find initial peak candidates using multiple detection criteria."""
        candidates = []

        # Normalize spectrum for consistent thresholding
        spectrum_norm = power_spectrum / np.max(power_spectrum)

        # Criterion 1: Prominence-based detection
        prominence_threshold = 0.1  # 10% of peak height
        peak_indices_1, properties_1 = find_peaks(
            spectrum_norm,
            prominence=prominence_threshold,
            distance=int(2.0 / self.grid_resolution)  # Minimum 2° separation
        )

        # Criterion 2: Height-based detection (above noise floor)
        noise_floor = np.mean(spectrum_norm) + 2 * np.std(spectrum_norm)
        height_threshold = max(noise_floor, 0.2)  # At least 20% or above noise
        peak_indices_2, properties_2 = find_peaks(
            spectrum_norm,
            height=height_threshold,
            distance=int(1.5 / self.grid_resolution)  # Minimum 1.5° separation
        )

        # Criterion 3: Relative maxima with width constraints
        peak_indices_3, properties_3 = find_peaks(
            spectrum_norm,
            width=1,  # Minimum width requirement
            rel_height=0.5  # Peak must be 50% above surroundings
        )

        # Combine candidates from all criteria
        all_indices = np.unique(np.concatenate([
            peak_indices_1, peak_indices_2, peak_indices_3
        ]))

        for idx in all_indices:
            candidate = {
                'index': idx,
                'angle': self.angular_grid[idx],
                'power': power_spectrum[idx],
                'normalized_power': spectrum_norm[idx],
                'detection_criteria': []
            }

            # Track which criteria detected this peak
            if idx in peak_indices_1:
                candidate['detection_criteria'].append('prominence')
            if idx in peak_indices_2:
                candidate['detection_criteria'].append('height')
            if idx in peak_indices_3:
                candidate['detection_criteria'].append('width')

            candidates.append(candidate)

        return candidates

    def _validate_physical_constraints(self, candidates: List[Dict],
                                     power_spectrum: np.ndarray,
                                     sample_cov: np.ndarray) -> List[Dict]:
        """Validate peaks against physical constraints."""
        valid_candidates = []

        for candidate in candidates:
            idx = candidate['index']
            steering_vec = self.steering_vectors[:, idx]

            # Physical constraint 1: Steering vector projection validation
            # A true target should have significant projection onto sample covariance
            projection_power = np.real(
                steering_vec.conj().T @ sample_cov @ steering_vec
            )
            normalized_projection = projection_power / np.trace(sample_cov)

            # Physical constraint 2: Coherence with dominant eigenspace
            eigenvals, eigenvecs = np.linalg.eigh(sample_cov)
            # Use top 50% of eigenspace (signal + some noise)
            n_dominant = max(1, self.n_sensors // 2)
            dominant_subspace = eigenvecs[:, -n_dominant:]

            subspace_coherence = np.sum(np.abs(
                dominant_subspace.conj().T @ steering_vec
            )**2) / n_dominant

            # Physical constraint 3: Spatial consistency
            # Check if peak has reasonable spatial characteristics
            spatial_consistency = self._assess_spatial_consistency(
                idx, power_spectrum
            )

            # Apply validation thresholds
            if (normalized_projection > 0.01 and  # Minimum projection requirement
                subspace_coherence > 0.1 and     # Minimum coherence requirement
                spatial_consistency > 0.5):      # Minimum spatial consistency

                candidate['projection_power'] = projection_power
                candidate['normalized_projection'] = normalized_projection
                candidate['subspace_coherence'] = subspace_coherence
                candidate['spatial_consistency'] = spatial_consistency
                candidate['physical_valid'] = True
                valid_candidates.append(candidate)
            else:
                candidate['physical_valid'] = False

        return valid_candidates

    def _assess_spatial_consistency(self, peak_idx: int,
                                  power_spectrum: np.ndarray) -> float:
        """Assess spatial consistency of peak (smooth falloff from peak)."""
        # Check power falloff in neighborhood
        neighborhood_size = min(5, len(power_spectrum) // 10)
        start_idx = max(0, peak_idx - neighborhood_size)
        end_idx = min(len(power_spectrum), peak_idx + neighborhood_size + 1)

        neighborhood = power_spectrum[start_idx:end_idx]
        peak_power = power_spectrum[peak_idx]

        # Calculate how well power decreases away from peak
        distances = np.abs(np.arange(start_idx, end_idx) - peak_idx)
        distances[distances == 0] = 0.1  # Avoid division by zero

        # Expected falloff (should decrease with distance)
        expected_falloff = peak_power / (1 + distances * 0.5)
        actual_power = neighborhood

        # Measure consistency (how well actual matches expected falloff)
        consistency = 1.0 - np.mean(np.abs(actual_power - expected_falloff) / peak_power)
        return max(0.0, consistency)

    def _apply_snr_thresholding(self, candidates: List[Dict],
                               power_spectrum: np.ndarray,
                               estimated_snr_db: float) -> List[Dict]:
        """Apply SNR-aware adaptive thresholding."""
        # Convert SNR to linear scale
        snr_linear = 10**(estimated_snr_db / 10)

        # Estimate noise floor from spectrum
        sorted_powers = np.sort(power_spectrum)
        # Use bottom 25% as noise estimate
        noise_floor = np.mean(sorted_powers[:len(sorted_powers)//4])

        # Adaptive threshold based on SNR
        min_signal_power = noise_floor * (1 + snr_linear * 0.1)

        # SNR-based dynamic threshold
        if estimated_snr_db < 5:
            threshold_factor = 0.3  # Lower threshold for low SNR
        elif estimated_snr_db < 10:
            threshold_factor = 0.5  # Moderate threshold
        else:
            threshold_factor = 0.7  # Higher threshold for high SNR

        dynamic_threshold = threshold_factor * np.max(power_spectrum)
        final_threshold = max(min_signal_power, dynamic_threshold)

        valid_candidates = []
        for candidate in candidates:
            if candidate['power'] >= final_threshold:
                candidate['snr_threshold_passed'] = True
                candidate['threshold_used'] = final_threshold
                valid_candidates.append(candidate)
            else:
                candidate['snr_threshold_passed'] = False

        return valid_candidates

    def _validate_eigenstructure_coherence(self, candidates: List[Dict],
                                         power_spectrum: np.ndarray,
                                         sample_cov: np.ndarray,
                                         confidence_level: float) -> List[Dict]:
        """Validate peaks against eigenstructure of sample covariance."""
        if len(candidates) == 0:
            return candidates

        # Eigendecomposition of sample covariance
        eigenvals, eigenvecs = np.linalg.eigh(sample_cov)
        eigenvals = np.real(eigenvals)
        eigenvals = eigenvals[::-1]  # Descending order
        eigenvecs = eigenvecs[:, ::-1]

        # Estimate number of signals using information theoretic criteria
        n_signals = self._estimate_signal_subspace_dimension(eigenvals)

        valid_candidates = []
        for candidate in candidates:
            idx = candidate['index']
            steering_vec = self.steering_vectors[:, idx]

            # Test coherence with signal subspace
            if n_signals > 0:
                signal_subspace = eigenvecs[:, :n_signals]
                signal_projection = np.sum(np.abs(
                    signal_subspace.conj().T @ steering_vec
                )**2)
                signal_coherence = signal_projection / n_signals
            else:
                signal_coherence = 0.0

            # Statistical test for eigenstructure consistency
            # Use chi-squared test for goodness of fit
            expected_projection = 1.0 / self.n_sensors  # Uniform expectation
            chi2_statistic = (signal_coherence - expected_projection)**2 / expected_projection
            p_value = 1 - chi2.cdf(chi2_statistic, df=1)

            # Accept peak if statistically significant
            if p_value < (1 - confidence_level) or signal_coherence > 0.2:
                candidate['signal_coherence'] = signal_coherence
                candidate['eigenstructure_p_value'] = p_value
                candidate['eigenstructure_valid'] = True
                valid_candidates.append(candidate)
            else:
                candidate['eigenstructure_valid'] = False

        return valid_candidates

    def _estimate_signal_subspace_dimension(self, eigenvals: np.ndarray) -> int:
        """Estimate signal subspace dimension using MDL criterion."""
        n = len(eigenvals)
        if n < 2:
            return 0

        # Minimum Description Length (MDL) criterion
        mdl_scores = []
        for k in range(n):
            if k == 0:
                mdl_scores.append(np.inf)
                continue

            # Geometric mean of noise eigenvalues
            noise_eigenvals = eigenvals[k:]
            if len(noise_eigenvals) == 0:
                geometric_mean = eigenvals[-1]
            else:
                geometric_mean = np.exp(np.mean(np.log(noise_eigenvals + 1e-12)))

            # Arithmetic mean of noise eigenvalues
            arithmetic_mean = np.mean(noise_eigenvals)

            # MDL score
            if geometric_mean > 0 and arithmetic_mean > 0:
                mdl_score = (n - k) * np.log(arithmetic_mean / geometric_mean)
                mdl_scores.append(mdl_score)
            else:
                mdl_scores.append(np.inf)

        # Find minimum MDL score
        estimated_k = np.argmin(mdl_scores)
        return min(estimated_k, n // 2)  # Cap at half the sensors

    def _assess_peak_quality(self, candidates: List[Dict],
                           power_spectrum: np.ndarray,
                           sample_cov: np.ndarray) -> List[Dict]:
        """Assess overall quality of each peak candidate."""
        for candidate in candidates:
            # Combine multiple quality metrics
            quality_score = 0.0

            # Power level contribution (30%)
            power_score = candidate['normalized_power']
            quality_score += 0.3 * power_score

            # Physical constraint contribution (25%)
            if 'normalized_projection' in candidate:
                physical_score = min(1.0, candidate['normalized_projection'] * 10)
                quality_score += 0.25 * physical_score

            # Eigenstructure coherence contribution (25%)
            if 'signal_coherence' in candidate:
                coherence_score = min(1.0, candidate['signal_coherence'] * 5)
                quality_score += 0.25 * coherence_score

            # Spatial consistency contribution (20%)
            if 'spatial_consistency' in candidate:
                spatial_score = candidate['spatial_consistency']
                quality_score += 0.2 * spatial_score

            # Multi-criteria detection bonus
            criteria_bonus = len(candidate['detection_criteria']) / 3.0
            quality_score += 0.1 * criteria_bonus

            candidate['quality_score'] = quality_score

        # Sort by quality score
        candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        return candidates

    def _select_final_peaks(self, candidates: List[Dict],
                          max_targets: int) -> List[Dict]:
        """Select final peaks ensuring spatial separation."""
        if len(candidates) == 0:
            return []

        final_peaks = []
        min_separation_deg = 2.0  # Minimum 2° separation

        for candidate in candidates:
            if len(final_peaks) >= max_targets:
                break

            # Check separation from already selected peaks
            too_close = False
            for selected_peak in final_peaks:
                separation = abs(candidate['angle'] - selected_peak['angle'])
                if separation < min_separation_deg:
                    too_close = True
                    break

            if not too_close:
                final_peaks.append(candidate)

        return final_peaks

    def _empty_detection_result(self) -> Dict:
        """Return empty detection result."""
        return {
            'angles': np.array([]),
            'powers': np.array([]),
            'quality_scores': np.array([]),
            'n_peaks': 0,
            'detection_summary': {
                'candidates_found': 0,
                'physical_valid': 0,
                'eigenstructure_valid': 0,
                'final_selected': 0
            }
        }

    def _format_detection_result(self, final_peaks: List[Dict],
                               power_spectrum: np.ndarray) -> Dict:
        """Format final detection result."""
        if len(final_peaks) == 0:
            return self._empty_detection_result()

        angles = np.array([peak['angle'] for peak in final_peaks])
        powers = np.array([peak['power'] for peak in final_peaks])
        quality_scores = np.array([peak['quality_score'] for peak in final_peaks])

        # Create detailed summary
        detection_summary = {
            'candidates_found': len(final_peaks),
            'physical_valid': sum(1 for p in final_peaks if p.get('physical_valid', False)),
            'eigenstructure_valid': sum(1 for p in final_peaks if p.get('eigenstructure_valid', False)),
            'final_selected': len(final_peaks),
            'average_quality': np.mean(quality_scores),
            'peak_details': final_peaks
        }

        return {
            'angles': angles,
            'powers': powers,
            'quality_scores': quality_scores,
            'n_peaks': len(final_peaks),
            'detection_summary': detection_summary
        }


def create_advanced_detector(n_sensors: int, angular_grid: np.ndarray) -> AdvancedPeakDetector:
    """Factory function to create advanced peak detector."""
    return AdvancedPeakDetector(n_sensors, angular_grid)
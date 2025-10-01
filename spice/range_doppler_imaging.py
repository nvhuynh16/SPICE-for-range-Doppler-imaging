"""
Range-Doppler Imaging using SPICE for FMCW Radar Systems.

This module implements comprehensive range-doppler imaging capabilities using
SPICE algorithms for enhanced resolution and target detection in FMCW radar systems.

The implementation includes signal generation, processing, and advanced imaging
techniques suitable for automotive, maritime, and airborne radar applications.

References
----------
.. [1] M. A. Richards, "Fundamentals of Radar Signal Processing," 2nd ed., 2014.
.. [2] V. S. Chernyak, "Fundamentals of Multisite Radar Systems," 1998.
.. [3] A. Meta et al., "Signal Processing for FMCW SAR," IEEE Trans. GRS, 2007.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

from spice_core import SPICEEstimator, SPICEConfig
from spice_variants import select_spice_variant


@dataclass
class FMCWRadarConfig:
    """Configuration parameters for FMCW radar system.

    Parameters
    ----------
    f_start : float, default=24e9
        Start frequency in Hz.
    bandwidth : float, default=1e9
        Chirp bandwidth in Hz.
    chirp_duration : float, default=1e-3
        Chirp duration in seconds.
    n_chirps : int, default=128
        Number of chirps in coherent processing interval.
    fs : float, default=10e6
        Sampling frequency in Hz.
    n_rx : int, default=4
        Number of receive antennas.
    c : float, default=3e8
        Speed of light in m/s.
    """
    f_start: float = 24e9
    bandwidth: float = 1e9
    chirp_duration: float = 1e-3
    n_chirps: int = 128
    fs: float = 10e6
    n_rx: int = 4
    c: float = 3e8

    @property
    def range_resolution(self) -> float:
        """Range resolution in meters."""
        return self.c / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Maximum unambiguous range in meters."""
        return self.c * self.fs * self.chirp_duration / (2 * self.bandwidth)

    @property
    def velocity_resolution(self) -> float:
        """Velocity resolution in m/s."""
        return self.c / (2 * self.f_start * self.n_chirps * self.chirp_duration)

    @property
    def max_velocity(self) -> float:
        """Maximum unambiguous velocity in m/s."""
        return self.c / (4 * self.f_start * self.chirp_duration)


class FMCWSignalGenerator:
    """
    FMCW Radar Signal Generator for Realistic Target Scenarios.

    Generates realistic FMCW radar signals including multiple targets,
    noise, clutter, and interference for algorithm validation.

    Parameters
    ----------
    config : FMCWRadarConfig
        Radar system configuration.

    Examples
    --------
    >>> config = FMCWRadarConfig()
    >>> generator = FMCWSignalGenerator(config)
    >>> targets = [{'range': 100, 'velocity': 15, 'rcs': 1.0, 'angle': 0}]
    >>> radar_data = generator.generate_radar_data(targets, snr_db=20)
    """

    def __init__(self, config: FMCWRadarConfig):
        """Initialize FMCW signal generator."""
        self.config = config
        self._setup_time_frequency_grids()

    def _setup_time_frequency_grids(self) -> None:
        """Set up time and frequency grids for signal generation."""
        # Time samples within each chirp
        self.n_samples = int(self.config.fs * self.config.chirp_duration)
        self.time_fast = np.linspace(0, self.config.chirp_duration, self.n_samples)

        # Time samples across chirps (slow time)
        self.time_slow = np.arange(self.config.n_chirps) * self.config.chirp_duration

        # Range and velocity grids
        self.range_bins = np.arange(self.n_samples) * self.config.range_resolution
        self.velocity_bins = np.linspace(
            -self.config.max_velocity,
            self.config.max_velocity,
            self.config.n_chirps
        )

    def generate_radar_data(self, targets: List[Dict],
                           snr_db: float = 20.0,
                           clutter_power_db: float = -30.0,
                           add_interference: bool = False) -> np.ndarray:
        """
        Generate realistic FMCW radar data with multiple targets.

        Parameters
        ----------
        targets : list of dict
            List of target dictionaries with keys:
            - 'range': Target range in meters
            - 'velocity': Target velocity in m/s
            - 'rcs': Radar cross-section in m²
            - 'angle': Target angle in degrees (optional)
        snr_db : float, default=20.0
            Signal-to-noise ratio in dB.
        clutter_power_db : float, default=-30.0
            Clutter power relative to signal in dB.
        add_interference : bool, default=False
            Whether to add interference from other radars.

        Returns
        -------
        radar_data : ndarray, shape (n_rx, n_chirps, n_samples)
            Complex radar data cube.

        Examples
        --------
        >>> targets = [
        ...     {'range': 75, 'velocity': 10, 'rcs': 1.0, 'angle': -10},
        ...     {'range': 150, 'velocity': -5, 'rcs': 0.5, 'angle': 0},
        ...     {'range': 225, 'velocity': 20, 'rcs': 2.0, 'angle': 15}
        ... ]
        >>> data = generator.generate_radar_data(targets, snr_db=15)
        """
        # Initialize data cube
        radar_data = np.zeros(
            (self.config.n_rx, self.config.n_chirps, self.n_samples),
            dtype=complex
        )

        # Generate signals for each target
        for target in targets:
            target_signal = self._generate_target_signal(target)

            # Add to all receive channels (with angular dependence if specified)
            if 'angle' in target:
                angular_response = self._compute_angular_response(target['angle'])
                for rx_idx in range(self.config.n_rx):
                    radar_data[rx_idx, :, :] += angular_response[rx_idx] * target_signal
            else:
                # Add to all channels equally if no angle specified
                for rx_idx in range(self.config.n_rx):
                    radar_data[rx_idx, :, :] += target_signal

        # Add noise
        noise_power = self._compute_noise_power(radar_data, snr_db)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*radar_data.shape) +
            1j * np.random.randn(*radar_data.shape)
        )
        radar_data += noise

        # Add clutter if specified
        if clutter_power_db > -100:
            clutter = self._generate_clutter(clutter_power_db)
            radar_data += clutter

        # Add interference if specified
        if add_interference:
            interference = self._generate_interference()
            radar_data += interference

        return radar_data

    def _generate_target_signal(self, target: Dict) -> np.ndarray:
        """Generate signal for a single target."""
        # Target parameters
        target_range = target['range']
        target_velocity = target['velocity']
        target_rcs = target['rcs']

        # Time delays and doppler shifts
        time_delay = 2 * target_range / self.config.c
        doppler_freq = 2 * target_velocity * self.config.f_start / self.config.c

        # Beat frequency due to range
        beat_freq = 2 * target_range * self.config.bandwidth / (
            self.config.c * self.config.chirp_duration
        )

        # Generate target signal
        signal_amplitude = np.sqrt(target_rcs)

        # Fast time signal (within each chirp)
        fast_time_signal = signal_amplitude * np.exp(
            1j * 2 * np.pi * beat_freq * self.time_fast
        )

        # Slow time modulation (across chirps) due to Doppler
        slow_time_modulation = np.exp(
            1j * 2 * np.pi * doppler_freq * self.time_slow
        )

        # Combine fast and slow time
        target_signal = np.outer(slow_time_modulation, fast_time_signal)

        return target_signal

    def _compute_angular_response(self, angle_deg: float) -> np.ndarray:
        """Compute angular response for multi-antenna system."""
        if self.config.n_rx == 1:
            return np.array([1.0])

        # Assume uniform linear array with half-wavelength spacing
        wavelength = self.config.c / self.config.f_start
        element_spacing = wavelength / 2

        # Phase delays for each antenna element
        angle_rad = np.deg2rad(angle_deg)
        phase_delays = np.arange(self.config.n_rx) * (
            2 * np.pi * element_spacing * np.sin(angle_rad) / wavelength
        )

        return np.exp(1j * phase_delays)

    def _compute_noise_power(self, signal_data: np.ndarray, snr_db: float) -> float:
        """Compute noise power for specified SNR."""
        signal_power = np.mean(np.abs(signal_data)**2)
        snr_linear = 10**(snr_db / 10)
        return signal_power / snr_linear

    def _generate_clutter(self, clutter_power_db: float) -> np.ndarray:
        """Generate ground clutter with Rayleigh statistics."""
        clutter_power_linear = 10**(clutter_power_db / 10)

        # Rayleigh distributed clutter
        clutter_amplitude = np.sqrt(clutter_power_linear) * np.random.rayleigh(
            scale=1.0, size=(self.config.n_rx, self.config.n_chirps, self.n_samples)
        )

        # Random phase
        clutter_phase = 2 * np.pi * np.random.random(
            (self.config.n_rx, self.config.n_chirps, self.n_samples)
        )

        return clutter_amplitude * np.exp(1j * clutter_phase)

    def _generate_interference(self) -> np.ndarray:
        """Generate interference from other radar systems."""
        # Simplified interference model
        interference_power = 0.1  # Relative to signal

        # Frequency-shifted interference
        freq_offset = self.config.bandwidth * 0.1  # 10% frequency offset

        interference_signal = np.sqrt(interference_power) * np.exp(
            1j * 2 * np.pi * freq_offset * self.time_fast
        )

        # Broadcast to all dimensions
        interference = np.broadcast_to(
            interference_signal,
            (self.config.n_rx, self.config.n_chirps, self.n_samples)
        ).copy()

        return interference


class RangeDopplerProcessor:
    """
    Range-Doppler Processor with SPICE Enhancement.

    Processes FMCW radar data to generate high-resolution range-doppler images
    using both conventional FFT methods and advanced SPICE algorithms.

    Parameters
    ----------
    config : FMCWRadarConfig
        Radar system configuration.
    spice_config : SPICEConfig, optional
        Configuration for SPICE processing.

    Examples
    --------
    >>> processor = RangeDopplerProcessor(radar_config)
    >>> rd_map = processor.process_conventional(radar_data)
    >>> spice_map = processor.process_spice(radar_data)
    >>> targets = processor.detect_targets(spice_map)
    """

    def __init__(self, config: FMCWRadarConfig,
                 spice_config: Optional[SPICEConfig] = None):
        """Initialize range-doppler processor."""
        self.config = config
        self.spice_config = spice_config or SPICEConfig()
        self._setup_processing_parameters()

    def _setup_processing_parameters(self) -> None:
        """Set up processing parameters and windows."""
        # Window functions for sidelobe suppression
        self.range_window = signal.windows.hamming(
            int(self.config.fs * self.config.chirp_duration)
        )
        self.doppler_window = signal.windows.hamming(self.config.n_chirps)

        # FFT sizes (can be larger than data for zero-padding)
        self.n_range_fft = int(2**np.ceil(np.log2(len(self.range_window))))
        self.n_doppler_fft = int(2**np.ceil(np.log2(len(self.doppler_window))))

    def process_conventional(self, radar_data: np.ndarray,
                           channel: int = 0) -> np.ndarray:
        """
        Process radar data using conventional 2D FFT method.

        Parameters
        ----------
        radar_data : array_like, shape (n_rx, n_chirps, n_samples)
            Raw radar data cube.
        channel : int, default=0
            Receive channel to process.

        Returns
        -------
        rd_map : ndarray, shape (n_doppler_fft, n_range_fft)
            Range-doppler map (power in dB).

        Notes
        -----
        Conventional processing applies 2D FFT with windowing:
        1. Range FFT on each chirp (fast-time dimension)
        2. Doppler FFT across chirps (slow-time dimension)
        """
        # Extract single channel data
        channel_data = radar_data[channel, :, :]

        # Apply range windowing
        windowed_data = channel_data * self.range_window[np.newaxis, :]

        # Range FFT (fast-time)
        range_fft = fft.fft(windowed_data, n=self.n_range_fft, axis=1)

        # Apply doppler windowing
        range_fft = range_fft * self.doppler_window[:, np.newaxis]

        # Doppler FFT (slow-time)
        rd_complex = fft.fftshift(fft.fft(range_fft, n=self.n_doppler_fft, axis=0), axes=0)

        # Convert to power (dB)
        rd_power = np.abs(rd_complex)**2
        rd_map_db = 10 * np.log10(rd_power + 1e-12)

        return rd_map_db

    def process_spice(self, radar_data: np.ndarray,
                     variant: str = 'standard',
                     range_bins: Optional[List[int]] = None) -> np.ndarray:
        """
        Process radar data using SPICE algorithm for enhanced resolution.

        Parameters
        ----------
        radar_data : array_like, shape (n_rx, n_chirps, n_samples)
            Raw radar data cube.
        variant : str, default='standard'
            SPICE variant to use: 'standard', 'fast', 'weighted'.
        range_bins : list of int, optional
            Specific range bins to process. If None, processes all bins.

        Returns
        -------
        spice_map : ndarray, shape (grid_size, n_range_bins)
            SPICE-enhanced range-doppler map.

        Notes
        -----
        SPICE processing provides enhanced angular/doppler resolution by:
        1. Forming sample covariance matrices across receive channels
        2. Applying SPICE algorithm for sparse estimation
        3. Combining results across range bins
        """
        n_rx, n_chirps, n_samples = radar_data.shape

        # Determine range bins to process
        if range_bins is None:
            range_bins = list(range(n_samples))

        # Initialize SPICE-enhanced map
        spice_map = np.zeros((self.spice_config.grid_size, len(range_bins)))

        # Process each range bin with SPICE
        for idx, range_bin in enumerate(range_bins):
            # Extract data for current range bin across all antennas and chirps
            range_data = radar_data[:, :, range_bin]  # Shape: (n_rx, n_chirps)

            # Compute sample covariance matrix
            sample_cov = range_data @ range_data.conj().T / n_chirps

            # Apply SPICE algorithm
            spice_estimator = select_spice_variant(variant, n_rx)

            try:
                power_spectrum, _ = spice_estimator.fit(sample_cov)
                spice_map[:, idx] = power_spectrum
            except Exception as e:
                warnings.warn(f"SPICE failed for range bin {range_bin}: {e}")
                # Fallback to conventional beamforming
                power_spectrum = self._fallback_beamforming(sample_cov, spice_estimator)
                spice_map[:, idx] = power_spectrum

        # Convert to dB scale
        spice_map_db = 10 * np.log10(spice_map + 1e-12)

        return spice_map_db

    def _fallback_beamforming(self, sample_cov: np.ndarray,
                             spice_estimator: SPICEEstimator) -> np.ndarray:
        """Fallback to conventional beamforming if SPICE fails."""
        power_spectrum = np.zeros(self.spice_config.grid_size)

        for i in range(self.spice_config.grid_size):
            steering_vec = spice_estimator.steering_vectors[:, i:i+1]
            power_spectrum[i] = np.real(
                steering_vec.conj().T @ sample_cov @ steering_vec
            ).item()

        return power_spectrum

    def detect_targets(self, rd_map: np.ndarray,
                      threshold_db: float = -20.0,
                      min_separation: Tuple[float, float] = (2.0, 5.0)) -> List[Dict]:
        """
        Detect targets in range-doppler map using peak detection.

        Parameters
        ----------
        rd_map : array_like
            Range-doppler map in dB.
        threshold_db : float, default=-20.0
            Detection threshold relative to peak in dB.
        min_separation : tuple of float, default=(2.0, 5.0)
            Minimum separation (doppler_bins, range_bins) between targets.

        Returns
        -------
        targets : list of dict
            Detected targets with estimated parameters:
            - 'range': Range in meters
            - 'velocity': Velocity in m/s
            - 'power_db': Signal power in dB
            - 'doppler_bin': Doppler bin index
            - 'range_bin': Range bin index
        """
        # Find peak power
        max_power = np.max(rd_map)
        threshold_absolute = max_power + threshold_db

        # Find potential target locations
        target_candidates = np.where(rd_map > threshold_absolute)
        doppler_indices = target_candidates[0]
        range_indices = target_candidates[1]

        # Apply peak detection with minimum separation
        targets = []
        used_locations = set()

        # Sort candidates by power (descending)
        powers = rd_map[doppler_indices, range_indices]
        sort_order = np.argsort(powers)[::-1]

        for idx in sort_order:
            dop_idx = doppler_indices[idx]
            rng_idx = range_indices[idx]

            # Check if this location is too close to existing targets
            too_close = False
            for used_dop, used_rng in used_locations:
                if (abs(dop_idx - used_dop) < min_separation[0] and
                    abs(rng_idx - used_rng) < min_separation[1]):
                    too_close = True
                    break

            if not too_close:
                # Convert indices to physical parameters
                target = self._indices_to_target_params(
                    dop_idx, rng_idx, rd_map[dop_idx, rng_idx]
                )
                targets.append(target)
                used_locations.add((dop_idx, rng_idx))

        return targets

    def _indices_to_target_params(self, doppler_idx: int, range_idx: int,
                                 power_db: float) -> Dict:
        """Convert indices to physical target parameters."""
        # Range calculation
        target_range = range_idx * self.config.range_resolution

        # Velocity calculation (with proper Doppler bin mapping)
        velocity_idx = doppler_idx - self.n_doppler_fft // 2  # Center at zero velocity
        target_velocity = velocity_idx * self.config.velocity_resolution

        return {
            'range': target_range,
            'velocity': target_velocity,
            'power_db': power_db,
            'doppler_bin': doppler_idx,
            'range_bin': range_idx
        }

    def compute_range_profile(self, radar_data: np.ndarray,
                             channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute range profile by coherent integration across Doppler.

        Parameters
        ----------
        radar_data : array_like
            Raw radar data.
        channel : int, default=0
            Receive channel to process.

        Returns
        -------
        range_profile : ndarray
            Range profile in dB.
        range_bins : ndarray
            Range bins in meters.
        """
        # Process single channel
        channel_data = radar_data[channel, :, :]

        # Apply windowing and range FFT
        windowed_data = channel_data * self.range_window[np.newaxis, :]
        range_fft = fft.fft(windowed_data, n=self.n_range_fft, axis=1)

        # Coherent integration across chirps
        range_profile = np.mean(np.abs(range_fft)**2, axis=0)

        # Convert to dB
        range_profile_db = 10 * np.log10(range_profile + 1e-12)

        # Range axis
        range_bins = np.arange(self.n_range_fft) * self.config.range_resolution

        return range_profile_db, range_bins

    def compute_doppler_profile(self, radar_data: np.ndarray,
                               range_bin: int, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Doppler profile at specific range bin.

        Parameters
        ----------
        radar_data : array_like
            Raw radar data.
        range_bin : int
            Range bin index to analyze.
        channel : int, default=0
            Receive channel to process.

        Returns
        -------
        doppler_profile : ndarray
            Doppler profile in dB.
        velocity_bins : ndarray
            Velocity bins in m/s.
        """
        # Extract data at specific range bin
        range_data = radar_data[channel, :, range_bin]

        # Apply windowing and Doppler FFT
        windowed_data = range_data * self.doppler_window
        doppler_fft = fft.fftshift(fft.fft(windowed_data, n=self.n_doppler_fft))

        # Convert to dB
        doppler_profile = 10 * np.log10(np.abs(doppler_fft)**2 + 1e-12)

        # Velocity axis
        velocity_bins = np.linspace(
            -self.config.max_velocity,
            self.config.max_velocity,
            self.n_doppler_fft
        )

        return doppler_profile, velocity_bins

    def estimate_target_parameters(self, radar_data: np.ndarray,
                                  target_location: Tuple[int, int]) -> Dict:
        """
        Estimate detailed target parameters using super-resolution techniques.

        Parameters
        ----------
        radar_data : array_like
            Raw radar data.
        target_location : tuple of int
            (doppler_bin, range_bin) of target.

        Returns
        -------
        parameters : dict
            Detailed target parameters including confidence intervals.
        """
        doppler_bin, range_bin = target_location

        # Extract data around target location
        doppler_window = slice(max(0, doppler_bin - 2), min(self.config.n_chirps, doppler_bin + 3))
        range_window = slice(max(0, range_bin - 2), min(len(radar_data[0, 0, :]), range_bin + 3))

        target_data = radar_data[:, doppler_window, range_window]

        # Use SPICE for parameter estimation
        spice_estimator = select_spice_variant('standard', self.config.n_rx)

        # Process each range-doppler cell
        refined_parameters = {
            'range_estimate': range_bin * self.config.range_resolution,
            'velocity_estimate': (doppler_bin - self.config.n_chirps // 2) * self.config.velocity_resolution,
            'power_estimate': 0.0,
            'confidence_interval': {'range': [0, 0], 'velocity': [0, 0]}
        }

        # Additional processing would go here for refined estimates
        # This is a placeholder for the full implementation

        return refined_parameters


def compare_processing_methods(radar_data: np.ndarray, config: FMCWRadarConfig,
                              ground_truth: List[Dict]) -> Dict:
    """
    Compare conventional FFT and SPICE processing methods.

    Parameters
    ----------
    radar_data : array_like
        Raw radar data.
    config : FMCWRadarConfig
        Radar configuration.
    ground_truth : list of dict
        True target parameters for validation.

    Returns
    -------
    comparison : dict
        Comparison of conventional and SPICE processing methods.
    """
    processor = RangeDopplerProcessor(config)

    # Conventional processing
    conventional_map = processor.process_conventional(radar_data)
    conventional_targets = processor.detect_targets(conventional_map)

    # SPICE processing
    spice_map = processor.process_spice(radar_data, variant='standard')
    spice_targets = processor.detect_targets(spice_map)

    # Compute performance metrics
    comparison = {
        'conventional': {
            'map': conventional_map,
            'targets': conventional_targets,
            'metrics': _compute_detection_metrics(conventional_targets, ground_truth)
        },
        'spice': {
            'map': spice_map,
            'targets': spice_targets,
            'metrics': _compute_detection_metrics(spice_targets, ground_truth)
        }
    }

    return comparison


def _compute_detection_metrics(detected_targets: List[Dict],
                              ground_truth: List[Dict]) -> Dict:
    """Compute detection performance metrics."""
    # Match detected targets to ground truth
    matched_pairs = []
    false_alarms = []
    missed_detections = ground_truth.copy()

    detection_threshold = {'range': 10.0, 'velocity': 2.0}  # Matching thresholds

    for detected in detected_targets:
        matched = False
        for idx, true_target in enumerate(missed_detections):
            range_error = abs(detected['range'] - true_target['range'])
            velocity_error = abs(detected['velocity'] - true_target['velocity'])

            if (range_error < detection_threshold['range'] and
                velocity_error < detection_threshold['velocity']):
                matched_pairs.append((detected, true_target))
                missed_detections.pop(idx)
                matched = True
                break

        if not matched:
            false_alarms.append(detected)

    # Compute metrics
    n_true_targets = len(ground_truth)
    n_detected = len(detected_targets)
    n_correct = len(matched_pairs)
    n_false_alarms = len(false_alarms)
    n_missed = len(missed_detections)

    metrics = {
        'detection_rate': n_correct / n_true_targets if n_true_targets > 0 else 0,
        'false_alarm_rate': n_false_alarms / n_detected if n_detected > 0 else 0,
        'precision': n_correct / n_detected if n_detected > 0 else 0,
        'recall': n_correct / n_true_targets if n_true_targets > 0 else 0,
        'n_true_targets': n_true_targets,
        'n_detected': n_detected,
        'n_correct': n_correct,
        'n_false_alarms': n_false_alarms,
        'n_missed': n_missed
    }

    return metrics
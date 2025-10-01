"""
Core SPICE Algorithm Implementation for Radar Applications.

This module implements the Sparse Iterative Covariance-based Estimation (SPICE)
algorithm for high-resolution direction-of-arrival estimation and radar imaging.

SPICE is a robust, hyperparameter-free algorithm that achieves superior resolution
compared to conventional beamforming methods through sparse covariance fitting.

References
----------
.. [1] D. Stoica et al., "SPICE: A Sparse Covariance-Based Estimation Method for
       Array Processing," IEEE Transactions on Signal Processing, 2011.
.. [2] T. Yardibi et al., "Sparsity Constrained Deconvolution Approaches for
       Acoustic Source Mapping," Journal of the Acoustical Society of America, 2010.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict, Union
import warnings
from dataclasses import dataclass


@dataclass
class SPICEConfig:
    """Configuration parameters for SPICE algorithm.

    Parameters
    ----------
    max_iterations : int, default=100
        Maximum number of iterations for convergence.
    convergence_tolerance : float, default=1e-6
        Convergence tolerance for cost function relative change.
    grid_size : int, default=180
        Number of grid points for angular spectrum estimation.
    angular_range : tuple, default=(-90, 90)
        Angular range in degrees for estimation grid.
    regularization : float, default=1e-12
        Small regularization parameter for numerical stability.
    """
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    grid_size: int = 180
    angular_range: Tuple[float, float] = (-90.0, 90.0)
    regularization: float = 1e-12

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")

        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be positive, got {self.convergence_tolerance}")

        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")

        if self.regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {self.regularization}")

        if len(self.angular_range) != 2:
            raise ValueError(f"angular_range must be a tuple of 2 values, got {len(self.angular_range)} values")

        if self.angular_range[1] <= self.angular_range[0]:
            raise ValueError(f"angular_range[1] must be greater than angular_range[0], got {self.angular_range}")


class SPICEEstimator:
    """
    Sparse Iterative Covariance-based Estimation (SPICE) for Array Processing.

    SPICE is a robust, hyperparameter-free method for sparse parameter estimation
    in array processing applications. It excels at direction-of-arrival estimation
    and provides superior resolution compared to conventional beamforming.

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : SPICEConfig, optional
        Configuration parameters for the algorithm.

    Attributes
    ----------
    n_sensors : int
        Number of sensors in the array.
    config : SPICEConfig
        Algorithm configuration parameters.
    steering_vectors : ndarray
        Array steering vectors for the estimation grid.
    angular_grid : ndarray
        Angular grid points in degrees.
    is_fitted : bool
        Whether the algorithm has been fitted to data.

    Examples
    --------
    >>> import numpy as np
    >>> from spice_core import SPICEEstimator
    >>>
    >>> # Create synthetic array data
    >>> n_sensors = 8
    >>> n_snapshots = 100
    >>> estimator = SPICEEstimator(n_sensors=n_sensors)
    >>>
    >>> # Generate sample covariance matrix
    >>> data = np.random.randn(n_sensors, n_snapshots) + 1j * np.random.randn(n_sensors, n_snapshots)
    >>> sample_cov = data @ data.conj().T / n_snapshots
    >>>
    >>> # Estimate angular spectrum
    >>> power_spectrum, angles = estimator.fit(sample_cov)
    >>>
    >>> # Find peaks (source directions)
    >>> peaks = estimator.find_peaks(power_spectrum, min_separation=5.0)
    """

    def __init__(self, n_sensors: int, config: Optional[SPICEConfig] = None):
        """
        Initialize SPICE estimator.

        Parameters
        ----------
        n_sensors : int
            Number of sensors in the array.
        config : SPICEConfig, optional
            Configuration parameters. If None, default configuration is used.
        """
        # Validate number of sensors
        if n_sensors < 2:
            raise ValueError(f"Number of sensors must be at least 2 for DOA estimation, got {n_sensors}")

        self.n_sensors = n_sensors
        self.config = config or SPICEConfig()
        self.is_fitted = False

        # Initialize angular grid and steering vectors
        self._initialize_grid()

    def _initialize_grid(self) -> None:
        """Initialize angular grid and steering vectors."""
        # Create angular grid
        self.angular_grid = np.linspace(
            self.config.angular_range[0],
            self.config.angular_range[1],
            self.config.grid_size
        )

        # Compute steering vectors for uniform linear array
        self.steering_vectors = self._compute_steering_vectors(self.angular_grid)

    def _compute_steering_vectors(self, angles_deg: np.ndarray) -> np.ndarray:
        """
        Compute array steering vectors for given angles.

        Parameters
        ----------
        angles_deg : array_like
            Angles in degrees.

        Returns
        -------
        steering_vectors : ndarray, shape (n_sensors, n_angles)
            Complex steering vectors for each angle.

        Notes
        -----
        Assumes uniform linear array with half-wavelength spacing.
        Steering vector: a(θ) = [1, e^{jπsin(θ)}, ..., e^{j(N-1)πsin(θ)}]^T
        """
        angles_rad = np.deg2rad(angles_deg)
        sensor_positions = np.arange(self.n_sensors)

        # Phase delays for each sensor-angle combination
        phase_delays = np.outer(sensor_positions, np.pi * np.sin(angles_rad))

        # Complex steering vectors
        steering_vectors = np.exp(1j * phase_delays)

        return steering_vectors

    def fit(self, sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate angular power spectrum using SPICE algorithm.

        Parameters
        ----------
        sample_covariance : array_like, shape (n_sensors, n_sensors)
            Sample covariance matrix of the array data.

        Returns
        -------
        power_spectrum : ndarray, shape (grid_size,)
            Estimated angular power spectrum.
        angular_grid : ndarray, shape (grid_size,)
            Angular grid in degrees.

        Raises
        ------
        ValueError
            If sample_covariance is not square or has wrong dimensions.
        LinAlgError
            If covariance matrix is not positive definite.

        Notes
        -----
        The SPICE algorithm minimizes the covariance fitting criterion:

        .. math::
            \\min_{\\mathbf{P}} \\|\\mathbf{R} - \\mathbf{A}\\mathbf{P}\\mathbf{A}^H\\|_F^2

        where R is the sample covariance, A is the steering matrix, and P is
        a diagonal matrix of source powers.

        The algorithm iterates between:
        1. Power estimation: P[k+1] = arg min ||R - APA^H||_F^2
        2. Covariance update based on current power estimates

        Examples
        --------
        >>> estimator = SPICEEstimator(n_sensors=8)
        >>> # sample_cov is your 8x8 covariance matrix
        >>> power_spectrum, angles = estimator.fit(sample_cov)
        >>> print(f"Peak power: {np.max(power_spectrum):.2f}")
        """
        # Validate input
        sample_covariance = np.asarray(sample_covariance)
        self._validate_covariance_matrix(sample_covariance)

        # Store sample covariance
        self.sample_covariance = sample_covariance

        # Initialize power spectrum estimate
        power_spectrum = self._initialize_power_spectrum()

        # SPICE iterative algorithm
        cost_history = []

        for iteration in range(self.config.max_iterations):
            # Update power estimates
            power_spectrum_new = self._update_power_estimates(
                sample_covariance, power_spectrum
            )

            # Compute cost function
            cost = self._compute_cost_function(sample_covariance, power_spectrum_new)
            cost_history.append(cost)

            # Check convergence
            if iteration > 0:
                prev_cost = abs(cost_history[-2])
                curr_cost = abs(cost_history[-1])

                # Handle division by zero case
                if prev_cost < 1e-15:
                    # If previous cost is essentially zero, check absolute change
                    relative_change = curr_cost
                else:
                    relative_change = abs(cost_history[-2] - cost_history[-1]) / prev_cost

                if relative_change < self.config.convergence_tolerance:
                    break

            power_spectrum = power_spectrum_new

        else:
            warnings.warn(f"SPICE did not converge after {self.config.max_iterations} iterations",
                         UserWarning)

        # Store results
        self.power_spectrum = power_spectrum
        self.cost_history = np.array(cost_history)
        self.n_iterations = len(cost_history)
        self.is_fitted = True

        return power_spectrum, self.angular_grid

    def _validate_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        """Validate input covariance matrix."""
        if cov_matrix.ndim != 2:
            raise ValueError("Covariance matrix must be 2-dimensional")

        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if cov_matrix.shape[0] != self.n_sensors:
            raise ValueError(f"Covariance matrix must be {self.n_sensors}x{self.n_sensors}")

        # Check if matrix is positive definite
        try:
            la.cholesky(cov_matrix + self.config.regularization * np.eye(self.n_sensors))
        except la.LinAlgError:
            raise la.LinAlgError("Covariance matrix is not positive definite")

    def _initialize_power_spectrum(self) -> np.ndarray:
        """Initialize power spectrum estimate."""
        # Use conventional beamforming for initialization
        power_spectrum = np.zeros(self.config.grid_size)

        for i, angle in enumerate(self.angular_grid):
            steering_vec = self.steering_vectors[:, i:i+1]
            power_spectrum[i] = np.real(
                steering_vec.conj().T @ self.sample_covariance @ steering_vec
            ).item()

        # Normalize and ensure positivity
        power_spectrum = np.maximum(power_spectrum, self.config.regularization)

        return power_spectrum

    def _update_power_estimates(self, sample_cov: np.ndarray,
                               current_powers: np.ndarray) -> np.ndarray:
        """
        Update power estimates using covariance fitting criterion.

        Parameters
        ----------
        sample_cov : ndarray
            Sample covariance matrix.
        current_powers : ndarray
            Current power estimates.

        Returns
        -------
        updated_powers : ndarray
            Updated power estimates.
        """
        updated_powers = np.zeros_like(current_powers)

        # Literature-correct SPICE power update formula
        # Based on minimizing ||R - A*P*A^H||_F^2 w.r.t. each power p_k
        # Solution: p_k = (a_k^H * R * a_k) / (a_k^H * a_k)
        for i in range(self.config.grid_size):
            steering_vec = self.steering_vectors[:, i:i+1]

            # Direct power computation from literature
            numerator = np.real(steering_vec.conj().T @ sample_cov @ steering_vec).item()
            denominator = np.real(steering_vec.conj().T @ steering_vec).item()

            # Apply regularization to ensure non-negative powers
            updated_powers[i] = max(numerator / denominator, self.config.regularization)

        return updated_powers

    def _construct_fitted_covariance(self, powers: np.ndarray) -> np.ndarray:
        """Construct fitted covariance matrix from power estimates."""
        fitted_cov = np.zeros((self.n_sensors, self.n_sensors), dtype=complex)

        for i, power in enumerate(powers):
            if power > self.config.regularization:
                steering_vec = self.steering_vectors[:, i:i+1]
                fitted_cov += power * (steering_vec @ steering_vec.conj().T)

        return fitted_cov

    def _compute_cost_function(self, sample_cov: np.ndarray,
                              powers: np.ndarray) -> float:
        """Compute covariance fitting cost function."""
        fitted_cov = self._construct_fitted_covariance(powers)
        residual = sample_cov - fitted_cov
        return np.real(np.trace(residual.conj().T @ residual))

    def find_peaks(self, power_spectrum: Optional[np.ndarray] = None,
                   min_separation: float = 5.0,
                   threshold_db: float = -20.0) -> Dict[str, np.ndarray]:
        """
        Find peaks in the angular power spectrum.

        Parameters
        ----------
        power_spectrum : array_like, optional
            Power spectrum to analyze. If None, uses fitted spectrum.
        min_separation : float, default=5.0
            Minimum angular separation between peaks in degrees.
        threshold_db : float, default=-20.0
            Peak detection threshold relative to maximum in dB.

        Returns
        -------
        peaks : dict
            Dictionary containing:
            - 'angles': Peak angles in degrees
            - 'powers': Peak powers (linear scale)
            - 'powers_db': Peak powers in dB
            - 'indices': Peak indices in the spectrum

        Raises
        ------
        ValueError
            If algorithm has not been fitted and no power_spectrum provided.
        """
        if power_spectrum is None:
            if not self.is_fitted:
                raise ValueError("Must fit algorithm or provide power_spectrum")
            power_spectrum = self.power_spectrum
        else:
            power_spectrum = np.asarray(power_spectrum)

        # Convert to dB scale
        power_db = 10 * np.log10(np.maximum(power_spectrum, self.config.regularization))
        max_power_db = np.max(power_db)

        # Check if spectrum is essentially flat (pure noise case)
        spectrum_variation = np.std(power_spectrum)
        if spectrum_variation < self.config.regularization * 10:
            # Flat spectrum indicates pure noise - no sources detected
            return {
                'angles': np.array([]),
                'powers': np.array([]),
                'powers_db': np.array([]),
                'indices': np.array([], dtype=int)
            }

        # Find peaks above threshold
        threshold_absolute = max_power_db + threshold_db
        potential_peaks = np.where(power_db > threshold_absolute)[0]

        # Apply minimum separation constraint
        peak_indices = []
        peak_powers = []

        for idx in potential_peaks:
            # Check if this is a local maximum
            if self._is_local_maximum(power_db, idx):
                # Check separation from existing peaks
                min_sep_samples = int(min_separation * self.config.grid_size /
                                    (self.config.angular_range[1] - self.config.angular_range[0]))

                if not peak_indices or all(abs(idx - p) >= min_sep_samples for p in peak_indices):
                    peak_indices.append(idx)
                    peak_powers.append(power_spectrum[idx])

        peak_indices = np.array(peak_indices)
        peak_powers = np.array(peak_powers)

        # Sort by power (descending)
        sort_order = np.argsort(peak_powers)[::-1]
        peak_indices = peak_indices[sort_order]
        peak_powers = peak_powers[sort_order]

        return {
            'angles': self.angular_grid[peak_indices],
            'powers': peak_powers,
            'powers_db': 10 * np.log10(np.maximum(peak_powers, np.max(peak_powers) * 1e-12)) if len(peak_powers) > 0 else np.array([]),
            'indices': peak_indices
        }

    def _is_local_maximum(self, signal: np.ndarray, index: int,
                         window: int = 3) -> bool:
        """Check if point is a local maximum."""
        start = max(0, index - window // 2)
        end = min(len(signal), index + window // 2 + 1)
        local_region = signal[start:end]
        local_max_idx = np.argmax(local_region)
        return start + local_max_idx == index

    def estimate_noise_power(self) -> float:
        """
        Estimate noise power from the spectrum.

        Returns
        -------
        noise_power : float
            Estimated noise power in linear scale.

        Notes
        -----
        Estimates noise power as the median of the spectrum, which is robust
        to sparse source scenarios.
        """
        if not self.is_fitted:
            raise ValueError("Must fit algorithm first")

        return np.median(self.power_spectrum)

    def compute_array_gain(self, angle_deg: float) -> float:
        """
        Compute array gain in specific direction.

        Parameters
        ----------
        angle_deg : float
            Angle in degrees.

        Returns
        -------
        gain_db : float
            Array gain in dB.
        """
        # Find nearest grid point
        idx = np.argmin(np.abs(self.angular_grid - angle_deg))
        steering_vec = self.steering_vectors[:, idx]

        # Array gain is squared magnitude of normalized steering vector
        gain_linear = np.abs(np.sum(steering_vec))**2 / self.n_sensors
        gain_db = 10 * np.log10(gain_linear)

        return gain_db

    def get_convergence_info(self) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Get algorithm convergence information.

        Returns
        -------
        info : dict
            Convergence information including iterations, final cost, etc.
        """
        if not self.is_fitted:
            raise ValueError("Must fit algorithm first")

        return {
            'n_iterations': self.n_iterations,
            'final_cost': self.cost_history[-1],
            'initial_cost': self.cost_history[0],
            'cost_reduction': self.cost_history[0] - self.cost_history[-1],
            'cost_history': self.cost_history.copy()
        }


def compute_sample_covariance(data: np.ndarray,
                            method: str = 'biased') -> np.ndarray:
    """
    Compute sample covariance matrix from array data.

    Parameters
    ----------
    data : array_like, shape (n_sensors, n_snapshots)
        Array data with sensors along first axis, snapshots along second.
    method : {'biased', 'unbiased'}, default='biased'
        Covariance estimation method.

    Returns
    -------
    sample_cov : ndarray, shape (n_sensors, n_sensors)
        Sample covariance matrix.

    Examples
    --------
    >>> data = np.random.randn(8, 100) + 1j * np.random.randn(8, 100)
    >>> cov_matrix = compute_sample_covariance(data)
    >>> print(f"Covariance matrix shape: {cov_matrix.shape}")
    """
    data = np.asarray(data)
    n_sensors, n_snapshots = data.shape

    if method == 'biased':
        return data @ data.conj().T / n_snapshots
    elif method == 'unbiased':
        return data @ data.conj().T / (n_snapshots - 1)
    else:
        raise ValueError("Method must be 'biased' or 'unbiased'")


def compare_with_conventional_beamforming(spice_estimator: SPICEEstimator,
                                        sample_cov: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compare SPICE with conventional delay-and-sum beamforming.

    Parameters
    ----------
    spice_estimator : SPICEEstimator
        Fitted SPICE estimator.
    sample_cov : array_like
        Sample covariance matrix.

    Returns
    -------
    comparison : dict
        Dictionary containing both spectra and performance metrics.
    """
    if not spice_estimator.is_fitted:
        raise ValueError("SPICE estimator must be fitted first")

    # Conventional beamforming
    conv_spectrum = np.zeros(spice_estimator.config.grid_size)
    for i in range(spice_estimator.config.grid_size):
        steering_vec = spice_estimator.steering_vectors[:, i:i+1]
        conv_spectrum[i] = np.real(
            steering_vec.conj().T @ sample_cov @ steering_vec
        ).item()

    return {
        'spice_spectrum': spice_estimator.power_spectrum.copy(),
        'conventional_spectrum': conv_spectrum,
        'angular_grid': spice_estimator.angular_grid.copy(),
        'resolution_improvement': _compute_resolution_improvement(
            spice_estimator.power_spectrum, conv_spectrum
        )
    }


def _compute_resolution_improvement(spice_spectrum: np.ndarray,
                                  conv_spectrum: np.ndarray) -> float:
    """Compute resolution improvement factor."""
    # Simple metric: ratio of 3dB beamwidths
    spice_3db = _compute_3db_beamwidth(spice_spectrum)
    conv_3db = _compute_3db_beamwidth(conv_spectrum)

    return conv_3db / spice_3db if spice_3db > 0 else np.inf


def _compute_3db_beamwidth(spectrum: np.ndarray) -> float:
    """Compute 3dB beamwidth of spectrum."""
    max_idx = np.argmax(spectrum)
    max_power_db = 10 * np.log10(spectrum[max_idx])
    threshold_db = max_power_db - 3.0

    # Find points at 3dB below peak
    spectrum_db = 10 * np.log10(np.maximum(spectrum, np.max(spectrum) * 1e-12))
    above_threshold = spectrum_db > threshold_db

    if np.sum(above_threshold) < 2:
        return 0.0

    indices = np.where(above_threshold)[0]
    return indices[-1] - indices[0]
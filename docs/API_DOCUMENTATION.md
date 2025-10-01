# SPICE Radar API Documentation

## Table of Contents
1. [Core SPICE Algorithm](#core-spice-algorithm)
2. [SPICE Variants](#spice-variants)
3. [Range-Doppler Imaging](#range-doppler-imaging)
4. [Visualization Utilities](#visualization-utilities)
5. [Testing Framework](#testing-framework)
6. [Examples and Tutorials](#examples-and-tutorials)

---

## Core SPICE Algorithm

### `spice_core.SPICEEstimator`

The main class implementing the Sparse Iterative Covariance-based Estimation algorithm.

#### Constructor
```python
SPICEEstimator(n_sensors: int, config: Optional[SPICEConfig] = None)
```

**Parameters:**
- `n_sensors` (int): Number of sensors in the array
- `config` (SPICEConfig, optional): Algorithm configuration parameters

**Example:**
```python
from spice_core import SPICEEstimator, SPICEConfig

# Basic usage
estimator = SPICEEstimator(n_sensors=8)

# Advanced configuration
config = SPICEConfig(max_iterations=200, convergence_tolerance=1e-8)
estimator = SPICEEstimator(n_sensors=8, config=config)
```

#### Methods

##### `fit(sample_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Estimate angular power spectrum using SPICE algorithm.

**Parameters:**
- `sample_covariance` (ndarray): Sample covariance matrix, shape (n_sensors, n_sensors)

**Returns:**
- `power_spectrum` (ndarray): Estimated angular power spectrum
- `angular_grid` (ndarray): Angular grid in degrees

**Example:**
```python
import numpy as np

# Generate sample covariance matrix
data = np.random.randn(8, 100) + 1j * np.random.randn(8, 100)
sample_cov = data @ data.conj().T / 100

# Estimate spectrum
power_spectrum, angles = estimator.fit(sample_cov)
```

##### `find_peaks(power_spectrum: Optional[np.ndarray] = None, min_separation: float = 5.0, threshold_db: float = -20.0) -> Dict[str, np.ndarray]`

Find peaks in the angular power spectrum.

**Parameters:**
- `power_spectrum` (ndarray, optional): Power spectrum to analyze
- `min_separation` (float): Minimum angular separation between peaks in degrees
- `threshold_db` (float): Peak detection threshold relative to maximum in dB

**Returns:**
- Dictionary containing:
  - `'angles'`: Peak angles in degrees
  - `'powers'`: Peak powers (linear scale)
  - `'powers_db'`: Peak powers in dB
  - `'indices'`: Peak indices in the spectrum

**Example:**
```python
peaks = estimator.find_peaks(power_spectrum, min_separation=3.0, threshold_db=-15.0)
print(f"Detected {len(peaks['angles'])} sources at angles: {peaks['angles']}")
```

##### `get_convergence_info() -> Dict[str, Union[int, float, np.ndarray]]`

Get algorithm convergence information.

**Returns:**
- Dictionary containing convergence metrics

**Example:**
```python
conv_info = estimator.get_convergence_info()
print(f"Converged in {conv_info['n_iterations']} iterations")
print(f"Final cost: {conv_info['final_cost']:.2e}")
```

### `spice_core.SPICEConfig`

Configuration dataclass for SPICE algorithm parameters.

**Attributes:**
```python
@dataclass
class SPICEConfig:
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    grid_size: int = 180
    angular_range: Tuple[float, float] = (-90.0, 90.0)
    regularization: float = 1e-12
```

### Utility Functions

##### `compute_sample_covariance(data: np.ndarray, method: str = 'biased') -> np.ndarray`

Compute sample covariance matrix from array data.

**Parameters:**
- `data` (ndarray): Array data, shape (n_sensors, n_snapshots)
- `method` (str): 'biased' or 'unbiased' estimation

**Returns:**
- `sample_cov` (ndarray): Sample covariance matrix

---

## SPICE Variants

### `spice_variants.WeightedSPICEEstimator`

**Research Framework**: Weighted SPICE implementation framework.

#### Constructor
```python
WeightedSPICEEstimator(n_sensors: int, config: Optional[WeightedSPICEConfig] = None)
```

#### Framework Features
- Adaptive weighting structure for research
- Configurable weighting method framework
- Educational implementation of weighting concepts

**Important Note**: Current implementation provides basic weighting functionality and serves as a foundation for research into adaptive weighting strategies.

**Example:**
```python
from spice_variants import WeightedSPICEEstimator, WeightedSPICEConfig

# Research framework usage
config = WeightedSPICEConfig(weighting_method='adaptive')
estimator_weighted = WeightedSPICEEstimator(n_sensors=8, config=config)
```

### `spice_variants.FastSPICEEstimator`

**Research Framework**: Optimized SPICE implementation framework.

#### Constructor
```python
FastSPICEEstimator(n_sensors: int, config: Optional[SPICEConfig] = None)
```

#### Framework Features
- Structure for exploiting Toeplitz properties
- Framework for computational optimization research
- Foundation for real-time application development

**Important Note**: Current implementation provides basic structure for optimization research.

### Variant Selection

##### `select_spice_variant(scenario: str, n_sensors: int, **kwargs) -> SPICEEstimator`

Factory function to select SPICE implementation variant.

**Parameters:**
- `scenario` (str): 'standard', 'fast', 'weighted', 'quantized'
- `n_sensors` (int): Number of sensors
- `**kwargs`: Additional configuration parameters

**Important Note**: Advanced variants currently provide research frameworks. For validated performance, use 'standard' scenario.

**Example:**
```python
from spice_variants import select_spice_variant

# For validated implementation
estimator = select_spice_variant('standard', n_sensors=8)

# For research frameworks
estimator = select_spice_variant('weighted', n_sensors=8)  # Research framework
```

---

## Range-Doppler Imaging

### `range_doppler_imaging.FMCWRadarConfig`

Configuration for FMCW radar system parameters.

```python
@dataclass
class FMCWRadarConfig:
    f_start: float = 24e9      # Start frequency (Hz)
    bandwidth: float = 1e9     # Bandwidth (Hz)
    chirp_duration: float = 1e-3  # Chirp duration (s)
    n_chirps: int = 128        # Number of chirps
    fs: float = 10e6          # Sampling frequency (Hz)
    n_rx: int = 4             # Number of receive antennas
    c: float = 3e8            # Speed of light (m/s)
```

**Properties:**
- `range_resolution`: Range resolution in meters
- `max_range`: Maximum unambiguous range
- `velocity_resolution`: Velocity resolution in m/s
- `max_velocity`: Maximum unambiguous velocity

### `range_doppler_imaging.FMCWSignalGenerator`

Generate realistic FMCW radar signals for algorithm validation.

#### Constructor
```python
FMCWSignalGenerator(config: FMCWRadarConfig)
```

#### Methods

##### `generate_radar_data(targets: List[Dict], snr_db: float = 20.0, clutter_power_db: float = -30.0, add_interference: bool = False) -> np.ndarray`

Generate realistic FMCW radar data with multiple targets.

**Parameters:**
- `targets` (list): Target dictionaries with 'range', 'velocity', 'rcs', 'angle'
- `snr_db` (float): Signal-to-noise ratio in dB
- `clutter_power_db` (float): Clutter power relative to signal in dB
- `add_interference` (bool): Whether to add interference

**Returns:**
- `radar_data` (ndarray): Complex radar data cube, shape (n_rx, n_chirps, n_samples)

**Example:**
```python
from range_doppler_imaging import FMCWRadarConfig, FMCWSignalGenerator

config = FMCWRadarConfig()
generator = FMCWSignalGenerator(config)

targets = [
    {'range': 100, 'velocity': 15, 'rcs': 1.0, 'angle': 0},
    {'range': 200, 'velocity': -10, 'rcs': 0.5, 'angle': 20}
]

radar_data = generator.generate_radar_data(targets, snr_db=20)
```

### `range_doppler_imaging.RangeDopplerProcessor`

Process FMCW radar data for range-doppler imaging with SPICE enhancement.

#### Constructor
```python
RangeDopplerProcessor(config: FMCWRadarConfig, spice_config: Optional[SPICEConfig] = None)
```

#### Methods

##### `process_conventional(radar_data: np.ndarray, channel: int = 0) -> np.ndarray`

Process radar data using conventional 2D FFT method.

**Parameters:**
- `radar_data` (ndarray): Raw radar data cube
- `channel` (int): Receive channel to process

**Returns:**
- `rd_map` (ndarray): Range-doppler map in dB

##### `process_spice(radar_data: np.ndarray, variant: str = 'standard', range_bins: Optional[List[int]] = None) -> np.ndarray`

Process radar data using SPICE algorithm for enhanced resolution.

**Parameters:**
- `radar_data` (ndarray): Raw radar data cube
- `variant` (str): SPICE variant to use
- `range_bins` (list, optional): Specific range bins to process

**Returns:**
- `spice_map` (ndarray): SPICE-enhanced range-doppler map

##### `detect_targets(rd_map: np.ndarray, threshold_db: float = -20.0, min_separation: Tuple[float, float] = (2.0, 5.0)) -> List[Dict]`

Detect targets in range-doppler map using peak detection.

**Parameters:**
- `rd_map` (ndarray): Range-doppler map in dB
- `threshold_db` (float): Detection threshold relative to peak in dB
- `min_separation` (tuple): Minimum separation (doppler_bins, range_bins)

**Returns:**
- `targets` (list): Detected targets with parameters

**Example:**
```python
from range_doppler_imaging import RangeDopplerProcessor

processor = RangeDopplerProcessor(config)

# Conventional processing
conventional_map = processor.process_conventional(radar_data)

# SPICE-enhanced processing
spice_map = processor.process_spice(radar_data, variant='standard')

# Target detection
targets = processor.detect_targets(spice_map, threshold_db=-15.0)
```

---

## Visualization Utilities

### `visualization_utils.ProfessionalPlotter`

Professional plotting utilities for radar signal processing visualization.

#### Constructor
```python
ProfessionalPlotter(style: str = 'professional', color_scheme: str = 'viridis', save_format: str = 'png', dpi: int = 300)
```

**Parameters:**
- `style` (str): 'professional', 'academic', 'presentation', 'github'
- `color_scheme` (str): 'viridis', 'plasma', 'jet', 'custom'
- `save_format` (str): Default save format
- `dpi` (int): Resolution for saved figures

#### Methods

##### `plot_angular_spectrum(spectrum: np.ndarray, angles: np.ndarray, peaks: Optional[Dict] = None, comparison_spectrum: Optional[np.ndarray] = None, title: str = "Angular Spectrum Analysis", save_path: Optional[str] = None) -> plt.Figure`

Plot angular spectrum with professional formatting.

##### `plot_range_doppler_map(rd_map: np.ndarray, range_bins: np.ndarray, velocity_bins: np.ndarray, targets: Optional[List[Dict]] = None, title: str = "Range-Doppler Map", dynamic_range_db: float = 60.0, save_path: Optional[str] = None) -> plt.Figure`

Plot professional range-doppler map.

##### `plot_snr_performance(snr_range: np.ndarray, performance_data: Dict[str, List], metrics: List[str] = ['detection_rate', 'rmse_angle'], title: str = "SNR Performance Analysis", save_path: Optional[str] = None) -> plt.Figure`

Plot SNR performance curves for multiple algorithms.

**Example:**
```python
from visualization_utils import ProfessionalPlotter

plotter = ProfessionalPlotter(style='github')

# Plot angular spectrum
fig1 = plotter.plot_angular_spectrum(power_spectrum, angles, peaks=peaks)

# Plot range-doppler map
fig2 = plotter.plot_range_doppler_map(rd_map, range_bins, velocity_bins, targets=targets)

# Save figures
plotter.save_figure(fig1, 'angular_spectrum.png')
plotter.save_figure(fig2, 'range_doppler_map.png')
```

---

## Testing Framework

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_spice_core.py -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run performance tests
pytest test_integration.py::TestSPICEIntegration::test_snr_performance_curve -v
```

### Test Categories

1. **Unit Tests** (`test_spice_core.py`)
   - Core SPICE algorithm functionality
   - Individual component testing
   - Edge case validation

2. **Signal Generation Tests** (`test_signal_generation.py`)
   - FMCW signal generation
   - Target simulation
   - Noise modeling

3. **Integration Tests** (`test_integration.py`)
   - End-to-end pipeline testing
   - Performance analysis
   - Algorithm comparison

### Custom Test Fixtures

```python
import pytest
from spice_core import SPICEEstimator

@pytest.fixture
def sample_radar_data():
    """Generate sample radar data for testing."""
    # Implementation details...

@pytest.fixture
def known_targets():
    """Define known target parameters for testing."""
    return {
        'ranges': np.array([75, 150, 225]),
        'velocities': np.array([10, -15, 5]),
        'rcs': np.array([1.0, 0.5, 2.0])
    }
```

---

## Examples and Tutorials

### Basic SPICE Usage

```python
import numpy as np
from spice_core import SPICEEstimator

# Generate synthetic data
n_sensors = 8
n_snapshots = 100
angles = np.array([-20, 0, 25])  # True source angles
powers = np.array([1.0, 0.8, 1.2])  # Source powers

# Create steering matrix
steering_matrix = np.exp(1j * np.pi * np.sin(np.deg2rad(angles))[:, np.newaxis] * np.arange(n_sensors))

# Generate source signals
source_signals = np.sqrt(powers)[:, np.newaxis] * (
    np.random.randn(len(angles), n_snapshots) +
    1j * np.random.randn(len(angles), n_snapshots)
)

# Received signals
received_signals = steering_matrix.T @ source_signals

# Add noise
snr_db = 20
signal_power = np.mean(np.abs(received_signals)**2)
noise_power = signal_power / (10**(snr_db/10))
noise = np.sqrt(noise_power/2) * (
    np.random.randn(n_sensors, n_snapshots) +
    1j * np.random.randn(n_sensors, n_snapshots)
)
received_signals += noise

# Compute sample covariance
sample_cov = received_signals @ received_signals.conj().T / n_snapshots

# Apply SPICE
estimator = SPICEEstimator(n_sensors)
power_spectrum, angular_grid = estimator.fit(sample_cov)
peaks = estimator.find_peaks(power_spectrum)

print(f"True angles: {angles}")
print(f"Detected angles: {peaks['angles']}")
```

### Complete FMCW Radar Example

```python
from range_doppler_imaging import (
    FMCWRadarConfig, FMCWSignalGenerator, RangeDopplerProcessor
)
from visualization_utils import ProfessionalPlotter

# Configure radar
config = FMCWRadarConfig(
    f_start=24e9,
    bandwidth=1e9,
    n_chirps=128,
    n_rx=4
)

# Define targets
targets = [
    {'range': 100, 'velocity': 15, 'rcs': 1.0, 'angle': -10},
    {'range': 200, 'velocity': -8, 'rcs': 0.5, 'angle': 5}
]

# Generate radar data
generator = FMCWSignalGenerator(config)
radar_data = generator.generate_radar_data(targets, snr_db=20)

# Process data
processor = RangeDopplerProcessor(config)
conventional_map = processor.process_conventional(radar_data)
spice_map = processor.process_spice(radar_data, variant='standard')

# Detect targets
conv_targets = processor.detect_targets(conventional_map)
spice_targets = processor.detect_targets(spice_map)

# Visualize results
plotter = ProfessionalPlotter()
fig1 = plotter.plot_range_doppler_map(
    conventional_map,
    np.arange(config.max_range/conventional_map.shape[1] * conventional_map.shape[1]),
    np.linspace(-config.max_velocity, config.max_velocity, conventional_map.shape[0]),
    targets=conv_targets,
    title="Conventional Range-Doppler Processing"
)

fig2 = plotter.plot_range_doppler_map(
    spice_map,
    np.arange(config.max_range/spice_map.shape[1] * spice_map.shape[1]),
    np.linspace(-config.max_velocity, config.max_velocity, spice_map.shape[0]),
    targets=spice_targets,
    title="SPICE-Enhanced Range-Doppler Processing"
)

print(f"Conventional method detected {len(conv_targets)} targets")
print(f"SPICE method detected {len(spice_targets)} targets")
```

### Performance Analysis Example

```python
from spice_demonstration import SPICEDemonstrator

# Create demonstrator
demonstrator = SPICEDemonstrator(save_figures=True)

# Run comprehensive analysis
results = demonstrator.demonstrate_all()

# Access specific results
snr_analysis = results['snr_analysis']
failure_threshold = snr_analysis['failure_threshold']['spice']
print(f"SPICE failure threshold: {failure_threshold} dB")

# Super-resolution analysis
resolution_results = results['super_resolution']
avg_resolution = np.mean(resolution_results['resolution_performance']['spice'])
print(f"Average resolution score: {avg_resolution:.2f}")
```

---

## Error Handling and Best Practices

### Common Error Scenarios

1. **Singular Covariance Matrix**
```python
try:
    power_spectrum, angles = estimator.fit(sample_cov)
except np.linalg.LinAlgError as e:
    print(f"Covariance matrix is not positive definite: {e}")
    # Add regularization
    regularized_cov = sample_cov + 1e-10 * np.eye(sample_cov.shape[0])
    power_spectrum, angles = estimator.fit(regularized_cov)
```

2. **Insufficient Data**
```python
if n_snapshots < 2 * n_sensors:
    warnings.warn("Insufficient snapshots for reliable covariance estimation")
```

3. **Low SNR Scenarios**
```python
# Check SNR before applying SPICE
estimated_snr = estimate_snr(radar_data)
if estimated_snr < 0:  # Below 0 dB
    print("Warning: Low SNR detected. SPICE may fail.")
    # Use conventional processing as fallback
    result = processor.process_conventional(radar_data)
else:
    result = processor.process_spice(radar_data)
```

### Performance Optimization Tips

1. **Use Fast Variants for Real-time Applications**
```python
# For real-time processing
estimator = select_spice_variant('fast', n_sensors)
```

2. **Limit Range Bins for Computational Efficiency**
```python
# Process only relevant range bins
relevant_bins = list(range(50, 150))  # 50-150m range
spice_map = processor.process_spice(radar_data, range_bins=relevant_bins)
```

3. **Configure Appropriate Grid Size**
```python
# Balance resolution vs computation
config = SPICEConfig(grid_size=90)  # Reduce from default 180
estimator = SPICEEstimator(n_sensors, config)
```

---

This API documentation provides comprehensive coverage of all major classes, methods, and usage patterns in the SPICE radar implementation. For additional examples and advanced usage, refer to the demonstration scripts and test files.
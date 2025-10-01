"""
SPICE - Sparse Iterative Covariance-based Estimation

A comprehensive implementation of SPICE for radar signal processing with
advanced enhancements for stability, performance, and educational value.
"""

__version__ = "1.0.0"
__author__ = "SPICE Implementation Team"

# Core SPICE implementations
from .spice_core import SPICEEstimator, SPICEConfig
from .spice_enhanced import EnhancedSPICEEstimator, create_enhanced_spice
from .spice_stable import StableSPICEEstimator
from .spice_variants import WeightedSPICEEstimator, FastSPICEEstimator

# Advanced algorithms
from .advanced_peak_detection import (
    detect_peaks_advanced,
    PhysicalConstraints,
    PeakDetectionConfig
)
from .improved_enhanced_spice import ImprovedEnhancedSPICE
from .enhanced_snr_estimation import estimate_snr_enhanced, SNREstimationMethod

# Signal processing utilities
from .coprime_signal_design import generate_coprime_signal, CoprimeConfig
from .range_doppler_imaging import RangeDopplerProcessor
from .visualization_utils import plot_spectrum, plot_range_doppler

__all__ = [
    # Core estimators
    'SPICEEstimator',
    'SPICEConfig',
    'EnhancedSPICEEstimator',
    'create_enhanced_spice',
    'StableSPICEEstimator',
    'WeightedSPICEEstimator',
    'FastSPICEEstimator',
    'ImprovedEnhancedSPICE',

    # Peak detection
    'detect_peaks_advanced',
    'PhysicalConstraints',
    'PeakDetectionConfig',

    # Signal processing
    'generate_coprime_signal',
    'CoprimeConfig',
    'RangeDopplerProcessor',

    # SNR estimation
    'estimate_snr_enhanced',
    'SNREstimationMethod',

    # Visualization
    'plot_spectrum',
    'plot_range_doppler',
]

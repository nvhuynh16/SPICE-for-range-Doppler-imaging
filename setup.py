"""
Setup script for SPICE Range-Doppler Imaging Package.

Professional implementation of Sparse Iterative Covariance-based Estimation
for FMCW radar applications with comprehensive testing and documentation.

Author: Professional Radar Engineer
Date: 2025
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f.readlines()
                       if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pytest>=6.0.0",
        "pytest-cov>=3.0.0"
    ]

# Extract core requirements (exclude optional and development deps)
core_requirements = [req for req in requirements if not any(
    optional in req for optional in ['numba', 'jupyter', 'black', 'flake8', 'mypy', 'sphinx']
)]

setup(
    name="spice-radar",
    version="1.0.0",
    author="Professional Radar Engineer",
    author_email="professional.radar.engineer@email.com",
    description="Professional SPICE implementation for range-doppler radar imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/radar-engineer/spice-radar",
    project_urls={
        "Bug Reports": "https://github.com/radar-engineer/spice-radar/issues",
        "Source": "https://github.com/radar-engineer/spice-radar",
        "Documentation": "https://github.com/radar-engineer/spice-radar/wiki",
    },

    # Package discovery
    packages=find_packages(),
    py_modules=[
        "spice_core",
        "spice_variants",
        "range_doppler_imaging",
        "spice_demonstration",
        "visualization_utils"
    ],

    # Dependencies
    install_requires=core_requirements,

    # Optional dependencies
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0"
        ],
        "performance": [
            "numba>=0.56.0",
            "scikit-learn>=1.0.0"
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "kaleido>=0.2.1"
        ],
        "all": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "numba>=0.56.0",
            "scikit-learn>=1.0.0",
            "ipywidgets>=7.6.0",
            "kaleido>=0.2.1"
        ]
    },

    # Python version requirement
    python_requires=">=3.8",

    # Package classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],

    # Keywords for package discovery
    keywords=[
        "radar", "signal-processing", "spice", "range-doppler",
        "fmcw", "doa", "super-resolution", "sparse-estimation",
        "array-processing", "beamforming", "automotive-radar"
    ],

    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "spice-demo=spice_demonstration:main",
            "spice-test=pytest:main",
        ],
    },

    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },

    # Zip safety
    zip_safe=False,

    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=3.0.0"
    ],
)

# Post-installation information
print("""
[COMPLETE] SPICE for Range-Doppler Imaging Installation Complete!

Professional implementation of SPICE algorithm for radar applications.

Quick Start:
-----------
1. Run comprehensive demonstration:
   python -c "from spice_demonstration import main; main()"

2. Execute test suite:
   pytest -v

3. View documentation:
   python -c "import spice_core; help(spice_core.SPICEEstimator)"

Features:
--------
[OK] Core SPICE algorithm with variants (Fast, Weighted, Quantized)
[OK] FMCW radar range-doppler processing
[OK] Comprehensive testing and validation
[OK] Professional visualization utilities
[OK] Detailed performance analysis (strengths and weaknesses)

Example Usage:
-------------
from spice_core import SPICEEstimator
from range_doppler_imaging import FMCWRadarConfig

# Create SPICE estimator
estimator = SPICEEstimator(n_sensors=8)

# Configure FMCW radar
config = FMCWRadarConfig(f_start=24e9, bandwidth=1e9)

For detailed examples and documentation, see README.md

Author: Professional Radar Engineer
Repository: https://github.com/radar-engineer/spice-radar
""")
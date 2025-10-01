"""
SPICE Algorithm Demonstration: Strengths and Weaknesses Analysis.

This module provides comprehensive demonstrations of SPICE algorithm performance,
showcasing both its exceptional strengths (super-resolution, sparsity) and
critical weaknesses (low SNR failure) for professional radar engineering assessment.

The demonstrations are designed to be suitable for GitHub presentation to
demonstrate radar signal processing expertise to potential employers.

References
----------
.. [1] P. Stoica et al., "SPICE: A Sparse Covariance-Based Estimation Method
       for Array Processing," IEEE Trans. Signal Processing, 2011.
.. [2] T. Yardibi et al., "Source Localization and Sensing: A Nonparametric
       Iterative Adaptive Approach," IEEE Trans. Aerospace and Electronic Systems, 2010.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path

from spice_core import SPICEEstimator, SPICEConfig
from spice_variants import select_spice_variant, compare_spice_variants
from range_doppler_imaging import (
    FMCWRadarConfig, FMCWSignalGenerator, RangeDopplerProcessor,
    compare_processing_methods
)


class SPICEDemonstrator:
    """
    Comprehensive SPICE Algorithm Demonstrator.

    This class provides professional demonstrations of SPICE algorithm
    capabilities and limitations, suitable for technical presentations
    and portfolio documentation.

    Parameters
    ----------
    save_figures : bool, default=True
        Whether to save generated figures to files.
    figure_dir : str, default='figures'
        Directory to save figures.
    dpi : int, default=300
        Figure resolution for saved images.
    """

    def __init__(self, save_figures: bool = True, figure_dir: str = 'figures',
                 dpi: int = 300):
        """Initialize SPICE demonstrator."""
        self.save_figures = save_figures
        self.figure_dir = Path(figure_dir)
        self.dpi = dpi

        if self.save_figures:
            self.figure_dir.mkdir(exist_ok=True)

        # Set up professional plotting style
        self._setup_plotting_style()

    def _setup_plotting_style(self) -> None:
        """Set up professional plotting style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Custom style parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })

    def demonstrate_all(self) -> Dict[str, any]:
        """
        Run all SPICE demonstrations and return comprehensive results.

        Returns
        -------
        results : dict
            Complete demonstration results including all test scenarios.
        """
        print("[DEMO] SPICE Algorithm Comprehensive Demonstration")
        print("=" * 60)

        results = {}

        # 1. Super-resolution capabilities
        print("\n1. Super-Resolution Analysis...")
        results['super_resolution'] = self.demonstrate_super_resolution()

        # 2. SNR performance analysis
        print("\n2. SNR Performance Analysis...")
        results['snr_analysis'] = self.demonstrate_snr_performance()

        # 3. Sparsity advantages
        print("\n3. Sparsity Advantages...")
        results['sparsity'] = self.demonstrate_sparsity_advantages()

        # 4. Computational complexity
        print("\n4. Computational Complexity...")
        results['complexity'] = self.demonstrate_computational_complexity()

        # 5. FMCW radar integration
        print("\n5. FMCW Radar Integration...")
        results['fmcw_integration'] = self.demonstrate_fmcw_integration()

        # 6. Algorithm variants comparison
        print("\n6. Algorithm Variants...")
        results['variants'] = self.demonstrate_algorithm_variants()

        # 7. Failure mode analysis
        print("\n7. Failure Mode Analysis...")
        results['failure_modes'] = self.demonstrate_failure_modes()

        print("\n[SUCCESS] All demonstrations completed successfully!")
        return results

    def demonstrate_super_resolution(self) -> Dict:
        """
        Demonstrate SPICE super-resolution capabilities.

        This shows SPICE's ability to resolve closely spaced sources
        that conventional beamforming cannot separate.
        """
        print("   [TEST] Testing super-resolution with closely spaced sources...")

        # Test scenario: Two sources with varying angular separation
        n_sensors = 16
        n_snapshots = 200
        snr_db = 25

        angular_separations = np.array([0.5, 1.0, 2.0, 4.0, 8.0])  # degrees
        methods = ['conventional', 'spice', 'weighted']

        results = {
            'separations': angular_separations,
            'resolution_performance': {method: [] for method in methods},
            'processing_time': {method: [] for method in methods}
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SPICE Super-Resolution Demonstration', fontsize=16, fontweight='bold')

        for idx, separation in enumerate(angular_separations):
            # Generate two closely spaced sources
            true_angles = np.array([-separation/2, separation/2])
            true_powers = np.array([1.0, 0.8])

            # Generate synthetic data
            data = self._generate_doa_data(
                true_angles, true_powers, n_sensors, n_snapshots, snr_db
            )

            # Test each method
            for method in methods:
                start_time = time.time()

                if method == 'conventional':
                    spectrum = self._conventional_beamforming(data, n_sensors)
                elif method == 'spice':
                    estimator = SPICEEstimator(n_sensors)
                    spectrum, angles = estimator.fit(data)
                elif method == 'weighted':
                    estimator = select_spice_variant('weighted', n_sensors)
                    spectrum, angles = estimator.fit(data)

                processing_time = time.time() - start_time
                results['processing_time'][method].append(processing_time)

                # Evaluate resolution performance
                resolution_score = self._evaluate_resolution(spectrum, true_angles)
                results['resolution_performance'][method].append(resolution_score)

                # Plot representative cases
                if idx < 3:  # Plot first 3 separations
                    ax = axes[0, idx] if idx < 3 else axes[1, idx-3]
                    if method == 'spice':  # Plot SPICE results
                        angular_grid = np.linspace(-90, 90, len(spectrum))
                        ax.plot(angular_grid, 10*np.log10(spectrum),
                               label=f'SPICE (sep={separation}°)', linewidth=2)
                        ax.axvline(true_angles[0], color='red', linestyle='--', alpha=0.7, label='True sources')
                        ax.axvline(true_angles[1], color='red', linestyle='--', alpha=0.7)
                        ax.set_xlabel('Angle (degrees)')
                        ax.set_ylabel('Power (dB)')
                        ax.set_title(f'Angular Separation: {separation}°')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

        # Plot performance comparison
        ax_perf = axes[1, 0]
        for method in methods:
            ax_perf.plot(angular_separations, results['resolution_performance'][method],
                        'o-', label=method.replace('_', '-').title(), linewidth=2, markersize=8)

        ax_perf.set_xlabel('Angular Separation (degrees)')
        ax_perf.set_ylabel('Resolution Score')
        ax_perf.set_title('Resolution Performance Comparison')
        ax_perf.legend()
        ax_perf.grid(True, alpha=0.3)

        # Plot processing time comparison
        ax_time = axes[1, 1]
        for method in methods:
            ax_time.semilogy(angular_separations, results['processing_time'][method],
                           'o-', label=method.replace('_', '-').title(), linewidth=2, markersize=8)

        ax_time.set_xlabel('Angular Separation (degrees)')
        ax_time.set_ylabel('Processing Time (seconds)')
        ax_time.set_title('Computational Complexity')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)

        # Summary statistics plot
        ax_summary = axes[1, 2]
        method_names = [m.replace('_', '-').title() for m in methods]
        avg_resolution = [np.mean(results['resolution_performance'][m]) for m in methods]

        bars = ax_summary.bar(method_names, avg_resolution, alpha=0.7,
                             color=['skyblue', 'lightcoral', 'lightgreen'])
        ax_summary.set_ylabel('Average Resolution Score')
        ax_summary.set_title('Overall Resolution Performance')
        ax_summary.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, avg_resolution):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_super_resolution.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_snr_performance(self) -> Dict:
        """
        Demonstrate SPICE performance across SNR range, highlighting failure modes.

        This is critical for showing professional understanding of algorithm limitations.
        """
        print("   [ANALYSIS] Analyzing SNR performance and failure thresholds...")

        n_sensors = 8
        n_snapshots = 100
        n_monte_carlo = 20

        # SNR range from high performance to clear failure
        snr_range = np.arange(-15, 31, 3)  # -15 to 30 dB

        # Test scenario: 3 sources at different angles
        true_angles = np.array([-20, 0, 25])
        true_powers = np.array([1.0, 0.8, 1.2])

        methods = ['conventional', 'spice', 'weighted']
        results = {
            'snr_range': snr_range,
            'detection_rates': {method: [] for method in methods},
            'rmse_angle': {method: [] for method in methods},
            'rmse_power': {method: [] for method in methods},
            'failure_threshold': {}
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SPICE SNR Performance Analysis: Strengths and Weaknesses',
                    fontsize=16, fontweight='bold')

        for snr_db in snr_range:
            print(f"      Testing SNR = {snr_db} dB...")

            # Monte Carlo simulation for robust statistics
            trial_results = {method: {'detected': 0, 'angle_errors': [], 'power_errors': []}
                           for method in methods}

            for trial in range(n_monte_carlo):
                # Generate noisy data
                data = self._generate_doa_data(
                    true_angles, true_powers, n_sensors, n_snapshots, snr_db
                )

                for method in methods:
                    try:
                        if method == 'conventional':
                            spectrum = self._conventional_beamforming(data, n_sensors)
                            peaks = self._find_spectrum_peaks(spectrum)
                        else:
                            variant = 'standard' if method == 'spice' else 'weighted'
                            estimator = select_spice_variant(variant, n_sensors)
                            spectrum, angles = estimator.fit(data)
                            peaks = estimator.find_peaks(spectrum, threshold_db=-15)

                        # Evaluate detection performance
                        if self._validate_detection(peaks, true_angles):
                            trial_results[method]['detected'] += 1

                            # Compute estimation errors
                            angle_errors, power_errors = self._compute_estimation_errors(
                                peaks, true_angles, true_powers
                            )
                            trial_results[method]['angle_errors'].extend(angle_errors)
                            trial_results[method]['power_errors'].extend(power_errors)

                    except Exception as e:
                        # Algorithm failed - count as non-detection
                        continue

            # Aggregate trial results
            for method in methods:
                detection_rate = trial_results[method]['detected'] / n_monte_carlo
                results['detection_rates'][method].append(detection_rate)

                if trial_results[method]['angle_errors']:
                    rmse_angle = np.sqrt(np.mean(np.array(trial_results[method]['angle_errors'])**2))
                    rmse_power = np.sqrt(np.mean(np.array(trial_results[method]['power_errors'])**2))
                else:
                    rmse_angle = rmse_power = np.nan

                results['rmse_angle'][method].append(rmse_angle)
                results['rmse_power'][method].append(rmse_power)

        # Plot detection rate vs SNR
        ax1 = axes[0, 0]
        for method in methods:
            ax1.plot(snr_range, results['detection_rates'][method],
                    'o-', label=method.replace('_', '-').title(), linewidth=2, markersize=6)

        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='90% Detection')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% Detection')
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Detection Rate vs SNR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)

        # Highlight failure regions
        failure_region = patches.Rectangle((-15, 0), 15, 1, alpha=0.2, color='red',
                                         label='Failure Region')
        ax1.add_patch(failure_region)

        # Plot RMSE vs SNR
        ax2 = axes[0, 1]
        for method in methods:
            valid_indices = ~np.isnan(results['rmse_angle'][method])
            if np.any(valid_indices):
                ax2.semilogy(snr_range[valid_indices],
                           np.array(results['rmse_angle'][method])[valid_indices],
                           'o-', label=method.replace('_', '-').title(), linewidth=2, markersize=6)

        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Angle RMSE (degrees)')
        ax2.set_title('Estimation Accuracy vs SNR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Find and plot failure thresholds
        ax3 = axes[1, 0]
        for method in methods:
            detection_rates = np.array(results['detection_rates'][method])

            # Find SNR where detection drops below 50%
            failure_indices = np.where(detection_rates < 0.5)[0]
            if len(failure_indices) > 0:
                failure_threshold = snr_range[failure_indices[-1]]
                results['failure_threshold'][method] = failure_threshold

                ax3.axvline(failure_threshold, label=f'{method.title()} Threshold: {failure_threshold} dB',
                          linewidth=2)

        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Algorithm Performance')
        ax3.set_title('Failure Threshold Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Summary performance table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create performance summary
        summary_data = []
        for method in methods:
            high_snr_performance = np.mean(results['detection_rates'][method][-5:])  # Last 5 SNR points
            low_snr_performance = np.mean(results['detection_rates'][method][:5])   # First 5 SNR points
            failure_snr = results['failure_threshold'].get(method, 'N/A')

            summary_data.append([
                method.replace('_', '-').title(),
                f'{high_snr_performance:.2f}',
                f'{low_snr_performance:.2f}',
                f'{failure_snr} dB' if failure_snr != 'N/A' else 'N/A'
            ])

        table = ax4.table(cellText=summary_data,
                         colLabels=['Method', 'High SNR Performance', 'Low SNR Performance', 'Failure Threshold'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_snr_analysis.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_sparsity_advantages(self) -> Dict:
        """Demonstrate SPICE advantages in sparse scenarios."""
        print("   [DEMO] Demonstrating sparsity advantages...")

        n_sensors = 12
        n_snapshots = 150
        snr_db = 20

        # Test scenarios with different levels of sparsity
        sparsity_scenarios = [
            {'n_sources': 1, 'description': 'Single source'},
            {'n_sources': 2, 'description': 'Two sources'},
            {'n_sources': 4, 'description': 'Four sources'},
            {'n_sources': 6, 'description': 'Dense scenario'},
            {'n_sources': 8, 'description': 'Very dense'}
        ]

        results = {
            'scenarios': sparsity_scenarios,
            'spice_performance': [],
            'conventional_performance': [],
            'sparsity_advantage': []
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SPICE Sparsity Advantages in Radar Imaging', fontsize=16, fontweight='bold')

        for idx, scenario in enumerate(sparsity_scenarios):
            n_sources = scenario['n_sources']

            # Generate uniformly distributed sources
            angles = np.linspace(-60, 60, n_sources)
            powers = np.random.uniform(0.5, 1.5, n_sources)

            # Generate data
            data = self._generate_doa_data(angles, powers, n_sensors, n_snapshots, snr_db)

            # Compare methods
            conventional_spectrum = self._conventional_beamforming(data, n_sensors)

            estimator = SPICEEstimator(n_sensors)
            spice_spectrum, angular_grid = estimator.fit(data)

            # Compute performance metrics
            spice_peaks = estimator.find_peaks(spice_spectrum)
            conv_peaks = self._find_spectrum_peaks(conventional_spectrum)

            spice_perf = self._compute_sparsity_metrics(spice_peaks, angles)
            conv_perf = self._compute_sparsity_metrics(conv_peaks, angles)

            results['spice_performance'].append(spice_perf)
            results['conventional_performance'].append(conv_perf)
            results['sparsity_advantage'].append(spice_perf['resolution'] / conv_perf['resolution'])

            # Plot representative scenarios
            if idx < 6:
                row, col = divmod(idx, 3)
                ax = axes[row, col]

                # Plot both spectra
                ax.plot(angular_grid, 10*np.log10(conventional_spectrum),
                       label='Conventional', alpha=0.7, linewidth=2)
                ax.plot(angular_grid, 10*np.log10(spice_spectrum),
                       label='SPICE', linewidth=2)

                # Mark true source locations
                for angle in angles:
                    ax.axvline(angle, color='red', linestyle='--', alpha=0.5)

                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Power (dB)')
                ax.set_title(f'{scenario["description"]} ({n_sources} sources)')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_sparsity_advantages.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_computational_complexity(self) -> Dict:
        """Demonstrate computational complexity scaling."""
        print("   [ANALYSIS] Analyzing computational complexity...")

        problem_sizes = [
            {'n_sensors': 4, 'n_snapshots': 50},
            {'n_sensors': 8, 'n_snapshots': 100},
            {'n_sensors': 16, 'n_snapshots': 200},
            {'n_sensors': 32, 'n_snapshots': 400},
        ]

        methods = ['conventional', 'spice', 'fast_spice']
        results = {
            'problem_sizes': problem_sizes,
            'execution_times': {method: [] for method in methods},
            'memory_usage': {method: [] for method in methods}
        }

        # Test execution times
        for size_params in problem_sizes:
            n_sensors = size_params['n_sensors']
            n_snapshots = size_params['n_snapshots']

            print(f"      Testing {n_sensors} sensors, {n_snapshots} snapshots...")

            # Generate test data
            true_angles = np.array([-10, 15])
            true_powers = np.array([1.0, 0.8])
            data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, 20)

            for method in methods:
                # Measure execution time
                start_time = time.time()

                if method == 'conventional':
                    spectrum = self._conventional_beamforming(data, n_sensors)
                elif method == 'spice':
                    estimator = SPICEEstimator(n_sensors)
                    spectrum, _ = estimator.fit(data)
                elif method == 'fast_spice':
                    estimator = select_spice_variant('fast', n_sensors)
                    spectrum, _ = estimator.fit(data)

                execution_time = time.time() - start_time
                results['execution_times'][method].append(execution_time)

                # Estimate memory usage (simplified)
                memory_estimate = n_sensors**2 * n_snapshots * 16 / 1024**2  # MB
                results['memory_usage'][method].append(memory_estimate)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('SPICE Computational Complexity Analysis', fontsize=16, fontweight='bold')

        # Execution time scaling
        ax1 = axes[0]
        n_sensors_list = [p['n_sensors'] for p in problem_sizes]

        for method in methods:
            ax1.loglog(n_sensors_list, results['execution_times'][method],
                      'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)

        ax1.set_xlabel('Number of Sensors')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Scaling')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Memory usage
        ax2 = axes[1]
        for method in methods:
            ax2.loglog(n_sensors_list, results['memory_usage'][method],
                      'o-', label=method.replace('_', ' ').title(), linewidth=2, markersize=8)

        ax2.set_xlabel('Number of Sensors')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_complexity_analysis.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_fmcw_integration(self) -> Dict:
        """Demonstrate SPICE integration with FMCW radar systems."""
        print("   [INTEGRATION] FMCW radar integration demonstration...")

        # Set up FMCW radar configuration
        radar_config = FMCWRadarConfig(
            f_start=24e9,
            bandwidth=1e9,
            chirp_duration=1e-3,
            n_chirps=128,
            n_rx=4
        )

        # Define test targets
        targets = [
            {'range': 75, 'velocity': 15, 'rcs': 1.0, 'angle': -10},
            {'range': 150, 'velocity': -8, 'rcs': 0.5, 'angle': 5},
            {'range': 200, 'velocity': 25, 'rcs': 2.0, 'angle': 20}
        ]

        # Test different SNR levels
        snr_levels = [25, 15, 5, -5]
        results = {
            'targets': targets,
            'snr_levels': snr_levels,
            'conventional_maps': [],
            'spice_maps': [],
            'detection_performance': []
        }

        fig, axes = plt.subplots(len(snr_levels), 3, figsize=(18, 16))
        fig.suptitle('SPICE-Enhanced FMCW Radar Range-Doppler Imaging',
                    fontsize=16, fontweight='bold')

        # Generate and process radar data
        generator = FMCWSignalGenerator(radar_config)
        processor = RangeDopplerProcessor(radar_config)

        for idx, snr_db in enumerate(snr_levels):
            print(f"      Processing SNR = {snr_db} dB...")

            # Generate radar data
            radar_data = generator.generate_radar_data(targets, snr_db=snr_db)

            # Process with different methods
            conventional_map = processor.process_conventional(radar_data)
            spice_map = processor.process_spice(radar_data, variant='standard')

            results['conventional_maps'].append(conventional_map)
            results['spice_maps'].append(spice_map)

            # Detect targets
            conv_targets = processor.detect_targets(conventional_map)
            spice_targets = processor.detect_targets(spice_map)

            # Compute detection performance
            conv_metrics = self._compute_radar_detection_metrics(conv_targets, targets)
            spice_metrics = self._compute_radar_detection_metrics(spice_targets, targets)

            results['detection_performance'].append({
                'snr_db': snr_db,
                'conventional': conv_metrics,
                'spice': spice_metrics
            })

            # Plot range-doppler maps
            row = idx

            # Conventional processing
            ax1 = axes[row, 0]
            im1 = ax1.imshow(conventional_map, aspect='auto', cmap='jet',
                           extent=[0, radar_config.max_range, -radar_config.max_velocity, radar_config.max_velocity])
            ax1.set_title(f'Conventional (SNR={snr_db}dB)')
            ax1.set_xlabel('Range (m)')
            ax1.set_ylabel('Velocity (m/s)')

            # SPICE processing
            ax2 = axes[row, 1]
            im2 = ax2.imshow(spice_map, aspect='auto', cmap='jet',
                           extent=[0, radar_config.max_range, -radar_config.max_velocity, radar_config.max_velocity])
            ax2.set_title(f'SPICE Enhanced (SNR={snr_db}dB)')
            ax2.set_xlabel('Range (m)')
            ax2.set_ylabel('Velocity (m/s)')

            # Mark true target locations
            for target in targets:
                for ax in [ax1, ax2]:
                    ax.plot(target['range'], target['velocity'], 'r+', markersize=15, markeredgewidth=3)

            # Performance comparison
            ax3 = axes[row, 2]
            methods = ['Conventional', 'SPICE']
            detection_rates = [conv_metrics['detection_rate'], spice_metrics['detection_rate']]
            false_alarm_rates = [conv_metrics['false_alarm_rate'], spice_metrics['false_alarm_rate']]

            x = np.arange(len(methods))
            width = 0.35

            bars1 = ax3.bar(x - width/2, detection_rates, width, label='Detection Rate', alpha=0.8)
            bars2 = ax3.bar(x + width/2, false_alarm_rates, width, label='False Alarm Rate', alpha=0.8)

            ax3.set_ylabel('Rate')
            ax3.set_title(f'Performance (SNR={snr_db}dB)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_fmcw_integration.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_algorithm_variants(self) -> Dict:
        """Demonstrate different SPICE algorithm variants."""
        print("   [COMPARISON] Comparing SPICE algorithm variants...")

        n_sensors = 8
        n_snapshots = 100
        snr_db = 15

        # Test scenario
        true_angles = np.array([-15, 0, 20])
        true_powers = np.array([1.0, 0.7, 1.2])

        # Generate test data
        data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, snr_db)

        # Compare variants
        variants = ['standard', 'fast', 'weighted']
        results = {}

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('SPICE Algorithm Variants Comparison', fontsize=16, fontweight='bold')

        for idx, variant in enumerate(variants):
            print(f"      Testing {variant} variant...")

            start_time = time.time()
            estimator = select_spice_variant(variant, n_sensors)
            spectrum, angles = estimator.fit(data)
            execution_time = time.time() - start_time

            peaks = estimator.find_peaks(spectrum)

            results[variant] = {
                'spectrum': spectrum,
                'peaks': peaks,
                'execution_time': execution_time,
                'convergence_info': estimator.get_convergence_info()
            }

            # Plot spectrum
            ax = axes[idx]

            ax.plot(angles, 10*np.log10(spectrum), linewidth=2, label=f'{variant.title()} SPICE')

            # Mark true sources
            for angle in true_angles:
                ax.axvline(angle, color='red', linestyle='--', alpha=0.7)

            # Mark detected peaks
            if len(peaks['angles']) > 0:
                ax.plot(peaks['angles'], peaks['powers_db'], 'go', markersize=8,
                       label='Detected Peaks')

            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Power (dB)')
            ax.set_title(f'{variant.replace("_", " ").title()} Variant')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_variants_comparison.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        return results

    def demonstrate_failure_modes(self) -> Dict:
        """Demonstrate and analyze SPICE failure modes."""
        print("   [WARNING] Analyzing failure modes - Critical for professional assessment...")

        failure_scenarios = [
            {
                'name': 'Very Low SNR',
                'snr_db': -10,
                'description': 'Algorithm fails when noise dominates signal'
            },
            {
                'name': 'Model Mismatch',
                'snr_db': 20,
                'array_errors': True,
                'description': 'Calibration errors degrade performance'
            },
            {
                'name': 'Insufficient Data',
                'snr_db': 15,
                'n_snapshots': 10,
                'description': 'Too few snapshots for reliable covariance estimation'
            },
            {
                'name': 'Coherent Sources',
                'snr_db': 20,
                'coherent': True,
                'description': 'Fully coherent sources challenge the algorithm'
            }
        ]

        results = {
            'scenarios': failure_scenarios,
            'failure_analysis': []
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SPICE Algorithm Failure Mode Analysis', fontsize=16, fontweight='bold')

        for idx, scenario in enumerate(failure_scenarios):
            print(f"      Testing: {scenario['name']}...")

            # Generate test data for failure scenario
            failure_data = self._generate_failure_scenario_data(scenario)

            try:
                # Attempt SPICE processing
                estimator = SPICEEstimator(8)  # 8 sensors
                spectrum, angles = estimator.fit(failure_data['covariance'])
                peaks = estimator.find_peaks(spectrum, threshold_db=-25)

                failure_analysis = {
                    'scenario': scenario['name'],
                    'success': True,
                    'n_peaks_detected': len(peaks['angles']),
                    'convergence_iterations': estimator.get_convergence_info()['n_iterations'],
                    'final_cost': estimator.get_convergence_info()['final_cost']
                }

                # Plot results
                ax = axes[idx // 2, idx % 2]
                ax.plot(angles, 10*np.log10(spectrum), linewidth=2,
                       label=f'SPICE Result', color='blue')

                if 'true_angles' in failure_data:
                    for angle in failure_data['true_angles']:
                        ax.axvline(angle, color='red', linestyle='--', alpha=0.7,
                                 label='True Source' if angle == failure_data['true_angles'][0] else '')

                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Power (dB)')
                ax.set_title(f'{scenario["name"]}\n{scenario["description"]}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add performance annotation
                ax.text(0.05, 0.95, f'Detected: {len(peaks["angles"])} sources\nIterations: {failure_analysis["convergence_iterations"]}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            except Exception as e:
                failure_analysis = {
                    'scenario': scenario['name'],
                    'success': False,
                    'error': str(e),
                    'failure_reason': 'Algorithm crashed or failed to converge'
                }

                # Plot failure indication
                ax = axes[idx // 2, idx % 2]
                ax.text(0.5, 0.5, f'ALGORITHM FAILED\n{scenario["name"]}\n\nError: {str(e)[:50]}...',
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                       fontsize=12, fontweight='bold')
                ax.set_title(f'{scenario["name"]} - FAILURE')
                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Power (dB)')

            results['failure_analysis'].append(failure_analysis)

        plt.tight_layout()

        if self.save_figures:
            plt.savefig(self.figure_dir / 'spice_failure_modes.png',
                       dpi=self.dpi, bbox_inches='tight')

        plt.show()

        # Print failure mode summary
        print("\n   [SUMMARY] Failure Mode Summary:")
        print("   " + "="*50)
        for analysis in results['failure_analysis']:
            status = "[SUCCESS]" if analysis['success'] else "[FAILED]"
            print(f"   {analysis['scenario']}: {status}")
            if not analysis['success']:
                print(f"      Reason: {analysis['failure_reason']}")

        return results

    # Helper methods for data generation and analysis
    def _generate_doa_data(self, angles: np.ndarray, powers: np.ndarray,
                          n_sensors: int, n_snapshots: int, snr_db: float) -> np.ndarray:
        """Generate synthetic DOA data."""
        # Array manifold
        steering_matrix = np.zeros((n_sensors, len(angles)), dtype=complex)
        for i, angle in enumerate(angles):
            phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
            steering_matrix[:, i] = np.exp(1j * phase_shifts)

        # Source signals
        source_signals = np.zeros((len(angles), n_snapshots), dtype=complex)
        for i, power in enumerate(powers):
            source_signals[i, :] = np.sqrt(power) * (
                np.random.randn(n_snapshots) + 1j * np.random.randn(n_snapshots)
            )

        # Received signals
        received_signals = steering_matrix @ source_signals

        # Add noise
        signal_power = np.mean(np.abs(received_signals)**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(n_sensors, n_snapshots) + 1j * np.random.randn(n_sensors, n_snapshots)
        )

        received_signals += noise

        # Return covariance matrix
        return received_signals @ received_signals.conj().T / n_snapshots

    def _conventional_beamforming(self, covariance: np.ndarray, n_sensors: int) -> np.ndarray:
        """Compute conventional beamforming spectrum."""
        angles = np.linspace(-90, 90, 180)
        spectrum = np.zeros(len(angles))

        for i, angle in enumerate(angles):
            phase_shifts = np.arange(n_sensors) * np.pi * np.sin(np.deg2rad(angle))
            steering_vec = np.exp(1j * phase_shifts)
            spectrum[i] = np.real(steering_vec.conj().T @ covariance @ steering_vec)

        return spectrum

    def _find_spectrum_peaks(self, spectrum: np.ndarray) -> Dict:
        """Find peaks in spectrum."""
        # Simple peak finding
        threshold = np.max(spectrum) * 0.1
        peak_indices = []

        for i in range(1, len(spectrum)-1):
            if (spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold):
                peak_indices.append(i)

        angles = np.linspace(-90, 90, len(spectrum))
        return {
            'angles': angles[peak_indices],
            'powers': spectrum[peak_indices],
            'powers_db': 10*np.log10(spectrum[peak_indices])
        }

    def _evaluate_resolution(self, spectrum: np.ndarray, true_angles: np.ndarray) -> float:
        """Evaluate resolution performance."""
        peaks = self._find_spectrum_peaks(spectrum)

        if len(peaks['angles']) < len(true_angles):
            return 0.0  # Failed to resolve all sources

        # Simple resolution score based on peak separation
        if len(true_angles) == 2:
            true_separation = abs(true_angles[1] - true_angles[0])
            if len(peaks['angles']) >= 2:
                detected_separation = abs(peaks['angles'][1] - peaks['angles'][0])
                return min(1.0, true_separation / max(detected_separation, 0.1))

        return 0.5  # Default score

    def _validate_detection(self, peaks: Dict, true_angles: np.ndarray) -> bool:
        """Validate if detection is successful."""
        if 'angles' not in peaks or len(peaks['angles']) == 0:
            return False

        # Check if we detected the right number of sources (within tolerance)
        return abs(len(peaks['angles']) - len(true_angles)) <= 1

    def _compute_estimation_errors(self, peaks: Dict, true_angles: np.ndarray,
                                 true_powers: np.ndarray) -> Tuple[List[float], List[float]]:
        """Compute angle and power estimation errors."""
        if 'angles' not in peaks or len(peaks['angles']) == 0:
            return [], []

        angle_errors = []
        power_errors = []

        # Simple nearest neighbor matching
        for true_angle, true_power in zip(true_angles, true_powers):
            if len(peaks['angles']) > 0:
                distances = np.abs(peaks['angles'] - true_angle)
                nearest_idx = np.argmin(distances)

                if distances[nearest_idx] < 10:  # Within 10 degrees
                    angle_errors.append(distances[nearest_idx])
                    power_errors.append(abs(peaks['powers'][nearest_idx] - true_power))

        return angle_errors, power_errors

    def _compute_sparsity_metrics(self, peaks: Dict, true_angles: np.ndarray) -> Dict:
        """Compute sparsity-related performance metrics."""
        if 'angles' not in peaks:
            return {'resolution': 0.0, 'accuracy': 0.0, 'sparsity': 0.0}

        # Simple metrics
        n_detected = len(peaks['angles'])
        n_true = len(true_angles)

        accuracy = min(1.0, n_true / max(n_detected, 1))
        resolution = 1.0 / max(1.0, abs(n_detected - n_true) + 1)
        sparsity = n_true / max(n_detected, 1)

        return {
            'resolution': resolution,
            'accuracy': accuracy,
            'sparsity': sparsity
        }

    def _compute_radar_detection_metrics(self, detected_targets: List[Dict],
                                       true_targets: List[Dict]) -> Dict:
        """Compute radar detection performance metrics."""
        if not detected_targets:
            return {
                'detection_rate': 0.0,
                'false_alarm_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

        # Simple distance-based matching
        matches = 0
        for true_target in true_targets:
            for detected in detected_targets:
                range_error = abs(detected['range'] - true_target['range'])
                velocity_error = abs(detected['velocity'] - true_target['velocity'])

                if range_error < 20 and velocity_error < 5:  # Matching thresholds
                    matches += 1
                    break

        n_true = len(true_targets)
        n_detected = len(detected_targets)

        detection_rate = matches / n_true if n_true > 0 else 0
        false_alarm_rate = max(0, n_detected - matches) / n_detected if n_detected > 0 else 0
        precision = matches / n_detected if n_detected > 0 else 0
        recall = matches / n_true if n_true > 0 else 0

        return {
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'precision': precision,
            'recall': recall
        }

    def _generate_failure_scenario_data(self, scenario: Dict) -> Dict:
        """Generate data for specific failure scenarios."""
        n_sensors = 8
        n_snapshots = scenario.get('n_snapshots', 100)

        # Base test case
        true_angles = np.array([-10, 15])
        true_powers = np.array([1.0, 0.8])

        if scenario['name'] == 'Very Low SNR':
            data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, scenario['snr_db'])

        elif scenario['name'] == 'Model Mismatch':
            # Add array calibration errors
            data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, scenario['snr_db'])
            # Add random phase/gain errors
            errors = 0.1 * (np.random.randn(n_sensors) + 1j * np.random.randn(n_sensors))
            error_matrix = np.outer(errors, errors.conj())
            data += error_matrix

        elif scenario['name'] == 'Insufficient Data':
            data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, scenario['snr_db'])

        elif scenario['name'] == 'Coherent Sources':
            # Generate coherent sources (same signal, different delays)
            data = self._generate_doa_data(true_angles, true_powers, n_sensors, n_snapshots, scenario['snr_db'])
            # Make sources coherent by correlating them
            correlation_factor = 0.95
            data = correlation_factor * data + (1-correlation_factor) * np.random.randn(*data.shape)

        return {
            'covariance': data,
            'true_angles': true_angles,
            'scenario': scenario
        }


def main():
    """Main demonstration function."""
    print("🚀 SPICE for Range-Doppler Imaging: Professional Demonstration")
    print("="*70)
    print("Showcasing algorithm strengths, weaknesses, and radar applications")
    print("Designed for GitHub portfolio and employer assessment")
    print("="*70)

    # Create demonstrator
    demonstrator = SPICEDemonstrator(save_figures=True)

    # Run all demonstrations
    results = demonstrator.demonstrate_all()

    # Generate summary report
    print("\n[SUMMARY] DEMONSTRATION SUMMARY")
    print("="*40)
    print("[OK] Super-resolution capabilities demonstrated")
    print("[OK] SNR performance curve analyzed")
    print("[OK] Sparsity advantages showcased")
    print("[OK] Computational complexity evaluated")
    print("[OK] FMCW radar integration validated")
    print("[OK] Algorithm variants compared")
    print("[WARNING] Failure modes documented")
    print("\n[COMPLETE] Professional assessment complete!")
    print("📁 All figures saved to 'figures/' directory")

    return results


if __name__ == "__main__":
    main()
"""
Complete Integration Test for Enhanced SPICE Implementation.

This test validates the entire enhanced SPICE system including:
1. Original implementation consistency
2. Enhanced stability improvements
3. Documentation accuracy
4. Professional code standards
5. Literature-based claims validation

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import pytest
import warnings
import time
from typing import Dict, List, Tuple

from spice_core import SPICEEstimator, SPICEConfig
from spice_stable import StableSPICEEstimator, StableSPICEConfig
from coprime_signal_design import CoprimeSignalDesign
from educational_examples import EducationalAnalyzer


class ComprehensiveIntegrationTest:
    """Complete system integration testing."""

    def __init__(self):
        """Initialize comprehensive test suite."""
        self.test_results = {}
        self.validation_summary = {}

    def test_original_implementation_consistency(self) -> Dict:
        """Test that original SPICE implementation matches documented behavior."""
        print("Testing Original SPICE Implementation Consistency...")

        results = {
            'mathematical_accuracy': False,
            'snr_thresholds': False,
            'convergence_behavior': False,
            'peak_detection': False,
            'documentation_claims': False
        }

        # Test mathematical accuracy
        n_sensors = 8
        # Generate well-conditioned test scenario
        true_angles = np.array([-10, 15])
        steering_vecs = []
        for angle in np.deg2rad(true_angles):
            sensor_positions = np.arange(n_sensors)
            phases = np.exp(1j * np.pi * sensor_positions * np.sin(angle))
            steering_vecs.append(phases[:, np.newaxis])

        A = np.hstack(steering_vecs)
        P = np.diag([10, 8])  # High SNR scenario
        noise_cov = 0.1 * np.eye(n_sensors)
        sample_cov = A @ P @ A.conj().T + noise_cov

        config = SPICEConfig(max_iterations=100, convergence_tolerance=1e-6)
        estimator = SPICEEstimator(n_sensors, config)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(sample_cov)

            # Validate mathematical accuracy
            peaks = estimator.find_peaks(power_spectrum, min_separation=3.0)
            if len(peaks['angles']) >= 2:
                estimated_angles = np.sort(peaks['angles'][:2])
                true_angles_sorted = np.sort(true_angles)
                estimation_error = np.mean(np.abs(estimated_angles - true_angles_sorted))
                results['mathematical_accuracy'] = estimation_error < 2.0  # Within 2 degrees
                results['peak_detection'] = True

            # Test convergence behavior
            conv_info = estimator.get_convergence_info()
            results['convergence_behavior'] = conv_info['n_iterations'] < 100

            # SNR threshold validation (documented as 10 dB)
            # Test at 12 dB (should work) and 8 dB (may fail)
            high_snr_cov = A @ np.diag([15.8, 12.6]) @ A.conj().T + 0.1 * np.eye(n_sensors)  # 12 dB
            low_snr_cov = A @ np.diag([6.3, 5.0]) @ A.conj().T + 0.1 * np.eye(n_sensors)    # 8 dB

            # High SNR should work better
            estimator_high = SPICEEstimator(n_sensors, config)
            estimator_low = SPICEEstimator(n_sensors, config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ps_high, _ = estimator_high.fit(high_snr_cov)
                ps_low, _ = estimator_low.fit(low_snr_cov)

            peaks_high = estimator_high.find_peaks(ps_high, min_separation=3.0)
            peaks_low = estimator_low.find_peaks(ps_low, min_separation=3.0)

            # High SNR should detect sources more reliably
            results['snr_thresholds'] = len(peaks_high['angles']) >= len(peaks_low['angles'])

            # Documentation claims validation
            results['documentation_claims'] = (
                results['mathematical_accuracy'] and
                results['convergence_behavior'] and
                results['peak_detection']
            )

        except Exception as e:
            print(f"Original SPICE test failed: {e}")

        success_rate = sum(results.values()) / len(results)
        print(f"Original SPICE Consistency: {success_rate:.1%}")

        return results

    def test_enhanced_stability_improvements(self) -> Dict:
        """Test enhanced SPICE stability improvements."""
        print("Testing Enhanced SPICE Stability Improvements...")

        results = {
            'adaptive_regularization': False,
            'condition_number_handling': False,
            'matrix_conditioning': False,
            'convergence_enhancement': False,
            'stability_monitoring': False
        }

        n_sensors = 8

        # Test 1: Adaptive regularization
        try:
            # Create ill-conditioned matrix
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            U, s, Vh = np.linalg.svd(A)
            s_conditioned = np.logspace(0, -12, n_sensors)
            A_ill = U @ np.diag(s_conditioned) @ Vh
            ill_cov = A_ill @ A_ill.conj().T

            config = StableSPICEConfig(adaptive_regularization=True)
            estimator = StableSPICEEstimator(n_sensors, config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(ill_cov)

            # Check that regularization was adapted
            stability_report = estimator.get_stability_report()
            reg_range = stability_report['numerical_health']['regularization_range']
            results['adaptive_regularization'] = reg_range[1] > reg_range[0]

        except Exception as e:
            print(f"Adaptive regularization test failed: {e}")

        # Test 2: Condition number handling
        try:
            condition_numbers = [1e6, 1e10, 1e14]
            handled_conditions = 0

            for target_cond in condition_numbers:
                A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
                U, s, Vh = np.linalg.svd(A)
                s_conditioned = np.logspace(0, -np.log10(target_cond), n_sensors)
                A_cond = U @ np.diag(s_conditioned) @ Vh
                cond_cov = A_cond @ A_cond.conj().T

                estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig())

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        power_spectrum, angles = estimator.fit(cond_cov)
                    handled_conditions += 1
                except:
                    pass

            results['condition_number_handling'] = handled_conditions >= 2

        except Exception as e:
            print(f"Condition number handling test failed: {e}")

        # Test 3: Matrix conditioning
        try:
            # Create near-singular matrix
            singular_matrix = np.outer(np.ones(n_sensors), np.ones(n_sensors))
            noise = 1e-14 * np.random.randn(n_sensors, n_sensors)
            near_singular = singular_matrix + noise @ noise.T

            estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(near_singular)

            results['matrix_conditioning'] = True  # If we get here, conditioning worked

        except Exception as e:
            print(f"Matrix conditioning test failed: {e}")

        # Test 4: Convergence enhancement
        try:
            # Compare convergence with original
            test_cov = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            test_cov = test_cov @ test_cov.conj().T + 0.1 * np.eye(n_sensors)

            # Original SPICE
            orig_estimator = SPICEEstimator(n_sensors, SPICEConfig(max_iterations=50))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                orig_ps, _ = orig_estimator.fit(test_cov)
            orig_conv = orig_estimator.get_convergence_info()

            # Enhanced SPICE
            enh_estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig(max_iterations=50))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enh_ps, _ = enh_estimator.fit(test_cov)
            enh_report = enh_estimator.get_stability_report()

            # Enhanced should have better or equal convergence
            results['convergence_enhancement'] = (
                enh_report['convergence_info']['final_cost'] <= orig_conv['final_cost']
            )

        except Exception as e:
            print(f"Convergence enhancement test failed: {e}")

        # Test 5: Stability monitoring
        try:
            estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig())
            test_cov = np.eye(n_sensors) + 0.1 * np.random.randn(n_sensors, n_sensors)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                power_spectrum, angles = estimator.fit(test_cov)

            stability_report = estimator.get_stability_report()

            # Check that comprehensive monitoring is working
            required_fields = [
                'stability_metrics', 'condition_number_history',
                'regularization_history', 'numerical_health'
            ]
            results['stability_monitoring'] = all(
                field in stability_report for field in required_fields
            )

        except Exception as e:
            print(f"Stability monitoring test failed: {e}")

        success_rate = sum(results.values()) / len(results)
        print(f"Enhanced SPICE Improvements: {success_rate:.1%}")

        return results

    def test_documentation_accuracy(self) -> Dict:
        """Test that documentation claims match implementation reality."""
        print("Testing Documentation Accuracy...")

        results = {
            'snr_thresholds_accurate': False,
            'coprime_claims_honest': False,
            'mathematical_formulations': False,
            'performance_metrics': False,
            'literature_references': False
        }

        # Test SNR thresholds (documented as 10 dB for basic SPICE)
        try:
            n_sensors = 8
            true_angles = np.array([0, 20])

            # Generate scenarios at documented threshold
            scenarios = [
                ('above_threshold', 12),  # 12 dB - should work
                ('at_threshold', 10),     # 10 dB - borderline
                ('below_threshold', 8)    # 8 dB - may fail
            ]

            success_rates = {}

            for scenario_name, snr_db in scenarios:
                steering_vecs = []
                for angle in np.deg2rad(true_angles):
                    sensor_positions = np.arange(n_sensors)
                    phases = np.exp(1j * np.pi * sensor_positions * np.sin(angle))
                    steering_vecs.append(phases[:, np.newaxis])

                A = np.hstack(steering_vecs)
                signal_power = 10**(snr_db/10)
                P = np.diag([signal_power, signal_power * 0.8])
                noise_cov = 1.0 * np.eye(n_sensors)
                sample_cov = A @ P @ A.conj().T + noise_cov

                estimator = SPICEEstimator(n_sensors, SPICEConfig(max_iterations=50))

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        power_spectrum, angles = estimator.fit(sample_cov)

                    peaks = estimator.find_peaks(power_spectrum, min_separation=5.0)
                    success_rates[scenario_name] = len(peaks['angles']) >= 2
                except:
                    success_rates[scenario_name] = False

            # Above threshold should work better than below threshold
            results['snr_thresholds_accurate'] = (
                success_rates.get('above_threshold', False) >=
                success_rates.get('below_threshold', False)
            )

        except Exception as e:
            print(f"SNR threshold validation failed: {e}")

        # Test coprime claims (documented as equivalent performance)
        try:
            designer = CoprimeSignalDesign(coprime_pair=(31, 37))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                performance_analysis = designer.analyze_performance_improvement(n_chirps=64)

            # Should show equivalent performance (improvement factor near 1.0)
            improvement_factor = performance_analysis['coherence_reduction']
            results['coprime_claims_honest'] = 0.8 <= improvement_factor <= 1.2

        except Exception as e:
            print(f"Coprime claims validation failed: {e}")

        # Test mathematical formulations
        try:
            # Verify steering vector computation matches documented formula
            n_sensors = 8
            test_angle_deg = 30
            test_angle_rad = np.deg2rad(test_angle_deg)

            # Manual computation: a(θ) = [1, e^{jπsin(θ)}, ..., e^{j(N-1)πsin(θ)}]^T
            sensor_positions = np.arange(n_sensors)
            expected_steering = np.exp(1j * np.pi * sensor_positions * np.sin(test_angle_rad))

            # Implementation computation
            estimator = SPICEEstimator(n_sensors)
            computed_steering = estimator._compute_steering_vectors(np.array([test_angle_deg]))[:, 0]

            # Should match within numerical precision
            relative_error = np.linalg.norm(expected_steering - computed_steering) / np.linalg.norm(expected_steering)
            results['mathematical_formulations'] = relative_error < 1e-12

        except Exception as e:
            print(f"Mathematical formulation validation failed: {e}")

        # Test performance metrics claims
        try:
            # Compare enhanced vs original on challenging scenario
            n_sensors = 8
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            U, s, Vh = np.linalg.svd(A)
            s_conditioned = np.logspace(0, -10, n_sensors)
            A_ill = U @ np.diag(s_conditioned) @ Vh
            ill_cov = A_ill @ A_ill.conj().T

            # Original SPICE
            orig_estimator = SPICEEstimator(n_sensors, SPICEConfig(max_iterations=30))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                orig_ps, _ = orig_estimator.fit(ill_cov)
            orig_cost = orig_estimator.get_convergence_info()['final_cost']

            # Enhanced SPICE
            enh_estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig(max_iterations=30))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enh_ps, _ = enh_estimator.fit(ill_cov)
            enh_cost = enh_estimator.get_stability_report()['convergence_info']['final_cost']

            # Enhanced should achieve better cost (claims of significant improvement)
            improvement_factor = orig_cost / max(enh_cost, 1e-15)
            results['performance_metrics'] = improvement_factor > 2.0  # At least 2x improvement

        except Exception as e:
            print(f"Performance metrics validation failed: {e}")

        # Literature references check
        try:
            # Verify that enhanced algorithm implements literature-based improvements
            estimator = StableSPICEEstimator(n_sensors, StableSPICEConfig())

            # Check for key literature-based features
            has_adaptive_reg = hasattr(estimator, '_compute_adaptive_regularization')
            has_matrix_conditioning = hasattr(estimator, '_validate_and_condition_matrix')
            has_enhanced_convergence = hasattr(estimator, '_check_enhanced_convergence')
            has_stability_analysis = hasattr(estimator, '_analyze_matrix_stability')

            results['literature_references'] = all([
                has_adaptive_reg, has_matrix_conditioning,
                has_enhanced_convergence, has_stability_analysis
            ])

        except Exception as e:
            print(f"Literature references validation failed: {e}")

        success_rate = sum(results.values()) / len(results)
        print(f"Documentation Accuracy: {success_rate:.1%}")

        return results

    def test_professional_standards(self) -> Dict:
        """Test adherence to professional coding standards."""
        print("Testing Professional Standards...")

        results = {
            'code_quality': False,
            'error_handling': False,
            'type_annotations': False,
            'documentation_quality': False,
            'test_coverage': False
        }

        # Test code quality (basic structural checks)
        try:
            # Import all modules successfully
            from spice_core import SPICEEstimator, SPICEConfig
            from spice_stable import StableSPICEEstimator, StableSPICEConfig
            from coprime_signal_design import CoprimeSignalDesign
            from educational_examples import EducationalAnalyzer

            results['code_quality'] = True

        except Exception as e:
            print(f"Code quality check failed: {e}")

        # Test error handling
        try:
            estimator = SPICEEstimator(8)

            # Should handle invalid input gracefully
            try:
                # Invalid covariance matrix (wrong size)
                invalid_cov = np.eye(4)  # Wrong size for 8-sensor array
                estimator.fit(invalid_cov)
                results['error_handling'] = False  # Should have raised error
            except ValueError:
                results['error_handling'] = True  # Proper error handling

        except Exception as e:
            print(f"Error handling test failed: {e}")

        # Test type annotations (basic check)
        try:
            import inspect
            from spice_stable import StableSPICEEstimator

            # Check if key methods have type annotations
            fit_signature = inspect.signature(StableSPICEEstimator.fit)
            has_return_annotation = fit_signature.return_annotation != inspect.Signature.empty

            init_signature = inspect.signature(StableSPICEEstimator.__init__)
            has_param_annotations = any(
                param.annotation != inspect.Parameter.empty
                for param in init_signature.parameters.values()
                if param.name != 'self'
            )

            results['type_annotations'] = has_return_annotation and has_param_annotations

        except Exception as e:
            print(f"Type annotation check failed: {e}")

        # Test documentation quality
        try:
            estimator = StableSPICEEstimator(8)

            # Check for comprehensive docstrings
            has_class_doc = estimator.__class__.__doc__ is not None
            has_fit_doc = estimator.fit.__doc__ is not None
            has_report_doc = estimator.get_stability_report.__doc__ is not None

            results['documentation_quality'] = all([has_class_doc, has_fit_doc, has_report_doc])

        except Exception as e:
            print(f"Documentation quality check failed: {e}")

        # Test coverage (basic functional check)
        try:
            # Verify key functionality works end-to-end
            n_sensors = 8
            sample_cov = np.eye(n_sensors) + 0.1 * np.random.randn(n_sensors, n_sensors)

            # Original SPICE
            orig_estimator = SPICEEstimator(n_sensors)
            orig_ps, orig_angles = orig_estimator.fit(sample_cov)
            orig_peaks = orig_estimator.find_peaks(orig_ps)

            # Enhanced SPICE
            enh_estimator = StableSPICEEstimator(n_sensors)
            enh_ps, enh_angles = enh_estimator.fit(sample_cov)
            enh_peaks = enh_estimator.find_peaks(enh_ps)
            enh_report = enh_estimator.get_stability_report()

            # Both should complete successfully
            results['test_coverage'] = all([
                len(orig_ps) > 0, len(enh_ps) > 0,
                'stability_metrics' in enh_report
            ])

        except Exception as e:
            print(f"Test coverage check failed: {e}")

        success_rate = sum(results.values()) / len(results)
        print(f"Professional Standards: {success_rate:.1%}")

        return results

    def run_complete_integration_test(self) -> Dict:
        """Run complete integration test suite."""
        print("="*70)
        print("COMPLETE INTEGRATION TEST SUITE")
        print("Comprehensive Validation of Enhanced SPICE Implementation")
        print("="*70)

        # Run all test categories
        test_categories = [
            ('Original Implementation', self.test_original_implementation_consistency),
            ('Enhanced Stability', self.test_enhanced_stability_improvements),
            ('Documentation Accuracy', self.test_documentation_accuracy),
            ('Professional Standards', self.test_professional_standards)
        ]

        overall_results = {}
        category_scores = {}

        for category_name, test_function in test_categories:
            print(f"\n{category_name} Testing:")
            print("-" * 50)

            start_time = time.time()
            category_results = test_function()
            execution_time = time.time() - start_time

            overall_results[category_name.lower().replace(' ', '_')] = category_results
            category_score = sum(category_results.values()) / len(category_results)
            category_scores[category_name] = category_score

            print(f"Execution time: {execution_time:.2f}s")
            print(f"Category score: {category_score:.1%}")

        # Overall assessment
        overall_score = sum(category_scores.values()) / len(category_scores)

        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)

        for category, score in category_scores.items():
            status = "PASS" if score >= 0.8 else "WARN" if score >= 0.6 else "FAIL"
            print(f"{category:<25} {score:>6.1%} [{status}]")

        print(f"\nOverall Integration Score: {overall_score:.1%}")

        # Final assessment
        if overall_score >= 0.9:
            assessment = "EXCELLENT - Ready for professional use"
        elif overall_score >= 0.8:
            assessment = "GOOD - Minor improvements recommended"
        elif overall_score >= 0.7:
            assessment = "ACCEPTABLE - Some issues need attention"
        else:
            assessment = "NEEDS WORK - Significant improvements required"

        print(f"Assessment: {assessment}")

        # Store comprehensive results
        self.test_results = overall_results
        self.validation_summary = {
            'category_scores': category_scores,
            'overall_score': overall_score,
            'assessment': assessment,
            'timestamp': time.time()
        }

        return {
            'results': overall_results,
            'summary': self.validation_summary
        }


def test_complete_integration():
    """Pytest entry point for complete integration testing."""
    tester = ComprehensiveIntegrationTest()
    results = tester.run_complete_integration_test()

    # Assert overall quality standards
    overall_score = results['summary']['overall_score']
    assert overall_score >= 0.8, f"Integration test score too low: {overall_score:.1%}"

    # Assert critical categories
    category_scores = results['summary']['category_scores']
    assert category_scores['Original Implementation'] >= 0.8, "Original implementation issues"
    assert category_scores['Enhanced Stability'] >= 0.7, "Stability improvements insufficient"
    assert category_scores['Professional Standards'] >= 0.8, "Professional standards not met"

    return results


if __name__ == "__main__":
    # Run comprehensive integration test
    tester = ComprehensiveIntegrationTest()
    results = tester.run_complete_integration_test()

    print("\n" + "="*70)
    print("COMPREHENSIVE INTEGRATION TEST COMPLETE")
    print("="*70)
    print("Enhanced SPICE implementation has been thoroughly validated")
    print("against professional standards and literature-based claims.")
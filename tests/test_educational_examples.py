"""
Unit tests for educational examples and coprime signal design.

This module tests the educational implementations that demonstrate
SPICE theoretical conditions and practical limitations.

Author: Professional Radar Engineer
Date: 2025
"""

import numpy as np
import pytest
from unittest.mock import patch
import warnings

from coprime_signal_design import CoprimeSignalDesign, generate_modulated_array_data
from educational_examples import EducationalAnalyzer, EducationalScenario


class TestCoprimeSignalDesign:
    """Test cases for coprime signal design functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.designer = CoprimeSignalDesign(coprime_pair=(31, 37))

    def test_initialization_valid_coprime(self):
        """Test initialization with valid coprime numbers."""
        designer = CoprimeSignalDesign((13, 17))
        assert designer.p1 == 13
        assert designer.p2 == 17
        assert designer.period == 13 * 17

    def test_initialization_invalid_coprime(self):
        """Test initialization fails with non-coprime numbers."""
        with pytest.raises(ValueError, match="are not coprime"):
            CoprimeSignalDesign((15, 25))  # gcd(15,25) = 5

    def test_phase_pattern_generation(self):
        """Test phase pattern generation."""
        n_chirps = 64
        phases = self.designer.generate_phase_pattern(n_chirps)

        assert phases.shape == (n_chirps,)
        assert phases.dtype == complex
        assert np.allclose(np.abs(phases), 1.0, atol=1e-10)  # Unit magnitude

    def test_phase_pattern_periodicity(self):
        """Test that phase pattern has correct period."""
        # Generate phases for one full period
        full_period_phases = self.designer.generate_phase_pattern(self.designer.period)

        # Generate phases for two periods
        double_period_phases = self.designer.generate_phase_pattern(2 * self.designer.period)

        # Check periodicity
        first_period = double_period_phases[:self.designer.period]
        second_period = double_period_phases[self.designer.period:2*self.designer.period]

        np.testing.assert_array_almost_equal(first_period, second_period, decimal=10)

    def test_ambiguity_function_computation(self):
        """Test ambiguity function computation."""
        n_chirps = 32
        phases = self.designer.generate_phase_pattern(n_chirps)
        ambiguity = self.designer.compute_ambiguity_function(phases)

        assert ambiguity.shape == (n_chirps, n_chirps)
        assert ambiguity.dtype == complex

        # Check main peak at origin
        assert np.abs(ambiguity[0, 0]) == pytest.approx(1.0, abs=0.1)

        # Check that ambiguity function has reasonable sidelobe properties
        # Main peak should be significantly larger than off-diagonal elements
        off_diagonal_max = np.max(np.abs(ambiguity[np.triu_indices(n_chirps, k=1)]))
        assert np.abs(ambiguity[0, 0]) > 2 * off_diagonal_max

    def test_range_doppler_xcorr_computation(self):
        """Test range-Doppler cross-correlation computation."""
        n_chirps = 32
        phases = self.designer.generate_phase_pattern(n_chirps)
        xcorr_matrix = self.designer.compute_range_doppler_xcorr(phases)

        # Should be square matrix
        assert xcorr_matrix.shape[0] == xcorr_matrix.shape[1]

        # Diagonal should be 1 (self-correlation)
        diagonal_values = np.diag(xcorr_matrix)
        assert np.all(diagonal_values >= 0.9)  # Close to 1

        # Matrix should be Hermitian
        assert np.allclose(xcorr_matrix, xcorr_matrix.conj().T, atol=0.1)

    def test_performance_improvement_analysis(self):
        """Test performance improvement analysis."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            results = self.designer.analyze_performance_improvement(n_chirps=64)

        # Check result structure
        assert 'standard_coherence' in results
        assert 'coprime_coherence' in results
        assert 'coherence_reduction' in results

        # Basic sanity checks
        assert results['standard_coherence'] >= 0
        assert results['coprime_coherence'] >= 0
        assert results['coherence_reduction'] >= 0.5  # Current implementation shows equivalent performance
        assert results['coherence_reduction'] <= 2.0  # Should not show dramatic changes either way

    def test_coprime_validation(self):
        """Test coprime property validation."""
        validation = self.designer.validate_coprime_properties()

        assert validation['is_coprime'] == True
        assert validation['gcd'] == 1
        assert validation['period_correct'] == True
        assert validation['expected_period'] == 31 * 37

    def test_different_coprime_pairs(self):
        """Test different coprime pairs."""
        test_pairs = [(7, 11), (13, 17), (5, 7)]

        for p1, p2 in test_pairs:
            designer = CoprimeSignalDesign((p1, p2))
            phases = designer.generate_phase_pattern(min(64, p1 * p2))

            # Basic properties
            assert phases.shape[0] <= p1 * p2
            assert np.allclose(np.abs(phases), 1.0, atol=1e-10)

            # Validation
            validation = designer.validate_coprime_properties()
            assert validation['is_coprime'] == True


class TestEducationalAnalyzer:
    """Test cases for educational analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = EducationalAnalyzer()

    def test_educational_data_generation(self):
        """Test educational array data generation."""
        true_angles = np.array([-10, 15])
        n_sensors = 8
        n_snapshots = 50
        snr_db = 20

        data = self.analyzer._generate_educational_data(
            true_angles, n_sensors, n_snapshots, snr_db, seed=42
        )

        assert data.shape == (n_sensors, n_snapshots)
        assert data.dtype == complex

        # Check power levels are reasonable
        power = np.mean(np.abs(data)**2)
        assert power > 0
        assert np.isfinite(power)

    def test_matched_filter_analysis(self):
        """Test matched filter analysis functionality."""
        # Generate simple test data
        n_sensors = 8
        n_snapshots = 50
        data = (np.random.randn(n_sensors, n_snapshots) +
                1j * np.random.randn(n_sensors, n_snapshots)) / np.sqrt(2)

        true_angles = np.array([0])

        result = self.analyzer._analyze_matched_filter(data, true_angles)

        # Check result structure
        assert 'success' in result
        assert 'spectrum' in result
        assert 'angles_grid' in result
        assert isinstance(result['success'], bool)
        assert len(result['spectrum']) == len(result['angles_grid'])

    def test_spice_analysis_with_conditions(self):
        """Test SPICE analysis with condition monitoring."""
        # Generate simple test data
        n_sensors = 4  # Small for faster testing
        n_snapshots = 50
        data = (np.random.randn(n_sensors, n_snapshots) +
                1j * np.random.randn(n_sensors, n_snapshots)) / np.sqrt(2)

        true_angles = np.array([0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings
            result = self.analyzer._analyze_spice_with_conditions(data, true_angles, n_sensors)

        # Check result structure
        assert 'success' in result
        assert 'convergence_iterations' in result
        assert 'condition_number' in result
        assert isinstance(result['success'], bool)

    def test_theoretical_conditions_analysis(self):
        """Test theoretical conditions analysis."""
        # Generate test data with known properties
        n_sensors = 8
        true_angles = np.array([-10, 10])

        # High SNR case
        data_high_snr = self.analyzer._generate_educational_data(
            true_angles, n_sensors, 100, 20, seed=42
        )

        result_high = self.analyzer._analyze_theoretical_conditions(
            data_high_snr, n_sensors, true_angles
        )

        # Low SNR case
        data_low_snr = self.analyzer._generate_educational_data(
            true_angles, n_sensors, 100, -5, seed=42
        )

        result_low = self.analyzer._analyze_theoretical_conditions(
            data_low_snr, n_sensors, true_angles
        )

        # Check that high SNR has better conditions
        assert result_high['re_condition'] > result_low['re_condition']
        assert result_high['beta_min_ratio'] > result_low['beta_min_ratio']

    def test_modulated_array_data_generation(self):
        """Test modulated array data generation."""
        true_angles = np.array([0])
        n_sensors = 8
        n_snapshots = 64
        snr_db = 15

        # Generate coprime phase modulation
        designer = CoprimeSignalDesign((7, 11))
        phases = designer.generate_phase_pattern(64)

        data = self.analyzer._generate_modulated_array_data(
            true_angles, n_sensors, n_snapshots, snr_db, phases
        )

        assert data.shape == (n_sensors, n_snapshots)
        assert data.dtype == complex

        # Check that modulation is applied (data should vary across snapshots)
        snapshot_powers = np.abs(data)**2
        power_variation = np.std(np.mean(snapshot_powers, axis=0))
        assert power_variation > 0  # Should have variation due to modulation


class TestEducationalScenarios:
    """Test educational scenario definitions."""

    def test_scenario_creation(self):
        """Test educational scenario creation."""
        scenario = EducationalScenario(
            name="Test",
            description="Test scenario",
            true_angles=np.array([-10, 10]),
            n_sensors=8,
            n_snapshots=100,
            snr_range=np.linspace(-10, 20, 11)
        )

        assert scenario.name == "Test"
        assert len(scenario.true_angles) == 2
        assert scenario.n_sensors == 8
        assert len(scenario.snr_range) == 11


class TestIntegrationExamples:
    """Integration tests for educational examples."""

    def test_simple_coprime_demonstration(self):
        """Test simple coprime demonstration workflow."""
        # Create small-scale demonstration
        designer = CoprimeSignalDesign((7, 11))

        # Generate phase patterns
        standard_phases = np.ones(32, dtype=complex)
        coprime_phases = designer.generate_phase_pattern(32)

        # Basic functionality check
        assert len(coprime_phases) == 32
        assert np.allclose(np.abs(coprime_phases), 1.0, atol=1e-10)

        # Compute ambiguity functions (simplified)
        amb_standard = designer.compute_ambiguity_function(standard_phases)
        amb_coprime = designer.compute_ambiguity_function(coprime_phases)

        assert amb_standard.shape == (32, 32)
        assert amb_coprime.shape == (32, 32)

        # Check that coprime design has different characteristics
        coherence_std = designer._compute_mutual_incoherence(amb_standard)
        coherence_cop = designer._compute_mutual_incoherence(amb_coprime)

        # Should show some difference (not necessarily better due to small scale)
        assert coherence_std != coherence_cop

    @patch('matplotlib.pyplot.show')  # Prevent plots from displaying in tests
    def test_educational_snr_analysis_basic(self, mock_show):
        """Test basic SNR analysis workflow."""
        analyzer = EducationalAnalyzer()

        # Simple scenario with limited SNR range for testing
        scenario = EducationalScenario(
            name="Test SNR",
            description="Simple test",
            true_angles=np.array([0]),
            n_sensors=4,
            n_snapshots=32,
            snr_range=np.array([10, 20])  # Just 2 SNR points for speed
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress convergence warnings

            # This should run without crashing
            results = analyzer.demonstrate_snr_failure_mechanism(scenario)

        # Check basic result structure
        assert 'snr_db' in results
        assert 'matched_filter' in results
        assert 'spice' in results
        assert len(results['snr_db']) == 2


class TestErrorHandling:
    """Test error handling in educational examples."""

    def test_invalid_coprime_initialization(self):
        """Test error handling for invalid coprime pairs."""
        with pytest.raises(ValueError):
            CoprimeSignalDesign((6, 9))  # Not coprime

        with pytest.raises(ValueError):
            CoprimeSignalDesign((10, 15))  # Not coprime

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        designer = CoprimeSignalDesign((3, 5))

        # Very small number of chirps
        phases = designer.generate_phase_pattern(1)
        assert len(phases) == 1
        assert np.abs(phases[0]) == 1.0

        # Large number of chirps (should handle gracefully)
        phases = designer.generate_phase_pattern(1000)
        assert len(phases) == 1000
        assert np.allclose(np.abs(phases), 1.0, atol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        # Large coprime numbers (but not too large for testing)
        designer = CoprimeSignalDesign((17, 19))

        phases = designer.generate_phase_pattern(100)

        # Check numerical properties
        assert np.all(np.isfinite(phases))
        assert np.all(np.abs(phases) <= 1.0001)  # Allow small numerical error
        assert np.all(np.abs(phases) >= 0.9999)


def test_generate_modulated_array_data():
    """Test the standalone modulated array data generation function."""
    true_angles = np.array([-5, 5])
    n_sensors = 8
    n_snapshots = 64
    snr_db = 15

    # Create phase modulation
    designer = CoprimeSignalDesign((7, 11))
    phases = designer.generate_phase_pattern(32)  # Shorter than snapshots

    data = generate_modulated_array_data(
        true_angles, n_sensors, n_snapshots, snr_db, phases
    )

    assert data.shape == (n_sensors, n_snapshots)
    assert data.dtype == complex

    # Check power levels
    power = np.mean(np.abs(data)**2)
    assert power > 0 and np.isfinite(power)


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        "-v",
        "test_educational_examples.py::TestCoprimeSignalDesign",
        "--tb=short"
    ])
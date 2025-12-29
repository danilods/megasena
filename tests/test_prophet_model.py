"""
Unit tests for ProphetPredictor class.

Note: These tests verify the class structure and methods.
Full functionality tests require the prophet library to be installed.
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.prophet_model import ProphetPredictor, MultiTargetProphetPredictor


class TestProphetPredictorStructure(unittest.TestCase):
    """Test ProphetPredictor class structure."""

    def test_class_exists(self):
        """Test that ProphetPredictor class exists."""
        self.assertIsNotNone(ProphetPredictor)

    def test_initialization(self):
        """Test that ProphetPredictor can be initialized."""
        predictor = ProphetPredictor()
        self.assertIsNotNone(predictor)

    def test_has_required_methods(self):
        """Test that all required methods exist."""
        predictor = ProphetPredictor()

        # Check for required methods
        required_methods = [
            'fit',
            'predict',
            'get_trend_analysis',
            'get_seasonality_analysis',
            'analyze_number_trend',
            'prepare_data',
            'get_forecast_with_intervals',
            'plot_components_data',
            'analyze_all_numbers_trends',
            'get_model_params',
            'evaluate_prediction_quality',
            '_create_model'
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(predictor, method_name),
                f"ProphetPredictor missing method: {method_name}"
            )
            self.assertTrue(
                callable(getattr(predictor, method_name)),
                f"{method_name} is not callable"
            )

    def test_initialization_parameters(self):
        """Test that initialization parameters are stored correctly."""
        predictor = ProphetPredictor(
            yearly_seasonality=True,
            weekly_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1,
            interval_width=0.90
        )

        self.assertEqual(predictor.yearly_seasonality, True)
        self.assertEqual(predictor.weekly_seasonality, False)
        self.assertEqual(predictor.seasonality_mode, 'multiplicative')
        self.assertEqual(predictor.changepoint_prior_scale, 0.1)
        self.assertEqual(predictor.interval_width, 0.90)

    def test_initial_state(self):
        """Test that initial state is correct."""
        predictor = ProphetPredictor()

        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.forecast)
        self.assertIsNone(predictor.components)
        self.assertIsNone(predictor.target_column)
        self.assertIsNone(predictor.data)

    def test_get_model_params(self):
        """Test get_model_params method."""
        predictor = ProphetPredictor(
            yearly_seasonality='auto',
            weekly_seasonality=False,
            changepoint_prior_scale=0.05
        )

        params = predictor.get_model_params()

        self.assertIsInstance(params, dict)
        self.assertIn('yearly_seasonality', params)
        self.assertIn('weekly_seasonality', params)
        self.assertIn('changepoint_prior_scale', params)
        self.assertEqual(params['yearly_seasonality'], 'auto')
        self.assertEqual(params['weekly_seasonality'], False)
        self.assertEqual(params['changepoint_prior_scale'], 0.05)


class TestMultiTargetProphetPredictor(unittest.TestCase):
    """Test MultiTargetProphetPredictor class."""

    def test_class_exists(self):
        """Test that MultiTargetProphetPredictor class exists."""
        self.assertIsNotNone(MultiTargetProphetPredictor)

    def test_initialization(self):
        """Test that MultiTargetProphetPredictor can be initialized."""
        predictor = MultiTargetProphetPredictor()
        self.assertIsNotNone(predictor)

    def test_has_required_methods(self):
        """Test that all required methods exist."""
        predictor = MultiTargetProphetPredictor()

        required_methods = [
            'fit_multiple_targets',
            'predict_all',
            'get_all_trends',
            'get_predictor'
        ]

        for method_name in required_methods:
            self.assertTrue(
                hasattr(predictor, method_name),
                f"MultiTargetProphetPredictor missing method: {method_name}"
            )

    def test_initial_state(self):
        """Test initial state."""
        predictor = MultiTargetProphetPredictor()

        self.assertIsInstance(predictor.predictors, dict)
        self.assertEqual(len(predictor.predictors), 0)

    def test_get_predictor_empty(self):
        """Test get_predictor when no predictors exist."""
        predictor = MultiTargetProphetPredictor()

        result = predictor.get_predictor('non_existent')
        self.assertIsNone(result)


class TestProphetAvailability(unittest.TestCase):
    """Test Prophet library availability handling."""

    def test_prophet_import(self):
        """Test that the module handles Prophet import gracefully."""
        # This should not raise an error even if Prophet is not installed
        try:
            from src.models.prophet_model import PROPHET_AVAILABLE
            self.assertIsInstance(PROPHET_AVAILABLE, bool)
        except Exception as e:
            self.fail(f"Module failed to import: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProphetPredictorStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTargetProphetPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestProphetAvailability))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)

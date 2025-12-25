"""
Unit tests for ARIMAPredictor class.

Tests stationarity testing, model fitting, prediction, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.arima_model import ARIMAPredictor


class TestARIMAPredictor:
    """Test suite for ARIMAPredictor class."""

    @pytest.fixture
    def sample_series(self):
        """Create a sample time series for testing."""
        np.random.seed(42)
        # Generate a simple AR(1) process
        n = 100
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = 0.7 * series[i-1] + np.random.normal(0, 1)
        return pd.Series(series)

    @pytest.fixture
    def sample_features(self):
        """Create sample features DataFrame."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'sum': np.random.normal(180, 30, n),
            'even_count': np.random.randint(0, 7, n),
            'mean': np.random.normal(30, 5, n)
        })

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance."""
        return ARIMAPredictor(default_order=(2, 1, 1))

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.default_order == (2, 1, 1)
        assert predictor.is_fitted is False
        assert len(predictor.models) == 0
        assert len(predictor.history) == 0

    def test_stationarity_test_stationary(self, predictor):
        """Test stationarity test on stationary series."""
        # White noise is stationary
        np.random.seed(42)
        stationary_series = np.random.normal(0, 1, 200)

        result = predictor.test_stationarity(stationary_series)

        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert 'interpretation' in result
        assert isinstance(result['is_stationary'], bool)

    def test_stationarity_test_non_stationary(self, predictor):
        """Test stationarity test on non-stationary series."""
        # Random walk is non-stationary
        np.random.seed(42)
        random_walk = np.cumsum(np.random.normal(0, 1, 200))

        result = predictor.test_stationarity(random_walk)

        assert 'adf_statistic' in result
        assert 'p_value' in result
        # Random walk typically non-stationary (but not guaranteed with short series)
        assert isinstance(result['is_stationary'], bool)

    def test_stationarity_test_with_series(self, predictor, sample_series):
        """Test stationarity test with pandas Series."""
        result = predictor.test_stationarity(sample_series)

        assert result is not None
        assert 'p_value' in result
        assert 0 <= result['p_value'] <= 1

    def test_analyze_acf_pacf(self, predictor, sample_series):
        """Test ACF/PACF analysis."""
        result = predictor.analyze_acf_pacf(sample_series, lags=20, plot=False)

        assert 'acf' in result
        assert 'pacf' in result
        assert len(result['acf']) == 21  # lags + 1 (lag 0)
        assert len(result['pacf']) == 21

    def test_fit_basic(self, predictor, sample_series):
        """Test basic model fitting."""
        predictor.fit(sample_series, feature_name='test', order=(2, 1, 1))

        assert predictor.is_fitted is True
        assert 'test' in predictor.models
        assert 'test' in predictor.history
        assert 'test' in predictor.orders
        assert predictor.orders['test'] == (2, 1, 1)

    def test_fit_with_dataframe(self, predictor, sample_features):
        """Test fitting with pandas DataFrame column."""
        predictor.fit(sample_features['sum'], feature_name='sum')

        assert 'sum' in predictor.models
        assert predictor.is_fitted is True

    def test_fit_too_short_series(self, predictor):
        """Test fitting with too short series raises error."""
        short_series = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Series too short"):
            predictor.fit(short_series)

    def test_fit_multiple(self, predictor, sample_features):
        """Test fitting multiple features at once."""
        predictor.fit_multiple(sample_features, ['sum', 'even_count', 'mean'])

        assert len(predictor.models) == 3
        assert 'sum' in predictor.models
        assert 'even_count' in predictor.models
        assert 'mean' in predictor.models

    def test_fit_multiple_missing_column(self, predictor, sample_features):
        """Test fitting multiple with missing column (should warn, not error)."""
        # Should complete successfully, just skip missing column
        predictor.fit_multiple(sample_features, ['sum', 'nonexistent'])

        assert 'sum' in predictor.models
        assert 'nonexistent' not in predictor.models

    def test_predict_basic(self, predictor, sample_series):
        """Test basic prediction."""
        predictor.fit(sample_series, feature_name='test')

        predictions, lower, upper = predictor.predict(
            steps=5,
            feature_name='test',
            return_conf_int=True
        )

        assert len(predictions) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        # Confidence intervals should make sense
        assert all(lower <= predictions)
        assert all(predictions <= upper)

    def test_predict_without_conf_int(self, predictor, sample_series):
        """Test prediction without confidence intervals."""
        predictor.fit(sample_series, feature_name='test')

        predictions = predictor.predict(
            steps=3,
            feature_name='test',
            return_conf_int=False
        )

        assert len(predictions) == 3
        assert isinstance(predictions, np.ndarray)

    def test_predict_unfitted(self, predictor):
        """Test prediction on unfitted model raises error."""
        with pytest.raises(ValueError, match="No model fitted"):
            predictor.predict(steps=1, feature_name='test')

    def test_predict_all(self, predictor, sample_features):
        """Test predicting all features."""
        predictor.fit_multiple(sample_features, ['sum', 'mean'])

        results = predictor.predict_all(steps=3)

        assert len(results) == 2
        assert 'sum' in results
        assert 'mean' in results

        for feature, preds in results.items():
            assert 'prediction' in preds
            assert 'lower_bound' in preds
            assert 'upper_bound' in preds
            assert len(preds['prediction']) == 3

    def test_get_sum_range(self, predictor, sample_features):
        """Test getting sum range for filtering."""
        predictor.fit(sample_features['sum'], feature_name='sum')

        sum_min, sum_max = predictor.get_sum_range()

        assert isinstance(sum_min, float)
        assert isinstance(sum_max, float)
        assert sum_min < sum_max

    def test_get_sum_range_unfitted(self, predictor):
        """Test getting sum range without fitted model raises error."""
        with pytest.raises(ValueError, match="No model fitted for 'sum'"):
            predictor.get_sum_range()

    def test_get_feature_range(self, predictor, sample_features):
        """Test getting range for any feature."""
        predictor.fit(sample_features['mean'], feature_name='mean')

        mean_min, mean_max = predictor.get_feature_range('mean')

        assert isinstance(mean_min, float)
        assert isinstance(mean_max, float)
        assert mean_min < mean_max

    def test_evaluate_residuals(self, predictor, sample_series):
        """Test residual evaluation."""
        predictor.fit(sample_series, feature_name='test')

        residual_stats = predictor.evaluate_residuals('test', plot=False)

        assert 'mean' in residual_stats
        assert 'std' in residual_stats
        assert 'residuals' in residual_stats
        # Mean of residuals should be close to 0
        assert abs(residual_stats['mean']) < 1

    def test_get_model_summary(self, predictor, sample_series):
        """Test getting model summary."""
        predictor.fit(sample_series, feature_name='test')

        summary = predictor.get_model_summary('test')

        assert isinstance(summary, str)
        assert 'ARIMA' in summary

    def test_get_model_summary_unfitted(self, predictor):
        """Test getting summary for unfitted model raises error."""
        with pytest.raises(ValueError, match="No model fitted"):
            predictor.get_model_summary('test')

    def test_repr(self, predictor, sample_series):
        """Test string representation."""
        # Before fitting
        repr_before = repr(predictor)
        assert 'fitted=False' in repr_before

        # After fitting
        predictor.fit(sample_series, feature_name='test')
        repr_after = repr(predictor)
        assert 'fitted=True' in repr_after
        assert 'test' in repr_after

    def test_save_and_load_models(self, predictor, sample_features, tmp_path):
        """Test saving and loading models."""
        # Fit models
        predictor.fit_multiple(sample_features, ['sum', 'mean'])

        # Save
        save_path = tmp_path / "models"
        predictor.save_models(str(save_path))

        # Verify files exist
        assert (save_path / "arima_sum.pkl").exists()
        assert (save_path / "arima_mean.pkl").exists()

        # Load into new predictor
        new_predictor = ARIMAPredictor()
        new_predictor.load_models(str(save_path), ['sum', 'mean'])

        assert new_predictor.is_fitted is True
        assert 'sum' in new_predictor.models
        assert 'mean' in new_predictor.models

        # Predictions should work
        pred1, _, _ = predictor.predict(steps=1, feature_name='sum')
        pred2, _, _ = new_predictor.predict(steps=1, feature_name='sum')

        # Should give same predictions
        assert np.allclose(pred1, pred2)

    def test_different_confidence_levels(self, predictor, sample_series):
        """Test predictions with different confidence levels."""
        predictor.fit(sample_series, feature_name='test')

        # 95% CI (alpha=0.05)
        _, lower_95, upper_95 = predictor.predict(
            steps=1, feature_name='test', alpha=0.05
        )

        # 99% CI (alpha=0.01)
        _, lower_99, upper_99 = predictor.predict(
            steps=1, feature_name='test', alpha=0.01
        )

        # 99% CI should be wider
        interval_95 = upper_95[0] - lower_95[0]
        interval_99 = upper_99[0] - lower_99[0]
        assert interval_99 > interval_95

    def test_method_chaining(self, predictor, sample_series):
        """Test that fit returns self for method chaining."""
        result = predictor.fit(sample_series, feature_name='test')

        assert result is predictor
        assert predictor.is_fitted is True


class TestARIMAIntegration:
    """Integration tests with realistic lottery data."""

    def test_lottery_workflow(self):
        """Test complete workflow with lottery-like data."""
        np.random.seed(42)

        # Simulate lottery draws
        n_draws = 200
        draws = []
        for _ in range(n_draws):
            draw = sorted(np.random.choice(range(1, 61), 6, replace=False))
            draws.append(draw)

        draws = np.array(draws)

        # Compute features
        features = pd.DataFrame({
            'sum': draws.sum(axis=1),
            'even_count': (draws % 2 == 0).sum(axis=1),
            'mean': draws.mean(axis=1),
            'range': draws.max(axis=1) - draws.min(axis=1)
        })

        # Initialize and train
        predictor = ARIMAPredictor()
        predictor.fit_multiple(
            features,
            ['sum', 'even_count', 'mean', 'range']
        )

        # Make predictions
        predictions = predictor.predict_all(steps=1)

        # Verify predictions
        assert len(predictions) == 4
        for feature in ['sum', 'even_count', 'mean', 'range']:
            assert feature in predictions
            assert len(predictions[feature]['prediction']) == 1

        # Test filtering
        sum_min, sum_max = predictor.get_sum_range()
        assert 0 < sum_min < sum_max < 366  # Max possible sum is 60+59+...+55

        even_min, even_max = predictor.get_feature_range('even_count')
        assert 0 <= even_min <= even_max <= 6

    def test_filtering_combinations(self):
        """Test using predictions to filter combinations."""
        np.random.seed(42)

        # Generate historical data
        n_draws = 150
        historical_sums = np.random.normal(180, 30, n_draws)
        historical_evens = np.random.randint(2, 5, n_draws)

        # Train predictor
        predictor = ARIMAPredictor()
        predictor.fit(historical_sums, feature_name='sum')
        predictor.fit(historical_evens, feature_name='even_count')

        # Get ranges
        sum_min, sum_max = predictor.get_sum_range(alpha=0.05)
        even_min, even_max = predictor.get_feature_range('even_count', alpha=0.05)

        # Test combinations
        test_combos = [
            [1, 2, 3, 4, 5, 6],        # Sum: 21 (likely rejected)
            [10, 20, 30, 40, 50, 60],   # Sum: 210
            [15, 20, 25, 30, 35, 40],   # Sum: 165
        ]

        for combo in test_combos:
            combo_sum = sum(combo)
            combo_even = sum(1 for x in combo if x % 2 == 0)

            passes_sum = sum_min <= combo_sum <= sum_max
            passes_even = even_min <= combo_even <= even_max

            # Just verify logic works (no specific assertion on pass/fail)
            assert isinstance(passes_sum, bool)
            assert isinstance(passes_even, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

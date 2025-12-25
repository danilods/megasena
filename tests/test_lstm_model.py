"""
Unit tests for the LSTM model.

Tests cover:
- Model initialization
- Sequence creation
- Model building
- Training
- Prediction
- Model persistence
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import LSTMPredictor
from config.settings import TOTAL_NUMBERS


class TestLSTMPredictor:
    """Test suite for LSTMPredictor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample binary encoded lottery data."""
        np.random.seed(42)
        n_draws = 50
        data = np.zeros((n_draws, TOTAL_NUMBERS))

        # Simulate lottery draws
        for i in range(n_draws):
            numbers = np.random.choice(TOTAL_NUMBERS, size=6, replace=False)
            data[i, numbers] = 1

        return data

    @pytest.fixture
    def predictor(self):
        """Create a fresh LSTMPredictor instance."""
        return LSTMPredictor(lookback=10)

    def test_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert predictor.lookback == 10
        assert predictor.model is None
        assert predictor.history is None

    def test_initialization_custom_lookback(self):
        """Test initialization with custom lookback."""
        predictor = LSTMPredictor(lookback=15)
        assert predictor.lookback == 15

    def test_create_sequences(self, predictor, sample_data):
        """Test sequence creation from data."""
        X, y = predictor.create_sequences(sample_data, lookback=10)

        # Check shapes
        assert X.shape == (40, 10, TOTAL_NUMBERS)
        assert y.shape == (40, TOTAL_NUMBERS)

        # Check that sequences are correct
        # First sequence should be draws 0-9, target should be draw 10
        np.testing.assert_array_equal(X[0], sample_data[0:10])
        np.testing.assert_array_equal(y[0], sample_data[10])

    def test_create_sequences_insufficient_data(self, predictor):
        """Test that error is raised with insufficient data."""
        insufficient_data = np.zeros((5, TOTAL_NUMBERS))

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.create_sequences(insufficient_data, lookback=10)

    def test_build_model(self, predictor):
        """Test model building."""
        input_shape = (10, TOTAL_NUMBERS)
        model = predictor.build_model(input_shape)

        # Check that model is created
        assert model is not None
        assert predictor.model is not None

        # Check model architecture
        assert len(model.layers) == 5  # 2 LSTM, 2 Dropout, 1 Dense

        # Check output shape
        assert model.output_shape == (None, TOTAL_NUMBERS)

    def test_predict_proba_single_sequence(self, predictor, sample_data):
        """Test probability prediction for single sequence."""
        # Build model
        predictor.build_model((10, TOTAL_NUMBERS))

        # Create a single sequence
        sequence = sample_data[:10]

        # Get predictions
        proba = predictor.predict_proba(sequence)

        # Check output shape and values
        assert proba.shape == (TOTAL_NUMBERS,)
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Probabilities
        assert np.abs(proba.sum() - TOTAL_NUMBERS / 2) < TOTAL_NUMBERS  # Reasonable sum

    def test_predict_proba_multiple_sequences(self, predictor, sample_data):
        """Test probability prediction for multiple sequences."""
        # Build model
        predictor.build_model((10, TOTAL_NUMBERS))

        # Create multiple sequences
        X, _ = predictor.create_sequences(sample_data, lookback=10)

        # Get predictions
        proba = predictor.predict_proba(X[:5])

        # Check output shape
        assert proba.shape == (5, TOTAL_NUMBERS)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_top_k(self, predictor, sample_data):
        """Test top-k prediction."""
        # Build model
        predictor.build_model((10, TOTAL_NUMBERS))

        # Get single sequence
        sequence = sample_data[:10]

        # Get top 6 predictions
        top_6 = predictor.predict_top_k(sequence, k=6)

        # Check output
        assert len(top_6) == 6
        assert all(isinstance(item, tuple) for item in top_6)
        assert all(len(item) == 2 for item in top_6)

        # Check that numbers are in valid range
        numbers = [num for num, _ in top_6]
        assert all(1 <= num <= TOTAL_NUMBERS for num in numbers)

        # Check that probabilities are in descending order
        probabilities = [prob for _, prob in top_6]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_predict_top_k_invalid(self, predictor, sample_data):
        """Test that invalid k raises error."""
        predictor.build_model((10, TOTAL_NUMBERS))
        sequence = sample_data[:10]

        with pytest.raises(ValueError, match="k must be between"):
            predictor.predict_top_k(sequence, k=0)

        with pytest.raises(ValueError, match="k must be between"):
            predictor.predict_top_k(sequence, k=61)

    def test_fit_basic(self, predictor, sample_data):
        """Test basic model training."""
        # Create sequences
        X, y = predictor.create_sequences(sample_data, lookback=10)

        # Train model
        history = predictor.fit(
            X, y,
            epochs=2,
            batch_size=8,
            validation_split=0.2,
            verbose=0,
            patience=5,
            save_best=False
        )

        # Check that model was trained
        assert predictor.model is not None
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history

    def test_save_and_load_model(self, predictor, sample_data):
        """Test model saving and loading."""
        # Build and train a simple model
        X, y = predictor.create_sequences(sample_data, lookback=10)
        predictor.fit(X, y, epochs=1, verbose=0, save_best=False)

        # Save model to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.keras"
            predictor.save_model(model_path)

            # Check file was created
            assert model_path.exists()

            # Load model in new predictor
            new_predictor = LSTMPredictor(lookback=10)
            new_predictor.load_model(model_path)

            # Check that loaded model works
            assert new_predictor.model is not None

            # Compare predictions
            test_sequence = sample_data[:10]
            original_pred = predictor.predict_proba(test_sequence)
            loaded_pred = new_predictor.predict_proba(test_sequence)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_save_model_without_training(self, predictor):
        """Test that saving without a model raises error."""
        with pytest.raises(ValueError, match="No model to save"):
            predictor.save_model("dummy_path.keras")

    def test_load_nonexistent_model(self, predictor):
        """Test that loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model("nonexistent_model.keras")

    def test_evaluate(self, predictor, sample_data):
        """Test model evaluation."""
        # Build and train model
        X, y = predictor.create_sequences(sample_data, lookback=10)
        predictor.fit(X, y, epochs=1, verbose=0, save_best=False)

        # Evaluate
        metrics = predictor.evaluate(X, y, verbose=0)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_get_model_summary(self, predictor):
        """Test getting model summary."""
        # Build model
        predictor.build_model((10, TOTAL_NUMBERS))

        # Get summary
        summary = predictor.get_model_summary()

        # Check that summary contains expected information
        assert isinstance(summary, str)
        assert 'lstm_layer_1' in summary
        assert 'lstm_layer_2' in summary
        assert 'dropout' in summary.lower()
        assert 'output_layer' in summary

    def test_get_model_summary_no_model(self, predictor):
        """Test that getting summary without model raises error."""
        with pytest.raises(ValueError, match="Model has not been built"):
            predictor.get_model_summary()

    def test_repr(self, predictor):
        """Test string representation."""
        repr_str = repr(predictor)

        assert 'LSTMPredictor' in repr_str
        assert 'lookback=10' in repr_str
        assert 'not trained' in repr_str

        # After building model
        predictor.build_model((10, TOTAL_NUMBERS))
        repr_str = repr(predictor)
        assert 'trained' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

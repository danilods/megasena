"""
LSTM Deep Learning model for Mega-Sena lottery prediction.

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting Mega-Sena lottery numbers based on historical draw patterns.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config.settings import (
    TOTAL_NUMBERS,
    LSTM_LOOKBACK,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_UNITS,
    DROPOUT_RATE,
    RANDOM_SEED,
    MODELS_PATH
)


class LSTMPredictor:
    """
    LSTM-based predictor for Mega-Sena lottery numbers.

    This class implements a multi-layer LSTM neural network that learns
    patterns from historical lottery draws to predict probabilities for
    each number in the lottery.

    Architecture:
        - Input: Sequences of past draws (configurable lookback)
        - LSTM Layer 1: 100 units with dropout
        - LSTM Layer 2: 50 units with dropout
        - Output: 60 neurons with sigmoid activation (one per number)

    Attributes:
        lookback (int): Number of past draws to use for prediction.
        model (keras.Model): The compiled LSTM model.
        history (keras.callbacks.History): Training history.
    """

    def __init__(self, lookback: int = LSTM_LOOKBACK):
        """
        Initialize the LSTM predictor.

        Args:
            lookback (int): Number of previous draws to consider for prediction.
                Defaults to LSTM_LOOKBACK from settings.
        """
        self.lookback = lookback
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None

        # Configure GPU settings
        self._configure_gpu()

        # Set random seeds for reproducibility
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

    def _configure_gpu(self) -> None:
        """
        Configure GPU settings and handle availability gracefully.

        Detects GPU availability and configures memory growth to prevent
        TensorFlow from allocating all GPU memory at once.
        """
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) detected and configured: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"GPU configuration warning: {e}")
        else:
            print("No GPU detected. Using CPU for training.")

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build and compile the LSTM model.

        Creates a sequential model with:
        - Multiple LSTM layers with dropout for regularization
        - Dense output layer with sigmoid activation
        - Binary crossentropy loss for multi-label classification
        - Adam optimizer

        Args:
            input_shape (Tuple[int, int]): Shape of input data (timesteps, features).

        Returns:
            keras.Model: Compiled LSTM model.
        """
        model = Sequential([
            # First LSTM layer
            LSTM(
                units=LSTM_UNITS[0],
                return_sequences=True,
                input_shape=input_shape,
                name='lstm_layer_1'
            ),
            Dropout(DROPOUT_RATE, name='dropout_1'),

            # Second LSTM layer
            LSTM(
                units=LSTM_UNITS[1],
                return_sequences=False,
                name='lstm_layer_2'
            ),
            Dropout(DROPOUT_RATE, name='dropout_2'),

            # Output layer with sigmoid activation for probabilities
            Dense(
                TOTAL_NUMBERS,
                activation='sigmoid',
                name='output_layer'
            )
        ])

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['binary_accuracy', 'precision', 'recall']
        )

        self.model = model

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        return model

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from preprocessed data.

        Transforms the data into sequences suitable for LSTM training.
        Each sequence consists of 'lookback' consecutive draws, and the
        target is the next draw after the sequence.

        Args:
            data (np.ndarray): Binary encoded draws (n_draws, 60).
                Each row is a binary vector where 1 indicates the number was drawn.
            lookback (int, optional): Number of past draws per sequence.
                Defaults to self.lookback.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) where:
                - X: Input sequences of shape (samples, lookback, 60)
                - y: Target draws of shape (samples, 60)

        Raises:
            ValueError: If data has insufficient samples for creating sequences.
        """
        if lookback is None:
            lookback = self.lookback

        if len(data) < lookback + 1:
            raise ValueError(
                f"Insufficient data: need at least {lookback + 1} samples, "
                f"got {len(data)}"
            )

        X, y = [], []

        for i in range(len(data) - lookback):
            # Sequence of 'lookback' draws
            X.append(data[i:i + lookback])
            # Target is the next draw
            y.append(data[i + lookback])

        X = np.array(X)
        y = np.array(y)

        print(f"\nCreated sequences:")
        print(f"  Input shape (X): {X.shape}")
        print(f"  Target shape (y): {y.shape}")

        return X, y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        validation_split: float = 0.2,
        verbose: int = 1,
        patience: int = 10,
        save_best: bool = True,
        model_path: Optional[Union[str, Path]] = None
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.

        Trains the model with early stopping to prevent overfitting and
        optionally saves the best model during training.

        Args:
            X (np.ndarray): Input sequences of shape (samples, lookback, features).
            y (np.ndarray): Target labels of shape (samples, features).
            epochs (int): Maximum number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of data to use for validation.
            verbose (int): Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).
            patience (int): Number of epochs with no improvement for early stopping.
            save_best (bool): Whether to save the best model during training.
            model_path (str or Path, optional): Path to save the best model.
                Defaults to MODELS_PATH/lstm_best.keras.

        Returns:
            keras.callbacks.History: Training history object.
        """
        if self.model is None:
            # Build model with appropriate input shape
            input_shape = (X.shape[1], X.shape[2])  # (lookback, features)
            self.build_model(input_shape)

        # Setup callbacks
        callback_list = []

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)

        # Model checkpoint to save best model
        if save_best:
            if model_path is None:
                MODELS_PATH.mkdir(parents=True, exist_ok=True)
                model_path = MODELS_PATH / 'lstm_best.keras'
            else:
                model_path = Path(model_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = ModelCheckpoint(
                filepath=str(model_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)

        # Train the model
        print(f"\nTraining LSTM model...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation split: {validation_split}")

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose
        )

        print("\nTraining completed!")

        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over all 60 numbers.

        Args:
            X (np.ndarray): Input sequences of shape (samples, lookback, features)
                or (lookback, features) for a single prediction.

        Returns:
            np.ndarray: Probability distribution of shape (samples, 60)
                or (60,) for a single prediction. Each value represents
                the probability of that number being drawn.

        Raises:
            ValueError: If model has not been built or trained.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Handle single sequence input
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Get predictions
        predictions = self.model.predict(X, verbose=0)

        # Return single prediction if input was single sequence
        if predictions.shape[0] == 1:
            return predictions[0]

        return predictions

    def predict_top_k(
        self,
        X: np.ndarray,
        k: int = 6
    ) -> List[Tuple[int, float]]:
        """
        Get top K most likely numbers.

        Args:
            X (np.ndarray): Input sequence of shape (lookback, features)
                or (samples, lookback, features).
            k (int): Number of top predictions to return. Defaults to 6.

        Returns:
            List[Tuple[int, float]]: List of (number, probability) tuples
                sorted by probability in descending order. Numbers are
                in range [1, 60].

        Raises:
            ValueError: If k is not between 1 and 60.
        """
        if not 1 <= k <= TOTAL_NUMBERS:
            raise ValueError(f"k must be between 1 and {TOTAL_NUMBERS}, got {k}")

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Handle multiple predictions - use the last one
        if probabilities.ndim == 2:
            probabilities = probabilities[-1]

        # Get top k indices (0-based)
        top_k_indices = np.argsort(probabilities)[-k:][::-1]

        # Convert to 1-based numbering and create (number, probability) tuples
        top_k_predictions = [
            (int(idx + 1), float(probabilities[idx]))
            for idx in top_k_indices
        ]

        return top_k_predictions

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path (str or Path): Path where the model should be saved.
                Recommended extension: .keras or .h5

        Raises:
            ValueError: If model has not been built.
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        print(f"Model saved to: {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.

        Args:
            path (str or Path): Path to the saved model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = keras.models.load_model(str(path))
        print(f"Model loaded from: {path}")

    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.

        Returns:
            str: Model summary as a string.

        Raises:
            ValueError: If model has not been built.
        """
        if self.model is None:
            raise ValueError("Model has not been built.")

        # Capture model summary
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        summary = summary_buffer.getvalue()

        return summary

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 1
    ) -> dict:
        """
        Evaluate the model on test data.

        Args:
            X (np.ndarray): Test sequences of shape (samples, lookback, features).
            y (np.ndarray): True labels of shape (samples, features).
            verbose (int): Verbosity mode.

        Returns:
            dict: Dictionary containing evaluation metrics.

        Raises:
            ValueError: If model has not been built or trained.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Evaluate the model
        results = self.model.evaluate(X, y, verbose=verbose)

        # Get metric names
        metric_names = self.model.metrics_names

        # Create results dictionary
        metrics = dict(zip(metric_names, results))

        if verbose:
            print("\nEvaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        return metrics

    def __repr__(self) -> str:
        """String representation of the LSTMPredictor."""
        status = "trained" if self.model is not None else "not trained"
        return f"LSTMPredictor(lookback={self.lookback}, status='{status}')"

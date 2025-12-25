"""
DataPreprocessor module for Mega-Sena lottery data.

This module provides functionality to preprocess lottery data for machine learning,
including normalization, train/test splitting, and sequence generation.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    TOTAL_NUMBERS,
    NUMBERS_PER_DRAW,
    TRAIN_TEST_SPLIT,
    LSTM_LOOKBACK,
    RANDOM_SEED
)


class DataPreprocessor:
    """
    DataPreprocessor class for Mega-Sena lottery data.

    This class handles data preprocessing tasks including:
    - Normalization of ball numbers for neural networks
    - Train/test splitting with chronological ordering (avoiding data leakage)
    - Creating sliding window sequences for LSTM models
    - Inverse transformation of predictions back to original scale

    Attributes:
        scaler: MinMaxScaler instance for normalizing data.
        lookback: Number of previous draws to use for sequence generation.
        train_test_split: Fraction of data to use for training.
    """

    def __init__(
        self,
        lookback: Optional[int] = None,
        train_test_split: Optional[float] = None
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            lookback: Number of previous draws to use for LSTM sequences.
                     If not provided, uses LSTM_LOOKBACK from settings.
            train_test_split: Fraction of data to use for training (0-1).
                            If not provided, uses TRAIN_TEST_SPLIT from settings.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = lookback or LSTM_LOOKBACK
        self.train_test_split = train_test_split or TRAIN_TEST_SPLIT
        self._is_fitted = False

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize ball numbers to range [0, 1] using MinMaxScaler.

        The scaler is fitted on the range [1, TOTAL_NUMBERS] to ensure
        consistent scaling even for unseen data.

        Args:
            data: Array of shape (n_samples, n_features) containing ball numbers.
            fit: Whether to fit the scaler on the data. Set to False when
                normalizing test/validation data.

        Returns:
            Normalized array with values in range [0, 1].

        Raises:
            ValueError: If data has invalid shape.
        """
        if data.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {data.ndim}D array with shape {data.shape}"
            )

        if fit:
            # Fit on the theoretical range [1, TOTAL_NUMBERS] for all features
            # This ensures consistent scaling regardless of actual data
            fit_data = np.array([[1] * data.shape[1], [TOTAL_NUMBERS] * data.shape[1]])
            self.scaler.fit(fit_data)
            self._is_fitted = True

        if not self._is_fitted:
            raise ValueError(
                "Scaler not fitted. Call normalize() with fit=True first."
            )

        return self.scaler.transform(data)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            data: Normalized array of shape (n_samples, n_features) with
                 values in range [0, 1].

        Returns:
            Array with values transformed back to range [1, TOTAL_NUMBERS].

        Raises:
            ValueError: If scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise ValueError(
                "Scaler not fitted. Call normalize() with fit=True first."
            )

        return self.scaler.inverse_transform(data)

    def split_train_test(
        self,
        data: np.ndarray,
        split_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and test sets chronologically.

        This method maintains temporal ordering to avoid data leakage.
        The split is done chronologically, with earlier draws in the training
        set and later draws in the test set.

        Args:
            data: Array of shape (n_samples, n_features) to split.
            split_ratio: Optional custom split ratio. If not provided,
                        uses self.train_test_split.

        Returns:
            Tuple of (train_data, test_data).

        Raises:
            ValueError: If split_ratio is not between 0 and 1.
        """
        split_ratio = split_ratio or self.train_test_split

        if not 0 < split_ratio < 1:
            raise ValueError(
                f"split_ratio must be between 0 and 1, got {split_ratio}"
            )

        # Calculate split index (chronological split)
        split_idx = int(len(data) * split_ratio)

        train_data = data[:split_idx]
        test_data = data[split_idx:]

        return train_data, test_data

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM/time series models.

        Converts data into sequences where each input is the previous
        'lookback' draws and the target is the next draw.

        Args:
            data: Array of shape (n_samples, n_features) containing the data.
            lookback: Number of previous draws to use for each sequence.
                     If not provided, uses self.lookback.

        Returns:
            Tuple of (X, y) where:
            - X has shape (n_sequences, lookback, n_features)
            - y has shape (n_sequences, n_features)

        Raises:
            ValueError: If data has fewer samples than lookback.
        """
        lookback = lookback or self.lookback

        if len(data) <= lookback:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than "
                f"lookback ({lookback})"
            )

        X, y = [], []

        for i in range(len(data) - lookback):
            # Input: lookback previous draws
            X.append(data[i:i + lookback])
            # Target: next draw
            y.append(data[i + lookback])

        return np.array(X), np.array(y)

    def prepare_lstm_data(
        self,
        data: np.ndarray,
        lookback: Optional[int] = None,
        split_ratio: Optional[float] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training (end-to-end pipeline).

        This method performs the complete preprocessing pipeline:
        1. Chronological train/test split (to avoid data leakage)
        2. Normalization (fit on training data only)
        3. Sequence generation for both sets

        Args:
            data: Array of shape (n_samples, n_features) containing ball numbers.
            lookback: Number of previous draws to use for sequences.
            split_ratio: Fraction of data to use for training.
            normalize: Whether to normalize the data.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test) where:
            - X_train has shape (n_train_sequences, lookback, n_features)
            - y_train has shape (n_train_sequences, n_features)
            - X_test has shape (n_test_sequences, lookback, n_features)
            - y_test has shape (n_test_sequences, n_features)

        Raises:
            ValueError: If data is invalid or parameters are out of range.
        """
        lookback = lookback or self.lookback
        split_ratio = split_ratio or self.train_test_split

        # Step 1: Chronological split (BEFORE creating sequences)
        train_data, test_data = self.split_train_test(data, split_ratio)

        # Step 2: Normalize if requested
        if normalize:
            # Fit scaler on training data only
            train_data = self.normalize(train_data, fit=True)
            # Transform test data using fitted scaler
            test_data = self.normalize(test_data, fit=False)

        # Step 3: Create sequences for both sets
        X_train, y_train = self.create_sequences(train_data, lookback)
        X_test, y_test = self.create_sequences(test_data, lookback)

        return X_train, y_train, X_test, y_test

    def prepare_standard_split(
        self,
        data: np.ndarray,
        split_ratio: Optional[float] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with standard train/test split (for non-sequence models).

        This method performs chronological splitting and optional normalization
        without creating sequences.

        Args:
            data: Array of shape (n_samples, n_features) containing ball numbers.
            split_ratio: Fraction of data to use for training.
            normalize: Whether to normalize the data.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test) where:
            - Each array has shape (n_samples, n_features)

        Note:
            For standard supervised learning, you may want to use this with
            additional feature engineering. This method provides the basic split.
        """
        split_ratio = split_ratio or self.train_test_split

        # Chronological split
        train_data, test_data = self.split_train_test(data, split_ratio)

        # Normalize if requested
        if normalize:
            train_data = self.normalize(train_data, fit=True)
            test_data = self.normalize(test_data, fit=False)

        # For standard models, we might use shifted data
        # X = current draw, y = next draw
        X_train = train_data[:-1]  # All but last
        y_train = train_data[1:]   # All but first

        X_test = test_data[:-1]
        y_test = test_data[1:]

        return X_train, y_train, X_test, y_test

    def round_to_valid_numbers(self, predictions: np.ndarray) -> np.ndarray:
        """
        Round predictions to valid lottery numbers.

        Ensures predictions are integers in the range [1, TOTAL_NUMBERS].

        Args:
            predictions: Array of predicted values (possibly denormalized).

        Returns:
            Array of valid lottery numbers (integers in range [1, TOTAL_NUMBERS]).
        """
        # Round to nearest integer
        predictions = np.round(predictions)

        # Clip to valid range
        predictions = np.clip(predictions, 1, TOTAL_NUMBERS)

        return predictions.astype(int)

    def ensure_unique_numbers(
        self,
        predictions: np.ndarray,
        n_numbers: int = NUMBERS_PER_DRAW
    ) -> np.ndarray:
        """
        Ensure each predicted draw has exactly n_numbers unique numbers.

        If predictions contain duplicates, replaces them with random
        numbers that aren't already in the draw.

        Args:
            predictions: Array of shape (n_samples, n_features) containing
                        predicted lottery numbers.
            n_numbers: Expected number of unique numbers per draw.

        Returns:
            Array with guaranteed unique numbers per row.
        """
        np.random.seed(RANDOM_SEED)

        result = predictions.copy()

        for i in range(len(result)):
            draw = result[i]
            unique_nums = np.unique(draw)

            # If we have duplicates, replace them
            if len(unique_nums) < n_numbers:
                # Get all valid numbers not in the draw
                available = np.setdiff1d(
                    np.arange(1, TOTAL_NUMBERS + 1),
                    unique_nums
                )

                # Replace duplicates with random available numbers
                n_needed = n_numbers - len(unique_nums)
                replacements = np.random.choice(
                    available,
                    size=n_needed,
                    replace=False
                )

                # Create new draw with unique numbers
                new_draw = np.concatenate([unique_nums, replacements])
                np.random.shuffle(new_draw)

                result[i] = new_draw[:n_numbers]

        return result

    def postprocess_predictions(
        self,
        predictions: np.ndarray,
        denormalize: bool = True,
        ensure_valid: bool = True,
        ensure_unique: bool = True
    ) -> np.ndarray:
        """
        Complete postprocessing pipeline for model predictions.

        Args:
            predictions: Array of model predictions (possibly normalized).
            denormalize: Whether to denormalize the predictions.
            ensure_valid: Whether to round and clip to valid range.
            ensure_unique: Whether to ensure each draw has unique numbers.

        Returns:
            Array of valid lottery predictions.

        Raises:
            ValueError: If scaler not fitted and denormalize is True.
        """
        result = predictions.copy()

        # Step 1: Denormalize if needed
        if denormalize:
            result = self.denormalize(result)

        # Step 2: Round to valid integers
        if ensure_valid:
            result = self.round_to_valid_numbers(result)

        # Step 3: Ensure uniqueness
        if ensure_unique:
            result = self.ensure_unique_numbers(result)

        return result

    def get_split_info(self, total_samples: int) -> dict:
        """
        Get information about train/test split sizes.

        Args:
            total_samples: Total number of samples in the dataset.

        Returns:
            Dictionary with split information.
        """
        train_size = int(total_samples * self.train_test_split)
        test_size = total_samples - train_size

        # Account for sequence generation loss
        train_sequences = max(0, train_size - self.lookback)
        test_sequences = max(0, test_size - self.lookback)

        return {
            'total_samples': total_samples,
            'train_test_split': self.train_test_split,
            'train_size': train_size,
            'test_size': test_size,
            'lookback': self.lookback,
            'train_sequences': train_sequences,
            'test_sequences': test_sequences,
            'sequence_loss': self.lookback * 2  # Lost from both splits
        }

    def reset_scaler(self) -> None:
        """Reset the scaler to unfitted state."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False

    def __repr__(self) -> str:
        """
        String representation of the DataPreprocessor.

        Returns:
            String describing the preprocessor configuration.
        """
        return (
            f"DataPreprocessor(lookback={self.lookback}, "
            f"train_test_split={self.train_test_split}, "
            f"scaler_fitted={self._is_fitted})"
        )

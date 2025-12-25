"""
Feature engineering module for Mega-Sena lottery prediction.

This module provides the FeatureEngineer class that creates various features
from historical lottery draws to aid in prediction models.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from config.settings import (
    BALL_COLUMNS,
    TOTAL_NUMBERS,
    NUMBERS_PER_DRAW,
    CONTEST_COLUMN
)


class FeatureEngineer:
    """
    Feature engineering class for Mega-Sena lottery data.

    Creates comprehensive features from historical draws including statistical
    aggregations, parity analysis, consecutiveness patterns, quadrant distribution,
    gap analysis, and historical frequency metrics.

    Attributes:
        df (pd.DataFrame): Original DataFrame with lottery draws
        ball_columns (List[str]): List of column names containing ball numbers
        enhanced_df (pd.DataFrame): DataFrame with engineered features
    """

    def __init__(self, df: pd.DataFrame, ball_columns: Optional[List[str]] = None):
        """
        Initialize the FeatureEngineer with lottery data.

        Args:
            df (pd.DataFrame): DataFrame containing lottery draws
            ball_columns (Optional[List[str]]): Column names for ball numbers.
                Defaults to config.settings.BALL_COLUMNS
        """
        self.df = df.copy()
        self.ball_columns = ball_columns or BALL_COLUMNS
        self.enhanced_df = None

    def engineer_all_features(self) -> pd.DataFrame:
        """
        Create all features and return enhanced DataFrame.

        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        self.enhanced_df = self.df.copy()

        # Compute all feature groups
        self._add_aggregated_statistics()
        self._add_parity_features()
        self._add_consecutiveness_features()
        self._add_quadrant_features()
        self._add_gap_features()
        self._add_frequency_features()

        return self.enhanced_df

    def _add_aggregated_statistics(self) -> None:
        """
        Add aggregated statistical features for each draw.

        Features:
            - sum: Sum of all 6 numbers
            - mean: Mean of the 6 numbers
            - std: Standard deviation
            - min_num: Minimum number
            - max_num: Maximum number
            - range: Max - Min
        """
        # Extract ball values as numpy array for vectorized operations
        balls = self.enhanced_df[self.ball_columns].values

        # Calculate statistics using numpy vectorization
        self.enhanced_df['sum'] = np.sum(balls, axis=1)
        self.enhanced_df['mean'] = np.mean(balls, axis=1)
        self.enhanced_df['std'] = np.std(balls, axis=1, ddof=1)  # Sample std
        self.enhanced_df['min_num'] = np.min(balls, axis=1)
        self.enhanced_df['max_num'] = np.max(balls, axis=1)
        self.enhanced_df['range'] = self.enhanced_df['max_num'] - self.enhanced_df['min_num']

    def _add_parity_features(self) -> None:
        """
        Add parity (even/odd) features for each draw.

        Features:
            - even_count: Count of even numbers
            - odd_count: Count of odd numbers
            - parity_ratio: even_count / 6
        """
        balls = self.enhanced_df[self.ball_columns].values

        # Vectorized even/odd counting
        even_mask = (balls % 2 == 0)
        self.enhanced_df['even_count'] = np.sum(even_mask, axis=1)
        self.enhanced_df['odd_count'] = NUMBERS_PER_DRAW - self.enhanced_df['even_count']
        self.enhanced_df['parity_ratio'] = self.enhanced_df['even_count'] / NUMBERS_PER_DRAW

    def _add_consecutiveness_features(self) -> None:
        """
        Add consecutiveness features for each draw.

        Features:
            - consecutive_pairs: Count of consecutive number pairs (e.g., 23,24)
            - has_consecutive: Boolean if any consecutive pair exists
        """
        balls = self.enhanced_df[self.ball_columns].values

        # Sort balls for each row (should already be sorted, but ensure it)
        sorted_balls = np.sort(balls, axis=1)

        # Calculate differences between consecutive balls
        diffs = np.diff(sorted_balls, axis=1)

        # Count pairs where difference is exactly 1
        consecutive_pairs = np.sum(diffs == 1, axis=1)

        self.enhanced_df['consecutive_pairs'] = consecutive_pairs
        self.enhanced_df['has_consecutive'] = consecutive_pairs > 0

    def _add_quadrant_features(self) -> None:
        """
        Add quadrant distribution features.

        Divides 1-60 into 4 quadrants of 15 numbers each:
            - Q1: 1-15
            - Q2: 16-30
            - Q3: 31-45
            - Q4: 46-60

        Features:
            - q1_count, q2_count, q3_count, q4_count: Count in each quadrant
        """
        balls = self.enhanced_df[self.ball_columns].values

        # Define quadrant boundaries
        q1_mask = (balls >= 1) & (balls <= 15)
        q2_mask = (balls >= 16) & (balls <= 30)
        q3_mask = (balls >= 31) & (balls <= 45)
        q4_mask = (balls >= 46) & (balls <= 60)

        # Count numbers in each quadrant
        self.enhanced_df['q1_count'] = np.sum(q1_mask, axis=1)
        self.enhanced_df['q2_count'] = np.sum(q2_mask, axis=1)
        self.enhanced_df['q3_count'] = np.sum(q3_mask, axis=1)
        self.enhanced_df['q4_count'] = np.sum(q4_mask, axis=1)

    def _add_gap_features(self) -> None:
        """
        Add gap/delay analysis features.

        For each draw, calculates how many draws have passed since each
        number (1-60) last appeared. Creates 60 columns: gap_1 through gap_60.

        Features:
            - gap_{n}: Draws since number n last appeared (for n=1 to 60)
        """
        # Initialize gap tracking
        gaps = np.zeros((len(self.enhanced_df), TOTAL_NUMBERS), dtype=int)
        last_seen = np.full(TOTAL_NUMBERS, -1, dtype=int)  # -1 means never seen

        # Iterate through draws in chronological order (oldest to newest)
        for idx in range(len(self.enhanced_df) - 1, -1, -1):
            # Get drawn numbers for this row (subtract 1 for 0-indexing)
            drawn = self.enhanced_df.iloc[idx][self.ball_columns].values.astype(int) - 1

            # Update gaps for all numbers
            for num in range(TOTAL_NUMBERS):
                if last_seen[num] == -1:
                    # Number hasn't appeared yet (looking backwards)
                    gaps[idx, num] = len(self.enhanced_df) - idx
                else:
                    # Calculate gap from current position to last appearance
                    gaps[idx, num] = last_seen[num] - idx

            # Update last_seen for drawn numbers
            for num in drawn:
                last_seen[num] = idx

        # Add gap columns to dataframe
        for num in range(1, TOTAL_NUMBERS + 1):
            self.enhanced_df[f'gap_{num}'] = gaps[:, num - 1]

    def _add_frequency_features(
        self,
        windows: Optional[List[int]] = None
    ) -> None:
        """
        Add historical frequency features using rolling windows.

        Calculates rolling frequency of each number over configurable windows.
        Also identifies hot (frequently appearing) and cold (rarely appearing) numbers.

        Args:
            windows (Optional[List[int]]): Window sizes for rolling frequency.
                Defaults to [10, 50, 100]

        Features:
            - freq_{n}_window_{w}: Frequency of number n in last w draws
            - hot_numbers_{w}: Count of "hot" numbers (freq > avg) in last w draws
            - cold_numbers_{w}: Count of "cold" numbers (freq < avg) in last w draws
        """
        if windows is None:
            windows = [10, 50, 100]

        # Create binary matrix: row = draw, col = number (1-60)
        # Value = 1 if number was drawn, 0 otherwise
        draw_matrix = np.zeros((len(self.enhanced_df), TOTAL_NUMBERS), dtype=int)

        for idx in range(len(self.enhanced_df)):
            drawn = self.enhanced_df.iloc[idx][self.ball_columns].values.astype(int) - 1
            draw_matrix[idx, drawn] = 1

        # Calculate rolling frequencies for each window
        for window in windows:
            if window > len(self.enhanced_df):
                # Skip if window is larger than available data
                continue

            # Initialize frequency matrix for this window
            freq_matrix = np.zeros((len(self.enhanced_df), TOTAL_NUMBERS))

            for idx in range(len(self.enhanced_df)):
                # Look back at previous 'window' draws (not including current)
                start_idx = max(0, idx - window)

                # Calculate frequency for each number in the window
                if idx > 0:
                    freq_matrix[idx, :] = np.sum(draw_matrix[start_idx:idx, :], axis=0)

            # Add frequency columns for each number
            for num in range(1, TOTAL_NUMBERS + 1):
                self.enhanced_df[f'freq_{num}_window_{window}'] = freq_matrix[:, num - 1]

            # Calculate hot/cold numbers for this window
            # Average frequency = (window * 6) / 60 = window / 10
            avg_freq = (window * NUMBERS_PER_DRAW) / TOTAL_NUMBERS

            hot_counts = np.sum(freq_matrix > avg_freq, axis=1)
            cold_counts = np.sum(freq_matrix < avg_freq, axis=1)

            self.enhanced_df[f'hot_numbers_{window}'] = hot_counts
            self.enhanced_df[f'cold_numbers_{window}'] = cold_counts

    def get_current_gaps(self, reference_idx: Optional[int] = None) -> np.ndarray:
        """
        Get the current gap vector for all 60 numbers.

        Args:
            reference_idx (Optional[int]): Index to use as reference.
                If None, uses the last draw in the dataset.

        Returns:
            np.ndarray: Array of shape (60,) with gap for each number
        """
        if reference_idx is None:
            reference_idx = len(self.df) - 1

        gaps = np.zeros(TOTAL_NUMBERS, dtype=int)
        last_seen = np.full(TOTAL_NUMBERS, -1, dtype=int)

        # Iterate from reference point backwards
        for idx in range(reference_idx, -1, -1):
            drawn = self.df.iloc[idx][self.ball_columns].values.astype(int) - 1

            # Update last_seen for drawn numbers
            for num in drawn:
                if last_seen[num] == -1:
                    last_seen[num] = idx

        # Calculate gaps
        for num in range(TOTAL_NUMBERS):
            if last_seen[num] == -1:
                gaps[num] = reference_idx + 1  # Never appeared
            else:
                gaps[num] = reference_idx - last_seen[num]

        return gaps

    def get_rolling_frequency(
        self,
        number: int,
        window: int,
        reference_idx: Optional[int] = None
    ) -> int:
        """
        Get rolling frequency for a specific number.

        Args:
            number (int): Number to check (1-60)
            window (int): Window size (number of draws to look back)
            reference_idx (Optional[int]): Reference draw index.
                If None, uses the last draw.

        Returns:
            int: Count of times the number appeared in the window
        """
        if number < 1 or number > TOTAL_NUMBERS:
            raise ValueError(f"Number must be between 1 and {TOTAL_NUMBERS}")

        if reference_idx is None:
            reference_idx = len(self.df) - 1

        start_idx = max(0, reference_idx - window + 1)

        # Count appearances
        count = 0
        for idx in range(start_idx, reference_idx + 1):
            if number in self.df.iloc[idx][self.ball_columns].values:
                count += 1

        return count

    def get_hot_cold_numbers(
        self,
        window: int = 50,
        reference_idx: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Identify hot and cold numbers based on rolling frequency.

        Args:
            window (int): Window size for frequency calculation
            reference_idx (Optional[int]): Reference draw index.
                If None, uses the last draw.

        Returns:
            Tuple[List[int], List[int]]: (hot_numbers, cold_numbers)
                Lists of numbers sorted by frequency (descending for hot,
                ascending for cold)
        """
        if reference_idx is None:
            reference_idx = len(self.df) - 1

        # Calculate frequency for all numbers
        frequencies = {}
        for num in range(1, TOTAL_NUMBERS + 1):
            freq = self.get_rolling_frequency(num, window, reference_idx)
            frequencies[num] = freq

        # Average frequency
        avg_freq = (window * NUMBERS_PER_DRAW) / TOTAL_NUMBERS

        # Identify hot and cold
        hot = [num for num, freq in frequencies.items() if freq > avg_freq]
        cold = [num for num, freq in frequencies.items() if freq < avg_freq]

        # Sort by frequency
        hot.sort(key=lambda x: frequencies[x], reverse=True)
        cold.sort(key=lambda x: frequencies[x])

        return hot, cold

    def calculate_draw_features(self, draw_numbers: List[int]) -> Dict[str, float]:
        """
        Calculate features for a single draw (useful for prediction).

        Args:
            draw_numbers (List[int]): List of 6 drawn numbers

        Returns:
            Dict[str, float]: Dictionary of feature names and values
        """
        if len(draw_numbers) != NUMBERS_PER_DRAW:
            raise ValueError(f"Draw must contain exactly {NUMBERS_PER_DRAW} numbers")

        balls = np.array(draw_numbers)
        features = {}

        # Aggregated statistics
        features['sum'] = np.sum(balls)
        features['mean'] = np.mean(balls)
        features['std'] = np.std(balls, ddof=1)
        features['min_num'] = np.min(balls)
        features['max_num'] = np.max(balls)
        features['range'] = features['max_num'] - features['min_num']

        # Parity features
        even_count = np.sum(balls % 2 == 0)
        features['even_count'] = even_count
        features['odd_count'] = NUMBERS_PER_DRAW - even_count
        features['parity_ratio'] = even_count / NUMBERS_PER_DRAW

        # Consecutiveness
        sorted_balls = np.sort(balls)
        diffs = np.diff(sorted_balls)
        consecutive_pairs = np.sum(diffs == 1)
        features['consecutive_pairs'] = consecutive_pairs
        features['has_consecutive'] = consecutive_pairs > 0

        # Quadrant distribution
        features['q1_count'] = np.sum((balls >= 1) & (balls <= 15))
        features['q2_count'] = np.sum((balls >= 16) & (balls <= 30))
        features['q3_count'] = np.sum((balls >= 31) & (balls <= 45))
        features['q4_count'] = np.sum((balls >= 46) & (balls <= 60))

        return features

"""
DataLoader module for Mega-Sena lottery data.

This module provides functionality to load and validate Mega-Sena lottery
draw data from CSV files.
"""
from typing import List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from config.settings import (
    DATA_PATH,
    BALL_COLUMNS,
    DATE_COLUMN,
    CONTEST_COLUMN,
    TOTAL_NUMBERS,
    NUMBERS_PER_DRAW
)


class DataLoader:
    """
    DataLoader class for Mega-Sena lottery data.

    This class handles loading CSV data, validating data integrity,
    and providing convenient access to lottery draw information.

    Attributes:
        data_path: Path to the CSV file containing lottery data.
        data: Pandas DataFrame containing the loaded and validated data.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the DataLoader.

        Args:
            data_path: Optional path to the CSV file. If not provided,
                      uses the default DATA_PATH from settings.
        """
        if data_path is None:
            self.data_path = DATA_PATH
        elif isinstance(data_path, str):
            self.data_path = Path(data_path)
        else:
            self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load Mega-Sena data from CSV file.

        The CSV file has the following characteristics:
        - Uses semicolon (;) as separator
        - Has an extra header row "Tabela 1" that needs to be skipped
        - Contains dates in DD/MM/YYYY format (Brazilian format)

        Returns:
            DataFrame containing the loaded lottery data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is empty or has invalid format.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Read CSV with semicolon separator, skipping the "Tabela 1" row
        self.data = pd.read_csv(
            self.data_path,
            sep=';',
            skiprows=1,  # Skip "Tabela 1" header row
            encoding='utf-8'
        )

        if self.data.empty:
            raise ValueError(f"Loaded data is empty from {self.data_path}")

        # Parse dates in Brazilian format (DD/MM/YYYY)
        self.data[DATE_COLUMN] = pd.to_datetime(
            self.data[DATE_COLUMN],
            format='%d/%m/%Y',
            errors='coerce'
        )

        # Remove rows with invalid dates
        invalid_dates = self.data[DATE_COLUMN].isna().sum()
        if invalid_dates > 0:
            print(f"Warning: Removing {invalid_dates} rows with invalid dates")
            self.data = self.data.dropna(subset=[DATE_COLUMN])

        # Sort by date chronologically (oldest to newest)
        self.data = self.data.sort_values(DATE_COLUMN).reset_index(drop=True)

        # Validate data integrity
        self._validate_data()

        return self.data

    def _validate_data(self) -> None:
        """
        Validate the integrity of the loaded lottery data.

        Checks:
        - All ball columns exist
        - All balls are between 1 and TOTAL_NUMBERS (60)
        - Each draw has exactly NUMBERS_PER_DRAW (6) unique numbers

        Raises:
            ValueError: If data validation fails.
        """
        # Check if all ball columns exist
        missing_cols = set(BALL_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing ball columns: {missing_cols}")

        # Convert ball columns to numeric, coercing errors
        for col in BALL_COLUMNS:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Check for any NaN values in ball columns
        ball_data = self.data[BALL_COLUMNS]
        nan_rows = ball_data.isna().any(axis=1).sum()
        if nan_rows > 0:
            print(f"Warning: Found {nan_rows} rows with missing ball values")
            self.data = self.data.dropna(subset=BALL_COLUMNS).reset_index(drop=True)

        ball_data = self.data[BALL_COLUMNS]

        # Validate ball numbers are in valid range [1, TOTAL_NUMBERS]
        invalid_range = (
            (ball_data < 1) | (ball_data > TOTAL_NUMBERS)
        ).any(axis=1)

        if invalid_range.any():
            invalid_count = invalid_range.sum()
            print(f"Warning: Found {invalid_count} rows with balls outside range [1, {TOTAL_NUMBERS}]")
            # Show first few invalid rows for debugging
            print("First invalid rows:")
            print(self.data[invalid_range].head())
            # Remove invalid rows
            self.data = self.data[~invalid_range].reset_index(drop=True)

        # Validate each draw has exactly 6 unique numbers
        ball_data = self.data[BALL_COLUMNS]

        # Check for duplicate numbers in each row
        def has_duplicates(row):
            return len(set(row)) != NUMBERS_PER_DRAW

        has_dups = ball_data.apply(has_duplicates, axis=1)
        if has_dups.any():
            dup_count = has_dups.sum()
            print(f"Warning: Found {dup_count} rows with duplicate numbers")
            print("First rows with duplicates:")
            print(self.data[has_dups].head())
            # Remove rows with duplicates
            self.data = self.data[~has_dups].reset_index(drop=True)

        print(f"Data validation complete. Valid draws: {len(self.data)}")

    def get_ball_data(self) -> pd.DataFrame:
        """
        Get only the ball number columns.

        Returns:
            DataFrame containing only the 6 ball columns.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.data[BALL_COLUMNS].copy()

    def get_ball_array(self) -> np.ndarray:
        """
        Get ball numbers as a numpy array.

        Returns:
            Numpy array of shape (n_draws, 6) containing ball numbers.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        return self.get_ball_data().values

    def get_dates(self) -> pd.Series:
        """
        Get the draw dates.

        Returns:
            Series containing the draw dates.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.data[DATE_COLUMN].copy()

    def get_contests(self) -> pd.Series:
        """
        Get the contest numbers.

        Returns:
            Series containing the contest numbers.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.data[CONTEST_COLUMN].copy()

    def get_metadata_columns(self) -> List[str]:
        """
        Get list of metadata columns (non-ball columns).

        Returns:
            List of column names that are not ball columns.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return [col for col in self.data.columns if col not in BALL_COLUMNS]

    def get_metadata(self) -> pd.DataFrame:
        """
        Get all metadata columns (excluding ball numbers).

        Returns:
            DataFrame containing all non-ball columns.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        metadata_cols = self.get_metadata_columns()
        return self.data[metadata_cols].copy()

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get the date range of the loaded data.

        Returns:
            Tuple of (earliest_date, latest_date).

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        dates = self.data[DATE_COLUMN]
        return (dates.min(), dates.max())

    def get_draw_by_contest(self, contest_number: int) -> Optional[pd.Series]:
        """
        Get a specific draw by contest number.

        Args:
            contest_number: The contest number to retrieve.

        Returns:
            Series containing the draw data, or None if not found.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        result = self.data[self.data[CONTEST_COLUMN] == contest_number]

        if result.empty:
            return None

        return result.iloc[0]

    def get_recent_draws(self, n: int = 10) -> pd.DataFrame:
        """
        Get the most recent n draws.

        Args:
            n: Number of recent draws to retrieve.

        Returns:
            DataFrame containing the n most recent draws.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.data.tail(n).copy()

    def get_statistics(self) -> dict:
        """
        Get basic statistics about the loaded data.

        Returns:
            Dictionary containing various statistics about the lottery data.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        ball_data = self.get_ball_array()
        date_range = self.get_date_range()

        # Count frequency of each number across all draws
        all_numbers = ball_data.flatten()
        unique, counts = np.unique(all_numbers, return_counts=True)
        number_frequency = dict(zip(unique.astype(int), counts.astype(int)))

        return {
            'total_draws': len(self.data),
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d'),
                'end': date_range[1].strftime('%Y-%m-%d'),
                'days': (date_range[1] - date_range[0]).days
            },
            'ball_statistics': {
                'most_common': int(unique[counts.argmax()]),
                'least_common': int(unique[counts.argmin()]),
                'number_frequency': number_frequency
            }
        }

    def __len__(self) -> int:
        """
        Get the number of draws in the loaded data.

        Returns:
            Number of draws.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return len(self.data)

    def __repr__(self) -> str:
        """
        String representation of the DataLoader.

        Returns:
            String describing the DataLoader state.
        """
        if self.data is None:
            return f"DataLoader(data_path={self.data_path}, loaded=False)"
        else:
            return f"DataLoader(data_path={self.data_path}, draws={len(self.data)})"

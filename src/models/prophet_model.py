"""
Prophet-based prediction model for Mega-Sena lottery analysis.

This module uses Facebook's Prophet library to analyze long-term trends
and potential seasonality in lottery data.
"""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None  # Set to None for type hints
    warnings.warn("Prophet library not installed. ProphetPredictor will not be available.")

from config.settings import (
    BALL_COLUMNS,
    DATE_COLUMN,
    TOTAL_NUMBERS,
    NUMBERS_PER_DRAW,
    RANDOM_SEED
)


class ProphetPredictor:
    """
    Prophet-based predictor for analyzing trends and seasonality in lottery data.

    This class uses Facebook's Prophet library to:
    - Analyze long-term trends in number frequencies
    - Detect potential seasonality patterns (yearly, weekly)
    - Forecast future values with uncertainty intervals
    - Provide component analysis (trend, seasonality)

    Attributes:
        model: Trained Prophet model
        forecast: DataFrame containing predictions
        components: DataFrame containing trend and seasonality components
        target_column: Name of the target column being predicted
        data: Original training data in Prophet format
    """

    def __init__(
        self,
        yearly_seasonality: Union[bool, str, int] = 'auto',
        weekly_seasonality: Union[bool, str, int] = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays: Optional[pd.DataFrame] = None,
        growth: str = 'linear',
        interval_width: float = 0.95
    ):
        """
        Initialize the Prophet predictor.

        Args:
            yearly_seasonality: Enable yearly seasonality ('auto', True, False, or int)
            weekly_seasonality: Enable weekly seasonality ('auto', True, False, or int)
            daily_seasonality: Enable daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes (higher = more flexible)
            seasonality_prior_scale: Strength of seasonality (higher = stronger)
            holidays: Optional DataFrame with 'ds' and 'holiday' columns
            growth: 'linear' or 'logistic'
            interval_width: Width of uncertainty intervals (default 0.95 = 95%)

        Raises:
            ImportError: If Prophet library is not installed
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet library is required but not installed. "
                "Install it with: pip install prophet"
            )

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays = holidays
        self.growth = growth
        self.interval_width = interval_width

        self.model: Optional[Prophet] = None
        self.forecast: Optional[pd.DataFrame] = None
        self.components: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None

    def _create_model(self) -> Prophet:
        """
        Create a new Prophet model instance with configured parameters.

        Returns:
            Configured Prophet model
        """
        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays=self.holidays,
            growth=self.growth,
            interval_width=self.interval_width,
            uncertainty_samples=1000
        )

        return model

    def prepare_data(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        date_format: str = '%d/%m/%Y'
    ) -> pd.DataFrame:
        """
        Convert data to Prophet format (ds, y columns).

        Args:
            df: Input DataFrame
            date_column: Name of the date column
            target_column: Name of the target column to predict
            date_format: Format of the date column (default: dd/mm/yyyy)

        Returns:
            DataFrame with 'ds' (datetime) and 'y' (target) columns

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Create a copy to avoid modifying original
        prophet_df = pd.DataFrame()

        # Convert date column to datetime
        try:
            prophet_df['ds'] = pd.to_datetime(df[date_column], format=date_format)
        except Exception as e:
            raise ValueError(f"Error parsing dates: {str(e)}")

        # Set target column
        prophet_df['y'] = df[target_column].values

        # Remove any NaN values
        prophet_df = prophet_df.dropna()

        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

        return prophet_df

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str = DATE_COLUMN,
        date_format: str = '%d/%m/%Y',
        verbose: bool = False
    ) -> 'ProphetPredictor':
        """
        Train Prophet model on the provided data.

        Args:
            df: DataFrame containing lottery data
            target_column: Column name to predict (e.g., 'sum', 'number_1_freq')
            date_column: Name of the date column
            date_format: Format of the date column
            verbose: Whether to print training progress

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If data preparation fails
        """
        # Prepare data in Prophet format
        self.data = self.prepare_data(df, date_column, target_column, date_format)
        self.target_column = target_column

        # Create and train model
        self.model = self._create_model()

        # Suppress Prophet's verbose output if requested
        if not verbose:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        try:
            self.model.fit(self.data)
        except Exception as e:
            raise RuntimeError(f"Error training Prophet model: {str(e)}")

        # Generate forecast for historical data to get components
        self.forecast = self.model.predict(self.data)

        return self

    def predict(
        self,
        periods: int = 10,
        freq: str = 'W'
    ) -> pd.DataFrame:
        """
        Forecast future values.

        Args:
            periods: Number of future periods to forecast
            freq: Frequency of predictions ('D'=daily, 'W'=weekly, 'M'=monthly)

        Returns:
            DataFrame with predictions including:
                - ds: datetime
                - yhat: predicted value
                - yhat_lower: lower bound of prediction interval
                - yhat_upper: upper bound of prediction interval
                - trend: trend component
                - (seasonality components if present)

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Make predictions
        self.forecast = self.model.predict(future)

        # Return only future predictions
        future_forecast = self.forecast.tail(periods).copy()

        return future_forecast

    def get_trend_analysis(self) -> pd.DataFrame:
        """
        Extract trend component from the model.

        Returns:
            DataFrame with date and trend values

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.forecast is None:
            raise ValueError("Model must be trained before analyzing trend. Call fit() first.")

        trend_df = self.forecast[['ds', 'trend']].copy()
        trend_df.columns = ['date', 'trend']

        return trend_df

    def get_seasonality_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Extract seasonality components from the model.

        Returns:
            Dictionary with seasonality components:
                - 'yearly': Yearly seasonality (if present)
                - 'weekly': Weekly seasonality (if present)
                - 'daily': Daily seasonality (if present)

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.forecast is None:
            raise ValueError("Model must be trained before analyzing seasonality. Call fit() first.")

        seasonality = {}

        # Check for yearly seasonality
        if 'yearly' in self.forecast.columns:
            seasonality['yearly'] = self.forecast[['ds', 'yearly']].copy()
            seasonality['yearly'].columns = ['date', 'yearly_effect']

        # Check for weekly seasonality
        if 'weekly' in self.forecast.columns:
            seasonality['weekly'] = self.forecast[['ds', 'weekly']].copy()
            seasonality['weekly'].columns = ['date', 'weekly_effect']

        # Check for daily seasonality
        if 'daily' in self.forecast.columns:
            seasonality['daily'] = self.forecast[['ds', 'daily']].copy()
            seasonality['daily'].columns = ['date', 'daily_effect']

        return seasonality

    def analyze_number_trend(
        self,
        df: pd.DataFrame,
        number: int,
        date_column: str = DATE_COLUMN,
        date_format: str = '%d/%m/%Y',
        window_size: int = 10
    ) -> Dict[str, Union[str, float, pd.DataFrame]]:
        """
        Analyze if a specific number is trending up or down.

        Args:
            df: DataFrame containing lottery data
            number: The number to analyze (1-60)
            date_column: Name of the date column
            date_format: Format of the date column
            window_size: Number of recent draws to analyze for trend

        Returns:
            Dictionary containing:
                - 'number': The analyzed number
                - 'trend_direction': 'up', 'down', or 'stable'
                - 'trend_strength': Float indicating strength of trend
                - 'forecast': DataFrame with predictions
                - 'current_frequency': Recent frequency
                - 'predicted_frequency': Predicted future frequency

        Raises:
            ValueError: If number is out of range
        """
        if not 1 <= number <= TOTAL_NUMBERS:
            raise ValueError(f"Number must be between 1 and {TOTAL_NUMBERS}")

        # Calculate frequency of the number over time
        frequency_data = []

        for idx, row in df.iterrows():
            balls = [row[col] for col in BALL_COLUMNS]
            freq = 1 if number in balls else 0
            frequency_data.append({
                date_column: row[date_column],
                'frequency': freq
            })

        freq_df = pd.DataFrame(frequency_data)

        # Calculate rolling frequency
        freq_df['rolling_freq'] = freq_df['frequency'].rolling(
            window=window_size,
            min_periods=1
        ).mean()

        # Train model on rolling frequency
        self.fit(freq_df, 'rolling_freq', date_column, date_format, verbose=False)

        # Predict next period
        future_forecast = self.predict(periods=5)

        # Analyze trend
        trend_data = self.get_trend_analysis()

        # Calculate trend direction from last portion of trend
        recent_trend = trend_data['trend'].tail(window_size)
        trend_slope = np.polyfit(range(len(recent_trend)), recent_trend.values, 1)[0]

        # Determine trend direction
        if trend_slope > 0.001:
            direction = 'up'
        elif trend_slope < -0.001:
            direction = 'down'
        else:
            direction = 'stable'

        # Current vs predicted frequency
        current_freq = recent_trend.iloc[-1]
        predicted_freq = future_forecast['yhat'].mean()

        return {
            'number': number,
            'trend_direction': direction,
            'trend_strength': abs(trend_slope),
            'forecast': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'current_frequency': current_freq,
            'predicted_frequency': predicted_freq,
            'trend_slope': trend_slope
        }

    def get_forecast_with_intervals(self) -> pd.DataFrame:
        """
        Get forecast with uncertainty intervals.

        Returns:
            DataFrame with predictions and confidence intervals

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.forecast is None:
            raise ValueError("Model must be trained before getting forecast. Call fit() first.")

        columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']

        # Add seasonality columns if they exist
        for season in ['yearly', 'weekly', 'daily']:
            if season in self.forecast.columns:
                columns.append(season)

        forecast_df = self.forecast[columns].copy()
        forecast_df.columns = [
            'date' if col == 'ds' else
            'prediction' if col == 'yhat' else
            'lower_bound' if col == 'yhat_lower' else
            'upper_bound' if col == 'yhat_upper' else
            col
            for col in forecast_df.columns
        ]

        return forecast_df

    def plot_components_data(self) -> Dict[str, pd.DataFrame]:
        """
        Return data for plotting trend and seasonality components.

        This method returns the data needed to create visualizations
        of the trend and seasonality components.

        Returns:
            Dictionary containing DataFrames for each component:
                - 'trend': Trend over time
                - 'yearly': Yearly seasonality (if present)
                - 'weekly': Weekly seasonality (if present)
                - 'forecast': Full forecast with intervals

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.forecast is None:
            raise ValueError("Model must be trained before plotting. Call fit() first.")

        plot_data = {
            'trend': self.get_trend_analysis(),
            'forecast': self.get_forecast_with_intervals()
        }

        # Add seasonality components
        seasonality = self.get_seasonality_analysis()
        plot_data.update(seasonality)

        return plot_data

    def analyze_all_numbers_trends(
        self,
        df: pd.DataFrame,
        date_column: str = DATE_COLUMN,
        date_format: str = '%d/%m/%Y',
        window_size: int = 10,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Analyze trends for all numbers and return top trending numbers.

        Args:
            df: DataFrame containing lottery data
            date_column: Name of the date column
            date_format: Format of the date column
            window_size: Number of recent draws to analyze
            top_n: Number of top trending numbers to return

        Returns:
            DataFrame with trending numbers sorted by trend strength

        Raises:
            ValueError: If analysis fails
        """
        results = []

        for number in range(1, TOTAL_NUMBERS + 1):
            try:
                analysis = self.analyze_number_trend(
                    df, number, date_column, date_format, window_size
                )

                results.append({
                    'number': number,
                    'trend_direction': analysis['trend_direction'],
                    'trend_strength': analysis['trend_strength'],
                    'trend_slope': analysis['trend_slope'],
                    'current_frequency': analysis['current_frequency'],
                    'predicted_frequency': analysis['predicted_frequency']
                })
            except Exception as e:
                warnings.warn(f"Error analyzing number {number}: {str(e)}")
                continue

        results_df = pd.DataFrame(results)

        # Sort by trend strength and return top N
        results_df = results_df.sort_values('trend_strength', ascending=False)

        return results_df.head(top_n)

    def get_model_params(self) -> Dict[str, any]:
        """
        Get model configuration parameters.

        Returns:
            Dictionary with model parameters
        """
        return {
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'seasonality_mode': self.seasonality_mode,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'growth': self.growth,
            'interval_width': self.interval_width,
            'target_column': self.target_column
        }

    def evaluate_prediction_quality(self) -> Dict[str, float]:
        """
        Evaluate the quality of predictions on historical data.

        Returns:
            Dictionary with evaluation metrics:
                - 'mse': Mean Squared Error
                - 'mae': Mean Absolute Error
                - 'mape': Mean Absolute Percentage Error
                - 'coverage': Percentage of actuals within prediction intervals

        Raises:
            ValueError: If model hasn't been trained
        """
        if self.forecast is None or self.data is None:
            raise ValueError("Model must be trained before evaluation. Call fit() first.")

        # Merge forecast with actual data
        merged = self.data.merge(
            self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='inner'
        )

        actual = merged['y'].values
        predicted = merged['yhat'].values
        lower = merged['yhat_lower'].values
        upper = merged['yhat_upper'].values

        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100

        # Coverage: percentage of actuals within prediction interval
        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100

        return {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'coverage': float(coverage)
        }


class MultiTargetProphetPredictor:
    """
    Prophet predictor for multiple target variables.

    This class manages multiple ProphetPredictor instances to analyze
    different aspects of the lottery data simultaneously.
    """

    def __init__(self, **prophet_params):
        """
        Initialize multi-target predictor.

        Args:
            **prophet_params: Parameters to pass to each ProphetPredictor instance
        """
        self.prophet_params = prophet_params
        self.predictors: Dict[str, ProphetPredictor] = {}

    def fit_multiple_targets(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        date_column: str = DATE_COLUMN,
        date_format: str = '%d/%m/%Y',
        verbose: bool = False
    ) -> 'MultiTargetProphetPredictor':
        """
        Train Prophet models for multiple target columns.

        Args:
            df: DataFrame containing lottery data
            target_columns: List of column names to predict
            date_column: Name of the date column
            date_format: Format of the date column
            verbose: Whether to print training progress

        Returns:
            self (for method chaining)
        """
        for target in target_columns:
            if target not in df.columns:
                warnings.warn(f"Target column '{target}' not found in DataFrame, skipping")
                continue

            try:
                predictor = ProphetPredictor(**self.prophet_params)
                predictor.fit(df, target, date_column, date_format, verbose)
                self.predictors[target] = predictor

                if verbose:
                    print(f"Trained model for {target}")
            except Exception as e:
                warnings.warn(f"Error training model for {target}: {str(e)}")

        return self

    def predict_all(
        self,
        periods: int = 10,
        freq: str = 'W'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get predictions for all trained models.

        Args:
            periods: Number of future periods to forecast
            freq: Frequency of predictions

        Returns:
            Dictionary mapping target names to prediction DataFrames
        """
        predictions = {}

        for target, predictor in self.predictors.items():
            try:
                predictions[target] = predictor.predict(periods, freq)
            except Exception as e:
                warnings.warn(f"Error predicting {target}: {str(e)}")

        return predictions

    def get_all_trends(self) -> Dict[str, pd.DataFrame]:
        """
        Get trend analysis for all trained models.

        Returns:
            Dictionary mapping target names to trend DataFrames
        """
        trends = {}

        for target, predictor in self.predictors.items():
            try:
                trends[target] = predictor.get_trend_analysis()
            except Exception as e:
                warnings.warn(f"Error getting trend for {target}: {str(e)}")

        return trends

    def get_predictor(self, target: str) -> Optional[ProphetPredictor]:
        """
        Get a specific predictor by target name.

        Args:
            target: Target column name

        Returns:
            ProphetPredictor instance or None if not found
        """
        return self.predictors.get(target)

"""
ARIMA-based time series prediction for Mega-Sena aggregated features.

This module implements ARIMA (AutoRegressive Integrated Moving Average) models
to predict aggregated features such as sum of numbers, even/odd counts, and mean values.
It does NOT predict individual lottery numbers, but rather statistical properties
that can be used to filter unlikely combinations.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not installed. Auto ARIMA functionality will be limited.")

from config.settings import ARIMA_ORDER, RANDOM_SEED


class ARIMAPredictor:
    """
    ARIMA-based predictor for time series forecasting of lottery aggregated features.

    This class implements ARIMA models to predict statistical properties of lottery draws
    such as sum of numbers, even/odd counts, and mean values. These predictions can be
    used to filter unlikely number combinations.

    Attributes:
        models (Dict[str, ARIMA]): Dictionary of fitted ARIMA models for each feature
        history (Dict[str, pd.Series]): Historical data for each feature
        orders (Dict[str, Tuple[int, int, int]]): ARIMA orders (p, d, q) for each feature
        is_fitted (bool): Whether models have been trained
    """

    def __init__(self, default_order: Optional[Tuple[int, int, int]] = None):
        """
        Initialize the ARIMA predictor.

        Args:
            default_order: Default ARIMA order (p, d, q). If None, uses config setting.
        """
        self.models: Dict[str, ARIMA] = {}
        self.history: Dict[str, pd.Series] = {}
        self.orders: Dict[str, Tuple[int, int, int]] = {}
        self.is_fitted: bool = False
        self.default_order = default_order or ARIMA_ORDER

        # Store prediction results
        self._last_predictions: Dict[str, pd.DataFrame] = {}

    def test_stationarity(
        self,
        series: Union[pd.Series, np.ndarray],
        significance_level: float = 0.05
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Test if a time series is stationary using the Augmented Dickey-Fuller test.

        The ADF test checks the null hypothesis that a unit root is present in the
        time series. If the p-value is less than the significance level, we reject
        the null hypothesis and conclude the series is stationary.

        Args:
            series: Time series data to test
            significance_level: Significance level for the test (default: 0.05)

        Returns:
            Dictionary containing:
                - adf_statistic: The test statistic
                - p_value: P-value of the test
                - critical_values: Critical values at 1%, 5%, and 10%
                - is_stationary: Whether the series is stationary
                - interpretation: Human-readable interpretation
        """
        if isinstance(series, pd.Series):
            series = series.dropna().values

        # Perform ADF test
        result = adfuller(series, autolag='AIC')

        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]

        is_stationary = p_value < significance_level

        interpretation = (
            f"Series is {'stationary' if is_stationary else 'non-stationary'} "
            f"(p-value: {p_value:.4f}, significance: {significance_level})"
        )

        return {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'interpretation': interpretation,
            'lags_used': result[2],
            'n_observations': result[3]
        }

    def analyze_acf_pacf(
        self,
        series: Union[pd.Series, np.ndarray],
        lags: int = 40,
        plot: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Analyze Autocorrelation Function (ACF) and Partial ACF for order selection.

        ACF and PACF plots help identify appropriate ARIMA parameters:
        - ACF: Shows correlation between observations at different lags
        - PACF: Shows partial correlation controlling for shorter lags

        Args:
            series: Time series data to analyze
            lags: Number of lags to compute (default: 40)
            plot: Whether to plot ACF and PACF (default: False)

        Returns:
            Dictionary containing ACF and PACF values
        """
        if isinstance(series, pd.Series):
            series = series.dropna().values

        # Compute ACF and PACF
        acf_values = acf(series, nlags=lags, fft=False)
        pacf_values = pacf(series, nlags=lags, method='ywm')

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            plot_acf(series, lags=lags, ax=ax1)
            ax1.set_title('Autocorrelation Function (ACF)')

            plot_pacf(series, lags=lags, ax=ax2, method='ywm')
            ax2.set_title('Partial Autocorrelation Function (PACF)')

            plt.tight_layout()
            plt.show()

        return {
            'acf': acf_values,
            'pacf': pacf_values
        }

    def _auto_select_order(
        self,
        series: Union[pd.Series, np.ndarray],
        seasonal: bool = False
    ) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA order using auto_arima.

        Args:
            series: Time series data
            seasonal: Whether to use seasonal ARIMA (default: False)

        Returns:
            Tuple of (p, d, q) parameters
        """
        if not PMDARIMA_AVAILABLE:
            warnings.warn(
                "pmdarima not available. Using default order. "
                "Install with: pip install pmdarima"
            )
            return self.default_order

        if isinstance(series, pd.Series):
            series = series.dropna().values

        try:
            # Use auto_arima to find optimal parameters
            model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                d=None,  # Let auto_arima determine differencing
                seasonal=seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                random_state=RANDOM_SEED
            )

            order = model.order
            return order

        except Exception as e:
            warnings.warn(f"auto_arima failed: {e}. Using default order.")
            return self.default_order

    def fit(
        self,
        series: Union[pd.Series, np.ndarray],
        feature_name: str = 'default',
        order: Optional[Tuple[int, int, int]] = None,
        auto_order: bool = False
    ) -> 'ARIMAPredictor':
        """
        Train ARIMA model on a time series.

        Args:
            series: Time series data to fit
            feature_name: Name of the feature being modeled
            order: ARIMA order (p, d, q). If None, uses default_order
            auto_order: If True, automatically select order using auto_arima

        Returns:
            Self for method chaining
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)

        # Remove NaN values
        series = series.dropna()

        if len(series) < 10:
            raise ValueError(f"Series too short for ARIMA ({len(series)} points). Need at least 10.")

        # Store history
        self.history[feature_name] = series.copy()

        # Determine order
        if auto_order:
            selected_order = self._auto_select_order(series)
        else:
            selected_order = order or self.default_order

        self.orders[feature_name] = selected_order

        try:
            # Fit ARIMA model
            model = ARIMA(series, order=selected_order)
            fitted_model = model.fit()

            self.models[feature_name] = fitted_model
            self.is_fitted = True

            # Store model summary
            print(f"\n{feature_name} - ARIMA{selected_order} fitted successfully")
            print(f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

        except Exception as e:
            raise RuntimeError(f"Failed to fit ARIMA model for {feature_name}: {e}")

        return self

    def fit_multiple(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        auto_order: bool = False
    ) -> 'ARIMAPredictor':
        """
        Train ARIMA models for multiple features.

        Args:
            data: DataFrame containing feature columns
            feature_columns: List of column names to model
            auto_order: Whether to auto-select order for each feature

        Returns:
            Self for method chaining
        """
        for feature in feature_columns:
            if feature not in data.columns:
                warnings.warn(f"Feature '{feature}' not found in data. Skipping.")
                continue

            print(f"\nFitting ARIMA for feature: {feature}")
            self.fit(data[feature], feature_name=feature, auto_order=auto_order)

        return self

    def predict(
        self,
        steps: int = 1,
        feature_name: str = 'default',
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict next N steps with confidence intervals.

        Args:
            steps: Number of steps to predict
            feature_name: Name of feature to predict
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)

        Returns:
            If return_conf_int is True:
                Tuple of (predictions, lower_bound, upper_bound)
            Otherwise:
                Array of predictions
        """
        if feature_name not in self.models:
            raise ValueError(
                f"No model fitted for feature '{feature_name}'. "
                f"Available: {list(self.models.keys())}"
            )

        model = self.models[feature_name]

        # Get forecast
        forecast_result = model.forecast(steps=steps, alpha=alpha)

        if return_conf_int:
            # Get confidence intervals
            forecast_df = model.get_forecast(steps=steps, alpha=alpha)
            conf_int = forecast_df.conf_int()

            predictions = forecast_result
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values

            # Store predictions
            self._last_predictions[feature_name] = pd.DataFrame({
                'prediction': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })

            return predictions, lower_bound, upper_bound
        else:
            return forecast_result

    def predict_all(
        self,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict next N steps for all fitted features.

        Args:
            steps: Number of steps to predict
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary mapping feature names to prediction dictionaries containing:
                - 'prediction': Point predictions
                - 'lower_bound': Lower confidence interval
                - 'upper_bound': Upper confidence interval
        """
        results = {}

        for feature_name in self.models.keys():
            pred, lower, upper = self.predict(
                steps=steps,
                feature_name=feature_name,
                return_conf_int=True,
                alpha=alpha
            )

            results[feature_name] = {
                'prediction': pred,
                'lower_bound': lower,
                'upper_bound': upper
            }

        return results

    def get_sum_range(
        self,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Get predicted sum range for filtering unlikely combinations.

        This method predicts the expected range for the sum of drawn numbers,
        which can be used to filter out combinations with unlikely sums.

        Args:
            steps: Number of steps ahead (default: 1 for next draw)
            alpha: Significance level for confidence interval

        Returns:
            Tuple of (lower_bound, upper_bound) for the sum
        """
        if 'sum' not in self.models:
            raise ValueError(
                "No model fitted for 'sum' feature. "
                "Fit a model with feature_name='sum' first."
            )

        _, lower, upper = self.predict(
            steps=steps,
            feature_name='sum',
            return_conf_int=True,
            alpha=alpha
        )

        # Return range for the first step
        return float(lower[0]), float(upper[0])

    def get_feature_range(
        self,
        feature_name: str,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Get predicted range for any feature.

        Args:
            feature_name: Name of the feature
            steps: Number of steps ahead
            alpha: Significance level for confidence interval

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        _, lower, upper = self.predict(
            steps=steps,
            feature_name=feature_name,
            return_conf_int=True,
            alpha=alpha
        )

        return float(lower[0]), float(upper[0])

    def evaluate_residuals(
        self,
        feature_name: str = 'default',
        plot: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate model residuals for diagnostics.

        Args:
            feature_name: Name of feature to evaluate
            plot: Whether to plot residual diagnostics

        Returns:
            Dictionary containing residual statistics
        """
        if feature_name not in self.models:
            raise ValueError(f"No model fitted for feature '{feature_name}'")

        model = self.models[feature_name]
        residuals = model.resid

        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Residuals over time
            axes[0, 0].plot(residuals)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_title('Residuals over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Residuals')

            # Histogram
            axes[0, 1].hist(residuals, bins=30, edgecolor='black')
            axes[0, 1].set_title('Residuals Histogram')
            axes[0, 1].set_xlabel('Residual Value')
            axes[0, 1].set_ylabel('Frequency')

            # ACF of residuals
            plot_acf(residuals, lags=40, ax=axes[1, 0])
            axes[1, 0].set_title('ACF of Residuals')

            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')

            plt.tight_layout()
            plt.show()

        return {
            'mean': mean_residual,
            'std': std_residual,
            'residuals': residuals
        }

    def get_model_summary(self, feature_name: str = 'default') -> str:
        """
        Get detailed model summary.

        Args:
            feature_name: Name of feature to summarize

        Returns:
            Model summary as string
        """
        if feature_name not in self.models:
            raise ValueError(f"No model fitted for feature '{feature_name}'")

        return str(self.models[feature_name].summary())

    def save_models(self, base_path: str) -> None:
        """
        Save fitted models to disk.

        Args:
            base_path: Base directory path for saving models
        """
        import pickle
        from pathlib import Path

        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        for feature_name, model in self.models.items():
            file_path = base_path / f"arima_{feature_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {feature_name} model to {file_path}")

    def load_models(self, base_path: str, feature_names: List[str]) -> None:
        """
        Load fitted models from disk.

        Args:
            base_path: Base directory path containing saved models
            feature_names: List of feature names to load
        """
        import pickle
        from pathlib import Path

        base_path = Path(base_path)

        for feature_name in feature_names:
            file_path = base_path / f"arima_{feature_name}.pkl"

            if not file_path.exists():
                warnings.warn(f"Model file not found: {file_path}")
                continue

            with open(file_path, 'rb') as f:
                self.models[feature_name] = pickle.load(f)
            print(f"Loaded {feature_name} model from {file_path}")

        self.is_fitted = len(self.models) > 0

    def __repr__(self) -> str:
        """String representation of the predictor."""
        fitted_features = list(self.models.keys())
        return (
            f"ARIMAPredictor(fitted={self.is_fitted}, "
            f"features={fitted_features}, "
            f"default_order={self.default_order})"
        )

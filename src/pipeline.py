"""
Mega-Sena Prediction Pipeline
=============================

Main orchestrator that coordinates all prediction models and generates
optimized betting combinations.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config.settings import (
    DATA_PATH, BALL_COLUMNS, TOTAL_NUMBERS, NUMBERS_PER_DRAW,
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH_SIZE, TOP_K_VALUES,
    TRAIN_TEST_SPLIT, RANDOM_SEED, MODELS_PATH, RESULTS_PATH
)
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.utils.statistics import StatisticalAnalyzer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.wheeling import WheelingSystem


@dataclass
class PredictionResult:
    """Container for prediction results from all models."""
    model_name: str
    predictions: List[Tuple[int, float]]  # (number, probability)
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Combined prediction from all models."""
    top_numbers: List[int]
    probabilities: Dict[int, float]
    model_contributions: Dict[str, List[Tuple[int, float]]]
    recommended_bets: List[List[int]]
    expected_sum_range: Tuple[float, float]
    statistical_summary: Dict[str, Any]


class MegaSenaPipeline:
    """
    Main prediction pipeline that orchestrates all models.

    This pipeline:
    1. Loads and preprocesses historical data
    2. Engineers features for analysis
    3. Trains/uses ARIMA for aggregated predictions (sum ranges)
    4. Trains/uses Prophet for trend analysis
    5. Trains/uses LSTM for number probability predictions
    6. Combines predictions using ensemble methods
    7. Generates optimized betting combinations

    Example:
        >>> pipeline = MegaSenaPipeline()
        >>> pipeline.load_data()
        >>> pipeline.train_models()
        >>> predictions = pipeline.predict()
        >>> bets = pipeline.generate_bets(predictions)
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        use_arima: bool = True,
        use_prophet: bool = True,
        use_lstm: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the prediction pipeline.

        Args:
            data_path: Path to Mega-Sena CSV file
            use_arima: Whether to use ARIMA model
            use_prophet: Whether to use Prophet model
            use_lstm: Whether to use LSTM model
            verbose: Print progress messages
        """
        self.data_path = data_path or DATA_PATH
        self.use_arima = use_arima
        self.use_prophet = use_prophet
        self.use_lstm = use_lstm
        self.verbose = verbose

        # Components
        self.loader: Optional[DataLoader] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.analyzer: Optional[StatisticalAnalyzer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.wheeling: Optional[WheelingSystem] = None

        # Models
        self.arima_model = None
        self.prophet_model = None
        self.lstm_model = None

        # Data
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.train_data = None
        self.test_data = None

        # Ensure directories exist
        MODELS_PATH.mkdir(exist_ok=True)
        RESULTS_PATH.mkdir(exist_ok=True)

        self._log("MegaSena Pipeline initialized")

    def _log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[Pipeline] {message}")

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate the Mega-Sena historical data.

        Returns:
            DataFrame with validated lottery data
        """
        self._log("Loading data...")

        self.loader = DataLoader(str(self.data_path))
        self.raw_data = self.loader.load_data()

        self._log(f"Loaded {len(self.raw_data)} draws")
        self._log(f"Date range: {self.raw_data['Data do Sorteio'].min()} to {self.raw_data['Data do Sorteio'].max()}")

        return self.raw_data

    def engineer_features(self) -> pd.DataFrame:
        """
        Create all derived features from raw data.

        Returns:
            DataFrame with engineered features
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._log("Engineering features...")

        self.feature_engineer = FeatureEngineer(self.raw_data)
        self.features = self.feature_engineer.engineer_all_features()

        self._log(f"Created {len(self.features.columns) - len(self.raw_data.columns)} new features")

        return self.features

    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform statistical analysis on the data.

        Returns:
            Dictionary with analysis results
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._log("Analyzing data...")

        self.analyzer = StatisticalAnalyzer()

        # Frequency analysis
        frequencies = self.analyzer.frequency_distribution(self.raw_data, BALL_COLUMNS)

        # Chi-square test for uniformity
        chi_stat, p_value, conclusion = self.analyzer.chi_square_test(frequencies)

        # Hot/cold analysis
        hot_cold_result = self.analyzer.hot_cold_analysis(frequencies)
        hot_numbers = hot_cold_result.get('hot_numbers', [])
        cold_numbers = hot_cold_result.get('cold_numbers', [])

        # Sum distribution
        sum_stats = self.analyzer.sum_distribution(self.raw_data, BALL_COLUMNS)

        analysis = {
            'frequencies': frequencies,
            'chi_square': {'statistic': chi_stat, 'p_value': p_value, 'conclusion': conclusion},
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'sum_statistics': sum_stats,
            'total_draws': len(self.raw_data)
        }

        self._log(f"Chi-square test: {conclusion}")
        self._log(f"Hot numbers: {hot_numbers[:5] if hot_numbers else []}... Cold numbers: {cold_numbers[:5] if cold_numbers else []}...")

        return analysis

    def train_models(
        self,
        train_arima: bool = True,
        train_prophet: bool = True,
        train_lstm: bool = True
    ) -> Dict[str, bool]:
        """
        Train all prediction models.

        Args:
            train_arima: Whether to train ARIMA
            train_prophet: Whether to train Prophet
            train_lstm: Whether to train LSTM

        Returns:
            Dictionary indicating which models were successfully trained
        """
        if self.features is None:
            self.engineer_features()

        results = {'arima': False, 'prophet': False, 'lstm': False}

        # Prepare data for training
        self.preprocessor = DataPreprocessor()
        ball_data = self.loader.get_ball_array()

        # Split data
        split_idx = int(len(ball_data) * TRAIN_TEST_SPLIT)
        self.train_data = ball_data[:split_idx]
        self.test_data = ball_data[split_idx:]

        # Train ARIMA
        if train_arima and self.use_arima:
            results['arima'] = self._train_arima()

        # Train Prophet
        if train_prophet and self.use_prophet:
            results['prophet'] = self._train_prophet()

        # Train LSTM
        if train_lstm and self.use_lstm:
            results['lstm'] = self._train_lstm()

        return results

    def _train_arima(self) -> bool:
        """Train ARIMA model for sum prediction."""
        try:
            from src.models.arima_model import ARIMAPredictor

            self._log("Training ARIMA model...")

            # Prepare sum series
            sums = self.features['sum'].values[:int(len(self.features) * TRAIN_TEST_SPLIT)]

            self.arima_model = ARIMAPredictor()
            self.arima_model.fit(pd.Series(sums), feature_name='sum', auto_order=True)

            self._log("ARIMA model trained successfully")
            return True

        except ImportError as e:
            self._log(f"ARIMA not available: {e}")
            return False
        except Exception as e:
            self._log(f"ARIMA training failed: {e}")
            return False

    def _train_prophet(self) -> bool:
        """Train Prophet model for trend analysis."""
        try:
            from src.models.prophet_model import ProphetPredictor

            self._log("Training Prophet model...")

            # Prepare data with dates
            train_size = int(len(self.features) * TRAIN_TEST_SPLIT)
            prophet_df = self.features[['Data do Sorteio', 'sum']].iloc[:train_size].copy()
            prophet_df.columns = ['ds', 'y']

            self.prophet_model = ProphetPredictor(
                yearly_seasonality=False,
                weekly_seasonality=False
            )
            self.prophet_model.fit(prophet_df, target_column='y')

            self._log("Prophet model trained successfully")
            return True

        except ImportError as e:
            self._log(f"Prophet not available: {e}")
            return False
        except Exception as e:
            self._log(f"Prophet training failed: {e}")
            return False

    def _train_lstm(self) -> bool:
        """Train LSTM model for number prediction."""
        try:
            from src.models.lstm_model import LSTMPredictor

            self._log("Training LSTM model...")

            # Create binary encoding
            binary_data = np.zeros((len(self.train_data), TOTAL_NUMBERS))
            for i, draw in enumerate(self.train_data):
                for num in draw:
                    binary_data[i, int(num) - 1] = 1

            # Create sequences
            self.lstm_model = LSTMPredictor(lookback=LSTM_LOOKBACK)
            X, y = self.lstm_model.create_sequences(binary_data, LSTM_LOOKBACK)

            # Build and train
            self.lstm_model.build_model(input_shape=(LSTM_LOOKBACK, TOTAL_NUMBERS))
            self.lstm_model.fit(
                X, y,
                epochs=LSTM_EPOCHS,
                batch_size=LSTM_BATCH_SIZE,
                validation_split=0.2
            )

            self._log("LSTM model trained successfully")
            return True

        except ImportError as e:
            self._log(f"LSTM/TensorFlow not available: {e}")
            return False
        except Exception as e:
            self._log(f"LSTM training failed: {e}")
            return False

    def predict(self, top_k: int = 20) -> EnsemblePrediction:
        """
        Generate predictions using all trained models.

        Args:
            top_k: Number of top predictions to return

        Returns:
            EnsemblePrediction with combined results
        """
        self._log("Generating predictions...")

        model_predictions = {}
        sum_range = (100, 250)  # Default range

        # ARIMA prediction for sum range
        if self.arima_model is not None:
            try:
                sum_min, sum_max = self.arima_model.get_sum_range()
                sum_range = (sum_min, sum_max)
                self._log(f"ARIMA predicted sum range: {sum_min:.0f} - {sum_max:.0f}")
            except Exception as e:
                self._log(f"ARIMA prediction failed: {e}")

        # Prophet trend analysis
        if self.prophet_model is not None:
            try:
                forecast = self.prophet_model.predict(periods=1)
                self._log(f"Prophet trend forecast: {forecast['yhat'].iloc[-1]:.1f}")
            except Exception as e:
                self._log(f"Prophet prediction failed: {e}")

        # LSTM number predictions
        lstm_predictions = []
        if self.lstm_model is not None:
            try:
                # Get last sequence
                recent_data = self.loader.get_ball_array()[-LSTM_LOOKBACK:]
                binary_recent = np.zeros((LSTM_LOOKBACK, TOTAL_NUMBERS))
                for i, draw in enumerate(recent_data):
                    for num in draw:
                        binary_recent[i, int(num) - 1] = 1

                X_pred = binary_recent.reshape(1, LSTM_LOOKBACK, TOTAL_NUMBERS)
                lstm_predictions = self.lstm_model.predict_top_k(X_pred, k=top_k)
                model_predictions['lstm'] = lstm_predictions

                self._log(f"LSTM top 5 predictions: {[n for n, _ in lstm_predictions[:5]]}")
            except Exception as e:
                self._log(f"LSTM prediction failed: {e}")

        # Combine predictions
        if not model_predictions:
            # Fallback to frequency-based prediction
            frequencies = self.analyzer.frequency_distribution(self.raw_data, BALL_COLUMNS)
            sorted_nums = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            model_predictions['frequency'] = [(n, f/sum(frequencies.values())) for n, f in sorted_nums[:top_k]]

        # Calculate combined probabilities
        combined_probs = {}
        for model_name, predictions in model_predictions.items():
            for number, prob in predictions:
                if number not in combined_probs:
                    combined_probs[number] = 0
                combined_probs[number] += prob / len(model_predictions)

        # Sort by probability
        top_numbers = sorted(combined_probs.keys(), key=lambda x: combined_probs[x], reverse=True)[:top_k]

        # Generate recommended bets
        self.wheeling = WheelingSystem()
        recommended_bets = self._generate_smart_bets(top_numbers[:15], sum_range)

        return EnsemblePrediction(
            top_numbers=top_numbers,
            probabilities=combined_probs,
            model_contributions=model_predictions,
            recommended_bets=recommended_bets,
            expected_sum_range=sum_range,
            statistical_summary={
                'models_used': list(model_predictions.keys()),
                'prediction_count': len(top_numbers)
            }
        )

    def _generate_smart_bets(
        self,
        candidate_numbers: List[int],
        sum_range: Tuple[float, float],
        max_bets: int = 20
    ) -> List[List[int]]:
        """Generate optimized betting combinations."""
        try:
            # Generate abbreviated wheel
            combinations = self.wheeling.abbreviated_wheel(
                candidate_numbers,
                min_guarantee=4  # Guarantee Quadra
            )

            # Filter by sum range
            filtered = self.wheeling.filter_by_sum_range(
                combinations,
                min_sum=sum_range[0],
                max_sum=sum_range[1]
            )

            # Filter by parity (2-4 even numbers is most common)
            filtered = self.wheeling.filter_by_parity(
                filtered,
                min_even=2,
                max_even=4
            )

            # Limit number of bets
            return filtered[:max_bets]

        except Exception as e:
            self._log(f"Bet generation failed: {e}")
            return []

    def evaluate(self, actual_draw: List[int]) -> Dict[str, Any]:
        """
        Evaluate predictions against actual draw.

        Args:
            actual_draw: List of 6 drawn numbers

        Returns:
            Evaluation metrics
        """
        self.evaluator = ModelEvaluator()

        results = {}

        for model_name, predictions in self.last_prediction.model_contributions.items():
            predicted_numbers = [n for n, _ in predictions[:6]]
            metrics = self.evaluator.evaluate_prediction(predicted_numbers, actual_draw)
            results[model_name] = metrics

        return results

    def run_full_pipeline(self) -> EnsemblePrediction:
        """
        Run the complete prediction pipeline.

        Returns:
            EnsemblePrediction with all results
        """
        self._log("=" * 50)
        self._log("MEGA-SENA PREDICTION PIPELINE")
        self._log("=" * 50)

        # Step 1: Load data
        self.load_data()

        # Step 2: Engineer features
        self.engineer_features()

        # Step 3: Analyze data
        analysis = self.analyze_data()

        # Step 4: Train models
        training_results = self.train_models()
        self._log(f"Training results: {training_results}")

        # Step 5: Generate predictions
        prediction = self.predict()
        self.last_prediction = prediction

        # Step 6: Display results
        self._log("=" * 50)
        self._log("PREDICTION RESULTS")
        self._log("=" * 50)
        self._log(f"Top 10 predicted numbers: {prediction.top_numbers[:10]}")
        self._log(f"Expected sum range: {prediction.expected_sum_range}")
        self._log(f"Recommended bets: {len(prediction.recommended_bets)}")

        if prediction.recommended_bets:
            self._log("Sample bets:")
            for i, bet in enumerate(prediction.recommended_bets[:3]):
                self._log(f"  Bet {i+1}: {sorted(bet)}")

        return prediction

    def save_results(self, prediction: EnsemblePrediction, filename: str = "predictions.csv") -> Path:
        """Save prediction results to file."""
        output_path = RESULTS_PATH / filename

        # Create DataFrame with predictions
        df = pd.DataFrame({
            'number': prediction.top_numbers,
            'probability': [prediction.probabilities[n] for n in prediction.top_numbers]
        })

        df.to_csv(output_path, index=False)
        self._log(f"Results saved to {output_path}")

        return output_path


def main():
    """Main entry point for the pipeline."""
    pipeline = MegaSenaPipeline(verbose=True)
    prediction = pipeline.run_full_pipeline()
    pipeline.save_results(prediction)
    return prediction


if __name__ == "__main__":
    main()

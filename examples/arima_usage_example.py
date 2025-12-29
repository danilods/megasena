"""
Example usage of ARIMAPredictor for Mega-Sena lottery prediction.

This example demonstrates how to use the ARIMA model to predict aggregated
features of lottery draws and use those predictions for filtering.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.models.arima_model import ARIMAPredictor
from config.settings import DATA_PATH, BALL_COLUMNS


def load_and_prepare_data():
    """Load Mega-Sena data and compute aggregated features."""
    print("Loading data...")

    # Read CSV (skip first row which is "Tabela 1")
    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin-1')

    # Extract ball numbers
    balls = df[BALL_COLUMNS].values

    # Compute aggregated features
    features = pd.DataFrame({
        'sum': balls.sum(axis=1),
        'mean': balls.mean(axis=1),
        'even_count': (balls % 2 == 0).sum(axis=1),
        'odd_count': (balls % 2 != 0).sum(axis=1),
        'min': balls.min(axis=1),
        'max': balls.max(axis=1),
        'range': balls.max(axis=1) - balls.min(axis=1),
        'std': balls.std(axis=1)
    })

    print(f"Loaded {len(features)} draws")
    print(f"\nFeature statistics:")
    print(features.describe())

    return features


def example_basic_usage():
    """Example 1: Basic ARIMA model usage with a single feature."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic ARIMA Usage")
    print("="*80)

    features = load_and_prepare_data()

    # Initialize predictor
    predictor = ARIMAPredictor(default_order=(5, 1, 0))

    # Test stationarity of the sum feature
    print("\nTesting stationarity of 'sum' feature...")
    stationarity_result = predictor.test_stationarity(features['sum'])
    print(f"ADF Statistic: {stationarity_result['adf_statistic']:.4f}")
    print(f"P-value: {stationarity_result['p_value']:.4f}")
    print(f"Is Stationary: {stationarity_result['is_stationary']}")
    print(f"Interpretation: {stationarity_result['interpretation']}")

    # Fit model on sum feature
    print("\nFitting ARIMA model on 'sum' feature...")
    predictor.fit(features['sum'], feature_name='sum')

    # Make predictions
    print("\nPredicting next 5 draws...")
    predictions, lower, upper = predictor.predict(steps=5, feature_name='sum')

    print("\nPredictions with 95% confidence intervals:")
    for i in range(5):
        print(f"  Step {i+1}: {predictions[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")

    # Get sum range for filtering
    sum_min, sum_max = predictor.get_sum_range()
    print(f"\nRecommended sum range for next draw: [{sum_min:.2f}, {sum_max:.2f}]")


def example_multiple_features():
    """Example 2: Train models for multiple features."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Features")
    print("="*80)

    features = load_and_prepare_data()

    # Initialize predictor
    predictor = ARIMAPredictor()

    # Fit models for multiple features
    feature_list = ['sum', 'even_count', 'mean']
    print(f"\nFitting ARIMA models for features: {feature_list}")
    predictor.fit_multiple(features, feature_list)

    # Predict for all features
    print("\nPredicting next draw for all features...")
    all_predictions = predictor.predict_all(steps=1)

    print("\nPredictions:")
    for feature, preds in all_predictions.items():
        pred = preds['prediction'][0]
        lower = preds['lower_bound'][0]
        upper = preds['upper_bound'][0]
        print(f"  {feature:15s}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")


def example_auto_arima():
    """Example 3: Using auto_arima for automatic parameter selection."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Auto ARIMA")
    print("="*80)

    features = load_and_prepare_data()

    # Initialize predictor
    predictor = ARIMAPredictor()

    print("\nUsing auto_arima to select optimal parameters for 'sum'...")
    try:
        predictor.fit(features['sum'], feature_name='sum', auto_order=True)
        print(f"Selected order: {predictor.orders['sum']}")

        # Make prediction
        predictions, lower, upper = predictor.predict(steps=3, feature_name='sum')
        print("\nPredictions for next 3 draws:")
        for i in range(3):
            print(f"  Draw {i+1}: {predictions[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")

    except Exception as e:
        print(f"Note: {e}")
        print("Install pmdarima with: pip install pmdarima")


def example_acf_pacf_analysis():
    """Example 4: ACF/PACF analysis for parameter selection."""
    print("\n" + "="*80)
    print("EXAMPLE 4: ACF/PACF Analysis")
    print("="*80)

    features = load_and_prepare_data()

    predictor = ARIMAPredictor()

    print("\nAnalyzing ACF and PACF for 'sum' feature...")
    acf_pacf = predictor.analyze_acf_pacf(features['sum'], lags=20, plot=False)

    print(f"ACF values (first 10 lags): {acf_pacf['acf'][:10]}")
    print(f"PACF values (first 10 lags): {acf_pacf['pacf'][:10]}")
    print("\nNote: Set plot=True to visualize ACF and PACF plots")


def example_model_evaluation():
    """Example 5: Model evaluation and diagnostics."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Model Evaluation")
    print("="*80)

    features = load_and_prepare_data()

    predictor = ARIMAPredictor()
    predictor.fit(features['sum'], feature_name='sum')

    print("\nEvaluating model residuals...")
    residual_stats = predictor.evaluate_residuals('sum', plot=False)

    print(f"Mean of residuals: {residual_stats['mean']:.4f}")
    print(f"Std of residuals: {residual_stats['std']:.4f}")
    print("\nNote: Mean should be close to 0 for a good model")

    # Print model summary
    print("\nModel Summary:")
    print(predictor.get_model_summary('sum'))


def example_filtering_combinations():
    """Example 6: Using predictions to filter lottery combinations."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Filtering Combinations")
    print("="*80)

    features = load_and_prepare_data()

    # Train models
    predictor = ARIMAPredictor()
    predictor.fit_multiple(features, ['sum', 'even_count', 'mean'])

    # Get predicted ranges
    sum_min, sum_max = predictor.get_sum_range()
    even_min, even_max = predictor.get_feature_range('even_count')
    mean_min, mean_max = predictor.get_feature_range('mean')

    print("\nPredicted ranges for next draw:")
    print(f"  Sum:        [{sum_min:.2f}, {sum_max:.2f}]")
    print(f"  Even count: [{even_min:.2f}, {even_max:.2f}]")
    print(f"  Mean:       [{mean_min:.2f}, {mean_max:.2f}]")

    # Example: Generate random combination and check if it fits
    print("\nTesting sample combinations:")
    np.random.seed(42)

    for i in range(5):
        # Generate random 6 numbers from 1-60
        combo = sorted(np.random.choice(range(1, 61), 6, replace=False))
        combo_sum = sum(combo)
        combo_even = sum(1 for x in combo if x % 2 == 0)
        combo_mean = np.mean(combo)

        # Check if combination fits predicted ranges
        fits_sum = sum_min <= combo_sum <= sum_max
        fits_even = even_min <= combo_even <= even_max
        fits_mean = mean_min <= combo_mean <= mean_max

        fits_all = fits_sum and fits_even and fits_mean

        print(f"\n  Combination {i+1}: {combo}")
        print(f"    Sum: {combo_sum:.0f} {'✓' if fits_sum else '✗'}")
        print(f"    Even: {combo_even} {'✓' if fits_even else '✗'}")
        print(f"    Mean: {combo_mean:.2f} {'✓' if fits_mean else '✗'}")
        print(f"    Overall: {'PASS' if fits_all else 'REJECT'}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ARIMA PREDICTOR - USAGE EXAMPLES")
    print("="*80)

    try:
        example_basic_usage()
        example_multiple_features()
        example_auto_arima()
        example_acf_pacf_analysis()
        example_model_evaluation()
        example_filtering_combinations()

        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

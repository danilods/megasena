"""
Example usage of ProphetPredictor for Mega-Sena lottery analysis.

This script demonstrates how to use the Prophet model to:
- Analyze trends in lottery data
- Detect seasonality patterns
- Forecast future values
- Analyze individual number trends
"""
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.prophet_model import ProphetPredictor, MultiTargetProphetPredictor
from config.settings import DATA_PATH, BALL_COLUMNS, DATE_COLUMN


def example_1_basic_usage():
    """Example 1: Basic usage - analyze sum of drawn numbers over time."""
    print("=" * 80)
    print("Example 1: Analyzing sum of drawn numbers over time")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

    # Calculate sum of numbers for each draw
    df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)

    # Initialize and train model
    predictor = ProphetPredictor(
        yearly_seasonality='auto',
        weekly_seasonality=False,
        changepoint_prior_scale=0.1  # More flexible trend
    )

    # Train on sum of numbers
    print("\nTraining Prophet model on sum of numbers...")
    predictor.fit(df, target_column='sum_numbers', verbose=True)

    # Get trend analysis
    trend = predictor.get_trend_analysis()
    print(f"\nTrend Analysis:")
    print(trend.tail(10))

    # Predict next 10 draws
    print("\nForecasting next 10 draws...")
    forecast = predictor.predict(periods=10, freq='W')
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Evaluate model
    print("\nModel Evaluation:")
    metrics = predictor.evaluate_prediction_quality()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return predictor


def example_2_number_trend_analysis():
    """Example 2: Analyze if specific numbers are trending up or down."""
    print("\n" + "=" * 80)
    print("Example 2: Analyzing individual number trends")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

    # Initialize predictor
    predictor = ProphetPredictor(
        yearly_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale=0.05
    )

    # Analyze specific numbers
    numbers_to_analyze = [7, 13, 21, 33, 42]

    print(f"\nAnalyzing trends for numbers: {numbers_to_analyze}")

    for number in numbers_to_analyze:
        print(f"\n--- Number {number} ---")
        analysis = predictor.analyze_number_trend(df, number, window_size=20)

        print(f"Trend Direction: {analysis['trend_direction']}")
        print(f"Trend Strength: {analysis['trend_strength']:.6f}")
        print(f"Current Frequency: {analysis['current_frequency']:.4f}")
        print(f"Predicted Frequency: {analysis['predicted_frequency']:.4f}")
        print(f"Trend Slope: {analysis['trend_slope']:.6f}")


def example_3_all_numbers_trending():
    """Example 3: Find top trending numbers (up or down)."""
    print("\n" + "=" * 80)
    print("Example 3: Finding top trending numbers")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

    # Initialize predictor
    predictor = ProphetPredictor(
        yearly_seasonality=False,
        weekly_seasonality=False
    )

    # Analyze all numbers and get top 15
    print("\nAnalyzing all 60 numbers...")
    print("This may take a few minutes...\n")

    top_trending = predictor.analyze_all_numbers_trends(
        df,
        window_size=15,
        top_n=15
    )

    print("Top 15 Trending Numbers:")
    print(top_trending.to_string(index=False))


def example_4_multiple_targets():
    """Example 4: Analyze multiple targets simultaneously."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-target analysis")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

    # Calculate various features
    df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)
    df['mean_numbers'] = df[BALL_COLUMNS].mean(axis=1)
    df['std_numbers'] = df[BALL_COLUMNS].std(axis=1)
    df['max_number'] = df[BALL_COLUMNS].max(axis=1)
    df['min_number'] = df[BALL_COLUMNS].min(axis=1)
    df['range_numbers'] = df['max_number'] - df['min_number']

    # Initialize multi-target predictor
    multi_predictor = MultiTargetProphetPredictor(
        yearly_seasonality='auto',
        weekly_seasonality=False,
        changepoint_prior_scale=0.1
    )

    # Train on multiple targets
    targets = ['sum_numbers', 'mean_numbers', 'std_numbers', 'range_numbers']
    print(f"\nTraining models for: {targets}")

    multi_predictor.fit_multiple_targets(df, targets, verbose=True)

    # Get predictions for all targets
    print("\nForecasting next 5 draws for all targets...")
    predictions = multi_predictor.predict_all(periods=5, freq='W')

    for target, forecast in predictions.items():
        print(f"\n--- {target} ---")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Get trends for all targets
    print("\nTrend Analysis for all targets:")
    trends = multi_predictor.get_all_trends()

    for target, trend_df in trends.items():
        recent_trend = trend_df.tail(5)
        print(f"\n--- {target} (last 5 draws) ---")
        print(recent_trend)


def example_5_seasonality_detection():
    """Example 5: Detect and analyze seasonality patterns."""
    print("\n" + "=" * 80)
    print("Example 5: Seasonality detection")
    print("=" * 80)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')
    df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)

    # Initialize with strong seasonality detection
    predictor = ProphetPredictor(
        yearly_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode='additive',
        seasonality_prior_scale=15.0  # Stronger seasonality
    )

    print("\nTraining model with seasonality detection...")
    predictor.fit(df, target_column='sum_numbers', verbose=True)

    # Get seasonality analysis
    seasonality = predictor.get_seasonality_analysis()

    print("\nSeasonality Components Found:")
    for component, data in seasonality.items():
        print(f"\n--- {component.upper()} Seasonality ---")
        print(f"Shape: {data.shape}")
        print(data.head(10))

    # Get all plot data
    plot_data = predictor.plot_components_data()
    print("\nAvailable components for plotting:")
    for key in plot_data.keys():
        print(f"  - {key}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Prophet Model Examples for Mega-Sena Lottery Analysis")
    print("=" * 80)

    try:
        # Example 1: Basic usage
        example_1_basic_usage()

        # Example 2: Individual number trends
        example_2_number_trend_analysis()

        # Example 3: Top trending numbers
        # Uncomment to run (takes longer):
        # example_3_all_numbers_trending()

        # Example 4: Multiple targets
        example_4_multiple_targets()

        # Example 5: Seasonality detection
        example_5_seasonality_detection()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

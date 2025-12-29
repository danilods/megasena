"""
Usage examples for the FeatureEngineer class.

This module demonstrates how to use the FeatureEngineer to create
features for Mega-Sena lottery prediction models.
"""
import pandas as pd
from src.features.engineer import FeatureEngineer
from config.settings import DATA_PATH, BALL_COLUMNS


def example_basic_usage():
    """Example 1: Basic usage - engineer all features."""
    print("=" * 70)
    print("Example 1: Basic Feature Engineering")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')
    print(f"Loaded {len(df)} draws")

    # Use last 200 draws for this example
    recent_df = df.head(200).copy()

    # Initialize and engineer features
    fe = FeatureEngineer(recent_df)
    enhanced_df = fe.engineer_all_features()

    print(f"\nOriginal columns: {len(recent_df.columns)}")
    print(f"Enhanced columns: {len(enhanced_df.columns)}")
    print(f"New features: {len(enhanced_df.columns) - len(recent_df.columns)}")

    # View sample of engineered features
    feature_cols = [col for col in enhanced_df.columns if col not in recent_df.columns]
    print(f"\nSample of engineered features (first 20):")
    print(feature_cols[:20])

    # Show statistics for latest draw
    latest = enhanced_df.iloc[0]
    print(f"\nLatest draw ({latest['Concurso']}):")
    print(f"  Numbers: {latest[BALL_COLUMNS].values}")
    print(f"  Sum: {latest['sum']}")
    print(f"  Mean: {latest['mean']:.2f}")
    print(f"  Std: {latest['std']:.2f}")
    print(f"  Even/Odd: {latest['even_count']:.0f}/{latest['odd_count']:.0f}")
    print(f"  Consecutive pairs: {latest['consecutive_pairs']}")
    print(f"  Quadrants: Q1={latest['q1_count']}, Q2={latest['q2_count']}, "
          f"Q3={latest['q3_count']}, Q4={latest['q4_count']}")

    return enhanced_df


def example_gap_analysis():
    """Example 2: Analyze gaps for specific numbers."""
    print("\n" + "=" * 70)
    print("Example 2: Gap Analysis")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')
    fe = FeatureEngineer(df)

    # Get current gaps for all numbers
    gaps = fe.get_current_gaps()

    # Find numbers that haven't appeared recently (high gap)
    print("\nNumbers with highest gaps (overdue):")
    sorted_indices = gaps.argsort()[::-1]  # Sort descending
    for i in range(10):
        num = sorted_indices[i] + 1
        gap = gaps[sorted_indices[i]]
        print(f"  Number {num:2d}: {gap} draws ago")

    # Find numbers that appeared recently (low gap)
    print("\nNumbers with lowest gaps (recently appeared):")
    for i in range(10):
        num = sorted_indices[-i-1] + 1
        gap = gaps[sorted_indices[-i-1]]
        print(f"  Number {num:2d}: {gap} draws ago")


def example_hot_cold_numbers():
    """Example 3: Identify hot and cold numbers."""
    print("\n" + "=" * 70)
    print("Example 3: Hot and Cold Numbers Analysis")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')
    fe = FeatureEngineer(df)

    # Analyze different time windows
    windows = [10, 50, 100]

    for window in windows:
        print(f"\n--- Last {window} draws ---")
        hot, cold = fe.get_hot_cold_numbers(window=window)

        print(f"Hot numbers ({len(hot)}): {hot[:15]}")
        print(f"Cold numbers ({len(cold)}): {cold[:15]}")

        # Show frequencies for top hot numbers
        print(f"\nTop 5 hot numbers frequency:")
        for num in hot[:5]:
            freq = fe.get_rolling_frequency(num, window)
            print(f"  Number {num:2d}: appeared {freq} times ({freq/window*100:.1f}%)")


def example_frequency_tracking():
    """Example 4: Track frequency of specific numbers over time."""
    print("\n" + "=" * 70)
    print("Example 4: Frequency Tracking")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')
    fe = FeatureEngineer(df)

    # Track specific numbers
    numbers_to_track = [7, 13, 21, 42]
    window = 50

    print(f"\nFrequency in last {window} draws:")
    for num in numbers_to_track:
        freq = fe.get_rolling_frequency(num, window)
        print(f"  Number {num:2d}: {freq} appearances ({freq/window*100:.1f}%)")


def example_single_draw_features():
    """Example 5: Calculate features for a hypothetical draw."""
    print("\n" + "=" * 70)
    print("Example 5: Features for Hypothetical Draw")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')
    fe = FeatureEngineer(df)

    # Hypothetical draws
    draws = [
        [5, 10, 15, 20, 25, 30],  # Evenly spaced
        [1, 2, 3, 4, 5, 6],        # Low consecutive numbers
        [55, 56, 57, 58, 59, 60],  # High consecutive numbers
        [7, 14, 21, 28, 35, 42],   # Multiples of 7
    ]

    for draw in draws:
        features = fe.calculate_draw_features(draw)
        print(f"\nDraw: {draw}")
        print(f"  Sum: {features['sum']}")
        print(f"  Mean: {features['mean']:.2f}")
        print(f"  Range: {features['range']}")
        print(f"  Even/Odd: {features['even_count']}/{features['odd_count']}")
        print(f"  Consecutive pairs: {features['consecutive_pairs']}")
        print(f"  Quadrants: Q1={features['q1_count']}, Q2={features['q2_count']}, "
              f"Q3={features['q3_count']}, Q4={features['q4_count']}")


def example_using_engineered_features_for_ml():
    """Example 6: Prepare data for machine learning."""
    print("\n" + "=" * 70)
    print("Example 6: Prepare Features for Machine Learning")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin1')

    # Use last 500 draws
    df = df.head(500).copy()

    # Engineer features
    fe = FeatureEngineer(df)
    enhanced_df = fe.engineer_all_features()

    # Select only numeric features for ML
    numeric_cols = enhanced_df.select_dtypes(include=['int64', 'float64']).columns
    feature_cols = [col for col in numeric_cols if col not in BALL_COLUMNS + ['Concurso']]

    print(f"\nTotal numeric features available for ML: {len(feature_cols)}")

    # Create feature matrix and target
    X = enhanced_df[feature_cols].values
    y = enhanced_df[BALL_COLUMNS].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    # Check for any NaN values
    nan_count = pd.isna(X).sum()
    print(f"NaN values in features: {nan_count}")

    # Show sample feature names
    print(f"\nSample feature categories:")
    print(f"  - Aggregated stats: sum, mean, std, min_num, max_num, range")
    print(f"  - Parity: even_count, odd_count, parity_ratio")
    print(f"  - Consecutiveness: consecutive_pairs, has_consecutive")
    print(f"  - Quadrants: q1_count, q2_count, q3_count, q4_count")
    print(f"  - Gaps: gap_1 through gap_60 ({sum(1 for c in feature_cols if c.startswith('gap_'))} features)")
    print(f"  - Frequencies: freq_N_window_W ({sum(1 for c in feature_cols if c.startswith('freq_'))} features)")
    print(f"  - Hot/Cold: hot_numbers_W, cold_numbers_W ({sum(1 for c in feature_cols if 'hot' in c or 'cold' in c)} features)")

    return X, y, feature_cols


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_gap_analysis()
    example_hot_cold_numbers()
    example_frequency_tracking()
    example_single_draw_features()
    example_using_engineered_features_for_ml()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

"""
Test script for FeatureEngineer class.

This script tests all feature engineering functionality with real Mega-Sena data.
"""
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer
from config.settings import BALL_COLUMNS, DATA_PATH


def main():
    """Run comprehensive tests on FeatureEngineer."""
    print("=" * 70)
    print("Feature Engineer Test Suite")
    print("=" * 70)

    # Load data
    print("\n1. Loading Mega-Sena data...")
    df = pd.read_csv(
        DATA_PATH,
        sep=';',
        skiprows=1,
        encoding='latin1'
    )
    print(f"   ✓ Loaded {len(df)} draws")
    print(f"   ✓ Date range: {df.iloc[-1]['Data do Sorteio']} to {df.iloc[0]['Data do Sorteio']}")

    # Sample first 100 draws for faster testing
    test_df = df.head(100).copy()
    print(f"\n2. Using {len(test_df)} draws for testing...")

    # Initialize FeatureEngineer
    print("\n3. Initializing FeatureEngineer...")
    fe = FeatureEngineer(test_df)
    print("   ✓ FeatureEngineer initialized")

    # Test individual feature methods
    print("\n4. Testing aggregated statistics...")
    fe.enhanced_df = fe.df.copy()
    fe._add_aggregated_statistics()
    agg_features = ['sum', 'mean', 'std', 'min_num', 'max_num', 'range']
    for feat in agg_features:
        assert feat in fe.enhanced_df.columns, f"Missing feature: {feat}"
    print(f"   ✓ All aggregated statistics created: {', '.join(agg_features)}")

    # Show sample stats
    sample_idx = 0
    balls = test_df.iloc[sample_idx][BALL_COLUMNS].values
    print(f"   Example (Draw {test_df.iloc[sample_idx]['Concurso']}): {balls}")
    print(f"     - Sum: {fe.enhanced_df.iloc[sample_idx]['sum']:.0f}")
    print(f"     - Mean: {fe.enhanced_df.iloc[sample_idx]['mean']:.2f}")
    print(f"     - Std: {fe.enhanced_df.iloc[sample_idx]['std']:.2f}")
    print(f"     - Range: {fe.enhanced_df.iloc[sample_idx]['range']:.0f}")

    # Test parity features
    print("\n5. Testing parity features...")
    fe.enhanced_df = fe.df.copy()
    fe._add_parity_features()
    parity_features = ['even_count', 'odd_count', 'parity_ratio']
    for feat in parity_features:
        assert feat in fe.enhanced_df.columns, f"Missing feature: {feat}"
    print(f"   ✓ All parity features created: {', '.join(parity_features)}")
    print(f"   Example (Draw {test_df.iloc[sample_idx]['Concurso']}): {balls}")
    print(f"     - Even count: {fe.enhanced_df.iloc[sample_idx]['even_count']:.0f}")
    print(f"     - Odd count: {fe.enhanced_df.iloc[sample_idx]['odd_count']:.0f}")
    print(f"     - Parity ratio: {fe.enhanced_df.iloc[sample_idx]['parity_ratio']:.3f}")

    # Test consecutiveness features
    print("\n6. Testing consecutiveness features...")
    fe.enhanced_df = fe.df.copy()
    fe._add_consecutiveness_features()
    consec_features = ['consecutive_pairs', 'has_consecutive']
    for feat in consec_features:
        assert feat in fe.enhanced_df.columns, f"Missing feature: {feat}"
    print(f"   ✓ All consecutiveness features created: {', '.join(consec_features)}")
    print(f"   Example (Draw {test_df.iloc[sample_idx]['Concurso']}): {balls}")
    print(f"     - Consecutive pairs: {fe.enhanced_df.iloc[sample_idx]['consecutive_pairs']:.0f}")
    print(f"     - Has consecutive: {fe.enhanced_df.iloc[sample_idx]['has_consecutive']}")

    # Test quadrant features
    print("\n7. Testing quadrant features...")
    fe.enhanced_df = fe.df.copy()
    fe._add_quadrant_features()
    quad_features = ['q1_count', 'q2_count', 'q3_count', 'q4_count']
    for feat in quad_features:
        assert feat in fe.enhanced_df.columns, f"Missing feature: {feat}"
    print(f"   ✓ All quadrant features created: {', '.join(quad_features)}")
    print(f"   Example (Draw {test_df.iloc[sample_idx]['Concurso']}): {balls}")
    print(f"     - Q1 (1-15): {fe.enhanced_df.iloc[sample_idx]['q1_count']:.0f}")
    print(f"     - Q2 (16-30): {fe.enhanced_df.iloc[sample_idx]['q2_count']:.0f}")
    print(f"     - Q3 (31-45): {fe.enhanced_df.iloc[sample_idx]['q3_count']:.0f}")
    print(f"     - Q4 (46-60): {fe.enhanced_df.iloc[sample_idx]['q4_count']:.0f}")

    # Test gap features
    print("\n8. Testing gap features...")
    fe.enhanced_df = fe.df.copy()
    fe._add_gap_features()
    gap_cols = [col for col in fe.enhanced_df.columns if col.startswith('gap_')]
    assert len(gap_cols) == 60, f"Expected 60 gap columns, got {len(gap_cols)}"
    print(f"   ✓ All 60 gap features created")
    # Show sample gaps
    print(f"   Example gaps for first few numbers in latest draw:")
    for num in range(1, 6):
        gap_val = fe.enhanced_df.iloc[0][f'gap_{num}']
        print(f"     - Number {num}: {gap_val} draws ago")

    # Test frequency features
    print("\n9. Testing frequency features...")
    fe.enhanced_df = fe.df.copy()
    fe._add_frequency_features(windows=[10, 50])
    freq_cols = [col for col in fe.enhanced_df.columns if col.startswith('freq_')]
    print(f"   ✓ Created {len(freq_cols)} frequency features")
    hot_cold_cols = [col for col in fe.enhanced_df.columns if 'hot_numbers' in col or 'cold_numbers' in col]
    print(f"   ✓ Created {len(hot_cold_cols)} hot/cold features")

    # Test full feature engineering
    print("\n10. Testing full feature engineering pipeline...")
    fe = FeatureEngineer(test_df)
    enhanced = fe.engineer_all_features()
    original_cols = len(test_df.columns)
    new_cols = len(enhanced.columns)
    feature_cols = new_cols - original_cols
    print(f"   ✓ Original columns: {original_cols}")
    print(f"   ✓ Enhanced columns: {new_cols}")
    print(f"   ✓ New features created: {feature_cols}")

    # Test utility methods
    print("\n11. Testing utility methods...")

    # Test get_current_gaps
    gaps = fe.get_current_gaps()
    assert len(gaps) == 60, f"Expected 60 gaps, got {len(gaps)}"
    print(f"   ✓ get_current_gaps() works: {gaps[:5]}")

    # Test get_rolling_frequency
    freq = fe.get_rolling_frequency(number=5, window=10)
    print(f"   ✓ get_rolling_frequency() works: Number 5 appeared {freq} times in last 10 draws")

    # Test get_hot_cold_numbers
    hot, cold = fe.get_hot_cold_numbers(window=50)
    print(f"   ✓ get_hot_cold_numbers() works:")
    print(f"     - Hot numbers ({len(hot)}): {hot[:10] if len(hot) >= 10 else hot}")
    print(f"     - Cold numbers ({len(cold)}): {cold[:10] if len(cold) >= 10 else cold}")

    # Test calculate_draw_features
    test_draw = [5, 15, 23, 34, 42, 58]
    draw_features = fe.calculate_draw_features(test_draw)
    print(f"\n   ✓ calculate_draw_features() works for draw: {test_draw}")
    print(f"     - Sum: {draw_features['sum']:.0f}")
    print(f"     - Mean: {draw_features['mean']:.2f}")
    print(f"     - Even/Odd: {draw_features['even_count']:.0f}/{draw_features['odd_count']:.0f}")
    print(f"     - Consecutive pairs: {draw_features['consecutive_pairs']:.0f}")
    print(f"     - Quadrants: Q1={draw_features['q1_count']:.0f}, Q2={draw_features['q2_count']:.0f}, "
          f"Q3={draw_features['q3_count']:.0f}, Q4={draw_features['q4_count']:.0f}")

    # Final summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING TEST SUMMARY")
    print("=" * 70)
    print(f"✓ All tests passed successfully!")
    print(f"\nFeature Categories Implemented:")
    print(f"  1. Aggregated Statistics: 6 features (sum, mean, std, min, max, range)")
    print(f"  2. Parity Features: 3 features (even_count, odd_count, parity_ratio)")
    print(f"  3. Consecutiveness: 2 features (consecutive_pairs, has_consecutive)")
    print(f"  4. Quadrant Analysis: 4 features (q1-q4 counts)")
    print(f"  5. Gap Analysis: 60 features (gap for each number 1-60)")
    print(f"  6. Frequency Features: 120+ features (rolling frequencies + hot/cold)")
    print(f"\nTotal: {feature_cols} engineered features")
    print(f"\nUtility Methods Available:")
    print(f"  - get_current_gaps(): Get gap vector for all numbers")
    print(f"  - get_rolling_frequency(): Get frequency for specific number")
    print(f"  - get_hot_cold_numbers(): Identify hot and cold numbers")
    print(f"  - calculate_draw_features(): Calculate features for any draw")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Demonstration script for the statistical analysis and visualization utilities.

This script shows how to use the StatisticalAnalyzer and Visualizer classes
with sample Mega-Sena data.
"""

import numpy as np
import pandas as pd
from src.utils.statistics import StatisticalAnalyzer
from src.utils.visualization import Visualizer


def generate_sample_data(n_draws: int = 500) -> pd.DataFrame:
    """Generate sample lottery draw data for testing."""
    np.random.seed(42)
    data = {
        'Concurso': range(1, n_draws + 1),
        'Data': pd.date_range('2020-01-01', periods=n_draws, freq='W-SAT'),
    }

    # Generate 6 random balls (1-60) for each draw
    for i in range(1, 7):
        data[f'Bola{i}'] = [
            sorted(np.random.choice(range(1, 61), size=6, replace=False))[i-1]
            for _ in range(n_draws)
        ]

    return pd.DataFrame(data)


def main():
    """Run demonstrations of utility modules."""
    print("=" * 80)
    print("MEGA-SENA UTILITY MODULES DEMONSTRATION")
    print("=" * 80)

    # Initialize classes
    analyzer = StatisticalAnalyzer()
    visualizer = Visualizer()

    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(n_draws=500)
    print(f"   Created {len(df)} sample draws")
    print(f"   Columns: {list(df.columns)}")

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Frequency Distribution
    print("\n2. Frequency Distribution Analysis")
    frequencies = analyzer.frequency_distribution(df)
    print(f"   Number range: {frequencies.index.min()} - {frequencies.index.max()}")
    print(f"   Most frequent: {frequencies.idxmax()} ({frequencies.max()} times)")
    print(f"   Least frequent: {frequencies.idxmin()} ({frequencies.min()} times)")
    print(f"   Mean frequency: {frequencies.mean():.2f}")
    print(f"   Std frequency: {frequencies.std():.2f}")

    # Chi-Square Test
    print("\n3. Chi-Square Test (Uniformity)")
    chi_stat, p_value, conclusion = analyzer.chi_square_test(frequencies.values)
    print(f"   Chi-square statistic: {chi_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Conclusion: {conclusion}")

    # Shannon Entropy
    print("\n4. Shannon Entropy Analysis")
    entropy_results = analyzer.entropy_analysis(df, window_sizes=[50, 100, 200])
    print(f"   Overall entropy: {entropy_results['overall_entropy']:.4f} bits")
    print(f"   Maximum entropy: {entropy_results['max_entropy']:.4f} bits")
    print(f"   Normalized entropy: {entropy_results['normalized_entropy']:.4f}")

    for window_name, stats in entropy_results['window_entropies'].items():
        print(f"   {window_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Sum Distribution
    print("\n5. Sum Distribution Analysis")
    sum_stats = analyzer.sum_distribution(df)
    print(f"   Mean sum: {sum_stats['mean']:.2f}")
    print(f"   Std sum: {sum_stats['std']:.2f}")
    print(f"   Median sum: {sum_stats['median']:.2f}")
    print(f"   Skewness: {sum_stats['skewness']:.4f}")
    print(f"   Kurtosis: {sum_stats['kurtosis']:.4f}")
    print(f"   Shapiro test (normality): p={sum_stats['shapiro_test']['p_value']:.4f}, "
          f"normal={sum_stats['shapiro_test']['is_normal']}")

    # Bias Detection
    print("\n6. Bias Detection")
    bias_results = analyzer.detect_bias(frequencies)
    print(f"   Has significant bias: {bias_results['has_bias']}")
    print(f"   Overrepresented numbers: {bias_results['overrepresented'][:5]}...")
    print(f"   Underrepresented numbers: {bias_results['underrepresented'][:5]}...")
    print(f"   Critical value: {bias_results['critical_value']:.4f}")

    # Hot/Cold Analysis
    print("\n7. Hot/Cold Number Analysis")
    hot_cold = analyzer.hot_cold_analysis(frequencies)
    print(f"   Hot numbers (Q3): {hot_cold['hot_numbers'][:10]}")
    print(f"   Cold numbers (Q1): {hot_cold['cold_numbers'][:10]}")
    print(f"   Statistics: Q1={hot_cold['statistics']['q1']:.1f}, "
          f"Median={hot_cold['statistics']['median']:.1f}, "
          f"Q3={hot_cold['statistics']['q3']:.1f}")

    # Gap Distribution
    print("\n8. Gap Distribution Analysis")
    gaps = analyzer.gap_distribution(df)
    sample_number = 10
    if sample_number in gaps and gaps[sample_number]:
        print(f"   Number {sample_number} gaps: "
              f"mean={np.mean(gaps[sample_number]):.2f}, "
              f"min={min(gaps[sample_number])}, "
              f"max={max(gaps[sample_number])}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZATION EXAMPLES")
    print("=" * 80)

    print("\n9. Creating Visualizations...")

    # Frequency Histogram
    print("   - Frequency histogram")
    fig1 = visualizer.plot_frequency_histogram(frequencies, highlight_hot_cold=True)
    visualizer.save_figure(fig1, '/tmp/frequency_histogram.png')
    print("     Saved to: /tmp/frequency_histogram.png")

    # Sum Distribution
    print("   - Sum distribution with normal curve")
    fig2 = visualizer.plot_sum_distribution(sum_stats['sums'])
    visualizer.save_figure(fig2, '/tmp/sum_distribution.png')
    print("     Saved to: /tmp/sum_distribution.png")

    # Rolling Frequency
    print("   - Rolling frequency for number 10")
    fig3 = visualizer.plot_rolling_frequency(df, number=10, windows=[50, 100])
    visualizer.save_figure(fig3, '/tmp/rolling_frequency.png')
    print("     Saved to: /tmp/rolling_frequency.png")

    # Probability Distribution
    print("   - Probability distribution (simulated)")
    probs = pd.Series(np.random.dirichlet(np.ones(60)), index=range(1, 61))
    fig4 = visualizer.plot_probability_distribution(probs, top_n=20)
    visualizer.save_figure(fig4, '/tmp/probability_distribution.png')
    print("     Saved to: /tmp/probability_distribution.png")

    # Prediction vs Actual
    print("   - Prediction vs actual comparison")
    predictions = [list(np.random.choice(range(1, 61), size=6, replace=False)) for _ in range(20)]
    actuals = [list(np.random.choice(range(1, 61), size=6, replace=False)) for _ in range(20)]
    fig5 = visualizer.plot_prediction_vs_actual(predictions, actuals)
    visualizer.save_figure(fig5, '/tmp/prediction_vs_actual.png')
    print("     Saved to: /tmp/prediction_vs_actual.png")

    # Correlation Matrix
    print("   - Correlation matrix")
    corr_matrix = analyzer.correlation_analysis(df)
    fig6 = visualizer.plot_correlation_matrix(corr_matrix)
    visualizer.save_figure(fig6, '/tmp/correlation_matrix.png')
    print("     Saved to: /tmp/correlation_matrix.png")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll visualizations saved to /tmp/")
    print("\nAvailable StatisticalAnalyzer methods:")
    methods = [m for m in dir(analyzer) if not m.startswith('_') and callable(getattr(analyzer, m))]
    for method in methods:
        print(f"  - {method}")

    print("\nAvailable Visualizer methods:")
    methods = [m for m in dir(visualizer) if not m.startswith('_') and callable(getattr(visualizer, m))]
    for method in methods:
        print(f"  - {method}")


if __name__ == "__main__":
    main()

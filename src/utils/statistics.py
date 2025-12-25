"""Statistical analysis utilities for Mega-Sena prediction system."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, chisquare
from collections import Counter


class StatisticalAnalyzer:
    """
    Statistical analysis tools for lottery draw analysis.

    Provides methods for chi-square tests, entropy calculations,
    distribution analysis, and anomaly detection.
    """

    def __init__(self):
        """Initialize the StatisticalAnalyzer."""
        pass

    # ==================== Chi-Square Tests ====================

    def chi_square_test(
        self,
        observed_frequencies: np.ndarray,
        expected_frequencies: Optional[np.ndarray] = None,
        significance: float = 0.05
    ) -> Tuple[float, float, str]:
        """
        Perform chi-square test to determine if distribution is uniform.

        Tests the null hypothesis that the observed frequencies follow
        the expected distribution (uniform by default).

        Args:
            observed_frequencies: Array of observed frequencies for each number
            expected_frequencies: Array of expected frequencies (default: uniform)
            significance: Significance level for the test (default: 0.05)

        Returns:
            Tuple containing:
                - chi_square_statistic: The chi-square test statistic
                - p_value: The p-value of the test
                - conclusion: String interpretation of the result

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> observed = np.array([45, 52, 48, 50, 55])
            >>> stat, p_val, conclusion = analyzer.chi_square_test(observed)
        """
        observed = np.array(observed_frequencies)

        if expected_frequencies is None:
            # Assume uniform distribution
            expected = np.full_like(observed, np.mean(observed), dtype=float)
        else:
            expected = np.array(expected_frequencies)

        # Perform chi-square test
        chi_stat, p_value = chisquare(observed, f_exp=expected)

        # Interpret result
        if p_value < significance:
            conclusion = (
                f"REJECT null hypothesis (p={p_value:.4f} < {significance}). "
                f"Distribution is NOT uniform - statistically significant deviation detected."
            )
        else:
            conclusion = (
                f"ACCEPT null hypothesis (p={p_value:.4f} >= {significance}). "
                f"Distribution appears uniform - no significant deviation."
            )

        return chi_stat, p_value, conclusion

    # ==================== Shannon Entropy ====================

    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Shannon entropy to measure randomness.

        Shannon entropy quantifies the unpredictability/randomness of a distribution.
        Higher entropy = more random/uniform distribution.
        Maximum entropy occurs with uniform distribution.

        Args:
            probabilities: Array of probabilities (must sum to 1)

        Returns:
            Entropy value in bits

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform
            >>> entropy = analyzer.calculate_entropy(probs)  # Returns 2.0
        """
        probs = np.array(probabilities)

        # Remove zero probabilities to avoid log(0)
        probs = probs[probs > 0]

        # Normalize if not already
        if not np.isclose(probs.sum(), 1.0):
            probs = probs / probs.sum()

        # Calculate Shannon entropy: H = -sum(p * log2(p))
        entropy = -np.sum(probs * np.log2(probs))

        return entropy

    def entropy_analysis(
        self,
        draws: pd.DataFrame,
        window_sizes: List[int] = [50, 100, 200]
    ) -> Dict[str, any]:
        """
        Analyze entropy over different time windows.

        Examines how randomness changes over time by calculating
        entropy in rolling windows.

        Args:
            draws: DataFrame containing draw history
            window_sizes: List of window sizes to analyze

        Returns:
            Dictionary containing entropy metrics for each window size

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> results = analyzer.entropy_analysis(df, [50, 100])
        """
        results = {
            'overall_entropy': None,
            'window_entropies': {},
            'max_entropy': None,
            'normalized_entropy': None
        }

        # Get all drawn numbers
        ball_columns = [col for col in draws.columns if col.startswith('Bola')]
        all_numbers = draws[ball_columns].values.flatten()

        # Calculate overall entropy
        freq = pd.Series(all_numbers).value_counts(normalize=True)
        results['overall_entropy'] = self.calculate_entropy(freq.values)

        # Calculate maximum possible entropy (uniform distribution)
        n_unique = len(freq)
        results['max_entropy'] = np.log2(n_unique)

        # Normalized entropy (0-1 scale)
        results['normalized_entropy'] = results['overall_entropy'] / results['max_entropy']

        # Calculate entropy for different windows
        for window in window_sizes:
            if len(draws) < window:
                continue

            window_entropies = []
            for i in range(len(draws) - window + 1):
                window_data = draws.iloc[i:i+window][ball_columns].values.flatten()
                freq = pd.Series(window_data).value_counts(normalize=True)
                entropy = self.calculate_entropy(freq.values)
                window_entropies.append(entropy)

            results['window_entropies'][f'window_{window}'] = {
                'mean': np.mean(window_entropies),
                'std': np.std(window_entropies),
                'min': np.min(window_entropies),
                'max': np.max(window_entropies),
                'values': window_entropies
            }

        return results

    # ==================== Distribution Analysis ====================

    def frequency_distribution(
        self,
        df: pd.DataFrame,
        ball_columns: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Calculate frequency distribution of each number.

        Args:
            df: DataFrame containing draw history
            ball_columns: List of column names containing ball numbers
                         (default: auto-detect columns starting with 'Bola')

        Returns:
            Series with number frequencies, sorted by number

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> freq = analyzer.frequency_distribution(df)
        """
        if ball_columns is None:
            ball_columns = [col for col in df.columns if col.startswith('Bola')]

        # Flatten all numbers
        all_numbers = df[ball_columns].values.flatten()

        # Count frequencies
        freq = pd.Series(all_numbers).value_counts().sort_index()

        return freq

    def gap_distribution(
        self,
        df: pd.DataFrame,
        ball_columns: Optional[List[str]] = None
    ) -> Dict[int, List[int]]:
        """
        Analyze gap patterns (draws between consecutive appearances).

        For each number, calculates the number of draws between
        consecutive appearances.

        Args:
            df: DataFrame containing draw history (sorted by date, oldest first)
            ball_columns: List of column names containing ball numbers

        Returns:
            Dictionary mapping number -> list of gaps

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> gaps = analyzer.gap_distribution(df)
            >>> print(gaps[10])  # Gaps for number 10
        """
        if ball_columns is None:
            ball_columns = [col for col in df.columns if col.startswith('Bola')]

        gaps = {}

        # Get all unique numbers
        all_numbers = df[ball_columns].values.flatten()
        unique_numbers = np.unique(all_numbers)

        for number in unique_numbers:
            # Find all draws where this number appeared
            appearances = []
            for idx, row in df.iterrows():
                if number in row[ball_columns].values:
                    appearances.append(idx)

            # Calculate gaps between consecutive appearances
            number_gaps = []
            for i in range(len(appearances) - 1):
                gap = appearances[i + 1] - appearances[i] - 1
                number_gaps.append(gap)

            gaps[number] = number_gaps

        return gaps

    def sum_distribution(
        self,
        df: pd.DataFrame,
        ball_columns: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Analyze sum patterns (should be approximately normal).

        The sum of randomly drawn numbers should follow a normal distribution
        due to the Central Limit Theorem.

        Args:
            df: DataFrame containing draw history
            ball_columns: List of column names containing ball numbers

        Returns:
            Dictionary containing sum statistics and normality test results

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> sum_stats = analyzer.sum_distribution(df)
        """
        if ball_columns is None:
            ball_columns = [col for col in df.columns if col.startswith('Bola')]

        # Calculate sum for each draw
        sums = df[ball_columns].sum(axis=1)

        # Perform Shapiro-Wilk normality test
        shapiro_stat, shapiro_p = stats.shapiro(sums)

        # Perform Kolmogorov-Smirnov test against normal distribution
        ks_stat, ks_p = stats.kstest(
            (sums - sums.mean()) / sums.std(),
            'norm'
        )

        results = {
            'sums': sums.values,
            'mean': sums.mean(),
            'std': sums.std(),
            'median': sums.median(),
            'min': sums.min(),
            'max': sums.max(),
            'skewness': stats.skew(sums),
            'kurtosis': stats.kurtosis(sums),
            'shapiro_test': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'ks_test': {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > 0.05
            }
        }

        return results

    # ==================== Anomaly Detection ====================

    def detect_bias(
        self,
        frequencies: pd.Series,
        significance: float = 0.05
    ) -> Dict[str, any]:
        """
        Detect statistically significant deviations from uniform distribution.

        Uses chi-square test and standardized residuals to identify
        numbers that appear significantly more or less than expected.

        Args:
            frequencies: Series of number frequencies
            significance: Significance level (default: 0.05)

        Returns:
            Dictionary containing:
                - biased_numbers: List of numbers with significant deviation
                - overrepresented: Numbers appearing too frequently
                - underrepresented: Numbers appearing too infrequently
                - standardized_residuals: Residuals for each number
                - chi_square_result: Chi-square test result

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> freq = analyzer.frequency_distribution(df)
            >>> bias = analyzer.detect_bias(freq)
        """
        observed = frequencies.values
        expected = np.mean(observed)

        # Perform chi-square test
        chi_stat, p_value, conclusion = self.chi_square_test(observed, significance=significance)

        # Calculate standardized residuals
        # Formula: (observed - expected) / sqrt(expected)
        residuals = (observed - expected) / np.sqrt(expected)

        # Critical value for significance level (two-tailed)
        critical_value = stats.norm.ppf(1 - significance / 2)

        # Identify biased numbers
        biased_mask = np.abs(residuals) > critical_value
        biased_indices = np.where(biased_mask)[0]
        biased_numbers = frequencies.index[biased_indices].tolist()

        # Separate over/under represented
        overrepresented = frequencies.index[residuals > critical_value].tolist()
        underrepresented = frequencies.index[residuals < -critical_value].tolist()

        results = {
            'has_bias': p_value < significance,
            'biased_numbers': biased_numbers,
            'overrepresented': overrepresented,
            'underrepresented': underrepresented,
            'standardized_residuals': pd.Series(residuals, index=frequencies.index),
            'chi_square_result': {
                'statistic': chi_stat,
                'p_value': p_value,
                'conclusion': conclusion
            },
            'critical_value': critical_value
        }

        return results

    def hot_cold_analysis(
        self,
        frequencies: pd.Series,
        window: Optional[int] = None
    ) -> Dict[str, List[int]]:
        """
        Identify hot and cold numbers based on frequency distribution.

        Hot numbers: Upper quartile (Q3) of frequencies
        Cold numbers: Lower quartile (Q1) of frequencies

        Args:
            frequencies: Series of number frequencies
            window: Optional window size for recent analysis (uses all data if None)

        Returns:
            Dictionary containing:
                - hot_numbers: List of frequently appearing numbers
                - cold_numbers: List of infrequently appearing numbers
                - warm_numbers: Numbers in middle range
                - statistics: Quartile boundaries and thresholds

        Example:
            >>> analyzer = StatisticalAnalyzer()
            >>> freq = analyzer.frequency_distribution(df)
            >>> hot_cold = analyzer.hot_cold_analysis(freq)
        """
        freq = frequencies.copy()

        # Calculate quartiles
        q1 = freq.quantile(0.25)
        q2 = freq.quantile(0.50)  # median
        q3 = freq.quantile(0.75)

        # Classify numbers
        hot_numbers = freq[freq >= q3].index.tolist()
        cold_numbers = freq[freq <= q1].index.tolist()
        warm_numbers = freq[(freq > q1) & (freq < q3)].index.tolist()

        results = {
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'warm_numbers': warm_numbers,
            'statistics': {
                'q1': q1,
                'median': q2,
                'q3': q3,
                'mean': freq.mean(),
                'std': freq.std(),
                'min': freq.min(),
                'max': freq.max()
            },
            'counts': {
                'hot': len(hot_numbers),
                'cold': len(cold_numbers),
                'warm': len(warm_numbers)
            }
        }

        return results

    # ==================== Helper Methods ====================

    def correlation_analysis(
        self,
        df: pd.DataFrame,
        ball_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze correlations between ball positions.

        Args:
            df: DataFrame containing draw history
            ball_columns: List of column names containing ball numbers

        Returns:
            Correlation matrix as DataFrame
        """
        if ball_columns is None:
            ball_columns = [col for col in df.columns if col.startswith('Bola')]

        return df[ball_columns].corr()

    def runs_test(
        self,
        sequence: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[float, float, str]:
        """
        Perform runs test to detect non-randomness in a sequence.

        A run is a sequence of adjacent equal values. Too few or too many
        runs suggests non-randomness.

        Args:
            sequence: Binary sequence or sequence to convert to binary
            threshold: Threshold for converting to binary (default: median)

        Returns:
            Tuple containing:
                - z_statistic: Standardized test statistic
                - p_value: Two-tailed p-value
                - conclusion: Interpretation of result
        """
        seq = np.array(sequence)

        # Convert to binary if not already
        if threshold is None:
            threshold = np.median(seq)

        binary = (seq > threshold).astype(int)

        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1

        # Count 0s and 1s
        n1 = np.sum(binary)
        n0 = len(binary) - n1

        # Expected runs and variance
        n = len(binary)
        expected_runs = (2 * n0 * n1) / n + 1
        variance_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))

        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        if p_value < 0.05:
            conclusion = f"REJECT randomness (p={p_value:.4f}). Sequence is NOT random."
        else:
            conclusion = f"ACCEPT randomness (p={p_value:.4f}). Sequence appears random."

        return z_stat, p_value, conclusion

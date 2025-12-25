"""Visualization utilities for Mega-Sena prediction system."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from datetime import datetime


class Visualizer:
    """
    Visualization tools for lottery analysis and predictions.

    Provides methods for creating frequency plots, distribution plots,
    time series plots, and prediction visualizations.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the Visualizer.

        Args:
            style: Matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
        """
        self.style = style
        self.colors = sns.color_palette("husl", 8)
        self.figsize_default = (12, 6)
        self.figsize_large = (14, 8)
        self.figsize_square = (10, 10)

        # Set default style
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass

        # Set seaborn defaults
        sns.set_palette("husl")

    # ==================== Frequency Plots ====================

    def plot_frequency_histogram(
        self,
        frequencies: pd.Series,
        title: str = "Number Frequency Distribution",
        xlabel: str = "Number",
        ylabel: str = "Frequency",
        highlight_hot_cold: bool = True,
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create bar chart of number frequencies.

        Args:
            frequencies: Series with number frequencies
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            highlight_hot_cold: Whether to highlight hot/cold numbers
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> fig = viz.plot_frequency_histogram(frequencies)
            >>> fig.savefig('frequency.png')
        """
        figsize = figsize or self.figsize_default
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by number
        freq_sorted = frequencies.sort_index()

        # Determine colors
        if highlight_hot_cold:
            q1 = freq_sorted.quantile(0.25)
            q3 = freq_sorted.quantile(0.75)

            colors = []
            for val in freq_sorted.values:
                if val >= q3:
                    colors.append('#e74c3c')  # Red for hot
                elif val <= q1:
                    colors.append('#3498db')  # Blue for cold
                else:
                    colors.append('#95a5a6')  # Gray for warm

            # Create bar plot
            bars = ax.bar(freq_sorted.index, freq_sorted.values, color=colors, alpha=0.8, edgecolor='black')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', label=f'Hot (≥ {q3:.0f})'),
                Patch(facecolor='#95a5a6', label='Warm'),
                Patch(facecolor='#3498db', label=f'Cold (≤ {q1:.0f})')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        else:
            bars = ax.bar(freq_sorted.index, freq_sorted.values,
                         color=self.colors[0], alpha=0.7, edgecolor='black')

        # Add mean line
        mean_freq = freq_sorted.mean()
        ax.axhline(mean_freq, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_freq:.1f}', alpha=0.7)

        # Formatting
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars (for smaller datasets)
        if len(freq_sorted) <= 60:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8, rotation=0)

        plt.tight_layout()
        return fig

    def plot_frequency_heatmap(
        self,
        df: pd.DataFrame,
        ball_columns: Optional[List[str]] = None,
        window: int = 50,
        title: str = "Number Frequency Heatmap Over Time",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create heatmap showing number frequency evolution over time.

        Args:
            df: DataFrame containing draw history
            ball_columns: List of column names containing ball numbers
            window: Rolling window size for frequency calculation
            title: Plot title
            figsize: Figure size (default: self.figsize_large)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> fig = viz.plot_frequency_heatmap(df, window=100)
        """
        figsize = figsize or self.figsize_large
        fig, ax = plt.subplots(figsize=figsize)

        if ball_columns is None:
            ball_columns = [col for col in df.columns if col.startswith('Bola')]

        # Get all unique numbers
        all_numbers = df[ball_columns].values.flatten()
        unique_numbers = sorted(np.unique(all_numbers))

        # Calculate rolling frequencies
        n_windows = len(df) - window + 1
        freq_matrix = np.zeros((len(unique_numbers), n_windows))

        for i in range(n_windows):
            window_data = df.iloc[i:i+window][ball_columns].values.flatten()
            freq = pd.Series(window_data).value_counts()

            for j, num in enumerate(unique_numbers):
                freq_matrix[j, i] = freq.get(num, 0)

        # Create heatmap
        im = ax.imshow(freq_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency', rotation=270, labelpad=20, fontweight='bold')

        # Formatting
        ax.set_xlabel('Window Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(Window Size: {window} draws)',
                    fontsize=14, fontweight='bold', pad=20)

        # Set y-axis ticks to show numbers
        step = max(1, len(unique_numbers) // 20)  # Show max 20 labels
        ax.set_yticks(range(0, len(unique_numbers), step))
        ax.set_yticklabels([unique_numbers[i] for i in range(0, len(unique_numbers), step)])

        plt.tight_layout()
        return fig

    # ==================== Distribution Plots ====================

    def plot_sum_distribution(
        self,
        sums: np.ndarray,
        title: str = "Sum Distribution with Normal Curve",
        xlabel: str = "Sum of Numbers",
        ylabel: str = "Frequency",
        bins: int = 30,
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create histogram of sums with normal curve overlay.

        Args:
            sums: Array of sum values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> sums = df[ball_cols].sum(axis=1).values
            >>> fig = viz.plot_sum_distribution(sums)
        """
        figsize = figsize or self.figsize_default
        fig, ax = plt.subplots(figsize=figsize)

        # Create histogram
        n, bins_edges, patches = ax.hist(sums, bins=bins, density=True,
                                         alpha=0.7, color=self.colors[0],
                                         edgecolor='black', label='Observed')

        # Fit normal distribution
        mu, sigma = np.mean(sums), np.std(sums)
        x = np.linspace(sums.min(), sums.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)

        # Plot normal curve
        ax.plot(x, normal_curve, 'r-', linewidth=2,
               label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')

        # Add vertical lines for mean and median
        ax.axvline(mu, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mu:.1f}', alpha=0.7)
        ax.axvline(np.median(sums), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(sums):.1f}', alpha=0.7)

        # Formatting
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f'n = {len(sums)}\nSkewness: {stats.skew(sums):.3f}\nKurtosis: {stats.kurtosis(sums):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_gap_heatmap(
        self,
        gaps: Dict[int, List[int]],
        title: str = "Gap Distribution Heatmap",
        max_numbers: int = 60,
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create heatmap of gap patterns per number.

        Args:
            gaps: Dictionary mapping number -> list of gaps
            title: Plot title
            max_numbers: Maximum numbers to display (for readability)
            figsize: Figure size (default: self.figsize_square)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> from src.utils.statistics import StatisticalAnalyzer
            >>> analyzer = StatisticalAnalyzer()
            >>> gaps = analyzer.gap_distribution(df)
            >>> fig = viz.plot_gap_heatmap(gaps)
        """
        figsize = figsize or self.figsize_square
        fig, ax = plt.subplots(figsize=figsize)

        # Sort numbers
        numbers = sorted(gaps.keys())[:max_numbers]

        # Calculate statistics for each number
        gap_stats = []
        for num in numbers:
            if gaps[num]:
                gap_stats.append({
                    'number': num,
                    'mean': np.mean(gaps[num]),
                    'std': np.std(gaps[num]),
                    'min': np.min(gaps[num]),
                    'max': np.max(gaps[num]),
                    'median': np.median(gaps[num])
                })
            else:
                gap_stats.append({
                    'number': num,
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
                })

        # Create matrix for heatmap
        stats_df = pd.DataFrame(gap_stats)
        matrix = stats_df[['mean', 'median', 'std', 'min', 'max']].T

        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.1f', cmap='coolwarm',
                   xticklabels=numbers, yticklabels=['Mean', 'Median', 'Std', 'Min', 'Max'],
                   cbar_kws={'label': 'Draws'}, ax=ax, linewidths=0.5)

        # Formatting
        ax.set_xlabel('Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Statistic', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    # ==================== Time Series Plots ====================

    def plot_rolling_frequency(
        self,
        df: pd.DataFrame,
        number: int,
        windows: List[int] = [50, 100, 200],
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot rolling average frequency for a specific number.

        Args:
            df: DataFrame containing draw history
            number: Number to analyze
            windows: List of window sizes for rolling average
            title: Plot title (default: auto-generated)
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> fig = viz.plot_rolling_frequency(df, number=10, windows=[50, 100])
        """
        figsize = figsize or self.figsize_default
        fig, ax = plt.subplots(figsize=figsize)

        ball_columns = [col for col in df.columns if col.startswith('Bola')]

        # Check if number appears in each draw
        appears = df[ball_columns].apply(lambda row: number in row.values, axis=1).astype(int)

        # Calculate rolling averages for different windows
        for i, window in enumerate(windows):
            if len(appears) >= window:
                rolling_freq = appears.rolling(window=window).mean() * 100  # Convert to percentage
                ax.plot(range(len(rolling_freq)), rolling_freq,
                       label=f'{window}-draw window',
                       linewidth=2, alpha=0.7, color=self.colors[i])

        # Add expected frequency line
        expected_freq = (len(ball_columns) / 60) * 100  # For Mega-Sena (6 balls from 60)
        ax.axhline(expected_freq, color='red', linestyle='--',
                  linewidth=2, label=f'Expected: {expected_freq:.2f}%', alpha=0.7)

        # Formatting
        title = title or f'Rolling Frequency for Number {number}'
        ax.set_xlabel('Draw Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_model_components(
        self,
        prophet_model,
        forecast: Optional[pd.DataFrame] = None,
        title: str = "Prophet Model Components",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot Prophet model decomposition (trend, seasonality).

        Args:
            prophet_model: Fitted Prophet model
            forecast: Forecast DataFrame (if None, uses model's forecast)
            title: Plot title
            figsize: Figure size (default: (14, 10))

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> from prophet import Prophet
            >>> model = Prophet()
            >>> model.fit(train_data)
            >>> fig = viz.plot_model_components(model)
        """
        figsize = figsize or (14, 10)

        # Use Prophet's built-in plotting function
        fig = prophet_model.plot_components(forecast)

        # Enhance the title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        return fig

    def plot_entropy_evolution(
        self,
        entropy_values: List[float],
        max_entropy: float,
        window_size: int,
        title: str = "Entropy Evolution Over Time",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot how entropy changes over time.

        Args:
            entropy_values: List of entropy values
            max_entropy: Maximum possible entropy (for normalization)
            window_size: Window size used for calculation
            title: Plot title
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or self.figsize_default
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize entropy
        normalized = np.array(entropy_values) / max_entropy

        # Plot entropy evolution
        ax.plot(range(len(entropy_values)), entropy_values,
               linewidth=2, label='Entropy', color=self.colors[0])

        # Add max entropy line
        ax.axhline(max_entropy, color='red', linestyle='--',
                  linewidth=2, label=f'Max Entropy: {max_entropy:.2f}', alpha=0.7)

        # Add mean line
        mean_entropy = np.mean(entropy_values)
        ax.axhline(mean_entropy, color='green', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_entropy:.2f}', alpha=0.7)

        # Formatting
        ax.set_xlabel('Window Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(Window Size: {window_size} draws)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add normalized entropy on secondary y-axis
        ax2 = ax.twinx()
        ax2.fill_between(range(len(normalized)), normalized, alpha=0.3,
                         color=self.colors[1], label='Normalized (0-1)')
        ax2.set_ylabel('Normalized Entropy', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='upper right')

        plt.tight_layout()
        return fig

    # ==================== Prediction Plots ====================

    def plot_probability_distribution(
        self,
        probabilities: pd.Series,
        top_n: int = 20,
        title: str = "Predicted Number Probabilities",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create bar chart of predicted probabilities.

        Args:
            probabilities: Series with number probabilities
            top_n: Number of top probabilities to show
            title: Plot title
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> probs = pd.Series({1: 0.15, 2: 0.12, 3: 0.10, ...})
            >>> fig = viz.plot_probability_distribution(probs, top_n=15)
        """
        figsize = figsize or self.figsize_default
        fig, ax = plt.subplots(figsize=figsize)

        # Sort and get top N
        probs_sorted = probabilities.sort_values(ascending=False).head(top_n)

        # Create bar plot
        bars = ax.bar(range(len(probs_sorted)), probs_sorted.values,
                     color=self.colors[0], alpha=0.7, edgecolor='black')

        # Color the top 6 differently (likely picks)
        for i in range(min(6, len(bars))):
            bars[i].set_color('#e74c3c')
            bars[i].set_alpha(0.9)

        # Add value labels
        for i, (idx, val) in enumerate(probs_sorted.items()):
            ax.text(i, val, f'{val:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

        # Formatting
        ax.set_xticks(range(len(probs_sorted)))
        ax.set_xticklabels(probs_sorted.index, rotation=0)
        ax.set_xlabel('Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(Top {top_n} Numbers)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        # Add threshold line
        threshold = 1.0 / 60  # Expected probability for uniform distribution
        ax.axhline(threshold, color='red', linestyle='--',
                  linewidth=2, label=f'Expected: {threshold:.4f}', alpha=0.7)
        ax.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def plot_prediction_vs_actual(
        self,
        predictions: List[List[int]],
        actuals: List[List[int]],
        dates: Optional[List] = None,
        title: str = "Prediction Performance Over Time",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Compare predictions vs actual results over time.

        Args:
            predictions: List of predicted number sets
            actuals: List of actual drawn number sets
            dates: Optional list of dates for x-axis
            title: Plot title
            figsize: Figure size (default: self.figsize_default)

        Returns:
            Matplotlib Figure object

        Example:
            >>> viz = Visualizer()
            >>> preds = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
            >>> acts = [[1, 2, 10, 11, 12, 13], [5, 6, 7, 8, 15, 16]]
            >>> fig = viz.plot_prediction_vs_actual(preds, acts)
        """
        figsize = figsize or self.figsize_default
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.5))

        # Calculate match counts
        match_counts = []
        for pred, actual in zip(predictions, actuals):
            pred_set = set(pred)
            actual_set = set(actual)
            matches = len(pred_set.intersection(actual_set))
            match_counts.append(matches)

        x_axis = dates if dates else range(len(predictions))

        # Plot 1: Match counts over time
        ax1.plot(x_axis, match_counts, marker='o', linewidth=2,
                markersize=6, color=self.colors[0], label='Matches')
        ax1.fill_between(x_axis, match_counts, alpha=0.3, color=self.colors[0])

        # Add horizontal lines for different prize tiers
        prize_tiers = {
            'Sena (6)': 6,
            'Quina (5)': 5,
            'Quadra (4)': 4
        }
        for i, (name, threshold) in enumerate(prize_tiers.items()):
            ax1.axhline(threshold, color=self.colors[i+1], linestyle='--',
                       linewidth=2, label=name, alpha=0.7)

        ax1.set_xlabel('Draw' if not dates else 'Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
        ax1.set_title('Number of Correct Predictions per Draw',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 6.5])

        # Plot 2: Distribution of matches
        match_dist = pd.Series(match_counts).value_counts().sort_index()
        ax2.bar(match_dist.index, match_dist.values,
               color=self.colors[0], alpha=0.7, edgecolor='black')

        for idx, val in match_dist.items():
            ax2.text(idx, val, str(val), ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        ax2.set_xlabel('Number of Matches', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Match Counts', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(range(0, 7))

        # Add statistics text
        stats_text = (f'Total Draws: {len(predictions)}\n'
                     f'Avg Matches: {np.mean(match_counts):.2f}\n'
                     f'Best: {max(match_counts)} matches\n'
                     f'4+ matches: {sum(1 for m in match_counts if m >= 4)}')
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Ball Position Correlation Matrix",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot correlation matrix as a heatmap.

        Args:
            correlation_matrix: DataFrame with correlation coefficients
            title: Plot title
            figsize: Figure size (default: self.figsize_square)

        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or self.figsize_square
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'},
                   ax=ax)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def plot_residuals(
        self,
        residuals: np.ndarray,
        title: str = "Residual Analysis",
        figsize: Optional[Tuple[int, int]] = None
    ) -> matplotlib.figure.Figure:
        """
        Create residual plots for model diagnostics.

        Args:
            residuals: Array of residuals (predicted - actual)
            title: Plot title
            figsize: Figure size (default: (14, 10))

        Returns:
            Matplotlib Figure object
        """
        figsize = figsize or (14, 10)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Residuals over time
        ax1.plot(residuals, marker='o', linestyle='-', alpha=0.6, color=self.colors[0])
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Index', fontweight='bold')
        ax1.set_ylabel('Residual', fontweight='bold')
        ax1.set_title('Residuals Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Histogram of residuals
        ax2.hist(residuals, bins=30, color=self.colors[1], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residual', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution of Residuals', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Autocorrelation
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(pd.Series(residuals), ax=ax4, color=self.colors[2])
        ax4.set_title('Autocorrelation of Residuals', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    # ==================== Utility Methods ====================

    def save_figure(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save figure to file.

        Args:
            fig: Matplotlib Figure object
            filepath: Path to save the figure
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box specification
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)

    def create_subplot_grid(
        self,
        n_plots: int,
        ncols: int = 2,
        figsize_per_plot: Tuple[int, int] = (6, 4)
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
        """
        Create a grid of subplots.

        Args:
            n_plots: Number of plots needed
            ncols: Number of columns
            figsize_per_plot: Size of each subplot

        Returns:
            Tuple of (figure, axes_array)
        """
        nrows = int(np.ceil(n_plots / ncols))
        figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes array for easier iteration
        if n_plots == 1:
            axes = np.array([axes])
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        return fig, axes

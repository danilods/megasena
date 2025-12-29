# Mega-Sena Utility Modules Guide

This guide provides comprehensive documentation for the statistical analysis and visualization utilities.

## Table of Contents

1. [Installation](#installation)
2. [StatisticalAnalyzer](#statisticalanalyzer)
3. [Visualizer](#visualizer)
4. [Usage Examples](#usage-examples)
5. [Quick Reference](#quick-reference)

---

## Installation

Required dependencies:
```bash
pip install numpy pandas scipy matplotlib seaborn
```

---

## StatisticalAnalyzer

The `StatisticalAnalyzer` class provides comprehensive statistical analysis tools for lottery draw analysis.

### Methods

#### 1. Chi-Square Test

Test if draw distribution is uniform:

```python
from src.utils.statistics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
frequencies = pd.Series([45, 52, 48, 50, 55])

chi_stat, p_value, conclusion = analyzer.chi_square_test(frequencies)
print(f"Chi-square: {chi_stat}, p-value: {p_value}")
print(conclusion)
```

**Parameters:**
- `observed_frequencies`: Array of observed frequencies for each number
- `expected_frequencies`: Optional array of expected frequencies (default: uniform)
- `significance`: Significance level (default: 0.05)

**Returns:**
- `chi_square_statistic`: The chi-square test statistic
- `p_value`: The p-value of the test
- `conclusion`: String interpretation of the result

---

#### 2. Shannon Entropy

Measure randomness of draws:

```python
# Calculate entropy from probabilities
probs = np.array([0.25, 0.25, 0.25, 0.25])
entropy = analyzer.calculate_entropy(probs)
print(f"Entropy: {entropy} bits")

# Analyze entropy over time windows
entropy_results = analyzer.entropy_analysis(df, window_sizes=[50, 100, 200])
print(f"Overall entropy: {entropy_results['overall_entropy']}")
print(f"Normalized: {entropy_results['normalized_entropy']}")
```

**Parameters (entropy_analysis):**
- `draws`: DataFrame containing draw history
- `window_sizes`: List of window sizes to analyze

**Returns:**
- Dictionary containing:
  - `overall_entropy`: Total entropy
  - `max_entropy`: Maximum possible entropy
  - `normalized_entropy`: Normalized value (0-1)
  - `window_entropies`: Statistics for each window size

---

#### 3. Frequency Distribution

Get frequency of each number:

```python
frequencies = analyzer.frequency_distribution(df)
print(frequencies.head())
```

**Parameters:**
- `df`: DataFrame containing draw history
- `ball_columns`: List of column names (default: auto-detect 'Bola*')

**Returns:**
- Series with number frequencies, sorted by number

---

#### 4. Gap Distribution

Analyze gaps between consecutive appearances:

```python
gaps = analyzer.gap_distribution(df)
print(f"Number 10 gaps: {gaps[10]}")
```

**Parameters:**
- `df`: DataFrame containing draw history (sorted by date, oldest first)
- `ball_columns`: List of column names

**Returns:**
- Dictionary mapping number -> list of gaps

---

#### 5. Sum Distribution

Analyze sum patterns (should be approximately normal):

```python
sum_stats = analyzer.sum_distribution(df)
print(f"Mean: {sum_stats['mean']}")
print(f"Std: {sum_stats['std']}")
print(f"Is normal: {sum_stats['shapiro_test']['is_normal']}")
```

**Returns:**
- Dictionary containing:
  - `sums`: Array of sum values
  - `mean`, `std`, `median`, `min`, `max`: Basic statistics
  - `skewness`, `kurtosis`: Distribution shape
  - `shapiro_test`: Shapiro-Wilk normality test results
  - `ks_test`: Kolmogorov-Smirnov test results

---

#### 6. Bias Detection

Detect statistically significant deviations:

```python
bias_results = analyzer.detect_bias(frequencies, significance=0.05)
print(f"Has bias: {bias_results['has_bias']}")
print(f"Overrepresented: {bias_results['overrepresented']}")
print(f"Underrepresented: {bias_results['underrepresented']}")
```

**Returns:**
- Dictionary containing:
  - `has_bias`: Boolean indicating significant deviation
  - `biased_numbers`: List of all biased numbers
  - `overrepresented`: Numbers appearing too frequently
  - `underrepresented`: Numbers appearing too infrequently
  - `standardized_residuals`: Residuals for each number
  - `chi_square_result`: Chi-square test details

---

#### 7. Hot/Cold Analysis

Identify hot and cold numbers:

```python
hot_cold = analyzer.hot_cold_analysis(frequencies)
print(f"Hot numbers: {hot_cold['hot_numbers']}")
print(f"Cold numbers: {hot_cold['cold_numbers']}")
print(f"Warm numbers: {hot_cold['warm_numbers']}")
```

**Returns:**
- Dictionary containing:
  - `hot_numbers`: Upper quartile (Q3) of frequencies
  - `cold_numbers`: Lower quartile (Q1) of frequencies
  - `warm_numbers`: Middle range
  - `statistics`: Quartile boundaries and thresholds

---

#### 8. Additional Methods

**Correlation Analysis:**
```python
corr_matrix = analyzer.correlation_analysis(df)
```

**Runs Test:**
```python
z_stat, p_value, conclusion = analyzer.runs_test(sequence)
```

---

## Visualizer

The `Visualizer` class provides comprehensive visualization tools.

### Methods

#### 1. Frequency Histogram

Create bar chart of number frequencies:

```python
from src.utils.visualization import Visualizer

viz = Visualizer()
fig = viz.plot_frequency_histogram(frequencies, highlight_hot_cold=True)
viz.save_figure(fig, 'frequency.png')
```

**Parameters:**
- `frequencies`: Series with number frequencies
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `highlight_hot_cold`: Whether to color hot/cold numbers (default: True)
- `figsize`: Figure size

**Returns:**
- Matplotlib Figure object

---

#### 2. Frequency Heatmap

Show frequency evolution over time:

```python
fig = viz.plot_frequency_heatmap(df, window=50)
```

**Parameters:**
- `df`: DataFrame containing draw history
- `ball_columns`: Column names
- `window`: Rolling window size
- `title`: Plot title
- `figsize`: Figure size

---

#### 3. Sum Distribution

Histogram with normal curve overlay:

```python
sums = df[ball_columns].sum(axis=1).values
fig = viz.plot_sum_distribution(sums)
```

**Parameters:**
- `sums`: Array of sum values
- `bins`: Number of histogram bins (default: 30)

---

#### 4. Gap Heatmap

Visualize gap patterns:

```python
gaps = analyzer.gap_distribution(df)
fig = viz.plot_gap_heatmap(gaps, max_numbers=60)
```

---

#### 5. Rolling Frequency

Plot rolling average frequency for a number:

```python
fig = viz.plot_rolling_frequency(df, number=10, windows=[50, 100, 200])
```

**Parameters:**
- `df`: DataFrame containing draw history
- `number`: Number to analyze
- `windows`: List of window sizes

---

#### 6. Prophet Model Components

Plot Prophet decomposition:

```python
from prophet import Prophet
model = Prophet()
model.fit(train_data)
forecast = model.predict(future)

fig = viz.plot_model_components(model, forecast)
```

---

#### 7. Probability Distribution

Bar chart of predicted probabilities:

```python
probs = pd.Series({1: 0.15, 2: 0.12, 3: 0.10, ...})
fig = viz.plot_probability_distribution(probs, top_n=20)
```

**Parameters:**
- `probabilities`: Series with number probabilities
- `top_n`: Number of top probabilities to show (default: 20)

---

#### 8. Prediction vs Actual

Compare predictions with actual results:

```python
predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
actuals = [[1, 2, 10, 11, 12, 13], [5, 6, 7, 8, 15, 16]]

fig = viz.plot_prediction_vs_actual(predictions, actuals)
```

**Parameters:**
- `predictions`: List of predicted number sets
- `actuals`: List of actual drawn number sets
- `dates`: Optional list of dates for x-axis

---

#### 9. Additional Visualizations

**Correlation Matrix:**
```python
corr_matrix = analyzer.correlation_analysis(df)
fig = viz.plot_correlation_matrix(corr_matrix)
```

**Residual Analysis:**
```python
residuals = predictions - actuals
fig = viz.plot_residuals(residuals)
```

**Entropy Evolution:**
```python
fig = viz.plot_entropy_evolution(entropy_values, max_entropy, window_size)
```

---

## Usage Examples

### Complete Analysis Workflow

```python
import pandas as pd
from src.utils.statistics import StatisticalAnalyzer
from src.utils.visualization import Visualizer

# Load data
df = pd.read_csv('megasena_draws.csv')

# Initialize
analyzer = StatisticalAnalyzer()
viz = Visualizer()

# 1. Frequency Analysis
frequencies = analyzer.frequency_distribution(df)
fig1 = viz.plot_frequency_histogram(frequencies)
viz.save_figure(fig1, 'frequency.png')

# 2. Test Uniformity
chi_stat, p_value, conclusion = analyzer.chi_square_test(frequencies)
print(conclusion)

# 3. Detect Bias
bias = analyzer.detect_bias(frequencies)
if bias['has_bias']:
    print(f"Overrepresented: {bias['overrepresented']}")
    print(f"Underrepresented: {bias['underrepresented']}")

# 4. Hot/Cold Analysis
hot_cold = analyzer.hot_cold_analysis(frequencies)
print(f"Hot numbers: {hot_cold['hot_numbers']}")
print(f"Cold numbers: {hot_cold['cold_numbers']}")

# 5. Sum Distribution
sum_stats = analyzer.sum_distribution(df)
fig2 = viz.plot_sum_distribution(sum_stats['sums'])
viz.save_figure(fig2, 'sums.png')

# 6. Entropy Analysis
entropy = analyzer.entropy_analysis(df)
print(f"Entropy: {entropy['overall_entropy']:.4f} bits")
print(f"Normalized: {entropy['normalized_entropy']:.4f}")

# 7. Gap Analysis
gaps = analyzer.gap_distribution(df)
fig3 = viz.plot_gap_heatmap(gaps)
viz.save_figure(fig3, 'gaps.png')

# 8. Correlation
corr = analyzer.correlation_analysis(df)
fig4 = viz.plot_correlation_matrix(corr)
viz.save_figure(fig4, 'correlation.png')
```

### Time Series Analysis

```python
# Analyze how a number's frequency changes over time
number = 10
fig = viz.plot_rolling_frequency(df, number, windows=[50, 100, 200])
viz.save_figure(fig, f'rolling_freq_{number}.png')

# Analyze entropy evolution
entropy_results = analyzer.entropy_analysis(df, window_sizes=[100])
entropy_values = entropy_results['window_entropies']['window_100']['values']
fig = viz.plot_entropy_evolution(
    entropy_values,
    entropy_results['max_entropy'],
    window_size=100
)
viz.save_figure(fig, 'entropy_evolution.png')
```

### Prediction Evaluation

```python
# Evaluate prediction performance
predictions = [...]  # List of predicted number sets
actuals = [...]      # List of actual results
dates = [...]        # Optional dates

fig = viz.plot_prediction_vs_actual(predictions, actuals, dates)
viz.save_figure(fig, 'prediction_performance.png')

# Analyze prediction probabilities
probabilities = model.predict_proba()  # Your model's probabilities
fig = viz.plot_probability_distribution(probabilities, top_n=20)
viz.save_figure(fig, 'probabilities.png')
```

---

## Quick Reference

### StatisticalAnalyzer Methods

| Method | Purpose | Key Output |
|--------|---------|------------|
| `chi_square_test()` | Test uniformity | Chi-square statistic, p-value |
| `calculate_entropy()` | Measure randomness | Entropy in bits |
| `entropy_analysis()` | Analyze entropy over time | Overall & window entropies |
| `frequency_distribution()` | Count occurrences | Series of frequencies |
| `gap_distribution()` | Analyze gaps | Dict of gap lists |
| `sum_distribution()` | Analyze sums | Stats & normality tests |
| `detect_bias()` | Find biased numbers | Over/under represented |
| `hot_cold_analysis()` | Classify numbers | Hot/cold/warm lists |
| `correlation_analysis()` | Ball correlations | Correlation matrix |
| `runs_test()` | Test randomness | Z-statistic, p-value |

### Visualizer Methods

| Method | Purpose | Best For |
|--------|---------|----------|
| `plot_frequency_histogram()` | Bar chart of frequencies | Overall frequency view |
| `plot_frequency_heatmap()` | Frequency over time | Temporal patterns |
| `plot_sum_distribution()` | Sum histogram | Normality verification |
| `plot_gap_heatmap()` | Gap statistics | Gap pattern analysis |
| `plot_rolling_frequency()` | Frequency trends | Single number analysis |
| `plot_model_components()` | Prophet decomposition | Time series models |
| `plot_probability_distribution()` | Prediction probabilities | Model outputs |
| `plot_prediction_vs_actual()` | Performance tracking | Model evaluation |
| `plot_correlation_matrix()` | Correlation heatmap | Dependency analysis |
| `plot_residuals()` | Residual diagnostics | Model validation |
| `plot_entropy_evolution()` | Entropy over time | Randomness trends |

---

## Testing

Run the demonstration script:

```bash
python3 test_utils.py
```

This will:
- Generate sample data
- Run all statistical analyses
- Create all visualizations
- Save plots to `/tmp/`

---

## Notes

1. **Data Format**: DataFrames should have columns named 'Bola1', 'Bola2', etc.
2. **Figure Management**: All plot methods return Figure objects for flexibility
3. **Styling**: Visualizer uses seaborn styling by default
4. **Performance**: Entropy analysis with large windows can be slow
5. **Statistical Significance**: Default significance level is 0.05 (95% confidence)

---

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Statistical tests
- **matplotlib**: Plotting framework
- **seaborn**: Enhanced visualizations

---

## Integration with Prediction System

These utilities integrate with the prediction system:

```python
from src.data.loader import MegaSenaLoader
from src.utils.statistics import StatisticalAnalyzer
from src.utils.visualization import Visualizer

# Load real data
loader = MegaSenaLoader()
df = loader.load_from_csv('data/megasena.csv')

# Analyze
analyzer = StatisticalAnalyzer()
viz = Visualizer()

# Perform analysis
frequencies = analyzer.frequency_distribution(df)
hot_cold = analyzer.hot_cold_analysis(frequencies)

# Visualize
fig = viz.plot_frequency_histogram(frequencies)
viz.save_figure(fig, 'results/frequency_analysis.png')
```

---

For more information, see the source code documentation in:
- `/home/user/megasena/src/utils/statistics.py`
- `/home/user/megasena/src/utils/visualization.py`

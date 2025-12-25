# ARIMA Prediction Model for Mega-Sena

## Overview

The `ARIMAPredictor` class implements ARIMA (AutoRegressive Integrated Moving Average) time series models to predict **aggregated features** of Mega-Sena lottery draws. It does NOT predict individual numbers, but rather statistical properties that can be used to filter unlikely combinations.

## Purpose

ARIMA models are used to predict:
- Sum of drawn numbers
- Count of even/odd numbers
- Mean value
- Range and spread statistics
- Other aggregated features

These predictions provide expected ranges that help filter combinations that fall outside typical patterns.

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `statsmodels>=0.14.0` - Core ARIMA implementation
- `pmdarima>=2.0.0` - Automatic parameter selection (optional but recommended)
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization
- `scipy>=1.10.0` - Statistical tests

## Quick Start

```python
from src.models.arima_model import ARIMAPredictor
import pandas as pd

# Load your feature data
features = pd.DataFrame({
    'sum': [150, 145, 160, 155, ...],
    'even_count': [3, 4, 2, 3, ...],
    'mean': [25.0, 24.2, 26.7, 25.8, ...]
})

# Initialize predictor
predictor = ARIMAPredictor()

# Fit model on sum feature
predictor.fit(features['sum'], feature_name='sum')

# Predict next draw with confidence intervals
predictions, lower, upper = predictor.predict(steps=1, feature_name='sum')
print(f"Expected sum: {predictions[0]:.2f} [{lower[0]:.2f}, {upper[0]:.2f}]")

# Get filtering range
sum_min, sum_max = predictor.get_sum_range()
```

## Class Reference

### Initialization

```python
ARIMAPredictor(default_order=(p, d, q))
```

**Parameters:**
- `default_order`: Tuple of (p, d, q) where:
  - `p`: Number of autoregressive terms
  - `d`: Degree of differencing
  - `q`: Number of moving average terms
  - Default: Uses `ARIMA_ORDER` from config.settings

### Key Methods

#### 1. Stationarity Testing

```python
test_stationarity(series, significance_level=0.05)
```

Performs Augmented Dickey-Fuller (ADF) test to check if a time series is stationary.

**Parameters:**
- `series`: Time series data (pd.Series or np.ndarray)
- `significance_level`: Threshold for p-value (default: 0.05)

**Returns:**
Dictionary containing:
- `adf_statistic`: Test statistic value
- `p_value`: Probability value
- `critical_values`: Critical values at 1%, 5%, 10%
- `is_stationary`: Boolean indicating stationarity
- `interpretation`: Human-readable result

**Example:**
```python
result = predictor.test_stationarity(features['sum'])
print(f"Is stationary: {result['is_stationary']}")
print(f"P-value: {result['p_value']:.4f}")
```

#### 2. ACF/PACF Analysis

```python
analyze_acf_pacf(series, lags=40, plot=False)
```

Analyzes autocorrelation to help select ARIMA parameters.

**Parameters:**
- `series`: Time series data
- `lags`: Number of lags to compute
- `plot`: Whether to display plots

**Returns:**
Dictionary with `acf` and `pacf` arrays

**Example:**
```python
acf_pacf = predictor.analyze_acf_pacf(features['sum'], lags=20, plot=True)
```

#### 3. Model Training

```python
fit(series, feature_name='default', order=None, auto_order=False)
```

Train ARIMA model on a time series.

**Parameters:**
- `series`: Time series data
- `feature_name`: Name identifier for the feature
- `order`: ARIMA order (p, d, q). If None, uses default
- `auto_order`: If True, uses auto_arima to select optimal parameters

**Returns:**
Self (for method chaining)

**Example:**
```python
# Manual order
predictor.fit(features['sum'], feature_name='sum', order=(5, 1, 0))

# Automatic order selection
predictor.fit(features['sum'], feature_name='sum', auto_order=True)
```

#### 4. Multiple Features Training

```python
fit_multiple(data, feature_columns, auto_order=False)
```

Train ARIMA models for multiple features at once.

**Parameters:**
- `data`: DataFrame containing all features
- `feature_columns`: List of column names to model
- `auto_order`: Whether to auto-select order for each

**Example:**
```python
predictor.fit_multiple(
    features,
    ['sum', 'even_count', 'mean'],
    auto_order=True
)
```

#### 5. Prediction

```python
predict(steps=1, feature_name='default', return_conf_int=True, alpha=0.05)
```

Predict future values with confidence intervals.

**Parameters:**
- `steps`: Number of time steps to predict
- `feature_name`: Which feature to predict
- `return_conf_int`: Whether to return confidence intervals
- `alpha`: Significance level (0.05 = 95% CI)

**Returns:**
- If `return_conf_int=True`: Tuple of (predictions, lower_bound, upper_bound)
- Otherwise: Array of predictions

**Example:**
```python
# Predict next 5 draws
preds, lower, upper = predictor.predict(steps=5, feature_name='sum')
for i in range(5):
    print(f"Draw {i+1}: {preds[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")
```

#### 6. Predict All Features

```python
predict_all(steps=1, alpha=0.05)
```

Predict all fitted features at once.

**Returns:**
Dictionary mapping feature names to prediction dictionaries:
```python
{
    'sum': {'prediction': [...], 'lower_bound': [...], 'upper_bound': [...]},
    'even_count': {'prediction': [...], 'lower_bound': [...], 'upper_bound': [...]},
    ...
}
```

#### 7. Get Filtering Ranges

```python
get_sum_range(steps=1, alpha=0.05)
get_feature_range(feature_name, steps=1, alpha=0.05)
```

Get predicted ranges for filtering combinations.

**Returns:**
Tuple of (lower_bound, upper_bound)

**Example:**
```python
sum_min, sum_max = predictor.get_sum_range()
even_min, even_max = predictor.get_feature_range('even_count')

# Filter combinations
if sum_min <= combination_sum <= sum_max:
    print("Combination passes sum filter")
```

#### 8. Model Diagnostics

```python
evaluate_residuals(feature_name='default', plot=False)
```

Evaluate model quality through residual analysis.

**Returns:**
Dictionary with residual statistics and residual array

```python
get_model_summary(feature_name='default')
```

Get detailed statistical summary of the fitted model.

#### 9. Model Persistence

```python
save_models(base_path)
load_models(base_path, feature_names)
```

Save and load fitted models.

**Example:**
```python
# Save
predictor.save_models('/home/user/megasena/models_saved')

# Load
new_predictor = ARIMAPredictor()
new_predictor.load_models('/home/user/megasena/models_saved', ['sum', 'even_count'])
```

## Complete Workflow Example

```python
from src.models.arima_model import ARIMAPredictor
import pandas as pd
from config.settings import DATA_PATH, BALL_COLUMNS

# 1. Load and prepare data
df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin-1')
balls = df[BALL_COLUMNS].values

# 2. Create aggregated features
features = pd.DataFrame({
    'sum': balls.sum(axis=1),
    'mean': balls.mean(axis=1),
    'even_count': (balls % 2 == 0).sum(axis=1),
    'odd_count': (balls % 2 != 0).sum(axis=1),
    'range': balls.max(axis=1) - balls.min(axis=1),
    'std': balls.std(axis=1)
})

# 3. Initialize predictor
predictor = ARIMAPredictor()

# 4. Test stationarity
for feature in ['sum', 'even_count', 'mean']:
    result = predictor.test_stationarity(features[feature])
    print(f"{feature}: {result['interpretation']}")

# 5. Train models (with auto parameter selection)
predictor.fit_multiple(
    features,
    ['sum', 'even_count', 'mean', 'range'],
    auto_order=True
)

# 6. Make predictions
predictions = predictor.predict_all(steps=3, alpha=0.05)

print("\nPredictions for next 3 draws:")
for feature, preds in predictions.items():
    print(f"\n{feature}:")
    for i in range(3):
        pred = preds['prediction'][i]
        lower = preds['lower_bound'][i]
        upper = preds['upper_bound'][i]
        print(f"  Draw {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")

# 7. Get filtering ranges for next draw
print("\nFiltering ranges for next draw:")
sum_min, sum_max = predictor.get_sum_range()
even_min, even_max = predictor.get_feature_range('even_count')

print(f"Sum: [{sum_min:.2f}, {sum_max:.2f}]")
print(f"Even count: [{even_min:.2f}, {even_max:.2f}]")

# 8. Evaluate model quality
residual_stats = predictor.evaluate_residuals('sum')
print(f"\nSum model residual mean: {residual_stats['mean']:.4f}")
print(f"Sum model residual std: {residual_stats['std']:.4f}")

# 9. Save models
predictor.save_models('/home/user/megasena/models_saved')
```

## Understanding ARIMA Parameters

### Order Selection

The ARIMA(p, d, q) parameters:

1. **p (AR order)**: Number of past values to use
   - Look at PACF plot: significant lags suggest p value
   - Higher p captures more complex patterns
   - Typical range: 0-5

2. **d (Differencing)**: How many times to difference the data
   - Use ADF test to determine if differencing needed
   - d=0: Stationary series
   - d=1: First difference needed (most common)
   - d=2: Second difference (rare)

3. **q (MA order)**: Number of past errors to use
   - Look at ACF plot: significant lags suggest q value
   - Typical range: 0-5

### Automatic Selection

Use `auto_order=True` to let the algorithm find optimal parameters:

```python
predictor.fit(series, feature_name='sum', auto_order=True)
print(f"Selected order: {predictor.orders['sum']}")
```

### Manual Selection Process

```python
# 1. Test stationarity
stat_result = predictor.test_stationarity(series)

# 2. Analyze ACF/PACF
acf_pacf = predictor.analyze_acf_pacf(series, lags=40, plot=True)

# 3. Based on analysis, select parameters
# Example: PACF cuts off at lag 5, ACF decays -> ARIMA(5, 1, 0)
predictor.fit(series, feature_name='sum', order=(5, 1, 0))
```

## Best Practices

1. **Check Stationarity First**
   ```python
   result = predictor.test_stationarity(series)
   if not result['is_stationary']:
       print("Series needs differencing (d >= 1)")
   ```

2. **Use Auto-ARIMA for Initial Exploration**
   ```python
   predictor.fit(series, auto_order=True)
   # Review suggested order
   print(predictor.orders)
   ```

3. **Validate with Residuals**
   ```python
   residuals = predictor.evaluate_residuals('sum', plot=True)
   # Residuals should be white noise (mean ≈ 0, no patterns)
   ```

4. **Use Appropriate Confidence Intervals**
   ```python
   # 95% CI (alpha=0.05) for general use
   # 99% CI (alpha=0.01) for conservative filtering
   _, lower, upper = predictor.predict(steps=1, alpha=0.01)
   ```

5. **Train on Sufficient Data**
   - Minimum: 50-100 observations
   - Recommended: 200+ observations
   - More data = better parameter estimation

6. **Regular Retraining**
   ```python
   # Retrain as new data arrives
   new_features = compute_features(new_draws)
   predictor.fit(new_features['sum'], feature_name='sum', auto_order=True)
   ```

## Integration with Lottery System

### Filtering Pipeline

```python
def filter_combinations(combinations, predictor):
    """Filter combinations using ARIMA predictions."""

    # Get predicted ranges
    sum_min, sum_max = predictor.get_sum_range(alpha=0.05)
    even_min, even_max = predictor.get_feature_range('even_count', alpha=0.05)
    mean_min, mean_max = predictor.get_feature_range('mean', alpha=0.05)

    filtered = []
    for combo in combinations:
        combo_sum = sum(combo)
        combo_even = sum(1 for x in combo if x % 2 == 0)
        combo_mean = np.mean(combo)

        # Check all criteria
        if (sum_min <= combo_sum <= sum_max and
            even_min <= combo_even <= even_max and
            mean_min <= combo_mean <= mean_max):
            filtered.append(combo)

    return filtered
```

### Ensemble with Other Models

```python
# Combine ARIMA with Prophet and LSTM
arima_pred = ARIMAPredictor()
prophet_pred = ProphetPredictor()
lstm_pred = LSTMPredictor()

# Train all
arima_pred.fit(features['sum'], feature_name='sum')
prophet_pred.fit(features['sum'])
lstm_pred.fit(features['sum'])

# Average predictions
arima_sum, arima_l, arima_u = arima_pred.predict(steps=1, feature_name='sum')
prophet_sum = prophet_pred.predict(steps=1)
lstm_sum = lstm_pred.predict(steps=1)

ensemble_sum = (arima_sum[0] + prophet_sum[0] + lstm_sum[0]) / 3
```

## Troubleshooting

### Issue: "Series too short for ARIMA"
**Solution**: Ensure at least 10 data points, preferably 50+

### Issue: "Convergence failed"
**Solution**:
- Try different order parameters
- Check for outliers in data
- Ensure series has variance (not constant)

### Issue: "Auto ARIMA not available"
**Solution**:
```bash
pip install pmdarima
```

### Issue: Poor predictions
**Solution**:
- Check residuals for patterns
- Try different orders
- Ensure stationarity
- Add more historical data

## Performance Considerations

- **Training Time**: O(n * iterations) where n is series length
- **Auto ARIMA**: Slower (tests multiple orders) but more accurate
- **Memory**: Stores full history for each feature
- **Prediction**: Fast (milliseconds for hundreds of steps)

## References

- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [pmdarima Auto ARIMA](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)
- [Time Series Analysis in Python](https://www.statsmodels.org/stable/tsa.html)

## License

Part of the Mega-Sena Prediction System project.

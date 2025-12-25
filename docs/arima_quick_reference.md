# ARIMA Model - Quick Reference Card

## Import
```python
from src.models.arima_model import ARIMAPredictor
```

## Basic Usage

### 1. Initialize
```python
predictor = ARIMAPredictor()  # Uses default order from config
# OR
predictor = ARIMAPredictor(default_order=(5, 1, 0))
```

### 2. Test Stationarity
```python
result = predictor.test_stationarity(series)
print(f"Is stationary: {result['is_stationary']}")
print(f"P-value: {result['p_value']:.4f}")
```

### 3. Analyze ACF/PACF
```python
predictor.analyze_acf_pacf(series, lags=20, plot=True)
```

### 4. Fit Model

**Single feature:**
```python
# Manual order
predictor.fit(series, feature_name='sum', order=(5, 1, 0))

# Auto order selection
predictor.fit(series, feature_name='sum', auto_order=True)
```

**Multiple features:**
```python
predictor.fit_multiple(
    features_df,
    ['sum', 'even_count', 'mean'],
    auto_order=True
)
```

### 5. Predict

**Single feature:**
```python
# With confidence intervals (95%)
pred, lower, upper = predictor.predict(steps=5, feature_name='sum')

# Without confidence intervals
pred = predictor.predict(steps=5, feature_name='sum', return_conf_int=False)

# Custom confidence level (99%)
pred, lower, upper = predictor.predict(steps=5, feature_name='sum', alpha=0.01)
```

**All features:**
```python
results = predictor.predict_all(steps=3)
# Returns: {'sum': {...}, 'even_count': {...}, ...}
```

### 6. Get Filtering Ranges
```python
sum_min, sum_max = predictor.get_sum_range()
even_min, even_max = predictor.get_feature_range('even_count')
```

### 7. Diagnostics
```python
# Residual analysis
stats = predictor.evaluate_residuals('sum', plot=True)

# Model summary
summary = predictor.get_model_summary('sum')
print(summary)
```

### 8. Save/Load
```python
# Save
predictor.save_models('/path/to/save')

# Load
new_predictor = ARIMAPredictor()
new_predictor.load_models('/path/to/save', ['sum', 'even_count'])
```

## Complete Example

```python
import pandas as pd
from src.models.arima_model import ARIMAPredictor

# Prepare features
features = pd.DataFrame({
    'sum': [150, 145, 160, ...],
    'even_count': [3, 4, 2, ...],
})

# Initialize and train
predictor = ARIMAPredictor()
predictor.fit_multiple(features, ['sum', 'even_count'], auto_order=True)

# Predict
results = predictor.predict_all(steps=1)

# Filter combinations
sum_min, sum_max = predictor.get_sum_range()
valid_combos = [c for c in combos if sum_min <= sum(c) <= sum_max]
```

## Common Patterns

### Check if model is fitted
```python
if predictor.is_fitted:
    pred = predictor.predict(steps=1, feature_name='sum')
```

### Iterate over predictions
```python
for feature, preds in predictor.predict_all(steps=5).items():
    for i, (p, l, u) in enumerate(zip(
        preds['prediction'],
        preds['lower_bound'],
        preds['upper_bound']
    )):
        print(f"{feature} step {i+1}: {p:.2f} [{l:.2f}, {u:.2f}]")
```

### Method chaining
```python
predictor = ARIMAPredictor().fit(series, feature_name='sum')
```

## ARIMA Order Guide

| Pattern | Suggested Order | Notes |
|---------|----------------|-------|
| Strong trend | (p, 1, q) | d=1 for differencing |
| Seasonal pattern | Use seasonal ARIMA | Not implemented yet |
| Random walk | (0, 1, 0) | First difference |
| Stationary | (p, 0, q) | No differencing |
| AR process | (p, d, 0) | Check PACF |
| MA process | (0, d, q) | Check ACF |

## Error Handling

```python
try:
    predictor.fit(series, feature_name='sum')
except ValueError as e:
    print(f"Series too short or invalid: {e}")

if 'sum' not in predictor.models:
    print("Model not fitted")
else:
    pred = predictor.predict(steps=1, feature_name='sum')
```

## Performance Tips

1. Use `auto_order=True` for initial exploration
2. Cache fitted models with `save_models()`
3. Batch predictions with `predict_all()`
4. Use appropriate `alpha` (0.05 = 95% CI is standard)
5. Ensure 50+ observations for reliable fitting

## Dependencies

```bash
pip install statsmodels>=0.14.0 pmdarima>=2.0.0
```

## Configuration

Set defaults in `/config/settings.py`:
```python
ARIMA_ORDER = (5, 1, 0)  # Default (p, d, q)
```

## Available Attributes

```python
predictor.models          # Dict of fitted models
predictor.history         # Dict of historical series
predictor.orders          # Dict of ARIMA orders
predictor.is_fitted       # Boolean
predictor.default_order   # Default (p, d, q)
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Series too short" | Need 10+ points, recommend 50+ |
| "Convergence failed" | Try different order or check for outliers |
| "pmdarima not available" | `pip install pmdarima` |
| Poor predictions | Check residuals, try auto_order, add more data |

## See Also

- Full documentation: `/docs/arima_model_guide.md`
- Examples: `/examples/arima_usage_example.py`
- Tests: `/tests/test_arima_model.py`

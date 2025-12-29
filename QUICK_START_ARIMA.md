# ARIMA Model - Quick Start Guide

## Installation

```bash
cd /home/user/megasena
pip install -r requirements.txt
```

## Basic Example (Copy & Paste Ready)

```python
from src.models.arima_model import ARIMAPredictor
import pandas as pd
import numpy as np

# Example: Load your lottery data
# (Replace with actual data loading)
features = pd.DataFrame({
    'sum': [150, 145, 160, 155, 148, 162, 157, 151],
    'even_count': [3, 4, 2, 3, 4, 3, 2, 3],
    'mean': [25.0, 24.2, 26.7, 25.8, 24.7, 27.0, 26.2, 25.2]
})

# Initialize predictor
predictor = ARIMAPredictor()

# Train on sum feature
predictor.fit(features['sum'], feature_name='sum', auto_order=True)

# Predict next draw
predictions, lower, upper = predictor.predict(steps=1, feature_name='sum')
print(f"Expected sum: {predictions[0]:.2f} [{lower[0]:.2f}, {upper[0]:.2f}]")

# Get filtering range
sum_min, sum_max = predictor.get_sum_range()
print(f"Filter combinations with sum between {sum_min:.2f} and {sum_max:.2f}")
```

## With Real Mega-Sena Data

```python
from src.models.arima_model import ARIMAPredictor
import pandas as pd
from config.settings import DATA_PATH, BALL_COLUMNS

# Load data
df = pd.read_csv(DATA_PATH, sep=';', skiprows=1, encoding='latin-1')
balls = df[BALL_COLUMNS].values

# Compute features
features = pd.DataFrame({
    'sum': balls.sum(axis=1),
    'even_count': (balls % 2 == 0).sum(axis=1),
    'mean': balls.mean(axis=1)
})

# Train models
predictor = ARIMAPredictor()
predictor.fit_multiple(features, ['sum', 'even_count', 'mean'], auto_order=True)

# Predict next draw
results = predictor.predict_all(steps=1)

# Display predictions
for feature, preds in results.items():
    pred = preds['prediction'][0]
    lower = preds['lower_bound'][0]
    upper = preds['upper_bound'][0]
    print(f"{feature:15s}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
```

## Next Steps

1. Read full documentation: `/home/user/megasena/docs/arima_model_guide.md`
2. See more examples: `/home/user/megasena/examples/arima_usage_example.py`
3. Run tests: `pytest /home/user/megasena/tests/test_arima_model.py -v`
4. Check quick reference: `/home/user/megasena/docs/arima_quick_reference.md`

## Common Tasks

### Test if data is stationary
```python
result = predictor.test_stationarity(features['sum'])
print(result['interpretation'])
```

### Visualize ACF/PACF
```python
predictor.analyze_acf_pacf(features['sum'], lags=20, plot=True)
```

### Save trained models
```python
predictor.save_models('/home/user/megasena/models_saved')
```

### Load saved models
```python
new_predictor = ARIMAPredictor()
new_predictor.load_models('/home/user/megasena/models_saved', ['sum', 'even_count'])
```

## File Locations

- Main code: `/home/user/megasena/src/models/arima_model.py`
- Examples: `/home/user/megasena/examples/arima_usage_example.py`
- Tests: `/home/user/megasena/tests/test_arima_model.py`
- Full guide: `/home/user/megasena/docs/arima_model_guide.md`
- Quick ref: `/home/user/megasena/docs/arima_quick_reference.md`

# Prophet Model Guide for Mega-Sena Lottery Analysis

## Overview

The `ProphetPredictor` class uses Facebook's Prophet library to analyze long-term trends and potential seasonality in Mega-Sena lottery data. Prophet is particularly useful for time series analysis with the following characteristics:

- **Trend Analysis**: Detect long-term increasing or decreasing patterns
- **Seasonality Detection**: Identify yearly, weekly, or daily patterns
- **Uncertainty Intervals**: Provide confidence bounds for predictions
- **Component Decomposition**: Separate trend, seasonality, and residual effects

## Installation

First, install the required Prophet library:

```bash
pip install prophet
```

Note: Prophet requires `pystan` which may have additional system dependencies. On Linux:

```bash
pip install pystan
pip install prophet
```

## Basic Usage

### 1. Analyzing Sum of Drawn Numbers

```python
import pandas as pd
from src.models.prophet_model import ProphetPredictor
from config.settings import DATA_PATH, BALL_COLUMNS

# Load data
df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')

# Calculate sum of numbers for each draw
df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)

# Initialize predictor
predictor = ProphetPredictor(
    yearly_seasonality='auto',  # Auto-detect yearly patterns
    weekly_seasonality=False,    # Disable weekly (lottery draws aren't weekly)
    changepoint_prior_scale=0.1  # Flexibility of trend changes
)

# Train model
predictor.fit(df, target_column='sum_numbers')

# Forecast next 10 draws
forecast = predictor.predict(periods=10, freq='W')
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
```

### 2. Analyzing Individual Number Trends

```python
# Analyze if number 7 is trending up or down
analysis = predictor.analyze_number_trend(df, number=7, window_size=20)

print(f"Number 7 Trend: {analysis['trend_direction']}")
print(f"Trend Strength: {analysis['trend_strength']:.6f}")
print(f"Current Frequency: {analysis['current_frequency']:.4f}")
print(f"Predicted Frequency: {analysis['predicted_frequency']:.4f}")
```

### 3. Finding Top Trending Numbers

```python
# Analyze all 60 numbers and find top 10 trending
top_trending = predictor.analyze_all_numbers_trends(
    df,
    window_size=15,
    top_n=10
)

print(top_trending)
```

### 4. Multi-Target Analysis

```python
from src.models.prophet_model import MultiTargetProphetPredictor

# Calculate multiple features
df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)
df['mean_numbers'] = df[BALL_COLUMNS].mean(axis=1)
df['std_numbers'] = df[BALL_COLUMNS].std(axis=1)
df['range_numbers'] = df[BALL_COLUMNS].max(axis=1) - df[BALL_COLUMNS].min(axis=1)

# Initialize multi-target predictor
multi_predictor = MultiTargetProphetPredictor(
    yearly_seasonality='auto',
    changepoint_prior_scale=0.1
)

# Train on multiple targets
targets = ['sum_numbers', 'mean_numbers', 'std_numbers', 'range_numbers']
multi_predictor.fit_multiple_targets(df, targets)

# Get predictions for all targets
predictions = multi_predictor.predict_all(periods=5)

# Get trends for all targets
trends = multi_predictor.get_all_trends()
```

## Class: ProphetPredictor

### Initialization Parameters

```python
ProphetPredictor(
    yearly_seasonality='auto',      # 'auto', True, False, or int (Fourier order)
    weekly_seasonality=False,       # 'auto', True, False, or int
    daily_seasonality=False,        # True or False
    seasonality_mode='additive',    # 'additive' or 'multiplicative'
    changepoint_prior_scale=0.05,   # Flexibility of trend (higher = more flexible)
    seasonality_prior_scale=10.0,   # Strength of seasonality (higher = stronger)
    holidays=None,                  # Optional DataFrame with custom holidays
    growth='linear',                # 'linear' or 'logistic'
    interval_width=0.95            # Width of uncertainty intervals (0-1)
)
```

### Key Methods

#### `fit(df, target_column, date_column='Data do Sorteio', date_format='%d/%m/%Y', verbose=False)`

Train the Prophet model on provided data.

**Parameters:**
- `df`: DataFrame with lottery data
- `target_column`: Column to predict
- `date_column`: Name of date column
- `date_format`: Format of dates
- `verbose`: Print training progress

**Returns:** self (for chaining)

#### `predict(periods=10, freq='W')`

Forecast future values.

**Parameters:**
- `periods`: Number of periods to forecast
- `freq`: Frequency ('D'=daily, 'W'=weekly, 'M'=monthly)

**Returns:** DataFrame with predictions and confidence intervals

#### `get_trend_analysis()`

Extract trend component.

**Returns:** DataFrame with date and trend values

#### `get_seasonality_analysis()`

Extract seasonality components.

**Returns:** Dictionary with 'yearly', 'weekly', 'daily' seasonality (if present)

#### `analyze_number_trend(df, number, window_size=10)`

Analyze if a specific number is trending up or down.

**Parameters:**
- `df`: DataFrame with lottery data
- `number`: Number to analyze (1-60)
- `window_size`: Number of recent draws to analyze

**Returns:** Dictionary with:
- `trend_direction`: 'up', 'down', or 'stable'
- `trend_strength`: Absolute value of trend slope
- `forecast`: Future predictions
- `current_frequency`: Recent frequency
- `predicted_frequency`: Predicted future frequency

#### `analyze_all_numbers_trends(df, window_size=10, top_n=10)`

Analyze trends for all 60 numbers.

**Parameters:**
- `df`: DataFrame with lottery data
- `window_size`: Analysis window
- `top_n`: Number of top results to return

**Returns:** DataFrame with top trending numbers

#### `evaluate_prediction_quality()`

Evaluate model performance on historical data.

**Returns:** Dictionary with metrics:
- `mse`: Mean Squared Error
- `mae`: Mean Absolute Error
- `mape`: Mean Absolute Percentage Error
- `coverage`: % of actuals within prediction intervals

#### `plot_components_data()`

Get data for plotting trend and seasonality components.

**Returns:** Dictionary with DataFrames for visualization

## Class: MultiTargetProphetPredictor

Manages multiple `ProphetPredictor` instances for analyzing different targets simultaneously.

### Key Methods

#### `fit_multiple_targets(df, target_columns, verbose=False)`

Train models for multiple target columns.

#### `predict_all(periods=10, freq='W')`

Get predictions for all trained models.

#### `get_all_trends()`

Get trend analysis for all models.

#### `get_predictor(target)`

Get a specific predictor by target name.

## Understanding Prophet Parameters

### Seasonality Configuration

- **yearly_seasonality**: Use 'auto' for Prophet to decide, or set to False if you believe there's no yearly pattern in lottery numbers
- **weekly_seasonality**: Generally False for lottery data (draws aren't on specific days)
- **daily_seasonality**: Generally False for lottery data

### Trend Flexibility

- **changepoint_prior_scale**: Controls how flexible the trend can be
  - Lower values (0.001-0.05): More conservative, smoother trends
  - Higher values (0.1-0.5): More flexible, can capture rapid changes
  - Default: 0.05

### Seasonality Strength

- **seasonality_prior_scale**: Controls strength of seasonality
  - Lower values: Weaker seasonality effect
  - Higher values: Stronger seasonality effect
  - Default: 10.0

### Growth Mode

- **linear**: Assumes constant rate of change (typical for most cases)
- **logistic**: For data with a cap or saturation point

## Important Considerations for Lottery Data

### 1. No True Seasonality

Lottery draws are typically random events with no true seasonality. However, Prophet can still detect:
- Long-term trends in number frequencies
- Patterns in aggregate statistics (sums, means)
- Changepoints where drawing behavior shifts

### 2. Interpretation

Prophet's "predictions" for lottery data should be interpreted as:
- **Trend continuation**: What would happen if current patterns continue
- **Statistical patterns**: Historical tendencies, not guaranteed outcomes
- **Probability shifts**: Slight changes in likelihood, not certainties

### 3. Best Practices

```python
# Conservative settings for lottery data
predictor = ProphetPredictor(
    yearly_seasonality=False,        # No true yearly pattern
    weekly_seasonality=False,        # No weekly pattern
    changepoint_prior_scale=0.05,    # Conservative trend flexibility
    seasonality_prior_scale=5.0,     # Lower seasonality strength
)
```

## Example Workflow

```python
import pandas as pd
from src.models.prophet_model import ProphetPredictor
from config.settings import DATA_PATH, BALL_COLUMNS

# 1. Load and prepare data
df = pd.read_csv(DATA_PATH, sep=';', encoding='latin1')
df['sum_numbers'] = df[BALL_COLUMNS].sum(axis=1)

# 2. Initialize predictor
predictor = ProphetPredictor(
    yearly_seasonality=False,
    weekly_seasonality=False,
    changepoint_prior_scale=0.1
)

# 3. Train model
predictor.fit(df, 'sum_numbers', verbose=True)

# 4. Evaluate model
metrics = predictor.evaluate_prediction_quality()
print(f"MAE: {metrics['mae']:.2f}")
print(f"Coverage: {metrics['coverage']:.1f}%")

# 5. Get trend analysis
trend = predictor.get_trend_analysis()
print(trend.tail(10))

# 6. Make predictions
forecast = predictor.predict(periods=10)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# 7. Analyze specific numbers
for number in [7, 13, 21, 42]:
    analysis = predictor.analyze_number_trend(df, number, window_size=20)
    print(f"Number {number}: {analysis['trend_direction']} "
          f"(strength: {analysis['trend_strength']:.6f})")
```

## Visualization Tips

The model provides data for various visualizations:

```python
# Get plotting data
plot_data = predictor.plot_components_data()

# Available components:
# - plot_data['forecast']: Full forecast with confidence intervals
# - plot_data['trend']: Trend component over time
# - plot_data['yearly']: Yearly seasonality (if enabled)

# Use with matplotlib or plotly to create visualizations
import matplotlib.pyplot as plt

trend_df = plot_data['trend']
plt.figure(figsize=(12, 6))
plt.plot(trend_df['date'], trend_df['trend'])
plt.title('Trend Analysis')
plt.xlabel('Date')
plt.ylabel('Trend Value')
plt.show()
```

## Error Handling

The Prophet model includes comprehensive error handling:

```python
from src.models.prophet_model import ProphetPredictor

try:
    predictor = ProphetPredictor()
    predictor.fit(df, 'invalid_column')
except ValueError as e:
    print(f"Error: {e}")  # Will catch invalid column names

try:
    predictor.predict(periods=10)
except ValueError as e:
    print(f"Error: {e}")  # Will catch if model not trained
```

## Performance Considerations

- **Training Time**: Prophet can be slow for large datasets (1000+ rows may take minutes)
- **Memory**: Each model instance requires storing the full dataset
- **Multi-Target**: For analyzing many targets, use `MultiTargetProphetPredictor` for better organization

## References

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Prophet Paper](https://peerj.com/preprints/3190/)
- [Time Series Forecasting](https://otexts.com/fpp3/)

## Next Steps

1. Run the examples in `/home/user/megasena/examples/prophet_example.py`
2. Experiment with different parameter values
3. Combine Prophet insights with other models (ARIMA, LSTM)
4. Create visualizations of trends and predictions

# Feature Engineering Module

## Overview

The `FeatureEngineer` class in `/home/user/megasena/src/features/engineer.py` provides comprehensive feature engineering capabilities for Mega-Sena lottery prediction.

## Features Created

### 1. Aggregated Statistics (6 features)
- `sum`: Sum of all 6 numbers in the draw
- `mean`: Mean of the 6 numbers
- `std`: Standard deviation of the 6 numbers
- `min_num`: Minimum number in the draw
- `max_num`: Maximum number in the draw
- `range`: Difference between max and min

### 2. Parity Features (3 features)
- `even_count`: Count of even numbers (0-6)
- `odd_count`: Count of odd numbers (0-6)
- `parity_ratio`: Ratio of even numbers (even_count / 6)

### 3. Consecutiveness (2 features)
- `consecutive_pairs`: Number of consecutive pairs (e.g., 23-24)
- `has_consecutive`: Boolean indicating if any consecutive pair exists

### 4. Quadrant Analysis (4 features)
Numbers 1-60 are divided into 4 equal quadrants:
- `q1_count`: Numbers in Q1 (1-15)
- `q2_count`: Numbers in Q2 (16-30)
- `q3_count`: Numbers in Q3 (31-45)
- `q4_count`: Numbers in Q4 (46-60)

### 5. Gap/Delay Analysis (60 features)
For each number 1-60:
- `gap_N`: Number of draws since number N last appeared
  - Example: `gap_7` = 5 means number 7 appeared 5 draws ago

### 6. Historical Frequency (180+ features)
Rolling frequency features for configurable windows (default: 10, 50, 100 draws):
- `freq_N_window_W`: How many times number N appeared in last W draws
  - Example: `freq_7_window_50` = count of number 7 in last 50 draws
- `hot_numbers_W`: Count of "hot" numbers (above average frequency)
- `cold_numbers_W`: Count of "cold" numbers (below average frequency)

## Total Features

When using all defaults (windows: 10, 50, 100):
- **261 total engineered features** per draw
- All features use numpy vectorization for efficiency
- No NaN values generated

## Usage

### Basic Usage

```python
import pandas as pd
from src.features.engineer import FeatureEngineer

# Load data
df = pd.read_csv('Mega-Sena.csv', sep=';', skiprows=1, encoding='latin1')

# Initialize and engineer features
fe = FeatureEngineer(df)
enhanced_df = fe.engineer_all_features()

# Now enhanced_df contains all original columns plus 261 new features
```

### Custom Window Sizes

```python
# Use custom frequency windows
fe = FeatureEngineer(df)
fe.enhanced_df = fe.df.copy()

# Add only specific feature groups
fe._add_aggregated_statistics()
fe._add_parity_features()
fe._add_frequency_features(windows=[20, 100, 200])
```

### Analyze Hot/Cold Numbers

```python
fe = FeatureEngineer(df)

# Get hot and cold numbers for last 50 draws
hot, cold = fe.get_hot_cold_numbers(window=50)
print(f"Hot numbers: {hot[:10]}")
print(f"Cold numbers: {cold[:10]}")
```

### Gap Analysis

```python
fe = FeatureEngineer(df)

# Get current gap for all 60 numbers
gaps = fe.get_current_gaps()

# Find most overdue numbers
import numpy as np
overdue_indices = gaps.argsort()[::-1][:10]
overdue_numbers = overdue_indices + 1
print(f"Most overdue: {overdue_numbers}")
```

### Calculate Features for New Draw

```python
fe = FeatureEngineer(df)

# Calculate features for a hypothetical draw
draw = [5, 12, 23, 34, 45, 56]
features = fe.calculate_draw_features(draw)

print(f"Sum: {features['sum']}")
print(f"Even/Odd: {features['even_count']}/{features['odd_count']}")
print(f"Consecutive pairs: {features['consecutive_pairs']}")
```

### Prepare for Machine Learning

```python
# Engineer all features
fe = FeatureEngineer(df)
enhanced_df = fe.engineer_all_features()

# Select numeric features for ML
from config.settings import BALL_COLUMNS

numeric_cols = enhanced_df.select_dtypes(include=['int64', 'float64']).columns
feature_cols = [col for col in numeric_cols
                if col not in BALL_COLUMNS + ['Concurso']]

# Create feature matrix and target
X = enhanced_df[feature_cols].values  # Shape: (n_draws, 263)
y = enhanced_df[BALL_COLUMNS].values  # Shape: (n_draws, 6)
```

## Performance

- Uses numpy vectorization throughout for efficiency
- Can process 2,954 historical draws in seconds
- Minimal memory overhead with proper data types
- Warning: DataFrame fragmentation warnings are expected but don't affect correctness

## Testing

Run the test suite:
```bash
python test_feature_engineer.py
```

See usage examples:
```bash
python examples_feature_engineer.py
```

## API Reference

### Main Methods

#### `engineer_all_features() -> pd.DataFrame`
Engineers all feature categories and returns the enhanced DataFrame.

#### `get_current_gaps(reference_idx: Optional[int] = None) -> np.ndarray`
Returns array of gaps for all 60 numbers from the reference draw.

#### `get_rolling_frequency(number: int, window: int, reference_idx: Optional[int] = None) -> int`
Gets rolling frequency for a specific number.

#### `get_hot_cold_numbers(window: int = 50, reference_idx: Optional[int] = None) -> Tuple[List[int], List[int]]`
Identifies hot and cold numbers based on rolling frequency.

#### `calculate_draw_features(draw_numbers: List[int]) -> Dict[str, float]`
Calculates features for a single draw (useful for prediction).

### Internal Methods

- `_add_aggregated_statistics()`
- `_add_parity_features()`
- `_add_consecutiveness_features()`
- `_add_quadrant_features()`
- `_add_gap_features()`
- `_add_frequency_features(windows: Optional[List[int]] = None)`

## Notes

- All array indexing uses `.astype(int)` for compatibility with numpy int64
- Gap calculations look backward in time from each draw
- Frequency windows exclude the current draw
- Hot/cold threshold = (window × 6) / 60 = expected frequency

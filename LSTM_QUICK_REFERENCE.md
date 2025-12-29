# LSTM Model - Quick Reference

## Installation
```bash
pip install tensorflow>=2.13.0 numpy pandas
```

## Basic Usage (5 Steps)

### 1. Import
```python
from src.models.lstm_model import LSTMPredictor
import numpy as np
```

### 2. Prepare Data
```python
# Convert draws to binary encoding (n_draws x 60)
binary_data = np.zeros((n_draws, 60))
for i, draw in enumerate(historical_draws):
    for number in draw:
        binary_data[i, number - 1] = 1
```

### 3. Create Predictor & Sequences
```python
predictor = LSTMPredictor(lookback=10)
X, y = predictor.create_sequences(binary_data)
```

### 4. Train
```python
predictor.fit(X, y, epochs=100, validation_split=0.2)
```

### 5. Predict
```python
predictions = predictor.predict_top_k(X[-1], k=6)
for number, prob in predictions:
    print(f"{number}: {prob:.2%}")
```

## Common Operations

### Save Model
```python
predictor.save_model('models_saved/my_lstm.keras')
```

### Load Model
```python
predictor = LSTMPredictor()
predictor.load_model('models_saved/my_lstm.keras')
```

### Get All Probabilities
```python
probs = predictor.predict_proba(sequence)  # Returns array of 60 probabilities
```

### Evaluate on Test Set
```python
metrics = predictor.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['binary_accuracy']:.4f}")
```

## Configuration (config/settings.py)
```python
LSTM_LOOKBACK = 10      # Past draws to consider
LSTM_EPOCHS = 100       # Max training epochs
LSTM_BATCH_SIZE = 32    # Training batch size
LSTM_UNITS = [100, 50]  # LSTM layer sizes
DROPOUT_RATE = 0.2      # Regularization
```

## Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `build_model(shape)` | Build LSTM architecture | keras.Model |
| `create_sequences(data)` | Make training sequences | X, y arrays |
| `fit(X, y, ...)` | Train the model | History |
| `predict_proba(X)` | Get probabilities (60) | np.array |
| `predict_top_k(X, k)` | Get top K numbers | List[Tuple] |
| `save_model(path)` | Save to disk | None |
| `load_model(path)` | Load from disk | None |
| `evaluate(X, y)` | Test performance | dict |

## Architecture
```
Input (lookback x 60)
    ↓
LSTM(100) + Dropout(0.2)
    ↓
LSTM(50) + Dropout(0.2)
    ↓
Dense(60, sigmoid)
```

## Files
- **Main**: `/home/user/megasena/src/models/lstm_model.py`
- **Example**: `/home/user/megasena/examples/lstm_example.py`
- **Tests**: `/home/user/megasena/tests/test_lstm_model.py`
- **Docs**: `/home/user/megasena/docs/LSTM_MODEL.md`

## Run Example
```bash
python examples/lstm_example.py
```

## Run Tests
```bash
pytest tests/test_lstm_model.py -v
```

## GPU Support
- Auto-detected and configured
- Falls back to CPU if unavailable
- Enable memory growth automatically

## Tips
- Start with 20-50 epochs for testing
- Use early stopping (enabled by default)
- Larger lookback = more context but more data needed
- Monitor validation loss - should decrease steadily
- Save best model during training (default)

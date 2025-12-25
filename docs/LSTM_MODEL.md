# LSTM Deep Learning Model for Mega-Sena Prediction

## Overview

The LSTM (Long Short-Term Memory) model is a deep learning approach for predicting Mega-Sena lottery numbers based on historical draw patterns. Unlike traditional statistical methods, LSTM networks can learn complex temporal dependencies in sequential data.

## Architecture

### Network Structure

```
Input Layer (lookback × 60)
    ↓
LSTM Layer 1 (100 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (50 units, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Output Layer (60 units, sigmoid activation)
```

### Key Features

- **Input Shape**: `(lookback, 60)` - Binary encoded sequences of past draws
- **LSTM Layers**: Two stacked LSTM layers (100 → 50 units) for capturing temporal patterns
- **Dropout Regularization**: 0.2 dropout rate to prevent overfitting
- **Output**: 60 neurons with sigmoid activation, producing probability for each number
- **Loss Function**: Binary crossentropy (multi-label classification)
- **Optimizer**: Adam with learning rate 0.001

## Data Preparation

### Binary Encoding

Lottery draws are converted to binary vectors:
- Each draw → vector of length 60
- Value = 1 if number was drawn, 0 otherwise
- Example: Draw [5, 12, 23, 34, 45, 56] → Vector with 1s at positions 4, 11, 22, 33, 44, 55

### Sequence Creation

Historical draws are transformed into training sequences using a sliding window:

```
Lookback = 10

Draw 1  ]
Draw 2  ]
...     ] → Sequence 1 → Target: Draw 11
Draw 10 ]

Draw 2  ]
Draw 3  ]
...     ] → Sequence 2 → Target: Draw 12
Draw 11 ]
```

This creates overlapping sequences that help the model learn temporal patterns.

## Usage

### Basic Usage

```python
from src.models.lstm_model import LSTMPredictor
import numpy as np

# Initialize predictor
predictor = LSTMPredictor(lookback=10)

# Prepare data (binary encoded draws)
# Shape: (n_draws, 60)
binary_data = prepare_binary_data(historical_draws)

# Create sequences
X, y = predictor.create_sequences(binary_data, lookback=10)

# Build model
predictor.build_model(input_shape=(X.shape[1], X.shape[2]))

# Train model
history = predictor.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    patience=10
)

# Make predictions
latest_sequence = X[-1]
top_6 = predictor.predict_top_k(latest_sequence, k=6)

print("Predicted numbers:")
for number, probability in top_6:
    print(f"Number {number}: {probability:.4f}")
```

### Advanced Usage

#### Custom Training Configuration

```python
# Train with custom settings
history = predictor.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_split=0.15,
    verbose=1,
    patience=15,
    save_best=True,
    model_path='models_saved/lstm_custom.keras'
)
```

#### Getting Probability Distributions

```python
# Get probabilities for all 60 numbers
probabilities = predictor.predict_proba(sequence)

# Find numbers with probability > 0.5
likely_numbers = [
    (i + 1, prob)
    for i, prob in enumerate(probabilities)
    if prob > 0.5
]
```

#### Model Persistence

```python
# Save trained model
predictor.save_model('models_saved/lstm_best.keras')

# Load model later
new_predictor = LSTMPredictor(lookback=10)
new_predictor.load_model('models_saved/lstm_best.keras')

# Continue making predictions
predictions = new_predictor.predict_top_k(sequence, k=10)
```

#### Evaluation

```python
# Evaluate on test set
metrics = predictor.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {metrics['loss']:.4f}")
print(f"Test Accuracy: {metrics['binary_accuracy']:.4f}")
print(f"Test Precision: {metrics['precision']:.4f}")
print(f"Test Recall: {metrics['recall']:.4f}")
```

## Configuration

All model hyperparameters can be configured in `config/settings.py`:

```python
# Model parameters
LSTM_LOOKBACK = 10          # Number of past draws to consider
LSTM_EPOCHS = 100           # Maximum training epochs
LSTM_BATCH_SIZE = 32        # Batch size for training
LSTM_UNITS = [100, 50]      # Units in each LSTM layer
DROPOUT_RATE = 0.2          # Dropout rate for regularization
```

## Training Callbacks

### Early Stopping

Automatically stops training when validation loss stops improving:
- Monitors: `val_loss`
- Patience: 10 epochs (configurable)
- Restores best weights automatically

### Model Checkpoint

Saves the best model during training:
- Monitors: `val_loss`
- Saves only when validation loss improves
- Default path: `models_saved/lstm_best.keras`

## GPU Support

The model automatically detects and uses GPU if available:

```python
# GPU configuration is handled automatically
predictor = LSTMPredictor()  # Will use GPU if available

# Check GPU availability
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

GPU memory growth is enabled to prevent TensorFlow from allocating all GPU memory at once.

## Performance Considerations

### Memory Usage

- **Lookback**: Larger lookback = more memory required
- **Batch Size**: Larger batches = more memory but faster training
- **GPU Memory**: Enable memory growth to share GPU with other processes

### Training Time

Typical training times (on GPU):
- 100 epochs, 1000 samples: ~2-5 minutes
- 200 epochs, 2000 samples: ~5-10 minutes
- 500 epochs, 5000 samples: ~20-30 minutes

On CPU, training time can be 5-10x longer.

### Optimization Tips

1. **Start with fewer epochs**: Use 20-50 epochs initially to verify setup
2. **Use early stopping**: Prevents wasting time on overfitting
3. **Monitor validation loss**: Should decrease and stabilize
4. **Adjust batch size**: Larger batches (64, 128) can speed up training on GPU
5. **Use model checkpoint**: Saves best model automatically

## Interpretation of Results

### Probability Outputs

The model outputs a probability for each number (1-60):
- **High probability (>0.6)**: Number is likely to appear
- **Medium probability (0.4-0.6)**: Uncertain
- **Low probability (<0.4)**: Number is unlikely to appear

### Selecting Final Predictions

Common strategies:
1. **Top-K**: Select K numbers with highest probabilities
2. **Threshold**: Select all numbers above a probability threshold
3. **Weighted Random**: Sample based on probability distribution

Example:
```python
# Strategy 1: Top-K
top_6 = predictor.predict_top_k(sequence, k=6)

# Strategy 2: Threshold
probabilities = predictor.predict_proba(sequence)
candidates = [
    (i + 1, prob)
    for i, prob in enumerate(probabilities)
    if prob > 0.5
]

# Strategy 3: Weighted Random
import random
numbers = list(range(1, 61))
weights = predictor.predict_proba(sequence)
selected = random.choices(numbers, weights=weights, k=6)
```

## Common Issues and Solutions

### Issue: Model underfitting (high training loss)

**Solutions:**
- Increase number of epochs
- Reduce dropout rate
- Add more LSTM layers
- Increase units per layer

### Issue: Model overfitting (validation loss increases)

**Solutions:**
- Enable early stopping (default)
- Increase dropout rate
- Reduce model complexity
- Add more training data

### Issue: Slow training

**Solutions:**
- Increase batch size
- Use GPU if available
- Reduce number of epochs
- Reduce lookback window

### Issue: Poor predictions

**Solutions:**
- Increase lookback window
- Train on more data
- Experiment with different architectures
- Combine with other models (ensemble)

## API Reference

### LSTMPredictor Class

#### `__init__(lookback: int = 10)`
Initialize the predictor with specified lookback window.

#### `build_model(input_shape: Tuple[int, int]) -> keras.Model`
Build and compile the LSTM model.

#### `create_sequences(data: np.ndarray, lookback: int = None) -> Tuple[np.ndarray, np.ndarray]`
Create training sequences from binary encoded data.

#### `fit(X, y, epochs=100, batch_size=32, validation_split=0.2, ...) -> History`
Train the model with early stopping and checkpointing.

#### `predict_proba(X: np.ndarray) -> np.ndarray`
Get probability distribution over all 60 numbers.

#### `predict_top_k(X: np.ndarray, k: int = 6) -> List[Tuple[int, float]]`
Get top K most likely numbers with their probabilities.

#### `save_model(path: Union[str, Path]) -> None`
Save the trained model to disk.

#### `load_model(path: Union[str, Path]) -> None`
Load a trained model from disk.

#### `evaluate(X, y, verbose=1) -> dict`
Evaluate the model on test data.

## Future Improvements

Potential enhancements:
1. **Attention mechanism**: Add attention layers to focus on important past draws
2. **Multi-head output**: Separate predictions for different number ranges
3. **Feature engineering**: Include additional features (day of week, month, etc.)
4. **Bidirectional LSTM**: Process sequences in both directions
5. **Ensemble methods**: Combine multiple LSTM models
6. **Hyperparameter tuning**: Automated search for optimal parameters

## References

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Time Series Prediction with LSTM](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

## License

This model is part of the Mega-Sena prediction project.
See LICENSE file for details.

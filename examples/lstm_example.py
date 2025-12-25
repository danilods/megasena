"""
Example usage of the LSTMPredictor for Mega-Sena lottery prediction.

This script demonstrates how to:
1. Load and prepare historical lottery data
2. Create sequences for LSTM training
3. Train the LSTM model
4. Make predictions
5. Save and load models
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import LSTMPredictor
from config.settings import TOTAL_NUMBERS, NUMBERS_PER_DRAW


def create_binary_encoded_draws(draws: list) -> np.ndarray:
    """
    Convert lottery draws to binary encoding.

    Args:
        draws (list): List of draws, where each draw is a list of 6 numbers.

    Returns:
        np.ndarray: Binary encoded array of shape (n_draws, 60).
    """
    encoded = np.zeros((len(draws), TOTAL_NUMBERS))

    for i, draw in enumerate(draws):
        for number in draw:
            encoded[i, number - 1] = 1  # Convert to 0-based indexing

    return encoded


def main():
    """Main execution function."""

    # Example: Simulated historical draws (replace with actual data)
    print("=" * 70)
    print("LSTM Predictor for Mega-Sena - Example Usage")
    print("=" * 70)

    # Simulate some historical draws (in practice, load from CSV)
    np.random.seed(42)
    num_historical_draws = 100

    print(f"\n1. Generating {num_historical_draws} simulated draws...")
    historical_draws = []
    for _ in range(num_historical_draws):
        draw = sorted(np.random.choice(range(1, 61), size=6, replace=False))
        historical_draws.append(draw)

    # Convert to binary encoding
    print("\n2. Converting draws to binary encoding...")
    binary_data = create_binary_encoded_draws(historical_draws)
    print(f"   Binary data shape: {binary_data.shape}")

    # Initialize the predictor
    print("\n3. Initializing LSTM Predictor...")
    predictor = LSTMPredictor(lookback=10)
    print(f"   {predictor}")

    # Create sequences
    print("\n4. Creating training sequences...")
    X, y = predictor.create_sequences(binary_data, lookback=10)
    print(f"   Sequences created: {len(X)} samples")

    # Split into train and test
    print("\n5. Splitting data into train/test sets...")
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Build the model
    print("\n6. Building LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    predictor.build_model(input_shape)

    # Train the model (with reduced epochs for demo)
    print("\n7. Training model (demo with 5 epochs)...")
    history = predictor.fit(
        X_train, y_train,
        epochs=5,  # Use more epochs in production
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        patience=3
    )

    # Evaluate on test set
    print("\n8. Evaluating model on test set...")
    metrics = predictor.evaluate(X_test, y_test)

    # Make predictions
    print("\n9. Making predictions...")

    # Get probabilities for all numbers
    test_sequence = X_test[0]  # Use first test sequence
    probabilities = predictor.predict_proba(test_sequence)
    print(f"   Probability distribution shape: {probabilities.shape}")
    print(f"   Min probability: {probabilities.min():.4f}")
    print(f"   Max probability: {probabilities.max():.4f}")

    # Get top 6 predictions
    print("\n10. Top 6 predicted numbers:")
    top_6 = predictor.predict_top_k(test_sequence, k=6)
    for number, prob in top_6:
        print(f"    Number {number:2d}: {prob:.4f} ({prob*100:.2f}%)")

    # Get top 10 predictions
    print("\n11. Top 10 predicted numbers:")
    top_10 = predictor.predict_top_k(test_sequence, k=10)
    for number, prob in top_10:
        print(f"    Number {number:2d}: {prob:.4f} ({prob*100:.2f}%)")

    # Save the model
    print("\n12. Saving model...")
    model_path = Path(__file__).parent.parent / "models_saved" / "lstm_example.keras"
    predictor.save_model(model_path)

    # Load the model
    print("\n13. Loading saved model...")
    new_predictor = LSTMPredictor(lookback=10)
    new_predictor.load_model(model_path)

    # Verify loaded model works
    print("\n14. Verifying loaded model...")
    new_predictions = new_predictor.predict_top_k(test_sequence, k=6)
    print("    Top 6 from loaded model:")
    for number, prob in new_predictions:
        print(f"    Number {number:2d}: {prob:.4f}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

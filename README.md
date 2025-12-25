# Mega-Sena Prediction System

A comprehensive Python system for analyzing and predicting Mega-Sena lottery results using statistical methods (ARIMA), Facebook Prophet, and Deep Learning (LSTM).

## Overview

This project implements a multi-model prediction pipeline that:

1. **Loads and validates** historical Mega-Sena data
2. **Engineers features** from raw lottery numbers (261 derived features)
3. **Performs statistical analysis** (Chi-square tests, entropy analysis)
4. **Trains multiple models**:
   - ARIMA for aggregated predictions (sum, even/odd counts)
   - Prophet for trend analysis
   - LSTM neural network for sequence patterns
5. **Generates predictions** with ensemble methods
6. **Creates optimized betting combinations** using wheeling systems

## Project Structure

```
megasena/
├── config/
│   └── settings.py          # Global configuration
├── src/
│   ├── data/
│   │   ├── loader.py         # Data loading and validation
│   │   └── preprocessor.py   # Data preprocessing for ML
│   ├── features/
│   │   └── engineer.py       # Feature engineering (261 features)
│   ├── models/
│   │   ├── arima_model.py    # ARIMA for sum/aggregate prediction
│   │   ├── prophet_model.py  # Prophet for trend analysis
│   │   └── lstm_model.py     # LSTM neural network
│   ├── evaluation/
│   │   ├── metrics.py        # Hit rate, Top-K accuracy, etc.
│   │   └── wheeling.py       # Betting combination optimization
│   ├── utils/
│   │   ├── statistics.py     # Statistical analysis tools
│   │   └── visualization.py  # Plotting utilities
│   └── pipeline.py           # Main orchestrator
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── Mega-Sena.csv            # Historical data
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Prediction (Frequency-Based)
```bash
python main.py --quick
```

### Statistical Analysis Only
```bash
python main.py --analyze
```

### Full Pipeline
```bash
python main.py
```

### Options
```bash
python main.py --help
python main.py --no-lstm      # Skip LSTM (faster)
python main.py --top-k 15     # Top 15 predictions
python main.py --max-bets 10  # Maximum 10 betting combinations
```

## Features

### Engineered Features (261 total)
- **Aggregated Statistics**: sum, mean, std, min, max, range
- **Parity Features**: even_count, odd_count, parity_ratio
- **Consecutiveness**: consecutive_pairs, has_consecutive
- **Quadrant Analysis**: q1_count through q4_count
- **Gap Analysis**: gap_1 through gap_60 (draws since last appearance)
- **Rolling Frequency**: frequency per number across multiple windows

### Models

| Model | Purpose | Predicts |
|-------|---------|----------|
| ARIMA | Time series forecasting | Sum ranges, even/odd counts |
| Prophet | Trend analysis | Long-term patterns |
| LSTM | Sequence learning | Number probabilities |

### Evaluation Metrics
- **Hit Rate**: Numbers predicted vs actually drawn
- **Top-K Accuracy**: Actual numbers within top K predictions
- **Precision@K / Recall@K**: Information retrieval metrics

## Example Output

```
MEGA-SENA QUICK PREDICTION (Frequency-Based)
============================================================

Top 10 numbers by historical frequency:

   1. Number 10:  345 times (1.95%)
   2. Number 53:  336 times (1.90%)
   3. Number  5:  322 times (1.82%)
   ...

Recommended numbers: [10, 53, 5, 37, 34, 38]
```

## Disclaimer

This system is for educational and research purposes only. Lottery outcomes are fundamentally random, and no prediction system can guarantee wins. The mathematical foundation (Law of Large Numbers, Chi-square tests) confirms that over time, all numbers should appear with equal probability. This tool helps:

1. **Audit randomness** of the lottery mechanism
2. **Filter unlikely combinations** based on statistical patterns
3. **Optimize betting strategies** using wheeling systems

Always gamble responsibly.

## License

MIT License

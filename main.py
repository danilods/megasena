#!/usr/bin/env python3
"""
Mega-Sena Lottery Prediction System
====================================

A comprehensive system for analyzing and predicting Mega-Sena lottery results
using statistical methods (ARIMA), Facebook Prophet, and Deep Learning (LSTM).

Usage:
    python main.py                    # Run full prediction pipeline
    python main.py --analyze          # Run statistical analysis only
    python main.py --train            # Train all models
    python main.py --predict          # Generate predictions (requires trained models)
    python main.py --quick            # Quick prediction using frequency analysis only

Author: Mega-Sena Analytics Team
Version: 1.0.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import MegaSenaPipeline, EnsemblePrediction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mega-Sena Lottery Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full pipeline (load, train, predict)
    python main.py --analyze          # Statistical analysis only
    python main.py --train            # Train all models
    python main.py --predict -k 15    # Predict top 15 numbers
    python main.py --quick            # Quick frequency-based prediction
    python main.py --no-lstm          # Skip LSTM (faster, less accurate)
        """
    )

    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Run statistical analysis only'
    )

    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train all models'
    )

    parser.add_argument(
        '--predict', '-p',
        action='store_true',
        help='Generate predictions'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick prediction using frequency analysis only'
    )

    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=20,
        help='Number of top predictions to return (default: 20)'
    )

    parser.add_argument(
        '--max-bets', '-b',
        type=int,
        default=20,
        help='Maximum number of betting combinations (default: 20)'
    )

    parser.add_argument(
        '--no-arima',
        action='store_true',
        help='Skip ARIMA model'
    )

    parser.add_argument(
        '--no-prophet',
        action='store_true',
        help='Skip Prophet model'
    )

    parser.add_argument(
        '--no-lstm',
        action='store_true',
        help='Skip LSTM model'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.csv',
        help='Output filename for predictions (default: predictions.csv)'
    )

    return parser.parse_args()


def run_analysis(pipeline: MegaSenaPipeline) -> None:
    """Run statistical analysis and display results."""
    print("\n" + "=" * 60)
    print("MEGA-SENA STATISTICAL ANALYSIS")
    print("=" * 60 + "\n")

    pipeline.load_data()
    analysis = pipeline.analyze_data()

    print(f"\nTotal draws analyzed: {analysis['total_draws']}")
    print(f"\nChi-Square Test for Uniformity:")
    print(f"  Statistic: {analysis['chi_square']['statistic']:.4f}")
    print(f"  P-value: {analysis['chi_square']['p_value']:.4f}")
    print(f"  Conclusion: {analysis['chi_square']['conclusion']}")

    print(f"\nHot Numbers (most frequent): {analysis['hot_numbers'][:10]}")
    print(f"Cold Numbers (least frequent): {analysis['cold_numbers'][:10]}")

    print(f"\nSum Statistics:")
    for key, value in analysis['sum_statistics'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")


def run_quick_prediction(pipeline: MegaSenaPipeline, top_k: int) -> None:
    """Run quick frequency-based prediction."""
    print("\n" + "=" * 60)
    print("MEGA-SENA QUICK PREDICTION (Frequency-Based)")
    print("=" * 60 + "\n")

    pipeline.load_data()
    pipeline.engineer_features()
    analysis = pipeline.analyze_data()

    # Get frequencies and sort
    frequencies = analysis['frequencies']
    total_freq = frequencies.sum()
    sorted_nums = frequencies.sort_values(ascending=False)

    print(f"Top {top_k} numbers by historical frequency:\n")
    for i, (num, freq) in enumerate(sorted_nums.head(top_k).items(), 1):
        pct = freq / total_freq * 100
        bar = "#" * int(pct * 2)
        print(f"  {i:2d}. Number {int(num):2d}: {freq:4d} times ({pct:.2f}%) {bar}")

    print(f"\nRecommended numbers: {list(sorted_nums.head(6).index.astype(int))}")
    print("\nNote: This is a simple frequency-based prediction.")
    print("For better predictions, use the full pipeline with ML models.")


def run_full_pipeline(
    pipeline: MegaSenaPipeline,
    top_k: int,
    output_file: str
) -> EnsemblePrediction:
    """Run the complete prediction pipeline."""
    prediction = pipeline.run_full_pipeline()
    pipeline.save_results(prediction, output_file)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATIONS")
    print("=" * 60)

    print(f"\nTop {min(top_k, len(prediction.top_numbers))} predicted numbers:")
    for i, num in enumerate(prediction.top_numbers[:top_k], 1):
        prob = prediction.probabilities.get(num, 0)
        print(f"  {i:2d}. Number {num:2d} (confidence: {prob:.4f})")

    print(f"\nExpected sum range: {prediction.expected_sum_range[0]:.0f} - {prediction.expected_sum_range[1]:.0f}")

    if prediction.recommended_bets:
        print(f"\nRecommended betting combinations ({len(prediction.recommended_bets)} bets):")
        for i, bet in enumerate(prediction.recommended_bets[:5], 1):
            bet_sum = sum(bet)
            print(f"  Bet {i}: {sorted(bet)} (sum: {bet_sum})")

        if len(prediction.recommended_bets) > 5:
            print(f"  ... and {len(prediction.recommended_bets) - 5} more bets")

        cost = len(prediction.recommended_bets) * 5.0
        print(f"\nTotal cost: R$ {cost:.2f}")

    print(f"\nResults saved to: {output_file}")

    return prediction


def main():
    """Main entry point."""
    args = parse_args()

    # Create pipeline with options
    pipeline = MegaSenaPipeline(
        use_arima=not args.no_arima,
        use_prophet=not args.no_prophet,
        use_lstm=not args.no_lstm,
        verbose=not args.quiet
    )

    try:
        if args.quick:
            run_quick_prediction(pipeline, args.top_k)

        elif args.analyze:
            run_analysis(pipeline)

        elif args.train:
            pipeline.load_data()
            pipeline.engineer_features()
            results = pipeline.train_models()
            print(f"\nTraining complete: {results}")

        elif args.predict:
            pipeline.load_data()
            pipeline.engineer_features()
            pipeline.train_models()
            prediction = pipeline.predict(top_k=args.top_k)

            print(f"\nTop {args.top_k} predictions:")
            for i, num in enumerate(prediction.top_numbers[:args.top_k], 1):
                print(f"  {i:2d}. Number {num}")

        else:
            # Full pipeline
            run_full_pipeline(pipeline, args.top_k, args.output)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

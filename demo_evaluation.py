"""
Demonstration of evaluation metrics and wheeling system for Mega-Sena.
"""
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.wheeling import WheelingSystem


def demo_metrics():
    """Demonstrate ModelEvaluator functionality."""
    print("=" * 70)
    print("MODEL EVALUATOR DEMONSTRATION")
    print("=" * 70)

    evaluator = ModelEvaluator()

    # Example 1: Single prediction evaluation
    print("\n1. Single Prediction Evaluation")
    print("-" * 70)
    predicted = [5, 12, 23, 34, 45, 56]
    actual = [5, 12, 20, 34, 48, 56]

    result = evaluator.evaluate_prediction(predicted, actual, model_name="Example Model")

    print(f"Predicted: {predicted}")
    print(f"Actual: {actual}")
    print(f"Hits: {result['hits']}")
    print(f"Hit Rate: {result['hit_rate']:.2%}")
    print(f"Matched Numbers: {sorted(result['matched_numbers'])}")
    print(f"Quadra: {result['is_quadra']}, Quina: {result['is_quina']}, Sena: {result['is_sena']}")

    # Example 2: Top-K evaluation
    print("\n2. Top-K Accuracy Evaluation")
    print("-" * 70)

    # Simulated probability distribution
    probabilities = {i: 1.0 / (i + 1) for i in range(1, 61)}  # Simple decreasing probabilities
    actual = [1, 2, 3, 10, 15, 20]

    topk_result = evaluator.evaluate_top_k(probabilities, actual, k_values=[10, 15, 20])

    print(f"Actual Numbers: {actual}")
    for k in [10, 15, 20]:
        print(f"\nTop-{k} Results:")
        print(f"  Predictions: {topk_result['top_k_predictions'][k]}")
        print(f"  Hits: {topk_result['top_k_hit_counts'][k]}")
        print(f"  Accuracy: {topk_result['top_k_accuracy'][k]:.2%}")
        print(f"  Precision@{k}: {topk_result['precision_at_k'][k]:.3f}")
        print(f"  Recall@{k}: {topk_result['recall_at_k'][k]:.3f}")
        print(f"  F1@{k}: {topk_result['f1_at_k'][k]:.3f}")

    # Example 3: Model comparison
    print("\n3. Model Comparison")
    print("-" * 70)

    predictions = {
        "Model A": [1, 2, 3, 4, 5, 6],
        "Model B": [5, 12, 20, 34, 48, 56],
        "Model C": [10, 20, 30, 40, 50, 60]
    }
    actual = [5, 12, 20, 34, 48, 56]

    comparison = evaluator.compare_models(predictions, actual)

    print(f"Actual Draw: {actual}\n")
    print("Rankings:")
    for rank, (model, hits) in enumerate(comparison['rankings'], 1):
        print(f"  {rank}. {model}: {hits} hits")

    print(f"\nBest Model: {comparison['best_model']}")
    print(f"Summary: Avg Hits = {comparison['summary']['avg_hits']:.2f}")

    # Example 4: Generate report
    print("\n4. Evaluation Report")
    print("-" * 70)
    print(evaluator.generate_report(detailed=False))


def demo_wheeling():
    """Demonstrate WheelingSystem functionality."""
    print("\n\n" + "=" * 70)
    print("WHEELING SYSTEM DEMONSTRATION")
    print("=" * 70)

    ws = WheelingSystem()

    # Example 1: Full wheel with 8 numbers
    print("\n1. Full Wheel (8 numbers)")
    print("-" * 70)
    candidates = [5, 12, 17, 23, 34, 42, 51, 58]
    combos = ws.full_wheel(candidates)

    print(f"Candidates: {candidates}")
    print(f"Number of combinations: {len(combos)}")
    print(f"Total cost: R$ {ws.calculate_cost(len(combos)):.2f}")
    print(f"\nFirst 5 combinations:")
    for i, combo in enumerate(combos[:5], 1):
        print(f"  {i}. {combo}")

    # Example 2: Abbreviated wheel
    print("\n2. Abbreviated Wheel (12 numbers, Quadra guarantee)")
    print("-" * 70)
    candidates = [3, 7, 11, 15, 21, 28, 33, 39, 44, 49, 55, 60]
    combos = ws.abbreviated_wheel(candidates, min_guarantee=4)

    print(f"Candidates: {candidates}")
    print(f"Number of combinations: {len(combos)}")
    print(f"Total cost: R$ {ws.calculate_cost(len(combos)):.2f}")
    print(f"Cost savings vs full wheel: R$ {ws.calculate_cost(924) - ws.calculate_cost(len(combos)):.2f}")

    # Example 3: Generate combinations with limit
    print("\n3. Generate Combinations with Max Bets Limit")
    print("-" * 70)
    candidates = list(range(1, 16))  # Numbers 1-15
    combos = ws.generate_combinations(candidates, guarantee_level=4, max_bets=20)

    print(f"Candidates: {len(candidates)} numbers")
    print(f"Number of combinations: {len(combos)}")
    print(f"Total cost: R$ {ws.calculate_cost(len(combos)):.2f}")

    # Example 4: Filtering combinations
    print("\n4. Filtering Combinations")
    print("-" * 70)
    candidates = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    combos = ws.full_wheel(candidates)
    print(f"Original combinations: {len(combos)}")

    # Filter by sum range
    filtered_sum = ws.filter_by_sum_range(combos, 120, 180)
    print(f"After sum filter (120-180): {len(filtered_sum)} combinations")

    # Filter by parity
    combos = ws.full_wheel(candidates)  # Reset
    filtered_parity = ws.filter_by_parity(combos, min_even=2, max_even=4)
    print(f"After parity filter (2-4 even): {len(filtered_parity)} combinations")

    # Filter by range coverage
    combos = ws.full_wheel(candidates)  # Reset
    filtered_range = ws.filter_by_range_coverage(combos, min_ranges=4)
    print(f"After range filter (4+ ranges): {len(filtered_range)} combinations")

    # Example 5: Coverage analysis
    print("\n5. Coverage Analysis")
    print("-" * 70)
    candidates = [2, 8, 14, 22, 31, 38, 45, 52]
    combos = ws.full_wheel(candidates)
    test_draw = [8, 14, 22, 31, 38, 45]  # 6 numbers from our candidates

    coverage = ws.analyze_coverage(combos, test_draw)

    print(f"Test draw: {test_draw}")
    print(f"Maximum hits in any bet: {coverage['max_hits']}")
    print(f"Average hits per bet: {coverage['average_hits']:.2f}")
    print(f"Guarantee met (4+ hits): {coverage['guarantee_met']}")
    print(f"Hit distribution: {coverage['hit_distribution']}")

    # Example 6: Get recommendations
    print("\n6. Wheeling Recommendations")
    print("-" * 70)

    for num_candidates in [8, 12, 15]:
        print(f"\n{num_candidates} candidate numbers:")
        rec = ws.get_recommendations(num_candidates)
        print(f"  Full wheel: {rec['full_wheel']['combinations']} bets, "
              f"R$ {rec['full_wheel']['cost']:.2f}")
        print(f"  Recommended: {rec['recommendation']}")

    # Example 7: Cost summary
    print("\n7. Cost Summary")
    print("-" * 70)
    summary = ws.get_cost_summary()
    print(f"Number of combinations: {summary['num_combinations']}")
    print(f"Total cost: R$ {summary['total_cost']:.2f}")
    print(f"Numbers wheeled: {summary['numbers_wheeled']}")
    print(f"Average number in combinations: {summary['average_number']:.2f}")


if __name__ == "__main__":
    demo_metrics()
    demo_wheeling()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

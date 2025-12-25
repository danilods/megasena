# Quick Start Guide - Evaluation & Wheeling

## Model Evaluation

### Basic Usage
```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate a single prediction
result = evaluator.evaluate_prediction(
    predicted=[5, 12, 23, 34, 45, 56],
    actual=[5, 12, 20, 34, 48, 56]
)
print(f"Hits: {result['hits']}, Quadra: {result['is_quadra']}")

# Top-K evaluation
probabilities = {i: some_score for i in range(1, 61)}
topk = evaluator.evaluate_top_k(probabilities, actual, k_values=[10, 20])

# Compare models
predictions = {
    "Model A": [1,2,3,4,5,6],
    "Model B": [5,12,20,34,48,56]
}
comparison = evaluator.compare_models(predictions, actual)

# Generate report
print(evaluator.generate_report())
```

## Wheeling System

### Basic Usage
```python
from src.evaluation.wheeling import WheelingSystem

ws = WheelingSystem()

# Full wheel (all combinations)
candidates = [5, 12, 17, 23, 34, 42, 51, 58]
combos = ws.full_wheel(candidates)  # 28 bets, R$ 140

# Abbreviated wheel (optimized)
candidates = [3, 7, 11, 15, 21, 28, 33, 39, 44, 49, 55, 60]
combos = ws.abbreviated_wheel(candidates, min_guarantee=4)  # ~50 bets

# With filters
combos = ws.filter_by_sum_range(combos, 150, 210)
combos = ws.filter_by_parity(combos, min_even=2, max_even=4)

# Cost summary
summary = ws.get_cost_summary()
print(f"Bets: {summary['num_combinations']}, Cost: R$ {summary['total_cost']:.2f}")
```

## Run Demo
```bash
python3 demo_evaluation.py
```

## Key Files
- `/home/user/megasena/src/evaluation/metrics.py` - ModelEvaluator (420 lines)
- `/home/user/megasena/src/evaluation/wheeling.py` - WheelingSystem (596 lines)
- `/home/user/megasena/EVALUATION_GUIDE.md` - Complete documentation
- `/home/user/megasena/demo_evaluation.py` - Working examples

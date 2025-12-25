# Evaluation Metrics and Wheeling System Guide

## Overview

This guide documents the evaluation metrics and wheeling system implementation for the Mega-Sena prediction project.

## Files Created

1. `/home/user/megasena/src/evaluation/metrics.py` - ModelEvaluator class
2. `/home/user/megasena/src/evaluation/wheeling.py` - WheelingSystem class

---

## ModelEvaluator Class

### Purpose
Evaluates the performance of prediction models against actual Mega-Sena draw results.

### Key Features

#### 1. Single Prediction Evaluation
```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()
result = evaluator.evaluate_prediction(
    predicted=[5, 12, 23, 34, 45, 56],
    actual=[5, 12, 20, 34, 48, 56],
    model_name="My Model"
)

print(f"Hits: {result['hits']}")  # 4
print(f"Hit Rate: {result['hit_rate']:.2%}")  # 66.67%
print(f"Quadra: {result['is_quadra']}")  # True
```

**Metrics Returned:**
- `hits`: Number of correct predictions
- `hit_rate`: Proportion of correct predictions (0-1)
- `matched_numbers`: Set of correctly predicted numbers
- `missed_numbers`: Numbers that were drawn but not predicted
- `false_positives`: Predicted numbers that were not drawn
- `is_quadra`, `is_quina`, `is_sena`: Prize category flags

#### 2. Top-K Accuracy
```python
probabilities = {1: 0.8, 2: 0.7, 3: 0.6, ...}  # Number -> probability
actual = [1, 2, 3, 10, 15, 20]

result = evaluator.evaluate_top_k(
    probabilities=probabilities,
    actual=actual,
    k_values=[10, 15, 20]
)

for k in [10, 15, 20]:
    print(f"Top-{k} Accuracy: {result['top_k_accuracy'][k]:.2%}")
    print(f"Precision@{k}: {result['precision_at_k'][k]:.3f}")
    print(f"Recall@{k}: {result['recall_at_k'][k]:.3f}")
```

**Metrics Returned:**
- `top_k_accuracy`: Proportion of actual numbers in top-K predictions
- `top_k_hit_counts`: Number of hits in top-K
- `precision_at_k`: Precision score (relevant hits / K)
- `recall_at_k`: Recall score (relevant hits / total relevant)
- `f1_at_k`: F1 score (harmonic mean of precision and recall)

#### 3. Model Comparison
```python
predictions = {
    "ARIMA": [1, 2, 3, 4, 5, 6],
    "LSTM": [5, 12, 20, 34, 48, 56],
    "Prophet": [10, 20, 30, 40, 50, 60]
}
actual = [5, 12, 20, 34, 48, 56]

comparison = evaluator.compare_models(predictions, actual)
print(f"Best Model: {comparison['best_model']}")
print("Rankings:", comparison['rankings'])
```

#### 4. Coverage Analysis
```python
predictions = [[1,2,3,4,5,6], [7,8,9,10,11,12], ...]
actual_draws = [[5,12,20,34,48,56], [3,8,15,22,41,58], ...]

coverage = evaluator.calculate_coverage_analysis(predictions, actual_draws)
print(f"Average Hits: {coverage['average_hits']:.2f}")
print(f"Quadra Rate: {coverage['quadra_rate']:.2%}")
```

#### 5. Generate Reports
```python
report = evaluator.generate_report(model_name="LSTM", detailed=True)
print(report)
```

Output:
```
Evaluation Report for LSTM
===========================

Total Evaluations: 10

Performance Metrics:
  Average Hits: 2.45 ± 1.23
  Min Hits: 0
  Max Hits: 5
  Median Hits: 2.0

Prize Categories:
  Quadra (4+ hits): 3 (30.00%)
  Quina (5+ hits): 1 (10.00%)
  Sena (6 hits): 0 (0.00%)

Hit Distribution:
  0 hits:    1 (10.00%) █████
  1 hits:    2 (20.00%) ██████████
  2 hits:    3 (30.00%) ███████████████
  ...
```

---

## WheelingSystem Class

### Purpose
Generate optimal bet combinations from a set of candidate numbers, ensuring coverage guarantees while minimizing costs.

### Key Features

#### 1. Full Wheel
Generates ALL possible 6-number combinations from candidate numbers.

```python
from src.evaluation.wheeling import WheelingSystem

ws = WheelingSystem()
candidates = [5, 12, 17, 23, 34, 42, 51, 58]  # 8 numbers
combos = ws.full_wheel(candidates)

print(f"Combinations: {len(combos)}")  # 28 (C(8,6))
print(f"Cost: R$ {ws.calculate_cost(len(combos)):.2f}")  # R$ 140.00
```

**Guarantee:** If all 6 drawn numbers are among your 8 candidates, you WILL hit the Sena (6/6).

#### 2. Abbreviated Wheel
Generates an optimized reduced set of combinations with minimum guarantee.

```python
candidates = [3, 7, 11, 15, 21, 28, 33, 39, 44, 49, 55, 60]  # 12 numbers
combos = ws.abbreviated_wheel(candidates, min_guarantee=4)

print(f"Combinations: {len(combos)}")  # ~50-100 (vs 924 for full wheel)
print(f"Cost: R$ {ws.calculate_cost(len(combos)):.2f}")
```

**Guarantee:** If all 6 drawn numbers are among your candidates, you WILL have at least one bet with 4+ matches (Quadra).

#### 3. Generate Combinations with Limits
```python
candidates = list(range(1, 16))  # 15 numbers
combos = ws.generate_combinations(
    candidates,
    guarantee_level=4,
    max_bets=20  # Limit to 20 bets
)

print(f"Generated {len(combos)} bets for R$ {ws.calculate_cost(len(combos)):.2f}")
```

#### 4. Filtering Combinations

**Filter by Sum Range:**
```python
combos = ws.full_wheel([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
filtered = ws.filter_by_sum_range(combos, min_sum=120, max_sum=180)
```

Statistical analysis shows most Mega-Sena draws have sums between 150-210.

**Filter by Parity (Even/Odd Balance):**
```python
filtered = ws.filter_by_parity(combos, min_even=2, max_even=4)
```

Most draws have 2-4 even numbers, rarely all even or all odd.

**Filter by Range Coverage:**
```python
filtered = ws.filter_by_range_coverage(combos, min_ranges=4)
```

Ensures combinations cover multiple decade ranges (1-10, 11-20, ..., 51-60).

#### 5. Coverage Analysis
```python
candidates = [2, 8, 14, 22, 31, 38, 45, 52]
combos = ws.full_wheel(candidates)
test_draw = [8, 14, 22, 31, 38, 45]  # Actual draw

coverage = ws.analyze_coverage(combos, test_draw)
print(f"Max hits in any bet: {coverage['max_hits']}")
print(f"Guarantee met: {coverage['guarantee_met']}")
print(f"Hit distribution: {coverage['hit_distribution']}")
```

#### 6. Cost Calculations
```python
# Calculate cost for any number of bets
cost = ws.calculate_cost(50)  # R$ 250.00

# Get cost summary for last generated wheel
summary = ws.get_cost_summary()
print(f"Total bets: {summary['num_combinations']}")
print(f"Total cost: R$ {summary['total_cost']:.2f}")
print(f"Numbers wheeled: {summary['numbers_wheeled']}")
```

#### 7. Recommendations
```python
rec = ws.get_recommendations(num_candidates=12)
print(f"Full wheel: {rec['full_wheel']['combinations']} bets")
print(f"Recommendation: {rec['recommendation']}")
```

---

## Bet Cost Structure

Mega-Sena bet costs (R$ 5.00 per 6-number combination):

| Candidates | Combinations | Cost (R$) |
|------------|--------------|-----------|
| 6          | 1            | 5.00      |
| 7          | 7            | 35.00     |
| 8          | 28           | 140.00    |
| 9          | 84           | 420.00    |
| 10         | 210          | 1,050.00  |
| 12         | 924          | 4,620.00  |
| 15         | 5,005        | 25,025.00 |

---

## Usage Examples

### Example 1: Evaluate Model Performance
```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate multiple predictions
predictions = [
    [1, 5, 12, 23, 34, 45],
    [2, 8, 15, 22, 41, 58],
    # ... more predictions
]

actuals = [
    [5, 12, 20, 34, 48, 56],
    [3, 8, 15, 22, 41, 58],
    # ... actual draws
]

# Evaluate each
for pred, actual in zip(predictions, actuals):
    evaluator.evaluate_prediction(pred, actual, model_name="LSTM")

# Generate report
print(evaluator.generate_report(model_name="LSTM"))
```

### Example 2: Create Optimized Betting Strategy
```python
from src.evaluation.wheeling import WheelingSystem

ws = WheelingSystem()

# Start with model predictions (top candidates)
top_candidates = [5, 12, 17, 20, 23, 28, 34, 38, 42, 45, 51, 58]

# Generate combinations
combos = ws.generate_combinations(top_candidates, guarantee_level=4, max_bets=30)

# Apply filters for better odds
combos = ws.filter_by_sum_range(combos, 150, 210)
combos = ws.filter_by_parity(combos, min_even=2, max_even=4)
combos = ws.filter_by_range_coverage(combos, min_ranges=4)

# Review
summary = ws.get_cost_summary()
print(f"Final strategy: {summary['num_combinations']} bets")
print(f"Total cost: R$ {summary['total_cost']:.2f}")

# Save for betting
for i, combo in enumerate(combos, 1):
    print(f"Bet {i}: {combo}")
```

---

## Technical Notes

### Performance Considerations

1. **Full Wheels** are computationally simple but expensive for large candidate sets:
   - 8 numbers: 28 bets (fast, affordable)
   - 12 numbers: 924 bets (fast, expensive)
   - 15 numbers: 5,005 bets (fast, very expensive)

2. **Abbreviated Wheels** use greedy coverage algorithm:
   - Efficient for up to 14 candidates
   - For 15+ candidates, uses simplified sampling strategy
   - Provides good coverage with significantly fewer bets

3. **Filtering** reduces bet count while maintaining quality:
   - Sum range filtering: Removes statistically unlikely combinations
   - Parity filtering: Removes imbalanced even/odd distributions
   - Range coverage: Ensures number diversity

### Type Hints and Documentation

Both modules include:
- Full type hints for all methods
- Comprehensive docstrings
- Example usage in docstrings
- Input validation with clear error messages

---

## Running the Demo

To see all features in action:

```bash
cd /home/user/megasena
python3 demo_evaluation.py
```

This demonstrates:
- All ModelEvaluator features
- All WheelingSystem features
- Real-world usage examples
- Performance metrics

---

## Integration with Other Modules

These evaluation modules integrate with:

1. **Prediction Models** (`src/models/`):
   - ARIMA, LSTM, Prophet models
   - Use ModelEvaluator to compare model performance

2. **Configuration** (`config/settings.py`):
   - Uses NUMBERS_PER_DRAW, TOTAL_NUMBERS
   - Uses TOP_K_VALUES for evaluation

3. **Future Enhancements**:
   - Visualization of evaluation metrics
   - Statistical analysis of wheeling coverage
   - Automated model selection based on performance

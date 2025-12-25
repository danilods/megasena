"""
Model evaluation metrics for Mega-Sena predictions.

This module provides comprehensive evaluation metrics for lottery prediction models,
including hit rates, top-K accuracy, precision/recall, and coverage analysis.
"""
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
from datetime import datetime

from config.settings import NUMBERS_PER_DRAW, TOTAL_NUMBERS, TOP_K_VALUES


class ModelEvaluator:
    """
    Evaluates prediction model performance for Mega-Sena lottery.

    Tracks various metrics including hit rates, top-K accuracy, precision/recall,
    and generates comprehensive performance reports.
    """

    def __init__(self):
        """Initialize the ModelEvaluator with empty history."""
        self.evaluation_history: List[Dict[str, Any]] = []
        self.model_comparisons: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def evaluate_prediction(
        self,
        predicted: List[int],
        actual: List[int],
        model_name: Optional[str] = None,
        contest_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction against actual draw results.

        Args:
            predicted: List of predicted numbers (typically 6 numbers)
            actual: List of actual drawn numbers (6 numbers)
            model_name: Optional name of the model making the prediction
            contest_id: Optional contest identifier

        Returns:
            Dictionary containing evaluation metrics:
                - hits: Number of correct predictions
                - hit_rate: Proportion of correct predictions (0-1)
                - predicted_set: Set of predicted numbers
                - actual_set: Set of actual numbers
                - matched_numbers: Set of correctly predicted numbers
                - missed_numbers: Numbers that were drawn but not predicted
                - false_positives: Predicted numbers that were not drawn
                - is_quadra: True if 4 numbers matched (Quadra prize)
                - is_quina: True if 5 numbers matched (Quina prize)
                - is_sena: True if all 6 numbers matched (Sena jackpot)
        """
        predicted_set = set(predicted)
        actual_set = set(actual)

        # Calculate matches
        matched = predicted_set & actual_set
        hits = len(matched)
        hit_rate = hits / NUMBERS_PER_DRAW

        # Calculate misses and false positives
        missed = actual_set - predicted_set
        false_positives = predicted_set - actual_set

        # Prize categories
        is_quadra = hits >= 4
        is_quina = hits >= 5
        is_sena = hits == 6

        result = {
            'hits': hits,
            'hit_rate': hit_rate,
            'predicted_set': predicted_set,
            'actual_set': actual_set,
            'matched_numbers': matched,
            'missed_numbers': missed,
            'false_positives': false_positives,
            'is_quadra': is_quadra,
            'is_quina': is_quina,
            'is_sena': is_sena,
            'model_name': model_name,
            'contest_id': contest_id,
            'timestamp': datetime.now()
        }

        # Store in history
        self.evaluation_history.append(result)
        if model_name:
            self.model_comparisons[model_name].append(result)

        return result

    def evaluate_top_k(
        self,
        probabilities: Dict[int, float],
        actual: List[int],
        k_values: Optional[List[int]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate top-K accuracy of probability predictions.

        Args:
            probabilities: Dictionary mapping numbers (1-60) to their probabilities
            actual: List of actual drawn numbers
            k_values: List of K values to evaluate (default: TOP_K_VALUES from settings)
            model_name: Optional name of the model

        Returns:
            Dictionary containing top-K metrics:
                - top_k_accuracy: Dict mapping K to accuracy (proportion of actual numbers in top-K)
                - top_k_hit_counts: Dict mapping K to number of hits in top-K
                - top_k_predictions: Dict mapping K to the top-K predicted numbers
                - precision_at_k: Dict mapping K to precision score
                - recall_at_k: Dict mapping K to recall score
                - f1_at_k: Dict mapping K to F1 score
        """
        if k_values is None:
            k_values = TOP_K_VALUES

        # Sort numbers by probability (descending)
        sorted_predictions = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        actual_set = set(actual)
        result = {
            'top_k_accuracy': {},
            'top_k_hit_counts': {},
            'top_k_predictions': {},
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'model_name': model_name
        }

        for k in k_values:
            # Get top K predictions
            top_k_numbers = [num for num, _ in sorted_predictions[:k]]
            top_k_set = set(top_k_numbers)

            # Calculate hits in top-K
            hits = len(actual_set & top_k_set)
            accuracy = hits / NUMBERS_PER_DRAW

            # Precision@K: proportion of top-K that are relevant
            precision = hits / k if k > 0 else 0

            # Recall@K: proportion of relevant items found in top-K
            recall = hits / NUMBERS_PER_DRAW

            # F1@K: harmonic mean of precision and recall
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            result['top_k_accuracy'][k] = accuracy
            result['top_k_hit_counts'][k] = hits
            result['top_k_predictions'][k] = top_k_numbers
            result['precision_at_k'][k] = precision
            result['recall_at_k'][k] = recall
            result['f1_at_k'][k] = f1

        return result

    def calculate_coverage_analysis(
        self,
        predictions: List[List[int]],
        actual_draws: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Analyze how well predictions cover actual results over multiple draws.

        Args:
            predictions: List of prediction sets
            actual_draws: List of actual draw results

        Returns:
            Dictionary containing coverage metrics:
                - total_predictions: Total number of predictions made
                - average_hits: Average number of hits per prediction
                - hit_distribution: Distribution of hit counts
                - coverage_rate: Overall coverage rate
                - best_prediction: Best performing prediction
                - worst_prediction: Worst performing prediction
                - quadra_rate: Proportion of predictions with 4+ hits
                - quina_rate: Proportion of predictions with 5+ hits
                - sena_rate: Proportion of predictions with 6 hits
        """
        if len(predictions) != len(actual_draws):
            raise ValueError("Number of predictions must match number of actual draws")

        hit_counts = []
        hit_distribution = defaultdict(int)
        quadra_count = 0
        quina_count = 0
        sena_count = 0

        evaluations = []

        for pred, actual in zip(predictions, actual_draws):
            eval_result = self.evaluate_prediction(pred, actual)
            hits = eval_result['hits']
            hit_counts.append(hits)
            hit_distribution[hits] += 1
            evaluations.append(eval_result)

            if eval_result['is_quadra']:
                quadra_count += 1
            if eval_result['is_quina']:
                quina_count += 1
            if eval_result['is_sena']:
                sena_count += 1

        total_predictions = len(predictions)
        avg_hits = np.mean(hit_counts) if hit_counts else 0
        std_hits = np.std(hit_counts) if hit_counts else 0

        # Find best and worst predictions
        best_idx = np.argmax(hit_counts) if hit_counts else 0
        worst_idx = np.argmin(hit_counts) if hit_counts else 0

        return {
            'total_predictions': total_predictions,
            'average_hits': avg_hits,
            'std_hits': std_hits,
            'min_hits': min(hit_counts) if hit_counts else 0,
            'max_hits': max(hit_counts) if hit_counts else 0,
            'hit_distribution': dict(hit_distribution),
            'coverage_rate': avg_hits / NUMBERS_PER_DRAW,
            'best_prediction': {
                'index': best_idx,
                'predicted': predictions[best_idx],
                'actual': actual_draws[best_idx],
                'hits': hit_counts[best_idx]
            },
            'worst_prediction': {
                'index': worst_idx,
                'predicted': predictions[worst_idx],
                'actual': actual_draws[worst_idx],
                'hits': hit_counts[worst_idx]
            },
            'quadra_rate': quadra_count / total_predictions if total_predictions > 0 else 0,
            'quina_rate': quina_count / total_predictions if total_predictions > 0 else 0,
            'sena_rate': sena_count / total_predictions if total_predictions > 0 else 0,
            'quadra_count': quadra_count,
            'quina_count': quina_count,
            'sena_count': sena_count
        }

    def compare_models(
        self,
        predictions_dict: Dict[str, List[int]],
        actual: List[int]
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same draw.

        Args:
            predictions_dict: Dictionary mapping model names to their predictions
            actual: Actual draw results

        Returns:
            Dictionary containing comparison metrics:
                - model_results: Individual results for each model
                - best_model: Name of best performing model
                - worst_model: Name of worst performing model
                - rankings: Models ranked by hit count
                - summary: Summary statistics across all models
        """
        model_results = {}

        for model_name, prediction in predictions_dict.items():
            result = self.evaluate_prediction(
                predicted=prediction,
                actual=actual,
                model_name=model_name
            )
            model_results[model_name] = result

        # Rank models by hits
        rankings = sorted(
            model_results.items(),
            key=lambda x: x[1]['hits'],
            reverse=True
        )

        best_model = rankings[0][0] if rankings else None
        worst_model = rankings[-1][0] if rankings else None

        # Calculate summary statistics
        all_hits = [r['hits'] for r in model_results.values()]

        return {
            'model_results': model_results,
            'best_model': best_model,
            'worst_model': worst_model,
            'rankings': [(name, result['hits']) for name, result in rankings],
            'summary': {
                'avg_hits': np.mean(all_hits) if all_hits else 0,
                'std_hits': np.std(all_hits) if all_hits else 0,
                'min_hits': min(all_hits) if all_hits else 0,
                'max_hits': max(all_hits) if all_hits else 0,
                'total_models': len(model_results)
            }
        }

    def generate_report(
        self,
        model_name: Optional[str] = None,
        detailed: bool = True
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            model_name: Optional model name to filter results
            detailed: If True, include detailed breakdown

        Returns:
            Formatted string report of evaluation metrics
        """
        # Filter evaluations if model_name provided
        if model_name:
            evaluations = self.model_comparisons.get(model_name, [])
            title = f"Evaluation Report for {model_name}"
        else:
            evaluations = self.evaluation_history
            title = "Overall Evaluation Report"

        if not evaluations:
            return f"{title}\n{'=' * len(title)}\nNo evaluations available."

        # Calculate aggregate statistics
        hit_counts = [e['hits'] for e in evaluations]
        quadra_count = sum(1 for e in evaluations if e['is_quadra'])
        quina_count = sum(1 for e in evaluations if e['is_quina'])
        sena_count = sum(1 for e in evaluations if e['is_sena'])

        total_evals = len(evaluations)

        report_lines = [
            title,
            "=" * len(title),
            f"\nTotal Evaluations: {total_evals}",
            f"\nPerformance Metrics:",
            f"  Average Hits: {np.mean(hit_counts):.2f} ± {np.std(hit_counts):.2f}",
            f"  Min Hits: {min(hit_counts)}",
            f"  Max Hits: {max(hit_counts)}",
            f"  Median Hits: {np.median(hit_counts):.1f}",
            f"\nPrize Categories:",
            f"  Quadra (4+ hits): {quadra_count} ({quadra_count/total_evals*100:.2f}%)",
            f"  Quina (5+ hits): {quina_count} ({quina_count/total_evals*100:.2f}%)",
            f"  Sena (6 hits): {sena_count} ({sena_count/total_evals*100:.2f}%)",
        ]

        # Hit distribution
        hit_dist = defaultdict(int)
        for hits in hit_counts:
            hit_dist[hits] += 1

        report_lines.append("\nHit Distribution:")
        for hits in sorted(hit_dist.keys()):
            count = hit_dist[hits]
            pct = count / total_evals * 100
            bar = '█' * int(pct / 2)
            report_lines.append(f"  {hits} hits: {count:4d} ({pct:5.2f}%) {bar}")

        if detailed and total_evals <= 20:
            report_lines.append("\nDetailed Results:")
            for i, eval_result in enumerate(evaluations, 1):
                contest = f"Contest {eval_result['contest_id']}" if eval_result['contest_id'] else f"Eval {i}"
                report_lines.append(
                    f"  {contest}: {eval_result['hits']} hits - "
                    f"Matched: {sorted(eval_result['matched_numbers'])}"
                )

        return "\n".join(report_lines)

    def get_statistics(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get summary statistics for evaluations.

        Args:
            model_name: Optional model name to filter results

        Returns:
            Dictionary of statistical metrics
        """
        evaluations = (
            self.model_comparisons.get(model_name, [])
            if model_name
            else self.evaluation_history
        )

        if not evaluations:
            return {}

        hit_counts = [e['hits'] for e in evaluations]

        return {
            'count': len(evaluations),
            'mean_hits': np.mean(hit_counts),
            'std_hits': np.std(hit_counts),
            'min_hits': min(hit_counts),
            'max_hits': max(hit_counts),
            'median_hits': np.median(hit_counts),
            'quadra_rate': sum(1 for e in evaluations if e['is_quadra']) / len(evaluations),
            'quina_rate': sum(1 for e in evaluations if e['is_quina']) / len(evaluations),
            'sena_rate': sum(1 for e in evaluations if e['is_sena']) / len(evaluations)
        }

    def clear_history(self):
        """Clear all evaluation history."""
        self.evaluation_history.clear()
        self.model_comparisons.clear()

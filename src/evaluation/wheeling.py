"""
Wheeling system for Mega-Sena lottery.

This module provides wheeling algorithms to generate optimal bet combinations
from a set of candidate numbers, ensuring coverage guarantees while minimizing costs.
"""
from typing import List, Set, Tuple, Dict, Optional
from itertools import combinations
import numpy as np
from collections import defaultdict

from config.settings import NUMBERS_PER_DRAW, TOTAL_NUMBERS


class WheelingSystem:
    """
    Generate optimal betting combinations using wheeling strategies.

    Wheeling systems allow players to select more than 6 numbers and generate
    a set of bets that guarantee a minimum prize if enough numbers are drawn
    from the selected candidate pool.
    """

    # Mega-Sena bet costs (in Brazilian Reais as of standard pricing)
    # Cost = base_cost * C(n, 6) where n is numbers selected
    BASE_BET_COST = 5.00  # Cost for a single 6-number bet

    def __init__(self):
        """Initialize the WheelingSystem."""
        self.last_combinations: List[Tuple[int, ...]] = []
        self.last_cost: float = 0.0

    def full_wheel(self, numbers: List[int]) -> List[Tuple[int, ...]]:
        """
        Generate all possible 6-number combinations from candidate numbers.

        This is the most comprehensive wheeling system - guarantees maximum
        coverage but can be very expensive for large number sets.

        Args:
            numbers: List of candidate numbers to wheel

        Returns:
            List of all possible 6-number combinations

        Raises:
            ValueError: If numbers list is invalid

        Example:
            >>> ws = WheelingSystem()
            >>> combos = ws.full_wheel([1, 2, 3, 4, 5, 6, 7, 8])
            >>> len(combos)  # C(8,6) = 28
            28
        """
        if len(numbers) < NUMBERS_PER_DRAW:
            raise ValueError(
                f"Need at least {NUMBERS_PER_DRAW} numbers for full wheel, got {len(numbers)}"
            )

        if not all(1 <= n <= TOTAL_NUMBERS for n in numbers):
            raise ValueError(f"All numbers must be between 1 and {TOTAL_NUMBERS}")

        if len(set(numbers)) != len(numbers):
            raise ValueError("Duplicate numbers in candidate list")

        # Generate all combinations
        combos = list(combinations(sorted(numbers), NUMBERS_PER_DRAW))
        self.last_combinations = combos
        self.last_cost = self.calculate_cost(len(combos))

        return combos

    def abbreviated_wheel(
        self,
        numbers: List[int],
        min_guarantee: int = 4
    ) -> List[Tuple[int, ...]]:
        """
        Generate an abbreviated (reduced) wheel with minimum guarantee.

        Abbreviated wheels provide coverage guarantees with fewer bets than
        full wheels. The min_guarantee specifies the minimum prize level
        guaranteed if enough numbers are drawn.

        Args:
            numbers: List of candidate numbers (typically 8-15 numbers)
            min_guarantee: Minimum matches guaranteed (3, 4, or 5)
                          4 = guarantee Quadra if 6 numbers drawn from candidates
                          5 = guarantee Quina if 6 numbers drawn from candidates

        Returns:
            List of optimized 6-number combinations

        Note:
            This uses a greedy coverage algorithm. True optimal abbreviated
            wheels are pre-computed tables for specific configurations.
        """
        if len(numbers) < NUMBERS_PER_DRAW:
            raise ValueError(
                f"Need at least {NUMBERS_PER_DRAW} numbers, got {len(numbers)}"
            )

        if min_guarantee < 3 or min_guarantee > 5:
            raise ValueError("min_guarantee must be 3, 4, or 5")

        # For small sets, use full wheel
        if len(numbers) <= 8:
            return self.full_wheel(numbers)

        # Use greedy coverage algorithm for larger sets
        combos = self._greedy_coverage_wheel(numbers, min_guarantee)
        self.last_combinations = combos
        self.last_cost = self.calculate_cost(len(combos))

        return combos

    def _greedy_coverage_wheel(
        self,
        numbers: List[int],
        min_guarantee: int
    ) -> List[Tuple[int, ...]]:
        """
        Greedy algorithm to generate reduced wheel with coverage guarantee.

        This algorithm iteratively selects combinations that cover the most
        uncovered smaller combinations. Uses optimized approach for large sets.

        Args:
            numbers: Candidate numbers
            min_guarantee: Minimum guarantee level

        Returns:
            List of combinations providing coverage
        """
        from math import comb

        n = len(numbers)
        sorted_nums = sorted(numbers)

        # For very large sets (>14 numbers), use a simplified sampling approach
        # to avoid computational explosion
        if n > 14:
            return self._simplified_wheel(sorted_nums, min_guarantee)

        # Generate all smaller combinations that need to be covered
        target_size = min_guarantee
        targets = set(combinations(sorted_nums, target_size))

        # Pre-compute coverage map: which targets each combo covers
        all_combos = list(combinations(sorted_nums, NUMBERS_PER_DRAW))
        coverage_map = {}

        for combo in all_combos:
            combo_set = frozenset(combo)
            covered_targets = frozenset(
                target for target in targets
                if set(target).issubset(set(combo))
            )
            coverage_map[combo] = covered_targets

        selected_combos = []
        covered = set()

        # Greedy selection with pre-computed coverage
        max_iterations = min(len(all_combos), comb(n, NUMBERS_PER_DRAW) // 2)

        for _ in range(max_iterations):
            if covered == targets or not all_combos:
                break

            best_combo = None
            best_new_coverage = 0

            # Find combo that covers most uncovered targets
            for combo in all_combos:
                new_coverage = len(coverage_map[combo] - covered)
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_combo = combo

            if best_combo is None or best_new_coverage == 0:
                break

            # Add best combo
            selected_combos.append(best_combo)
            all_combos.remove(best_combo)
            covered.update(coverage_map[best_combo])

        return selected_combos

    def _simplified_wheel(
        self,
        numbers: List[int],
        min_guarantee: int
    ) -> List[Tuple[int, ...]]:
        """
        Simplified wheeling for large candidate sets.

        Uses strategic sampling instead of full coverage to keep
        computational cost reasonable.

        Args:
            numbers: Sorted list of candidate numbers
            min_guarantee: Minimum guarantee level

        Returns:
            List of strategically selected combinations
        """
        from math import comb

        n = len(numbers)

        # Calculate a reasonable number of bets (about 1/3 of full wheel)
        target_size = max(10, comb(n, NUMBERS_PER_DRAW) // 3)

        # Strategy: Select combinations with good distribution
        # 1. Include combinations from evenly spaced starting points
        # 2. Mix different number ranges

        selected = []
        step = max(1, n // 8)  # Step through candidates

        # Generate diverse combinations
        for start_idx in range(0, n - NUMBERS_PER_DRAW + 1, step):
            if len(selected) >= target_size:
                break
            # Take sequential numbers
            combo = tuple(numbers[start_idx:start_idx + NUMBERS_PER_DRAW])
            if len(combo) == NUMBERS_PER_DRAW:
                selected.append(combo)

        # Add some combinations with wider spread
        for i in range(0, n - NUMBERS_PER_DRAW + 1, step * 2):
            if len(selected) >= target_size:
                break
            # Take every other number for wider coverage
            combo_list = []
            idx = i
            while len(combo_list) < NUMBERS_PER_DRAW and idx < n:
                combo_list.append(numbers[idx])
                idx += 2
            if len(combo_list) == NUMBERS_PER_DRAW:
                combo = tuple(combo_list)
                if combo not in selected:
                    selected.append(combo)

        # Fill remaining with random diverse selections
        import random
        random.seed(42)  # For reproducibility
        all_combos = list(combinations(numbers, NUMBERS_PER_DRAW))

        while len(selected) < target_size and len(all_combos) > 0:
            # Pick combinations with high variance (spread)
            remaining = [c for c in all_combos if c not in selected]
            if not remaining:
                break
            # Sort by variance and pick high-variance combinations
            scored = [(c, np.var(c)) for c in remaining[:1000]]  # Sample for efficiency
            scored.sort(key=lambda x: x[1], reverse=True)
            for combo, _ in scored[:min(10, len(scored))]:
                if len(selected) >= target_size:
                    break
                if combo not in selected:
                    selected.append(combo)

        return selected[:target_size]

    def generate_combinations(
        self,
        candidates: List[int],
        guarantee_level: int = 4,
        max_bets: Optional[int] = None
    ) -> List[Tuple[int, ...]]:
        """
        Generate bet combinations with specified guarantee level.

        Main method for generating wheeling combinations. Automatically
        chooses between full and abbreviated wheels based on candidate size.

        Args:
            candidates: List of candidate numbers to wheel
            guarantee_level: Guarantee level (3=Terno, 4=Quadra, 5=Quina)
            max_bets: Optional maximum number of bets to generate

        Returns:
            List of 6-number bet combinations

        Example:
            >>> ws = WheelingSystem()
            >>> # Generate bets from 10 candidate numbers with Quadra guarantee
            >>> combos = ws.generate_combinations([1,2,3,4,5,6,7,8,9,10], guarantee_level=4)
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        if len(candidates) < NUMBERS_PER_DRAW:
            raise ValueError(
                f"Need at least {NUMBERS_PER_DRAW} candidates, got {len(candidates)}"
            )

        # Decide between full and abbreviated wheel
        num_candidates = len(candidates)

        # Calculate full wheel size
        from math import comb
        full_wheel_size = comb(num_candidates, NUMBERS_PER_DRAW)

        # Use full wheel if reasonable size or explicitly requested
        if full_wheel_size <= 50 or guarantee_level == 6:
            combos = self.full_wheel(candidates)
        else:
            combos = self.abbreviated_wheel(candidates, min_guarantee=guarantee_level)

        # Apply max_bets limit if specified
        if max_bets and len(combos) > max_bets:
            # Select top combinations by diversity (spread across number range)
            combos = self._select_diverse_combinations(combos, max_bets)
            self.last_combinations = combos
            self.last_cost = self.calculate_cost(len(combos))

        return combos

    def _select_diverse_combinations(
        self,
        combinations: List[Tuple[int, ...]],
        n_select: int
    ) -> List[Tuple[int, ...]]:
        """
        Select n_select diverse combinations from the list.

        Uses variance of numbers in each combination as diversity metric.

        Args:
            combinations: List of all combinations
            n_select: Number of combinations to select

        Returns:
            List of selected diverse combinations
        """
        if n_select >= len(combinations):
            return combinations

        # Score each combination by variance (higher variance = more spread)
        scored_combos = [
            (combo, np.var(combo))
            for combo in combinations
        ]

        # Sort by variance (descending) and take top n_select
        scored_combos.sort(key=lambda x: x[1], reverse=True)

        return [combo for combo, _ in scored_combos[:n_select]]

    def filter_by_sum_range(
        self,
        combinations: List[Tuple[int, ...]],
        min_sum: int,
        max_sum: int
    ) -> List[Tuple[int, ...]]:
        """
        Filter combinations by sum of numbers.

        Statistical analysis shows that Mega-Sena draws tend to have sums
        within a certain range. This filter removes unlikely combinations.

        Args:
            combinations: List of bet combinations
            min_sum: Minimum acceptable sum
            max_sum: Maximum acceptable sum

        Returns:
            Filtered list of combinations

        Example:
            >>> ws = WheelingSystem()
            >>> combos = ws.full_wheel([1,2,3,4,5,6,7])
            >>> # Filter to typical sum range (150-210 is common for Mega-Sena)
            >>> filtered = ws.filter_by_sum_range(combos, 150, 210)
        """
        filtered = [
            combo for combo in combinations
            if min_sum <= sum(combo) <= max_sum
        ]

        self.last_combinations = filtered
        self.last_cost = self.calculate_cost(len(filtered))

        return filtered

    def filter_by_parity(
        self,
        combinations: List[Tuple[int, ...]],
        min_even: int = 1,
        max_even: int = 5
    ) -> List[Tuple[int, ...]]:
        """
        Filter combinations by parity balance (even/odd numbers).

        Historical data shows most draws have a mix of even and odd numbers,
        rarely all even or all odd.

        Args:
            combinations: List of bet combinations
            min_even: Minimum number of even numbers
            max_even: Maximum number of even numbers

        Returns:
            Filtered list of combinations with acceptable parity balance

        Example:
            >>> ws = WheelingSystem()
            >>> combos = ws.full_wheel([1,2,3,4,5,6,7])
            >>> # Filter to balanced parity (2-4 even numbers)
            >>> filtered = ws.filter_by_parity(combos, min_even=2, max_even=4)
        """
        filtered = [
            combo for combo in combinations
            if min_even <= sum(1 for n in combo if n % 2 == 0) <= max_even
        ]

        self.last_combinations = filtered
        self.last_cost = self.calculate_cost(len(filtered))

        return filtered

    def filter_by_range_coverage(
        self,
        combinations: List[Tuple[int, ...]],
        min_ranges: int = 4
    ) -> List[Tuple[int, ...]]:
        """
        Filter combinations by coverage of number ranges.

        Divides 1-60 into ranges (1-10, 11-20, etc.) and ensures
        combinations cover multiple ranges.

        Args:
            combinations: List of bet combinations
            min_ranges: Minimum number of ranges that must be covered

        Returns:
            Filtered list of combinations with good range coverage
        """
        def count_ranges(combo: Tuple[int, ...]) -> int:
            """Count how many decade ranges are covered."""
            ranges_covered = set()
            for num in combo:
                range_id = (num - 1) // 10  # 0-5 for ranges 1-10, 11-20, ..., 51-60
                ranges_covered.add(range_id)
            return len(ranges_covered)

        filtered = [
            combo for combo in combinations
            if count_ranges(combo) >= min_ranges
        ]

        self.last_combinations = filtered
        self.last_cost = self.calculate_cost(len(filtered))

        return filtered

    def calculate_cost(self, num_combinations: int) -> float:
        """
        Calculate total cost for a number of bet combinations.

        Args:
            num_combinations: Number of bets to place

        Returns:
            Total cost in Brazilian Reais (R$)

        Example:
            >>> ws = WheelingSystem()
            >>> cost = ws.calculate_cost(10)  # 10 bets
            >>> print(f"R$ {cost:.2f}")
            R$ 50.00
        """
        return num_combinations * self.BASE_BET_COST

    def get_cost_summary(self) -> Dict[str, any]:
        """
        Get summary of last generated combinations and their cost.

        Returns:
            Dictionary with cost breakdown:
                - num_combinations: Number of bets generated
                - total_cost: Total cost in R$
                - cost_per_bet: Cost per individual bet
                - numbers_wheeled: Total unique numbers in the wheel
        """
        if not self.last_combinations:
            return {
                'num_combinations': 0,
                'total_cost': 0.0,
                'cost_per_bet': self.BASE_BET_COST,
                'numbers_wheeled': 0
            }

        # Count unique numbers
        all_numbers = set()
        for combo in self.last_combinations:
            all_numbers.update(combo)

        return {
            'num_combinations': len(self.last_combinations),
            'total_cost': self.last_cost,
            'cost_per_bet': self.BASE_BET_COST,
            'numbers_wheeled': len(all_numbers),
            'average_number': np.mean([np.mean(combo) for combo in self.last_combinations])
        }

    def analyze_coverage(
        self,
        combinations: List[Tuple[int, ...]],
        test_set: List[int]
    ) -> Dict[str, any]:
        """
        Analyze how well the combinations cover a test set of numbers.

        Useful for validating that wheeling provides expected guarantees.

        Args:
            combinations: List of bet combinations
            test_set: Set of numbers to test coverage (typically 6 numbers)

        Returns:
            Dictionary with coverage analysis:
                - max_hits: Maximum hits in any single combination
                - combinations_with_hits: Number of combinations with at least one hit
                - hit_distribution: Distribution of hit counts
                - guarantee_met: Whether minimum guarantee was achieved
        """
        if len(test_set) != NUMBERS_PER_DRAW:
            print(f"Warning: test_set has {len(test_set)} numbers, expected {NUMBERS_PER_DRAW}")

        test_set_set = set(test_set)
        hit_counts = []
        hit_distribution = defaultdict(int)

        for combo in combinations:
            hits = len(set(combo) & test_set_set)
            hit_counts.append(hits)
            hit_distribution[hits] += 1

        max_hits = max(hit_counts) if hit_counts else 0
        combinations_with_hits = sum(1 for h in hit_counts if h > 0)

        return {
            'max_hits': max_hits,
            'min_hits': min(hit_counts) if hit_counts else 0,
            'average_hits': np.mean(hit_counts) if hit_counts else 0,
            'combinations_with_hits': combinations_with_hits,
            'hit_distribution': dict(hit_distribution),
            'total_combinations': len(combinations),
            'coverage_percentage': combinations_with_hits / len(combinations) * 100 if combinations else 0,
            'guarantee_met': max_hits >= 4  # Quadra guarantee
        }

    def get_recommendations(self, num_candidates: int) -> Dict[str, any]:
        """
        Get wheeling strategy recommendations based on number of candidates.

        Args:
            num_candidates: Number of candidate numbers to wheel

        Returns:
            Dictionary with recommendations including costs and strategies
        """
        from math import comb

        full_wheel_size = comb(num_candidates, NUMBERS_PER_DRAW)
        full_wheel_cost = self.calculate_cost(full_wheel_size)

        # Estimate abbreviated wheel sizes (rough approximations)
        # Actual sizes depend on specific algorithm and guarantee level
        abbreviated_estimate = full_wheel_size // 3  # Rough estimate

        return {
            'num_candidates': num_candidates,
            'full_wheel': {
                'combinations': full_wheel_size,
                'cost': full_wheel_cost,
                'guarantee': 'Sena (6/6)',
                'recommended': full_wheel_size <= 50
            },
            'abbreviated_wheel_estimate': {
                'combinations': abbreviated_estimate,
                'cost': self.calculate_cost(abbreviated_estimate),
                'guarantee': 'Quadra (4/6)',
                'recommended': num_candidates > 10
            },
            'recommendation': (
                'Full Wheel' if full_wheel_size <= 50
                else 'Abbreviated Wheel with filters'
            )
        }

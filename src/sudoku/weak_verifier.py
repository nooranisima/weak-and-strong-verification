"""
Rule-based weak verifier for 4x4 Sudoku.

Scores cell placements using constraint propagation to count valid candidates.
Score = 1/k where k = number of valid candidates (perfectly calibrated).
"""

from typing import Dict, List, Set

from .config import SudokuConfig


def _candidates_4x4(grid: List[List[int]], r: int, c: int) -> Set[int]:
    """Compute valid candidates for cell (r, c) using Sudoku constraints."""
    if grid[r][c] != 0:
        return set()
    vals = {1, 2, 3, 4}
    # Row constraint
    vals -= set(grid[r])
    # Column constraint
    vals -= {grid[i][c] for i in range(4)}
    # Box constraint
    br, bc = (r // 2) * 2, (c // 2) * 2
    vals -= {grid[i][j] for i in range(br, br + 2) for j in range(bc, bc + 2)}
    vals.discard(0)
    return vals


class WeakVerifier:
    """
    Rule-based weak verifier.

    Returns a calibrated confidence proxy based on the number of valid
    candidates at a given cell:
        - 0 candidates or illegal move → 0.0
        - 1 candidate (forced)        → 1.0
        - k candidates                → 1/k
    """

    def __init__(self, config: SudokuConfig):
        self.config = config
        self.call_count = 0

    def score_cell_placement(
        self,
        original_puzzle: List[List[int]],
        previous_steps: List[Dict],
        row: int,
        col: int,
        value: int,
        generator_reasoning: str = "",  # accepted but unused (interface compat)
    ) -> float:
        """
        Score a proposed cell placement.

        Args:
            original_puzzle: The original puzzle with 0s for empty cells.
            previous_steps: Previous moves [{"row": int, "col": int, "value": int}, ...].
            row, col, value: The proposed move.
            generator_reasoning: Ignored (kept for interface compatibility with
                tree generation code that passes this keyword arg).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        self.call_count += 1

        # Reconstruct current state
        current_state = [r[:] for r in original_puzzle]
        for step in previous_steps:
            current_state[step["row"]][step["col"]] = step["value"]

        # Cell already filled → invalid
        if current_state[row][col] != 0:
            return 0.0

        cands = _candidates_4x4(current_state, row, col)

        if value not in cands:
            return 0.0  # Illegal move

        k = len(cands)
        if k == 1:
            return 1.0  # Forced move

        return float(1.0 / k)

    def score_complete_solution(
        self, puzzle: List[List[int]], solution: List[List[int]]
    ) -> float:
        """Rule-check a complete solution (binary: 1.0 or 0.0)."""
        self.call_count += 1

        # Check clues preserved
        for r in range(4):
            for c in range(4):
                if puzzle[r][c] != 0 and solution[r][c] != puzzle[r][c]:
                    return 0.0

        target = {1, 2, 3, 4}
        for r in range(4):
            if set(solution[r]) != target:
                return 0.0
        for c in range(4):
            if {solution[r][c] for r in range(4)} != target:
                return 0.0

        for br in (0, 2):
            for bc in (0, 2):
                box = {
                    solution[r][c]
                    for r in range(br, br + 2)
                    for c in range(bc, bc + 2)
                }
                if box != target:
                    return 0.0

        return 1.0

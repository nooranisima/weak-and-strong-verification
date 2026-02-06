"""
Strong verifier for 4x4 Sudoku.

Deterministic checker: validates Sudoku rules AND matches ground truth.
"""

from typing import List, Tuple


class StrongVerifier:
    """Deterministic Sudoku verifier — checks validity AND correctness."""

    @staticmethod
    def verify_cell_correctness(
        true_solution: List[List[int]], row: int, col: int, value: int
    ) -> Tuple[bool, str]:
        """Check if a proposed value matches the ground truth."""
        true_value = true_solution[row][col]
        if value == true_value:
            return True, f"Correct! {value} matches ground truth"
        return False, f"Incorrect: placed {value} but should be {true_value}"

    @staticmethod
    def verify_complete_solution(
        puzzle: List[List[int]], solution: List[List[int]]
    ) -> Tuple[bool, str]:
        """Verify if a complete solution is correct."""
        if len(solution) != 4 or any(len(row) != 4 for row in solution):
            return False, "Grid is not 4x4"

        for r in range(4):
            for c in range(4):
                if solution[r][c] not in [1, 2, 3, 4]:
                    return False, f"Invalid value at ({r},{c}): {solution[r][c]}"

        for r in range(4):
            for c in range(4):
                if puzzle[r][c] != 0 and puzzle[r][c] != solution[r][c]:
                    return (
                        False,
                        f"Doesn't match clue at ({r},{c}): "
                        f"expected {puzzle[r][c]}, got {solution[r][c]}",
                    )

        for r in range(4):
            if sorted(solution[r]) != [1, 2, 3, 4]:
                return False, f"Row {r} invalid: {solution[r]}"

        for c in range(4):
            col_vals = [solution[r][c] for r in range(4)]
            if sorted(col_vals) != [1, 2, 3, 4]:
                return False, f"Column {c} invalid: {col_vals}"

        for box_r in range(2):
            for box_c in range(2):
                box = []
                for r in range(box_r * 2, box_r * 2 + 2):
                    for c in range(box_c * 2, box_c * 2 + 2):
                        box.append(solution[r][c])
                if sorted(box) != [1, 2, 3, 4]:
                    return False, f"Box ({box_r},{box_c}) invalid: {box}"

        return True, "Solution is correct!"

    @staticmethod
    def verify_partial_solution(
        puzzle: List[List[int]], current_grid: List[List[int]]
    ) -> Tuple[bool, str]:
        """Verify a partial solution has no conflicts."""
        if len(current_grid) != 4 or any(len(row) != 4 for row in current_grid):
            return False, "Grid is not 4x4"

        for r in range(4):
            for c in range(4):
                if current_grid[r][c] not in [0, 1, 2, 3, 4]:
                    return False, f"Invalid value at ({r},{c}): {current_grid[r][c]}"

        for r in range(4):
            for c in range(4):
                if puzzle[r][c] != 0 and current_grid[r][c] != 0:
                    if puzzle[r][c] != current_grid[r][c]:
                        return False, f"Conflicts with clue at ({r},{c})"

        for r in range(4):
            filled = [val for val in current_grid[r] if val != 0]
            if len(filled) != len(set(filled)):
                return False, f"Duplicate in row {r}"

        for c in range(4):
            filled = [current_grid[r][c] for r in range(4) if current_grid[r][c] != 0]
            if len(filled) != len(set(filled)):
                return False, f"Duplicate in column {c}"

        for box_r in range(2):
            for box_c in range(2):
                filled = []
                for r in range(box_r * 2, box_r * 2 + 2):
                    for c in range(box_c * 2, box_c * 2 + 2):
                        if current_grid[r][c] != 0:
                            filled.append(current_grid[r][c])
                if len(filled) != len(set(filled)):
                    return False, f"Duplicate in box ({box_r},{box_c})"

        return True, "Partial solution is valid"

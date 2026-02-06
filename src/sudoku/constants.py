"""
Sudoku constants, rules, and display utilities.
"""

from typing import List


# ============================================================================
# 4x4 Mini Sudoku Rules
# ============================================================================

SUDOKU_RULES = """4x4 Mini Sudoku Rules:
- The grid is 4x4, divided into four 2x2 boxes
- Fill each cell with a number from 1 to 4
- Each ROW must contain the numbers 1, 2, 3, 4 exactly once
- Each COLUMN must contain the numbers 1, 2, 3, 4 exactly once
- Each 2x2 BOX must contain the numbers 1, 2, 3, 4 exactly once
- Some cells are given as clues and cannot be changed
"""


# ============================================================================
# In-Context Example (used by generator)
# ============================================================================

EXAMPLE_PUZZLES = [
    {
        "puzzle": [[1, 0, 3, 0], [2, 0, 4, 1], [0, 2, 0, 4], [4, 3, 2, 0]],
        "solution": [[1, 2, 3, 4], [2, 3, 4, 1], [3, 1, 2, 4], [4, 3, 2, 1]],
        "explanation": (
            "Row 1 has 1,_ ,3,_ so missing {2,4}. Column 2 already has 2 "
            "so (0,1)=2, hence (0,3)=4. Continue similarly."
        ),
    },
]


# ============================================================================
# Display Utility
# ============================================================================


def format_grid(grid: List[List[int]], show_zeros: bool = True) -> str:
    """
    Format a 4x4 grid for display.

    Args:
        grid: 4x4 list of ints (0 = empty)
        show_zeros: If True, show 0s as '.'; if False, show raw numbers.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    for i, row in enumerate(grid):
        if show_zeros:
            row_str = " ".join(str(cell) if cell != 0 else "." for cell in row)
        else:
            row_str = " ".join(str(cell) for cell in row)
        lines.append(row_str)
        if i == 1:  # Separator after 2nd row
            lines.append("-" * 7)
    return "\n".join(lines)

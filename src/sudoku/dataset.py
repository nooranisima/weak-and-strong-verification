"""
Sudoku dataset loading from HuggingFace.

Uses the asadshahab/mini-sudoku dataset (4x4 Sudoku puzzles).
"""

from typing import Dict, List

from .constants import format_grid


def load_sudoku_dataset_hf(num_problems: int = 50) -> List[Dict]:
    """
    Load 4x4 Sudoku dataset from HuggingFace.

    Args:
        num_problems: Maximum number of puzzles to load.

    Returns:
        List of problem dicts with keys:
            id, puzzle, solution, puzzle_str, solution_str
    """
    from datasets import load_dataset

    print(f"Loading dataset from HuggingFace...")

    try:
        hf_dataset = load_dataset("asadshahab/mini-sudoku", split="train")

        problems = []
        for i, item in enumerate(hf_dataset):
            if i >= num_problems:
                break

            question_str = item["question"]
            answer_str = item["answer"]

            # Parse question: convert _ to 0
            puzzle_rows = []
            for row_str in question_str.split("\n"):
                row = [int(c) if c.isdigit() else 0 for c in row_str.split()]
                puzzle_rows.append(row)

            # Parse answer
            solution_rows = []
            for row_str in answer_str.split("\n"):
                row = [int(c) for c in row_str.split()]
                solution_rows.append(row)

            # Flat strings for compatibility
            puzzle_str = "".join(
                str(cell) for row in puzzle_rows for cell in row
            )
            solution_str = "".join(
                str(cell) for row in solution_rows for cell in row
            )

            problems.append(
                {
                    "id": f"puzzle_{i}",
                    "puzzle": puzzle_rows,
                    "solution": solution_rows,
                    "puzzle_str": puzzle_str,
                    "solution_str": solution_str,
                }
            )

        print(f"✅ Loaded {len(problems)} puzzles from HuggingFace")
        return problems

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using fallback sample puzzles...")
        return get_sample_puzzles(num_problems)


def get_sample_puzzles(num: int = 5) -> List[Dict]:
    """Fallback: return a few hardcoded sample puzzles."""
    samples = [
        {
            "puzzle": [[1, 2, 3, 4], [2, 0, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]],
            "solution": [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]],
        },
        {
            "puzzle": [[2, 3, 4, 1], [1, 4, 3, 2], [4, 1, 2, 3], [3, 2, 1, 4]],
            "solution": [[2, 3, 4, 1], [1, 4, 3, 2], [4, 1, 2, 3], [3, 2, 1, 4]],
        },
        {
            "puzzle": [[3, 2, 1, 4], [1, 4, 3, 2], [2, 3, 4, 1], [4, 1, 2, 3]],
            "solution": [[3, 2, 1, 4], [1, 4, 3, 2], [2, 3, 4, 1], [4, 1, 2, 3]],
        },
    ]

    return [
        {**s, "id": f"sample_{i}", "puzzle_str": "", "solution_str": ""}
        for i, s in enumerate(samples[:num])
    ]

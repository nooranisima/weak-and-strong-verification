"""MATH dataset loading utilities."""

from typing import List, Dict
from datasets import load_dataset


def load_math_dataset_by_difficulty(
    num_problems: int = 50,
    difficulty_levels: List[int] = None,
) -> List[Dict]:
    """
    Load MATH dataset filtered by specific difficulty levels.

    Args:
        num_problems: Number of problems to load
        difficulty_levels: List of levels to include (e.g., [1, 2] for easy)

    Returns:
        List of problem dictionaries
    """
    print(f"Loading MATH dataset (levels {difficulty_levels})...")
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")

    if difficulty_levels:
        dataset = dataset.filter(lambda x: x["level"] in difficulty_levels)

    problems = []
    for i, item in enumerate(dataset):
        if i >= num_problems:
            break
        problems.append({
            "id": item.get("unique_id", f"problem_{i}"),
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": item["answer"],
            "level": item["level"],
            "subject": item["subject"],
        })

    print(f"✓ Loaded {len(problems)} problems (levels {difficulty_levels})")
    return problems


def load_all_difficulty_datasets(
    num_problems_per_level: int = 50,
) -> Dict[str, List[Dict]]:
    """
    Load three datasets split by difficulty.

    Returns:
        Dictionary with keys 'easy', 'medium', 'hard'
    """
    datasets = {
        "easy": load_math_dataset_by_difficulty(num_problems_per_level, [1, 2]),
        "medium": load_math_dataset_by_difficulty(num_problems_per_level, [3, 4]),
        "hard": load_math_dataset_by_difficulty(num_problems_per_level, [5]),
    }

    print(f"\n✓ Loaded all datasets:")
    for name, data in datasets.items():
        print(f"  {name}: {len(data)} problems")

    return datasets


def sample_and_save_datasets(
    seed: int = 42,
    examples_per_difficulty: int = 200,
    difficulty_levels: List[int] = None,
) -> Dict[int, List[Dict]]:
    """
    Sample datasets by difficulty level (keyed by integer level).

    Args:
        seed: Random seed
        examples_per_difficulty: Number of problems per level
        difficulty_levels: Which difficulty levels to load

    Returns:
        {difficulty_int: [problem_dicts]}
    """
    import numpy as np

    np.random.seed(seed)

    if difficulty_levels is None:
        difficulty_levels = [2, 3, 5]

    datasets = {}
    for level in difficulty_levels:
        problems = load_math_dataset_by_difficulty(
            num_problems=examples_per_difficulty,
            difficulty_levels=[level],
        )
        datasets[level] = problems

    return datasets

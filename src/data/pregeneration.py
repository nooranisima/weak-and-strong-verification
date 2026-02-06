"""Pre-generate solutions, weak scores, and ground truth labels."""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


def print_pregeneration_stats(pregenerated_data: dict):
    """Print summary statistics for pre-generated data."""
    print(f"\n--- Summary Statistics ---")

    for difficulty in pregenerated_data["data"]:
        problems = pregenerated_data["data"][difficulty]

        all_weak_scores = []
        all_correct = []
        problems_with_at_least_one_correct = 0

        for prob_idx, prob_data in problems.items():
            attempts = prob_data["attempts"]
            for a in attempts:
                all_weak_scores.append(a["weak_score"])
                all_correct.append(a["is_correct"])
            if any(a["is_correct"] for a in attempts):
                problems_with_at_least_one_correct += 1

        n_problems = len(problems)
        n_attempts = len(all_weak_scores)

        print(f"\nDifficulty {difficulty}:")
        print(f"  Problems: {n_problems}")
        print(f"  Total attempts: {n_attempts}")
        print(
            f"  Overall accuracy: {100*sum(all_correct)/len(all_correct):.1f}%"
        )
        print(
            f"  Problems solvable (≥1 correct): "
            f"{problems_with_at_least_one_correct}/{n_problems} "
            f"({100*problems_with_at_least_one_correct/n_problems:.1f}%)"
        )
        print(
            f"  Weak score - mean: {np.mean(all_weak_scores):.3f}, "
            f"std: {np.std(all_weak_scores):.3f}"
        )

        if len(set(all_correct)) > 1:
            correct_scores = [
                s for s, c in zip(all_weak_scores, all_correct) if c
            ]
            incorrect_scores = [
                s for s, c in zip(all_weak_scores, all_correct) if not c
            ]
            print(
                f"  Weak score | correct: {np.mean(correct_scores):.3f}, "
                f"incorrect: {np.mean(incorrect_scores):.3f}"
            )


def pregenerate_all_data(
    datasets: Dict[int, List[dict]],
    llm,  # LLMWrapper
    num_problems: int = None,
    num_attempts: int = 5,
    difficulties: List[int] = None,
    save_dir: str = "./results",
    checkpoint_freq: int = 5,
    random_seed: int = None,
) -> dict:
    """
    Pre-generate all solutions, weak scores, and ground truth labels.
    Saves checkpoints automatically.

    Args:
        datasets: {difficulty_level: [problem_dicts]}
        llm: LLMWrapper instance
        num_problems: Max problems per difficulty (None = all)
        num_attempts: Number of solution attempts per problem
        difficulties: Which difficulty levels to process
        save_dir: Directory for saving results
        checkpoint_freq: Save checkpoint every N problems
        random_seed: Random seed for reproducibility

    Returns:
        Pre-generated data dictionary
    """
    os.makedirs(save_dir, exist_ok=True)

    if random_seed is not None:
        np.random.seed(random_seed)

    if difficulties is None:
        difficulties = list(datasets.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"{save_dir}/pregenerated_{timestamp}.json"
    checkpoint_path = f"{save_dir}/pregenerated_{timestamp}_checkpoint.json"

    pregenerated_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_attempts": num_attempts,
            "num_problems_per_difficulty": num_problems,
            "difficulties": difficulties,
            "generator_model": llm.config.generator_model,
            "strong_verifier_model": llm.config.strong_verifier_model,
            "weak_verifier": (
                type(llm.weak_verifier).__name__ if llm.weak_verifier else None
            ),
            "random_seed": random_seed,
            "save_path": final_path,
        },
        "data": {},
    }

    total_problems = 0
    total_attempts = 0
    start_time = time.time()

    for difficulty in difficulties:
        print(f"\n{'='*60}")
        print(f"DIFFICULTY {difficulty}")
        print(f"{'='*60}")

        problems = datasets[difficulty]
        if num_problems:
            problems = problems[:num_problems]

        pregenerated_data["data"][difficulty] = {}

        for prob_idx, problem in enumerate(problems):
            prob_data = {
                "problem_id": problem.get("id", f"diff{difficulty}_prob{prob_idx}"),
                "problem": problem["problem"],
                "ground_truth_answer": problem["answer"],
                "attempts": [],
            }

            for attempt_idx in range(num_attempts):
                reasoning, generated_answer, weak_score = llm.generate_solution(
                    problem["problem"], temperature=0.7
                )

                is_correct, explanation = llm.verify_answer(
                    problem=problem["problem"],
                    reasoning=reasoning,
                    generated_answer=generated_answer,
                    ground_truth_answer=problem["answer"],
                    is_ground_truth_call=True,
                )

                prob_data["attempts"].append({
                    "attempt_idx": attempt_idx,
                    "reasoning": reasoning,
                    "generated_answer": generated_answer,
                    "weak_score": weak_score,
                    "is_correct": is_correct,
                })

                total_attempts += 1

            pregenerated_data["data"][difficulty][prob_idx] = prob_data
            total_problems += 1

            if (prob_idx + 1) % checkpoint_freq == 0 or prob_idx == len(problems) - 1:
                elapsed = time.time() - start_time
                rate = total_attempts / elapsed if elapsed > 0 else 0

                correct_attempts = sum(
                    1 for a in prob_data["attempts"] if a["is_correct"]
                )
                avg_weak = np.mean([a["weak_score"] for a in prob_data["attempts"]])

                print(
                    f"  Problem {prob_idx+1}/{len(problems)} | "
                    f"Correct: {correct_attempts}/{num_attempts} | "
                    f"Avg weak: {avg_weak:.3f} | "
                    f"Rate: {rate:.1f} attempts/sec"
                )

                with open(checkpoint_path, "w") as f:
                    json.dump(pregenerated_data, f)
                print(f"    → Checkpoint saved")

    elapsed = time.time() - start_time

    pregenerated_data["metadata"]["total_problems"] = total_problems
    pregenerated_data["metadata"]["total_attempts"] = total_attempts
    pregenerated_data["metadata"]["generation_time_seconds"] = elapsed

    print(f"\n{'='*60}")
    print(f"PRE-GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total problems: {total_problems}")
    print(f"Total attempts: {total_attempts}")
    print(f"Time: {elapsed:.1f}s ({total_attempts/elapsed:.1f} attempts/sec)")

    print_pregeneration_stats(pregenerated_data)

    with open(final_path, "w") as f:
        json.dump(pregenerated_data, f, indent=2)
    print(f"\n✓ Final data saved to: {final_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint cleaned up")

    return pregenerated_data


def append_pregenerated_data(
    existing_path: str,
    datasets: Dict[int, List[dict]],
    llm,  # LLMWrapper
    num_additional_problems: int,
    num_attempts: int = 5,
    difficulties: List[int] = None,
    checkpoint_freq: int = 10,
    random_seed: int = None,
) -> dict:
    """
    Append more problems to existing pre-generated data.

    Args:
        existing_path: Path to existing pregenerated JSON file
        datasets: {difficulty: [problems]}
        llm: LLM wrapper
        num_additional_problems: How many MORE problems to add per difficulty
        num_attempts: Number of attempts per problem
        difficulties: Which difficulties (None = same as existing)
        checkpoint_freq: Save checkpoint every N problems
        random_seed: Random seed for reproducibility

    Returns:
        Updated pregenerated_data dict
    """
    print(f"Loading existing data from: {existing_path}")
    with open(existing_path, "r") as f:
        pregenerated_data = json.load(f)

    # Convert keys to integers
    new_data = {}
    for diff_key, diff_val in pregenerated_data["data"].items():
        diff_int = int(diff_key)
        new_data[diff_int] = {}
        for prob_key, prob_val in diff_val.items():
            prob_int = int(prob_key)
            new_data[diff_int][prob_int] = prob_val
    pregenerated_data["data"] = new_data

    existing_counts = {}
    for diff in pregenerated_data["data"]:
        existing_counts[diff] = len(pregenerated_data["data"][diff])

    print(f"Existing data:")
    for diff, count in existing_counts.items():
        print(f"  Difficulty {diff}: {count} problems")

    if difficulties is None:
        difficulties = list(pregenerated_data["data"].keys())

    if random_seed is not None:
        np.random.seed(random_seed)

    # Create backup
    backup_path = existing_path.replace(".json", "_backup.json")
    with open(backup_path, "w") as f:
        json.dump(pregenerated_data, f)
    print(f"✓ Backup saved to: {backup_path}")

    total_new_problems = 0
    total_new_attempts = 0
    start_time = time.time()

    for difficulty in difficulties:
        print(f"\n{'='*60}")
        print(f"DIFFICULTY {difficulty} - Adding {num_additional_problems} more problems")
        print(f"{'='*60}")

        start_idx = existing_counts.get(difficulty, 0)
        all_problems = datasets[difficulty]
        end_idx = start_idx + num_additional_problems

        if end_idx > len(all_problems):
            print(
                f"  ⚠️ Only {len(all_problems) - start_idx} problems remaining"
            )
            end_idx = len(all_problems)

        new_problems = all_problems[start_idx:end_idx]
        print(f"  Using problems {start_idx} to {end_idx-1} from dataset")

        if difficulty not in pregenerated_data["data"]:
            pregenerated_data["data"][difficulty] = {}

        for i, problem in enumerate(new_problems):
            prob_idx = start_idx + i

            prob_data = {
                "problem_id": problem.get("id", f"diff{difficulty}_prob{prob_idx}"),
                "problem": problem["problem"],
                "ground_truth_answer": problem["answer"],
                "attempts": [],
            }

            for attempt_idx in range(num_attempts):
                reasoning, generated_answer, weak_score = llm.generate_solution(
                    problem["problem"], temperature=0.7
                )

                is_correct, explanation = llm.verify_answer(
                    problem=problem["problem"],
                    reasoning=reasoning,
                    generated_answer=generated_answer,
                    ground_truth_answer=problem["answer"],
                    is_ground_truth_call=True,
                )

                prob_data["attempts"].append({
                    "attempt_idx": attempt_idx,
                    "reasoning": reasoning,
                    "generated_answer": generated_answer,
                    "weak_score": weak_score,
                    "is_correct": is_correct,
                })

                total_new_attempts += 1

            pregenerated_data["data"][difficulty][prob_idx] = prob_data
            total_new_problems += 1

            if (i + 1) % checkpoint_freq == 0 or i == len(new_problems) - 1:
                elapsed = time.time() - start_time
                rate = total_new_attempts / elapsed if elapsed > 0 else 0

                correct_attempts = sum(
                    1 for a in prob_data["attempts"] if a["is_correct"]
                )
                avg_weak = np.mean([a["weak_score"] for a in prob_data["attempts"]])

                print(
                    f"  Problem {prob_idx+1} (new: {i+1}/{len(new_problems)}) | "
                    f"Correct: {correct_attempts}/{num_attempts} | "
                    f"Avg weak: {avg_weak:.3f} | "
                    f"Rate: {rate:.1f} attempts/sec"
                )

                with open(existing_path, "w") as f:
                    json.dump(pregenerated_data, f)

    # Update metadata
    pregenerated_data["metadata"]["total_problems"] = sum(
        len(pregenerated_data["data"][d]) for d in pregenerated_data["data"]
    )
    pregenerated_data["metadata"]["total_attempts"] = (
        pregenerated_data["metadata"]["total_problems"] * num_attempts
    )
    pregenerated_data["metadata"]["last_updated"] = datetime.now().isoformat()
    pregenerated_data["metadata"].setdefault("append_history", [])
    pregenerated_data["metadata"]["append_history"].append({
        "timestamp": datetime.now().isoformat(),
        "added_problems": total_new_problems,
        "added_attempts": total_new_attempts,
    })

    with open(existing_path, "w") as f:
        json.dump(pregenerated_data, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"APPEND COMPLETE")
    print(f"{'='*60}")
    print(f"Added: {total_new_problems} problems, {total_new_attempts} attempts")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nNew totals:")
    for diff in pregenerated_data["data"]:
        count = len(pregenerated_data["data"][diff])
        print(f"  Difficulty {diff}: {count} problems")
    print(f"\n✓ Saved to: {existing_path}")
    print(f"✓ Backup at: {backup_path}")

    print_pregeneration_stats(pregenerated_data)

    return pregenerated_data

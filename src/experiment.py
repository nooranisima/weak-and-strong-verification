"""Run full experiment on pre-generated data."""

import json
from datetime import datetime
from typing import List, Tuple, Dict

from .algorithms import SimulatedAdaptiveRunner, SimulatedStrongBaseline, SimulatedWeakBaseline
from .utils import convert_results_for_json


def run_simulated_experiment(
    pregenerated_data: dict,
    alpha_beta_combinations: List[Tuple[float, float]],
    config: dict,
    save_path: str = None,
) -> dict:
    """
    Run all algorithm configurations on pre-generated data.

    Args:
        pregenerated_data: Output from pregenerate_all_data()
        alpha_beta_combinations: List of (alpha, beta) tuples to test
        config: Dict with algorithm hyperparameters
        save_path: Optional path to save results JSON

    Returns:
        Results dict with strong_baseline, weak_baseline, adaptive_warmup, adaptive_final
    """
    difficulties = list(pregenerated_data["data"].keys())
    max_attempts = config.get("max_attempts", 5)

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pregeneration_timestamp": pregenerated_data["metadata"]["timestamp"],
            "alpha_beta_combinations": alpha_beta_combinations,
            "config": config,
        },
        "strong_baseline": {},
        "weak_baseline": {},
        "adaptive_warmup": {},
        "adaptive_final": {},
    }

    # ========================================================================
    # 1. STRONG BASELINE
    # ========================================================================
    print("\n" + "=" * 60)
    print("STRONG BASELINE")
    print("=" * 60)

    for difficulty in difficulties:
        problems = pregenerated_data["data"][difficulty]
        runner = SimulatedStrongBaseline(max_attempts=max_attempts)

        records = []
        for prob_idx in sorted(problems.keys(), key=int):
            result = runner.solve_problem(problems[prob_idx])
            records.append(result)

        correct = sum(1 for r in records if r["final_answer_correct"])
        strong_calls = sum(r["num_strong_verifier_calls"] for r in records)

        results["strong_baseline"][difficulty] = {
            "problem_records": records,
            "summary": {
                "total": len(records),
                "correct": correct,
                "accuracy": correct / len(records),
                "total_strong_calls": strong_calls,
            },
        }

        print(
            f"  Difficulty {difficulty}: {correct}/{len(records)} "
            f"({100*correct/len(records):.1f}%), {strong_calls} strong calls"
        )

    # ========================================================================
    # 2. WEAK BASELINE
    # ========================================================================
    print("\n" + "=" * 60)
    print("WEAK BASELINE (threshold=0.5)")
    print("=" * 60)

    for difficulty in difficulties:
        problems = pregenerated_data["data"][difficulty]
        runner = SimulatedWeakBaseline(
            threshold=config.get("weak_baseline_threshold", 0.5),
            max_attempts=max_attempts,
        )

        records = []
        for prob_idx in sorted(problems.keys(), key=int):
            result = runner.solve_problem(problems[prob_idx])
            records.append(result)

        correct = sum(1 for r in records if r["final_answer_correct"])

        results["weak_baseline"][difficulty] = {
            "problem_records": records,
            "summary": {
                "total": len(records),
                "correct": correct,
                "accuracy": correct / len(records),
                "total_strong_calls": 0,
            },
        }

        print(
            f"  Difficulty {difficulty}: {correct}/{len(records)} "
            f"({100*correct/len(records):.1f}%), 0 strong calls"
        )

    # ========================================================================
    # 3. ADAPTIVE (warmup + final for each α, β)
    # ========================================================================
    print("\n" + "=" * 60)
    print("ADAPTIVE ALGORITHM")
    print("=" * 60)

    for difficulty in difficulties:
        print(f"\n--- Difficulty {difficulty} ---")
        problems = pregenerated_data["data"][difficulty]
        prob_keys = sorted(problems.keys(), key=int)

        results["adaptive_warmup"][difficulty] = {}
        results["adaptive_final"][difficulty] = {}

        for alpha, beta in alpha_beta_combinations:
            print(f"  α={alpha}, β={beta}:")

            adaptive_config = config.copy()
            adaptive_config["alpha"] = alpha
            adaptive_config["beta"] = beta

            # WARMUP phase
            warmup_runner = SimulatedAdaptiveRunner(adaptive_config)
            warmup_records = []

            for prob_idx in prob_keys:
                result = warmup_runner.solve_problem(problems[prob_idx])
                warmup_records.append(result)

            learned_thresholds = warmup_runner.get_thresholds()

            warmup_correct = sum(
                1 for r in warmup_records if r["final_answer_correct"]
            )
            warmup_calls = sum(
                r["num_strong_verifier_calls"] for r in warmup_records
            )

            results["adaptive_warmup"][difficulty][(alpha, beta)] = {
                "problem_records": warmup_records,
                "learned_thresholds": learned_thresholds,
                "tau_A_history": warmup_runner.tau_A_history,
                "tau_R_history": warmup_runner.tau_R_history,
                "empirical_errors": warmup_runner.get_empirical_errors(),
                "summary": {
                    "correct": warmup_correct,
                    "accuracy": warmup_correct / len(warmup_records),
                    "total_strong_calls": warmup_calls,
                },
            }

            print(
                f"    Warmup: τ_A={learned_thresholds['tau_A']:.3f}, "
                f"τ_R={learned_thresholds['tau_R']:.3f}"
            )

            # FINAL phase (use learned thresholds)
            final_runner = SimulatedAdaptiveRunner(adaptive_config)
            final_runner.set_thresholds(
                learned_thresholds["tau_A"], learned_thresholds["tau_R"]
            )
            final_runner.reset_tracking_only()

            final_records = []
            for prob_idx in prob_keys:
                result = final_runner.solve_problem(problems[prob_idx])
                final_records.append(result)

            final_correct = sum(
                1 for r in final_records if r["final_answer_correct"]
            )
            final_calls = sum(
                r["num_strong_verifier_calls"] for r in final_records
            )

            results["adaptive_final"][difficulty][(alpha, beta)] = {
                "problem_records": final_records,
                "initial_thresholds": learned_thresholds.copy(),
                "final_thresholds": final_runner.get_thresholds(),
                "tau_A_history": final_runner.tau_A_history,
                "tau_R_history": final_runner.tau_R_history,
                "empirical_errors": final_runner.get_empirical_errors(),
                "summary": {
                    "total": len(final_records),
                    "correct": final_correct,
                    "accuracy": final_correct / len(final_records),
                    "total_strong_calls": final_calls,
                },
            }

            print(
                f"    Final:  {final_correct}/{len(final_records)} "
                f"({100*final_correct/len(final_records):.1f}%), "
                f"{final_calls} strong calls"
            )

    # Save results
    if save_path:
        serializable = convert_results_for_json(results)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n✓ Results saved to {save_path}")

    return results

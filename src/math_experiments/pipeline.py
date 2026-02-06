"""Grid search, full pipeline, and accuracy-vs-latency plotting for MATH experiments."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

from ..algorithms import SimulatedAdaptiveRunnerWithRNG


# ============================================================================
# Constants
# ============================================================================

DIFF_COLORS = {2: "green", 3: "blue", 5: "red"}
DIFF_NAMES = {2: "Easy", 3: "Medium", 5: "Hard"}
DIFF_LABELS = {2: "Easy (Level 2)", 3: "Medium (Level 3)", 5: "Hard (Level 5)"}
DIFF_FILENAMES = {2: "easy", 3: "medium", 5: "hard"}

WEAK_COLORS = {
    1: "#FF6B6B",
    2: "#4ECDC4",
    3: "#9B59B6",
    4: "#F39C12",
    5: "#1ABC9C",
}


# ============================================================================
# Grid search
# ============================================================================


def grid_search_for_config(
    pregenerated_data: dict,
    difficulty: int,
    alpha: float,
    beta: float,
    eta_values: list = None,
    eta_R_values: list = None,
    tau_A_offsets: list = None,
    tau_R_offsets: list = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Grid search to find best config for a specific (α, β, difficulty).

    Searches over learning rates and initial threshold offsets.
    Returns dict with best eta, eta_R, tau_A_init, tau_R_init and final errors.
    """
    if eta_values is None:
        eta_values = [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.17, 0.2,
        ]
    if eta_R_values is None:
        eta_R_values = [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.17, 0.2,
        ]
    if tau_A_offsets is None:
        tau_A_offsets = [-0.1, -0.08, -0.05, -0.04, -0.02, 0, 0.02, 0.04, 0.05, 0.08, 0.1]
    if tau_R_offsets is None:
        tau_R_offsets = [-0.1, -0.08, -0.05, -0.04, -0.02, 0, 0.02, 0.04, 0.05, 0.08, 0.1]

    problems = pregenerated_data["data"][difficulty]
    prob_keys = sorted(problems.keys())

    tau_A_base = 1 - alpha
    tau_R_base = beta

    tau_A_values = [np.clip(tau_A_base + o, 0.1, 0.99) for o in tau_A_offsets]
    tau_R_values = [np.clip(tau_R_base + o, 0.01, 0.9) for o in tau_R_offsets]

    best_score = float("inf")
    best_config = None

    for eta in eta_values:
        for eta_R in eta_R_values:
            for tau_A_init in tau_A_values:
                for tau_R_init in tau_R_values:
                    if tau_R_init >= tau_A_init - 0.05:
                        continue

                    rng = np.random.RandomState(seed * 100 + difficulty)

                    config = {
                        "max_attempts": 5,
                        "eta": eta,
                        "eta_R": eta_R,
                        "tau_A_init": tau_A_init,
                        "tau_R_init": tau_R_init,
                        "alpha": alpha,
                        "beta": beta,
                        "P_a_init": 0.3,
                        "P_r_init": 0.3,
                        "P_a_min": 0.05,
                        "P_r_min": 0.05,
                    }

                    runner = SimulatedAdaptiveRunnerWithRNG(config, rng)

                    for prob_idx in prob_keys:
                        runner.solve_problem(problems[int(prob_idx)])

                    errors = runner.get_empirical_errors()
                    ae = errors["accept_error"]
                    re = errors["reject_error"]

                    score = abs(ae - alpha) + abs(re - beta)

                    if score < best_score:
                        best_score = score
                        best_config = {
                            "eta": eta,
                            "eta_R": eta_R,
                            "tau_A_init": tau_A_init,
                            "tau_R_init": tau_R_init,
                            "accept_error": ae,
                            "reject_error": re,
                            "score": score,
                        }

    if verbose and best_config:
        print(
            f"    Diff {difficulty}: η={best_config['eta']:.2f}, "
            f"η_R={best_config['eta_R']:.2f}, "
            f"τ_A₀={best_config['tau_A_init']:.2f}, "
            f"τ_R₀={best_config['tau_R_init']:.2f}, "
            f"ae={best_config['accept_error']:.3f}, "
            f"re={best_config['reject_error']:.3f}"
        )

    return best_config


# ============================================================================
# Full pipeline
# ============================================================================


def run_full_pipeline(
    pregenerated_data: dict,
    alpha_beta_values: List[Tuple[float, float]],
    seed: int = 42,
    n_runs: int = 3,
    verbose: bool = True,
    weak_baseline_n_values: Optional[List[int]] = None,
) -> dict:
    """
    Full pipeline:
    1. Compute baselines (strong + multiple weak Best-of-N)
    2. For each (α, β), run grid search to find best config per difficulty
    3. Run adaptive algorithm with best configs

    Args:
        pregenerated_data: Pre-generated data
        alpha_beta_values: List of (alpha, beta) tuples to test
        seed: Random seed
        n_runs: Number of runs to average
        verbose: Print grid search details
        weak_baseline_n_values: N values for Best-of-N weak baselines (e.g. [1,2,3,5])

    Returns:
        Results dict with strong_baseline, weak_baseline, adaptive, configs
    """
    difficulties = sorted(pregenerated_data["data"].keys())
    max_attempts = 5

    if weak_baseline_n_values is None:
        weak_baseline_n_values = [max_attempts]

    results = {
        "strong_baseline": {},
        "weak_baseline": {},
        "adaptive": {},
        "configs": {},
        "weak_baseline_n_values": weak_baseline_n_values,
    }

    for n_val in weak_baseline_n_values:
        results["weak_baseline"][n_val] = {}

    # ========================================================================
    # Baselines
    # ========================================================================
    print("Computing baselines...")

    for difficulty in difficulties:
        problems = pregenerated_data["data"][difficulty]
        prob_keys = sorted(problems.keys())
        n_problems = len(prob_keys)

        # Strong baseline
        strong_correct = 0
        strong_calls = 0
        strong_weak_calls = 0

        for prob_idx in prob_keys:
            for attempt in problems[prob_idx]["attempts"]:
                strong_calls += 1
                strong_weak_calls += 1
                if attempt["is_correct"]:
                    strong_correct += 1
                    break

        results["strong_baseline"][difficulty] = {
            "accuracy": strong_correct / n_problems,
            "strong_calls": strong_calls,
            "weak_calls": strong_weak_calls,
            "calls_per_problem": strong_calls / n_problems,
            "weak_calls_per_problem": strong_weak_calls / n_problems,
            "n_problems": n_problems,
        }

        # Weak Best-of-N for each N value
        for n_val in weak_baseline_n_values:
            rng = np.random.RandomState(seed + n_val)
            bon_correct = 0
            total_weak_calls = 0

            for prob_idx in prob_keys:
                attempts = problems[prob_idx]["attempts"]
                n_att = min(n_val, len(attempts))
                total_weak_calls += n_att

                best_score = -1
                best_indices = []
                for i in range(n_att):
                    score = attempts[i]["weak_score"]
                    if score > best_score:
                        best_score = score
                        best_indices = [i]
                    elif score == best_score:
                        best_indices.append(i)

                chosen_idx = rng.choice(best_indices)
                if attempts[chosen_idx]["is_correct"]:
                    bon_correct += 1

            results["weak_baseline"][n_val][difficulty] = {
                "accuracy": bon_correct / n_problems,
                "strong_calls": 0,
                "weak_calls": total_weak_calls,
                "calls_per_problem": 0,
                "weak_calls_per_problem": total_weak_calls / n_problems,
                "n_problems": n_problems,
            }

        # Summary
        print(
            f"  Difficulty {difficulty}: "
            f"Strong={results['strong_baseline'][difficulty]['accuracy']*100:.1f}% "
            f"@ {results['strong_baseline'][difficulty]['calls_per_problem']:.2f} c/p, ",
            end="",
        )
        for n_val in weak_baseline_n_values:
            acc = results["weak_baseline"][n_val][difficulty]["accuracy"] * 100
            print(f"BoN-{n_val}={acc:.1f}%, ", end="")
        print()

    # ========================================================================
    # Adaptive: grid search + run for each (α, β)
    # ========================================================================
    print("\nRunning grid search + experiments for each (α, β)...")

    for difficulty in difficulties:
        results["adaptive"][difficulty] = {}
        results["configs"][difficulty] = {}

    for alpha, beta in alpha_beta_values:
        print(f"\n  (α={alpha:.2f}, β={beta:.2f}):")

        best_configs = {}
        for difficulty in difficulties:
            best_config = grid_search_for_config(
                pregenerated_data, difficulty, alpha, beta, seed=seed, verbose=verbose
            )
            best_configs[difficulty] = best_config
            results["configs"][difficulty][(alpha, beta)] = best_config

        for difficulty in difficulties:
            problems = pregenerated_data["data"][difficulty]
            prob_keys = sorted(problems.keys())
            n_problems = len(prob_keys)

            best_config = best_configs[difficulty]

            accuracies = []
            strong_calls_list = []
            weak_calls_list = []

            for run_idx in range(n_runs):
                rng = np.random.RandomState(
                    seed * 1000 + difficulty * 100 + run_idx
                )

                config = {
                    "max_attempts": max_attempts,
                    "eta": best_config["eta"],
                    "eta_R": best_config["eta_R"],
                    "tau_A_init": best_config["tau_A_init"],
                    "tau_R_init": best_config["tau_R_init"],
                    "alpha": alpha,
                    "beta": beta,
                    "P_a_init": 0.1,
                    "P_r_init": 0.1,
                    "P_a_min": 0.05,
                    "P_r_min": 0.05,
                }

                runner = SimulatedAdaptiveRunnerWithRNG(config, rng)

                correct = 0
                total_strong_calls = 0
                total_weak_calls = 0

                for prob_idx in prob_keys:
                    result = runner.solve_problem(problems[int(prob_idx)])
                    if result["final_answer_correct"]:
                        correct += 1
                    total_strong_calls += result["num_strong_verifier_calls"]
                    total_weak_calls += result["num_attempts"]

                accuracies.append(correct / n_problems)
                strong_calls_list.append(total_strong_calls)
                weak_calls_list.append(total_weak_calls)

            results["adaptive"][difficulty][(alpha, beta)] = {
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "strong_calls_mean": np.mean(strong_calls_list),
                "strong_calls_std": np.std(strong_calls_list),
                "weak_calls_mean": np.mean(weak_calls_list),
                "weak_calls_std": np.std(weak_calls_list),
                "calls_per_problem_mean": np.mean(strong_calls_list) / n_problems,
                "calls_per_problem_std": np.std(strong_calls_list) / n_problems,
                "weak_calls_per_problem_mean": np.mean(weak_calls_list) / n_problems,
                "weak_calls_per_problem_std": np.std(weak_calls_list) / n_problems,
            }

    return results


# ============================================================================
# Accuracy vs Latency plotting
# ============================================================================


def plot_accuracy_vs_latency(
    results: dict,
    save_dir: str = None,
    filename: str = "accuracy_vs_latency.png",
    difficulties_to_plot: list = None,
    weak_n_values_to_plot: list = None,
    y_padding: float = 3,
):
    """
    Plot accuracy vs strong verifier calls/problem.
    Creates one figure per difficulty.

    Args:
        results: Output from run_full_pipeline
        save_dir: Directory to save figures
        filename: Base filename (difficulty suffix appended)
        difficulties_to_plot: Which difficulties to include
        weak_n_values_to_plot: Which BoN-N baselines to show
        y_padding: Padding around y-axis limits

    Returns:
        Dict of {difficulty: figure}
    """
    all_difficulties = sorted(results["strong_baseline"].keys())

    if difficulties_to_plot is not None:
        difficulties = [d for d in all_difficulties if d in difficulties_to_plot]
    else:
        difficulties = all_difficulties

    available_n_values = sorted(results.get("weak_baseline_n_values", [5]))
    if weak_n_values_to_plot is None:
        weak_n_values_to_plot = available_n_values

    figures = {}

    for difficulty in difficulties:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        color = DIFF_COLORS.get(difficulty, "black")

        strong = results["strong_baseline"][difficulty]
        adaptive = results["adaptive"][difficulty]

        strong_acc = strong["accuracy"] * 100
        strong_cpp = strong["calls_per_problem"]

        all_accs = [strong_acc]

        # Collect adaptive points
        all_points = []
        all_alphas = set()
        all_betas = set()

        for (alpha, beta), data in adaptive.items():
            all_alphas.add(alpha)
            all_betas.add(beta)
            all_points.append({
                "cpp": data["calls_per_problem_mean"],
                "acc": data["accuracy_mean"] * 100,
                "alpha": alpha,
                "beta": beta,
            })

        # Detect varying parameter
        if len(all_alphas) == 1 and len(all_betas) > 1:
            varying_param = "β"
            fixed_param = "α"
            fixed_value = list(all_alphas)[0]
            get_varying = lambda p: p["beta"]
        elif len(all_betas) == 1 and len(all_alphas) > 1:
            varying_param = "α"
            fixed_param = "β"
            fixed_value = list(all_betas)[0]
            get_varying = lambda p: p["alpha"]
        else:
            all_points = [p for p in all_points if p["alpha"] == p["beta"]]
            varying_param = "α=β"
            fixed_param = None
            fixed_value = None
            get_varying = lambda p: p["alpha"]

        # Monotonic filtering: start at peak, keep only decreasing
        all_points.sort(key=lambda x: get_varying(x))

        best_idx = 0
        best_acc = -1
        for i, p in enumerate(all_points):
            if p["acc"] > best_acc:
                best_acc = p["acc"]
                best_idx = i

        filtered_points = [all_points[best_idx]]
        for i in range(best_idx + 1, len(all_points)):
            point = all_points[i]
            last = filtered_points[-1]
            if point["acc"] <= last["acc"] and point["cpp"] <= last["cpp"]:
                filtered_points.append(point)
        filtered_points.sort(key=lambda x: x["cpp"])

        all_accs.extend([p["acc"] for p in filtered_points])

        # Plot weak baselines at x=0
        for n_val in sorted(weak_n_values_to_plot):
            if (
                n_val in results["weak_baseline"]
                and difficulty in results["weak_baseline"][n_val]
            ):
                weak_acc = results["weak_baseline"][n_val][difficulty]["accuracy"] * 100
                all_accs.append(weak_acc)
                ax.scatter(
                    0, weak_acc,
                    color=WEAK_COLORS.get(n_val, "gray"),
                    marker="o", s=120, zorder=9,
                    edgecolors="black", linewidths=0.5,
                    label=f"BoN-{n_val} ({weak_acc:.1f}%)",
                )

        # Strong baseline
        ax.scatter(
            strong_cpp, strong_acc,
            color="black", marker="*", s=400, zorder=10,
            label=f"Strong ({strong_acc:.1f}%)",
        )

        # Adaptive curve
        if filtered_points:
            x = [p["cpp"] for p in filtered_points]
            y = [p["acc"] for p in filtered_points]

            if fixed_param:
                adaptive_label = f"Adaptive ({fixed_param}={fixed_value:.2f})"
            else:
                adaptive_label = "Adaptive"

            ax.plot(
                x, y, "o-", color=color, markersize=9, linewidth=2.5,
                alpha=0.8, label=adaptive_label,
            )

            for i, p in enumerate(filtered_points):
                varying_value = get_varying(p)
                offset_y = 20 if i % 2 == 0 else -26
                ax.annotate(
                    f"{varying_value:.2f}",
                    (p["cpp"], p["acc"]),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    fontsize=12,
                    ha="center",
                    fontweight="bold",
                    arrowprops=dict(
                        arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3,
                    ),
                )

        ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)

        ax.set_xlabel("Strong Verifier Calls / Problem", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")

        title = DIFF_LABELS.get(difficulty, f"Difficulty {difficulty}")
        if fixed_param:
            title += f"\n(varying {varying_param}, {fixed_param}={fixed_value:.2f})"
        ax.set_title(title, fontsize=16, fontweight="bold")

        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

        y_min = max(0, min(all_accs) - y_padding)
        y_max = min(100, max(all_accs) + y_padding)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.1, strong_cpp * 1.1)

        plt.tight_layout()

        if save_dir:
            base = filename.replace(".png", "")
            diff_name = DIFF_FILENAMES.get(difficulty, f"diff{difficulty}")
            path = f"{save_dir}/{base}_{diff_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {path}")

        plt.show()
        figures[difficulty] = fig

    return figures

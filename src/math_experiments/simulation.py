"""Error convergence tracking, running, and plotting for MATH experiments."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from ..algorithms import SimulatedAdaptiveRunnerWithRNG


# ============================================================================
# Constants
# ============================================================================

DIFF_COLORS = {2: "green", 3: "blue", 5: "red"}
DIFF_NAMES = {2: "Easy", 3: "Medium", 5: "Hard"}
DIFF_LABELS = {2: "Easy (Level 2)", 3: "Medium (Level 3)", 5: "Hard (Level 5)"}


# ============================================================================
# Error convergence tracking
# ============================================================================


def run_with_error_tracking(
    pregenerated_data: dict,
    config: dict,
    alpha: float,
    beta: float,
) -> dict:
    """
    Run adaptive algorithm and track running-average error at each step.

    Args:
        pregenerated_data: Pre-generated data dict
        config: Algorithm configuration
        alpha: Target accept error rate
        beta: Target reject error rate

    Returns:
        Dict keyed by difficulty with running errors and threshold histories
    """
    difficulties = list(pregenerated_data["data"].keys())
    results = {}

    for difficulty in difficulties:
        problems = pregenerated_data["data"][difficulty]
        prob_keys = list(problems.keys())
        np.random.shuffle(prob_keys)

        run_config = config.copy()
        run_config["alpha"] = alpha
        run_config["beta"] = beta

        rng = np.random.RandomState(42)
        runner = SimulatedAdaptiveRunnerWithRNG(run_config, rng)

        for prob_idx in prob_keys:
            runner.solve_problem(problems[int(prob_idx)])

        results[difficulty] = _compute_running_errors(runner)

        print(
            f"  Difficulty {difficulty}: {results[difficulty]['n_steps']} steps, "
            f"final accept_err={results[difficulty]['accept_errors_running'][-1]:.3f}, "
            f"final reject_err={results[difficulty]['reject_errors_running'][-1]:.3f}"
        )

    return results


def run_with_error_tracking_per_diff_config(
    pregenerated_data: dict,
    configs: dict,
    alpha: float,
    beta: float,
    seed: int = 42,
) -> dict:
    """
    Run adaptive algorithm with PER-DIFFICULTY configurations.
    Each difficulty is completely independent.

    Args:
        pregenerated_data: Pre-generated data dict
        configs: Dict keyed by difficulty with per-difficulty config overrides
        alpha: Target accept error rate
        beta: Target reject error rate
        seed: Random seed

    Returns:
        Dict keyed by difficulty with running errors and threshold histories
    """
    difficulties = sorted(pregenerated_data["data"].keys())
    results = {}

    for difficulty in difficulties:
        rng = np.random.RandomState(seed * 100 + difficulty)

        if difficulty in configs:
            config = configs[difficulty].copy()
        else:
            config = {
                "max_attempts": 5,
                "eta": 0.05,
                "eta_R": 0.05,
                "tau_A_init": None,
                "tau_R_init": None,
                "P_a_init": 0.3,
                "P_r_init": 0.3,
                "P_a_min": 0.05,
                "P_r_min": 0.05,
            }

        config["alpha"] = alpha
        config["beta"] = beta

        tau_A_init = config.get("tau_A_init") or (1 - alpha)
        tau_R_init = config.get("tau_R_init") or beta

        print(
            f"  Difficulty {difficulty}: "
            f"η={config.get('eta', 0.05):.3f}, "
            f"η_R={config.get('eta_R', 0.05):.3f}, "
            f"τ_A₀={tau_A_init:.3f}, τ_R₀={tau_R_init:.3f}"
        )

        problems = pregenerated_data["data"][difficulty]
        prob_keys = sorted(problems.keys())

        runner = SimulatedAdaptiveRunnerWithRNG(config, rng)

        for prob_idx in prob_keys:
            runner.solve_problem(problems[int(prob_idx)])

        results[difficulty] = _compute_running_errors(runner)
        results[difficulty]["config"] = config

        print(
            f"           {results[difficulty]['n_steps']} steps, "
            f"final accept_err={results[difficulty]['accept_errors_running'][-1]:.3f}, "
            f"final reject_err={results[difficulty]['reject_errors_running'][-1]:.3f}"
        )

    return results


def _compute_running_errors(runner) -> dict:
    """Extract running error averages from a runner's ground truth history."""
    ground_truth = runner.all_step_ground_truth

    accept_errors_running = []
    incorrect_count = 0
    accepted_incorrect = 0

    for w_t, H_t, accepted in ground_truth:
        if not H_t:
            incorrect_count += 1
            if accepted:
                accepted_incorrect += 1
        accept_errors_running.append(
            accepted_incorrect / incorrect_count if incorrect_count > 0 else 0.0
        )

    reject_errors_running = []
    correct_count = 0
    rejected_correct = 0

    for w_t, H_t, accepted in ground_truth:
        if H_t:
            correct_count += 1
            if not accepted:
                rejected_correct += 1
        reject_errors_running.append(
            rejected_correct / correct_count if correct_count > 0 else 0.0
        )

    return {
        "accept_errors_running": accept_errors_running,
        "reject_errors_running": reject_errors_running,
        "tau_A_history": runner.tau_A_history,
        "tau_R_history": runner.tau_R_history,
        "n_steps": len(ground_truth),
    }


# ============================================================================
# Plotting: error convergence
# ============================================================================


def _subsample(data, n_points):
    """Subsample data to n_points without smoothing."""
    if len(data) <= n_points:
        return np.array(data), np.linspace(0, 1, len(data))
    indices = np.linspace(0, len(data) - 1, n_points).astype(int)
    x = indices / (len(data) - 1)
    return np.array(data)[indices], x


def plot_error_convergence(
    results: dict,
    alpha: float,
    beta: float,
    save_dir: str = None,
):
    """
    Plot running average error over time for all difficulties.
    Two side-by-side panels: accept error and reject error.
    """
    difficulties = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Error Convergence Over Time (α={alpha}, β={beta})",
        fontsize=14,
        fontweight="bold",
    )

    # Accept error
    ax = axes[0]
    for difficulty in difficulties:
        color = DIFF_COLORS.get(difficulty, "black")
        label = DIFF_LABELS.get(difficulty, f"Difficulty {difficulty}")
        errors = results[difficulty]["accept_errors_running"]
        steps = range(1, len(errors) + 1)
        ax.plot(steps, errors, color=color, linewidth=1.5, alpha=0.8, label=label)
    ax.axhline(
        y=alpha, color="black", linestyle="--", linewidth=2, label=f"Target α={alpha}"
    )
    ax.set_xlabel("Step t", fontsize=12)
    ax.set_ylabel("Running Avg Accept Error", fontsize=12)
    ax.set_title("Accept Error: P(accepted | incorrect)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(alpha * 3, 0.3))

    # Reject error
    ax = axes[1]
    for difficulty in difficulties:
        color = DIFF_COLORS.get(difficulty, "black")
        label = DIFF_LABELS.get(difficulty, f"Difficulty {difficulty}")
        errors = results[difficulty]["reject_errors_running"]
        steps = range(1, len(errors) + 1)
        ax.plot(steps, errors, color=color, linewidth=1.5, alpha=0.8, label=label)
    ax.axhline(
        y=beta, color="black", linestyle="--", linewidth=2, label=f"Target β={beta}"
    )
    ax.set_xlabel("Step t", fontsize=12)
    ax.set_ylabel("Running Avg Reject Error", fontsize=12)
    ax.set_title("Reject Error: P(rejected | correct)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    max_re = max(results[d]["reject_errors_running"][-1] for d in difficulties)
    ax.set_ylim(0, max(beta * 3, max_re * 1.2, 0.3))

    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/error_convergence_a{alpha}_b{beta}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


def plot_error_summary_with_convergence(
    results: dict,
    alpha: float,
    beta: float,
    n_points: int = 500,
    save_dir: str = None,
):
    """
    Bar chart (left) showing final errors.
    Two stacked convergence plots (right): accept top, reject bottom.
    """
    difficulties = sorted(results.keys())

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], hspace=0.35, wspace=0.3
    )

    # Left: Bar chart (spans both rows)
    ax_bar = fig.add_subplot(gs[:, 0])
    x_pos = np.arange(len(difficulties))
    width = 0.35

    ae_vals = [results[d]["accept_errors_running"][-1] for d in difficulties]
    re_vals = [results[d]["reject_errors_running"][-1] for d in difficulties]
    colors = [DIFF_COLORS[d] for d in difficulties]

    ax_bar.bar(
        x_pos - width / 2,
        ae_vals,
        width,
        label="Accept Error",
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )
    ax_bar.bar(
        x_pos + width / 2,
        re_vals,
        width,
        label="Reject Error",
        color=colors,
        alpha=0.4,
        edgecolor="black",
        hatch="//",
    )
    ax_bar.axhline(
        y=alpha, color="black", linestyle="--", linewidth=2, label=f"Target ({alpha})"
    )
    ax_bar.set_xlabel("Difficulty", fontsize=11)
    ax_bar.set_ylabel("Final Error Rate", fontsize=11)
    ax_bar.set_title("Final Error Rates", fontsize=12, fontweight="bold")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([DIFF_NAMES[d] for d in difficulties])
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_ylim(0, max(max(ae_vals), max(re_vals)) * 1.4)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Top right: Accept error over time
    ax_accept = fig.add_subplot(gs[0, 1])
    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        ae, x = _subsample(results[difficulty]["accept_errors_running"], n_points)
        ax_accept.plot(
            x, ae, color=color, linewidth=1.2, alpha=0.8, label=DIFF_NAMES[difficulty]
        )
    ax_accept.axhline(
        y=alpha, color="black", linestyle="--", linewidth=1.5, label=f"Target α={alpha}"
    )
    ax_accept.set_ylabel("Accept Error", fontsize=10)
    ax_accept.set_title(
        r"Cumulative Avg Error: $\frac{1}{t}\sum_{i=1}^{t} \mathrm{err}_i$",
        fontsize=11,
        fontweight="bold",
    )
    ax_accept.set_xlim(0, 1)
    ax_accept.set_ylim(0, 0.25)
    ax_accept.tick_params(labelsize=9)
    ax_accept.grid(True, alpha=0.3)
    ax_accept.legend(loc="upper right", fontsize=8)
    ax_accept.set_xticklabels([])

    # Bottom right: Reject error over time
    ax_reject = fig.add_subplot(gs[1, 1])
    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        re, x = _subsample(results[difficulty]["reject_errors_running"], n_points)
        ax_reject.plot(
            x, re, color=color, linewidth=1.2, alpha=0.8, label=DIFF_NAMES[difficulty]
        )
    ax_reject.axhline(
        y=beta, color="black", linestyle="--", linewidth=1.5, label=f"Target β={beta}"
    )
    ax_reject.set_xlabel("Fraction of Total Steps", fontsize=10)
    ax_reject.set_ylabel("Reject Error", fontsize=10)
    ax_reject.set_xlim(0, 1)
    ax_reject.set_ylim(0, 0.4)
    ax_reject.tick_params(labelsize=9)
    ax_reject.grid(True, alpha=0.3)
    ax_reject.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Error Rate Analysis (α=β={alpha})", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/error_summary_with_convergence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


def plot_accept_error_convergence(
    results: dict,
    alpha: float,
    n_points: int = 500,
    save_dir: str = None,
    filename: str = "accept_error_convergence.png",
):
    """Plot accept error convergence for all difficulties."""
    difficulties = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        ae, x = _subsample(results[difficulty]["accept_errors_running"], n_points)
        ax.plot(x, ae, color=color, linewidth=1.5, label=DIFF_NAMES[difficulty])
    ax.axhline(
        y=alpha, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Target α={alpha}",
    )
    ax.set_xlabel("Fraction of Total Steps", fontsize=11)
    ax.set_ylabel("Cumulative Error Rate", fontsize=11)
    ax.set_title("Accept Error Convergence", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(alpha * 3, 0.25))
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{filename}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


def plot_reject_error_convergence(
    results: dict,
    beta: float,
    n_points: int = 500,
    save_dir: str = None,
    filename: str = "reject_error_convergence.png",
):
    """Plot reject error convergence for all difficulties."""
    difficulties = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        re, x = _subsample(results[difficulty]["reject_errors_running"], n_points)
        ax.plot(x, re, color=color, linewidth=1.5, label=DIFF_NAMES[difficulty])
    ax.axhline(
        y=beta, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Target β={beta}",
    )
    ax.set_xlabel("Fraction of Total Steps", fontsize=11)
    ax.set_ylabel("Cumulative Error Rate", fontsize=11)
    ax.set_title("Reject Error Convergence", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(beta * 3, 0.4))
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{filename}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


def plot_both_error_convergence(
    results: dict,
    alpha: float,
    beta: float,
    n_points: int = 500,
    save_dir: str = None,
    filename: str = "error_convergence.png",
):
    """Plot both accept (solid) and reject (dotted) error on same figure."""
    difficulties = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(8, 4))

    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        ae, x = _subsample(results[difficulty]["accept_errors_running"], n_points)
        ax.plot(
            x, ae, color=color, linewidth=1.5, linestyle="-",
            label=f"{DIFF_NAMES[difficulty]} Accept",
        )

    for difficulty in difficulties:
        color = DIFF_COLORS[difficulty]
        re, x = _subsample(results[difficulty]["reject_errors_running"], n_points)
        ax.plot(
            x, re, color=color, linewidth=1.5, linestyle=":",
            label=f"{DIFF_NAMES[difficulty]} Reject",
        )

    ax.axhline(
        y=alpha, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"Target α={alpha}",
    )
    if beta != alpha:
        ax.axhline(
            y=beta, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Target β={beta}",
        )

    ax.set_xlabel("Fraction of Total Steps", fontsize=11)
    ax.set_ylabel("Cumulative Error Rate", fontsize=11)
    ax.set_title("Error Convergence", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(alpha * 3, beta * 2, 0.4))
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{filename}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig

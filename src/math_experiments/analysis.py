"""Weak verifier quality analysis: score distributions, calibration, and summary."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss


# ============================================================================
# Constants
# ============================================================================

DIFF_NAMES = {2: "Easy", 3: "Medium", 5: "Hard"}
DIFF_COLORS = {2: "green", 3: "blue", 5: "red"}


# ============================================================================
# Data collection
# ============================================================================


def collect_scores(pregenerated_data: dict) -> dict:
    """
    Collect weak verifier scores grouped by difficulty and correctness.

    Returns:
        Dict keyed by difficulty with keys: correct, incorrect, all, labels
    """
    difficulties = sorted([int(d) for d in pregenerated_data["data"].keys()])
    scores_by_diff = {
        d: {"correct": [], "incorrect": [], "all": [], "labels": []}
        for d in difficulties
    }

    for diff in difficulties:
        diff_key = str(diff) if str(diff) in pregenerated_data["data"] else diff
        problems = pregenerated_data["data"][diff_key]

        for prob_idx, prob_data in problems.items():
            for attempt in prob_data["attempts"]:
                score = attempt["weak_score"]
                is_correct = attempt["is_correct"]

                scores_by_diff[diff]["all"].append(score)
                scores_by_diff[diff]["labels"].append(1 if is_correct else 0)
                if is_correct:
                    scores_by_diff[diff]["correct"].append(score)
                else:
                    scores_by_diff[diff]["incorrect"].append(score)

    return scores_by_diff


# ============================================================================
# Score distribution plot
# ============================================================================


def plot_score_distribution(
    pregenerated_data: dict,
    save_dir: str = None,
    filename: str = "score_distribution.png",
):
    """
    Plot score distribution (correct vs incorrect) with separation and AUC metrics.
    One panel per difficulty.
    """
    scores_by_diff = collect_scores(pregenerated_data)
    difficulties = sorted(scores_by_diff.keys())

    fig, axes = plt.subplots(1, len(difficulties), figsize=(14, 4))
    if len(difficulties) == 1:
        axes = [axes]

    for ax, diff in zip(axes, difficulties):
        correct_scores = np.array(scores_by_diff[diff]["correct"])
        incorrect_scores = np.array(scores_by_diff[diff]["incorrect"])
        all_scores = np.array(scores_by_diff[diff]["all"])
        labels = np.array(scores_by_diff[diff]["labels"])

        bins = np.linspace(0, 1, 21)
        correct_counts, _ = np.histogram(correct_scores, bins=bins)
        incorrect_counts, _ = np.histogram(incorrect_scores, bins=bins)

        correct_pct = (
            100 * correct_counts / len(correct_scores) if len(correct_scores) > 0
            else correct_counts
        )
        incorrect_pct = (
            100 * incorrect_counts / len(incorrect_scores) if len(incorrect_scores) > 0
            else incorrect_counts
        )

        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_width = 0.022

        ax.bar(
            bin_centers - bar_width / 2, correct_pct, width=bar_width,
            alpha=0.7, color="green", label="Correct", edgecolor="black", linewidth=0.5,
        )
        ax.bar(
            bin_centers + bar_width / 2, incorrect_pct, width=bar_width,
            alpha=0.7, color="red", label="Incorrect", edgecolor="black", linewidth=0.5,
        )

        mean_correct = np.mean(correct_scores)
        mean_incorrect = np.mean(incorrect_scores)
        separation = mean_correct - mean_incorrect

        ax.axvline(
            x=mean_correct, color="darkgreen", linestyle="--", linewidth=2.5,
            label=f"μ_correct={mean_correct:.2f}",
        )
        ax.axvline(
            x=mean_incorrect, color="darkred", linestyle="--", linewidth=2.5,
            label=f"μ_incorrect={mean_incorrect:.2f}",
        )

        try:
            auc = roc_auc_score(labels, all_scores)
        except Exception:
            auc = 0.5

        ax.set_xlabel("Weak Verifier Score", fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_title(
            f"{DIFF_NAMES[diff]} (Level {diff})\nSeparation={separation:.2f}, AUC={auc:.2f}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        "Score Distribution: Correct vs Incorrect (Discrimination)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{filename}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


# ============================================================================
# Calibration plot
# ============================================================================


def plot_calibration(
    pregenerated_data: dict,
    save_dir: str = None,
    filename: str = "calibration.png",
):
    """
    Plot calibration: predicted score vs actual accuracy.
    One panel per difficulty with Brier score and AUC.
    """
    scores_by_diff = collect_scores(pregenerated_data)
    difficulties = sorted(scores_by_diff.keys())

    fig, axes = plt.subplots(1, len(difficulties), figsize=(14, 4))
    if len(difficulties) == 1:
        axes = [axes]

    for ax, diff in zip(axes, difficulties):
        all_scores = np.array(scores_by_diff[diff]["all"])
        labels = np.array(scores_by_diff[diff]["labels"])

        base_accuracy = np.mean(labels)

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        calibration = []
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (all_scores >= bins[i]) & (all_scores <= bins[i + 1])
            else:
                mask = (all_scores >= bins[i]) & (all_scores < bins[i + 1])

            n_in_bin = np.sum(mask)
            if n_in_bin > 0:
                calibration.append(np.mean(labels[mask]))
            else:
                calibration.append(np.nan)

        calibration = np.array(calibration)
        valid_mask = ~np.isnan(calibration)

        ax.bar(
            bin_centers[valid_mask], calibration[valid_mask],
            width=bin_width * 0.8, color=DIFF_COLORS[diff], alpha=0.7,
            edgecolor="black", linewidth=0.5, label="Observed accuracy",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")
        ax.axhline(
            y=base_accuracy, color="gray", linestyle=":", linewidth=2,
            label=f"Base rate = {base_accuracy:.0%}",
        )

        brier = brier_score_loss(labels, all_scores)
        try:
            auc = roc_auc_score(labels, all_scores)
        except Exception:
            auc = 0.5

        ax.set_xlabel("Predicted Score", fontsize=11)
        ax.set_ylabel("Actual Accuracy", fontsize=11)
        ax.set_title(
            f"{DIFF_NAMES[diff]} (Level {diff})\nBrier={brier:.3f}, AUC={auc:.2f}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        "Calibration: Predicted Score vs Actual Accuracy",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save_dir:
        path = f"{save_dir}/{filename}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {path}")

    plt.show()
    return fig


# ============================================================================
# Summary table
# ============================================================================


def print_weak_verifier_summary(pregenerated_data: dict):
    """Print summary table of weak verifier quality metrics."""
    scores_by_diff = collect_scores(pregenerated_data)
    difficulties = sorted(scores_by_diff.keys())

    print("\n" + "=" * 95)
    print("WEAK VERIFIER QUALITY SUMMARY")
    print("=" * 95)

    print(
        f"\n{'Difficulty':<10} | {'Base Acc':<10} | {'AUC':<8} | {'Brier':<8} | "
        f"{'μ_correct':<10} | {'μ_incorrect':<12} | {'Separation':<10}"
    )
    print("-" * 95)

    for diff in difficulties:
        all_scores = np.array(scores_by_diff[diff]["all"])
        labels = np.array(scores_by_diff[diff]["labels"])
        correct_scores = np.array(scores_by_diff[diff]["correct"])
        incorrect_scores = np.array(scores_by_diff[diff]["incorrect"])

        base_acc = np.mean(labels)
        brier = brier_score_loss(labels, all_scores)
        try:
            auc = roc_auc_score(labels, all_scores)
        except Exception:
            auc = 0.5

        mean_correct = np.mean(correct_scores)
        mean_incorrect = np.mean(incorrect_scores)
        separation = mean_correct - mean_incorrect

        print(
            f"{DIFF_NAMES[diff]:<10} | {base_acc:<10.1%} | {auc:<8.2f} | "
            f"{brier:<8.3f} | {mean_correct:<10.2f} | {mean_incorrect:<12.2f} | "
            f"{separation:<10.2f}"
        )

    print(
        """
================================================================================
METRICS EXPLANATION:
================================================================================
- Separation: Mean(correct) - Mean(incorrect). Higher = better discrimination.
- AUC: Probability that a random correct sample scores higher than incorrect.
       0.5 = random, 1.0 = perfect. Higher = better.
- Brier: Mean squared error of predictions. 0 = perfect, 1 = worst. Lower = better.
- Base Acc: Overall accuracy (% of correct attempts in dataset).
"""
    )

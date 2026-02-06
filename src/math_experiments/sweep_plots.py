"""
Multi-sweep plotting: overlay multiple α-fixed or β-fixed sweeps on one figure.
Used for appendix figures showing how accuracy-vs-latency changes across parameter values.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List


# ============================================================================
# Constants
# ============================================================================

DIFF_LABELS = {2: "Easy (Level 2)", 3: "Medium (Level 3)", 5: "Hard (Level 5)"}
DIFF_FILENAMES = {2: "easy", 3: "medium", 5: "hard"}

COLORS_FIXED = {
    0.01: "#e41a1c", 0.02: "#377eb8", 0.05: "#4daf4a", 0.07: "#984ea3",
    0.10: "#ff7f00", 0.15: "#f781bf", 0.20: "#00BFFF",
}
FALLBACK = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
WEAK_COLORS = {
    1: "#FF6B6B", 2: "#4ECDC4", 3: "#9B59B6", 4: "#F39C12", 5: "#1ABC9C",
}


# ============================================================================
# Annotation placement
# ============================================================================


def _place_annotations_smart(ax, all_annotations, strong_cpp):
    """Place annotations avoiding overlaps using grid-based grouping."""
    if not all_annotations:
        return
    all_annotations.sort(key=lambda a: (round(a["x"], 2), a["y"]))
    x_tol = strong_cpp * 0.12

    groups, cur = [], [all_annotations[0]]
    for ann in all_annotations[1:]:
        if abs(ann["x"] - cur[0]["x"]) < x_tol:
            cur.append(ann)
        else:
            groups.append(cur)
            cur = [ann]
    groups.append(cur)

    for group in groups:
        group.sort(key=lambda a: a["y"])
        n = len(group)
        for i, ann in enumerate(group):
            oy = (22 + (i // 2) * 18) * (1 if i % 2 == 0 else -1)
            ox = (i - n / 2) * 6
            ax.annotate(
                ann["text"], (ann["x"], ann["y"]),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=9, ha="center", fontweight="bold", color=ann["color"],
                arrowprops=dict(
                    arrowstyle="-", color=ann["color"], lw=0.6,
                    alpha=0.5, shrinkA=0, shrinkB=3,
                ),
            )


# ============================================================================
# Shared baseline setup
# ============================================================================


def _setup_baselines(ax, first_results, difficulty, weak_n_values_to_plot=None):
    """Plot baselines (shared across sweeps) and return (strong_acc, strong_cpp, all_accs)."""
    strong = first_results["strong_baseline"][difficulty]
    weak_baselines = first_results["weak_baseline"]
    avail = sorted(first_results.get("weak_baseline_n_values", list(weak_baselines.keys())))
    if weak_n_values_to_plot is None:
        weak_n_values_to_plot = avail

    strong_acc = strong["accuracy"] * 100
    strong_cpp = strong["calls_per_problem"]
    all_accs = [strong_acc]

    # Weak baselines at x=0
    for n_val in sorted(weak_n_values_to_plot):
        if n_val in weak_baselines and difficulty in weak_baselines[n_val]:
            weak_acc = weak_baselines[n_val][difficulty]["accuracy"] * 100
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
    ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)

    return strong_acc, strong_cpp, all_accs


def _filter_monotonic(all_points, get_varying):
    """Keep only points on the Pareto frontier (decreasing accuracy with decreasing cost)."""
    all_points.sort(key=lambda x: get_varying(x))

    best_idx = 0
    best_acc = -1
    for i, p in enumerate(all_points):
        if p["acc"] > best_acc:
            best_acc = p["acc"]
            best_idx = i

    filtered = [all_points[best_idx]]
    for i in range(best_idx + 1, len(all_points)):
        point = all_points[i]
        last = filtered[-1]
        if point["acc"] <= last["acc"] and point["cpp"] <= last["cpp"]:
            filtered.append(point)
    filtered.sort(key=lambda x: x["cpp"])
    return filtered


# ============================================================================
# Multi-alpha plot (fixed α, varying β)
# ============================================================================


def plot_accuracy_vs_latency_multi_alpha(
    results_dict: Dict[float, dict],
    save_dir: str = None,
    filename: str = "accuracy_vs_latency_multi_alpha.png",
    difficulties_to_plot: list = None,
    weak_n_values_to_plot: list = None,
    y_padding: float = 3,
):
    """
    Overlay multiple α-fixed sweeps (each varying β) on one figure per difficulty.

    Args:
        results_dict: {alpha_value: pipeline_results} — each pipeline_results has
                      fixed α and sweep over β
        save_dir: Where to save
        filename: Base filename
        difficulties_to_plot: Which difficulties
        weak_n_values_to_plot: Which BoN baselines to show
        y_padding: Y-axis padding

    Returns:
        Dict of {difficulty: figure}
    """
    first_results = list(results_dict.values())[0]
    all_difficulties = sorted(first_results["strong_baseline"].keys())
    if difficulties_to_plot is not None:
        difficulties = [d for d in all_difficulties if d in difficulties_to_plot]
    else:
        difficulties = all_difficulties

    figures = {}

    for difficulty in difficulties:
        fig, ax = plt.subplots(figsize=(8, 6))

        strong_acc, strong_cpp, all_accs = _setup_baselines(
            ax, first_results, difficulty, weak_n_values_to_plot
        )
        all_annotations = []

        for idx, (fixed_alpha, pipeline_results) in enumerate(
            sorted(results_dict.items())
        ):
            color = COLORS_FIXED.get(fixed_alpha, FALLBACK[idx % len(FALLBACK)])
            adaptive = pipeline_results["adaptive"][difficulty]

            points = []
            for (alpha, beta), data in adaptive.items():
                points.append({
                    "cpp": data["calls_per_problem_mean"],
                    "acc": data["accuracy_mean"] * 100,
                    "alpha": alpha,
                    "beta": beta,
                })

            get_varying = lambda p: p["beta"]
            filtered = _filter_monotonic(points, get_varying)
            all_accs.extend([p["acc"] for p in filtered])

            if filtered:
                x = [p["cpp"] for p in filtered]
                y = [p["acc"] for p in filtered]
                ax.plot(
                    x, y, "o-", color=color, markersize=7, linewidth=2, alpha=0.8,
                    label=f"α={fixed_alpha:.2f}",
                )
                for p in filtered:
                    all_annotations.append({
                        "x": p["cpp"], "y": p["acc"],
                        "text": f"{get_varying(p):.2f}", "color": color,
                    })

        _place_annotations_smart(ax, all_annotations, strong_cpp)

        ax.set_xlabel("Strong Verifier Calls / Problem", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
        title = DIFF_LABELS.get(difficulty, f"Difficulty {difficulty}")
        ax.set_title(
            f"{title}\n(Fixed α, varying β)", fontsize=16, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=10)
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


# ============================================================================
# Multi-beta plot (fixed β, varying α)
# ============================================================================


def plot_accuracy_vs_latency_multi_beta(
    results_dict: Dict[float, dict],
    save_dir: str = None,
    filename: str = "accuracy_vs_latency_multi_beta.png",
    difficulties_to_plot: list = None,
    weak_n_values_to_plot: list = None,
    y_padding: float = 3,
):
    """
    Overlay multiple β-fixed sweeps (each varying α) on one figure per difficulty.

    Args:
        results_dict: {beta_value: pipeline_results} — each pipeline_results has
                      fixed β and sweep over α
    """
    first_results = list(results_dict.values())[0]
    all_difficulties = sorted(first_results["strong_baseline"].keys())
    if difficulties_to_plot is not None:
        difficulties = [d for d in all_difficulties if d in difficulties_to_plot]
    else:
        difficulties = all_difficulties

    figures = {}

    for difficulty in difficulties:
        fig, ax = plt.subplots(figsize=(8, 6))

        strong_acc, strong_cpp, all_accs = _setup_baselines(
            ax, first_results, difficulty, weak_n_values_to_plot
        )
        all_annotations = []

        for idx, (fixed_beta, pipeline_results) in enumerate(
            sorted(results_dict.items())
        ):
            color = COLORS_FIXED.get(fixed_beta, FALLBACK[idx % len(FALLBACK)])
            adaptive = pipeline_results["adaptive"][difficulty]

            points = []
            for (alpha, beta), data in adaptive.items():
                points.append({
                    "cpp": data["calls_per_problem_mean"],
                    "acc": data["accuracy_mean"] * 100,
                    "alpha": alpha,
                    "beta": beta,
                })

            get_varying = lambda p: p["alpha"]
            filtered = _filter_monotonic(points, get_varying)
            all_accs.extend([p["acc"] for p in filtered])

            if filtered:
                x = [p["cpp"] for p in filtered]
                y = [p["acc"] for p in filtered]
                ax.plot(
                    x, y, "o-", color=color, markersize=7, linewidth=2, alpha=0.8,
                    label=f"β={fixed_beta:.2f}",
                )
                for p in filtered:
                    all_annotations.append({
                        "x": p["cpp"], "y": p["acc"],
                        "text": f"{get_varying(p):.2f}", "color": color,
                    })

        _place_annotations_smart(ax, all_annotations, strong_cpp)

        ax.set_xlabel("Strong Verifier Calls / Problem", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
        title = DIFF_LABELS.get(difficulty, f"Difficulty {difficulty}")
        ax.set_title(
            f"{title}\n(Fixed β, varying α)", fontsize=16, fontweight="bold"
        )
        ax.legend(loc="lower right", fontsize=10)
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

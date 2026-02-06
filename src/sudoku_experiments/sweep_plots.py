"""
Multi-sweep plotting: overlay multiple α-fixed or β-fixed sweeps on one figure.
"""

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Smart annotation placement
# ============================================================================

def _place_annotations_smart(ax, all_annotations, strong_cpp):
    all_annotations.sort(key=lambda a: (round(a["x"], 2), a["y"]))
    x_tol = strong_cpp * 0.12
    groups, cur = [], [all_annotations[0]]
    for ann in all_annotations[1:]:
        if abs(ann["x"] - cur[0]["x"]) < x_tol:
            cur.append(ann)
        else:
            groups.append(cur); cur = [ann]
    groups.append(cur)

    for group in groups:
        group.sort(key=lambda a: a["y"])
        n = len(group)
        for i, ann in enumerate(group):
            oy = (22 + (i // 2) * 18) * (1 if i % 2 == 0 else -1)
            ox = (i - n / 2) * 6
            ax.annotate(ann["text"], (ann["x"], ann["y"]),
                        textcoords="offset points", xytext=(ox, oy),
                        fontsize=9, ha="center", fontweight="bold", color=ann["color"],
                        arrowprops=dict(arrowstyle="-", color=ann["color"], lw=0.6,
                                        alpha=0.5, shrinkA=0, shrinkB=3))


# ============================================================================
# Common setup
# ============================================================================

_COLORS_FIXED = {
    0.01: "#e41a1c", 0.02: "#377eb8", 0.05: "#4daf4a", 0.07: "#984ea3",
    0.10: "#ff7f00", 0.15: "#f781bf", 0.20: "#00BFFF",
}
_FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_WEAK_COLORS = {1: "#FF6B6B", 2: "#4ECDC4", 3: "#9B59B6", 4: "#F39C12", 5: "#1ABC9C"}


def _setup_baselines(ax, first_results, weak_n_values_to_plot=None):
    strong = first_results["strong_baseline"]
    weak_baselines = first_results["weak_baseline"]
    avail = sorted(first_results.get("weak_baseline_n_values", list(weak_baselines.keys())))
    if weak_n_values_to_plot is None:
        weak_n_values_to_plot = avail

    strong_acc = strong["accuracy"] * 100
    strong_cpp = strong["calls_per_puzzle"]
    all_accs = [strong_acc]

    for n_val in sorted(weak_n_values_to_plot):
        if n_val in weak_baselines:
            wa = weak_baselines[n_val]["accuracy"] * 100
            all_accs.append(wa)
            ax.scatter(0, wa, color=_WEAK_COLORS.get(n_val, "gray"), marker="o", s=120,
                       zorder=9, edgecolors="black", linewidths=0.5, label=f"BoN-{n_val} ({wa:.1f}%)")
    ax.scatter(strong_cpp, strong_acc, color="black", marker="*", s=400, zorder=10,
               label=f"Strong ({strong_acc:.1f}%)")
    return strong_acc, strong_cpp, all_accs


def _filter_curve(all_points):
    if not all_points:
        return []
    best_i = max(range(len(all_points)), key=lambda i: all_points[i]["acc"])
    filt = [all_points[best_i]]
    for i in range(best_i + 1, len(all_points)):
        p = all_points[i]
        if p["acc"] <= filt[-1]["acc"] and p["cpp"] <= filt[-1]["cpp"]:
            filt.append(p)
    filt.sort(key=lambda x: x["cpp"])
    return filt


# ============================================================================
# Plot multiple alpha-fixed sweeps (varying β)
# ============================================================================

def plot_accuracy_vs_latency_multi_alpha_sudoku(
    results_dict, save_path=None, weak_n_values_to_plot=None,
    y_padding=3, annotate_labels=True,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    first_results = list(results_dict.values())[0]
    strong_acc, strong_cpp, all_accs = _setup_baselines(ax, first_results, weak_n_values_to_plot)
    all_annotations = []

    for idx, (alpha_val, results) in enumerate(sorted(results_dict.items())):
        color = _COLORS_FIXED.get(alpha_val, _FALLBACK[idx % len(_FALLBACK)])
        pts = sorted(
            [{"cpp": d["calls_per_puzzle_mean"], "acc": d["accuracy_mean"]*100,
              "alpha": a, "beta": b}
             for (a, b), d in results["adaptive"].items()],
            key=lambda x: x["beta"],
        )
        filt = _filter_curve(pts)
        all_accs.extend(p["acc"] for p in filt)
        if filt:
            ax.plot([p["cpp"] for p in filt], [p["acc"] for p in filt],
                    "o-", color=color, markersize=8, linewidth=2.5, alpha=0.85,
                    label=f"Adaptive (α={alpha_val})")
            for p in filt:
                all_annotations.append({"x": p["cpp"], "y": p["acc"],
                                        "text": f"{p['beta']:.2f}", "color": color})

    if annotate_labels and all_annotations:
        _place_annotations_smart(ax, all_annotations, strong_cpp)
    ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Strong Verifier Calls / Puzzle", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Sudoku: Multiple α values (varying β)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3); ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(max(0, min(all_accs) - y_padding - 3), min(100, max(all_accs) + y_padding + 3))
    ax.set_xlim(-0.15, strong_cpp * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ============================================================================
# Plot multiple beta-fixed sweeps (varying α)
# ============================================================================

def plot_sweep_results(
    results, fixed_param='alpha', fixed_value=None,
    save_path=None, weak_n_values_to_plot=None, y_padding=3,
):
    """
    Plot a single sweep result (one fixed α or β) as its own figure.

    Unlike plot_accuracy_vs_latency_sudoku (which only shows α=β points),
    this plots ALL adaptive points in the result — suitable for asymmetric sweeps.

    Args:
        results: Standard pipeline results dict from run_full_pipeline_sudoku.
        fixed_param: 'alpha' or 'beta' — which parameter was held constant.
        fixed_value: The numeric value that was held constant.
        save_path: Path to save the figure.
        weak_n_values_to_plot: Which BoN baselines to show.
        y_padding: Padding for y-axis limits.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    strong = results["strong_baseline"]
    weak_baselines = results["weak_baseline"]
    adaptive = results["adaptive"]

    avail = sorted(results.get("weak_baseline_n_values", list(weak_baselines.keys())))
    if weak_n_values_to_plot is None:
        weak_n_values_to_plot = avail

    strong_acc = strong["accuracy"] * 100
    strong_cpp = strong["calls_per_puzzle"]
    all_accs = [strong_acc]

    # Weak baselines
    for n_val in sorted(weak_n_values_to_plot):
        if n_val in weak_baselines:
            wa = weak_baselines[n_val]["accuracy"] * 100
            all_accs.append(wa)
            ax.scatter(0, wa, color=_WEAK_COLORS.get(n_val, "gray"), marker="o",
                       s=120, zorder=9, edgecolors="black", linewidths=0.5,
                       label=f"BoN-{n_val} ({wa:.1f}%)")

    ax.scatter(strong_cpp, strong_acc, color="black", marker="*", s=400,
               zorder=10, label=f"Strong ({strong_acc:.1f}%)")

    # All adaptive points (not just symmetric)
    pts = []
    for (a, b), d in adaptive.items():
        pts.append({
            "cpp": d["calls_per_puzzle_mean"],
            "acc": d["accuracy_mean"] * 100,
            "alpha": a, "beta": b,
        })

    # Sort by the varying parameter
    if fixed_param == 'alpha':
        pts.sort(key=lambda x: x["beta"])
        varying_key = "beta"
        vary_symbol = "β"
    else:
        pts.sort(key=lambda x: x["alpha"])
        varying_key = "alpha"
        vary_symbol = "α"

    filt = _filter_curve(pts)
    all_accs.extend(p["acc"] for p in filt)

    if filt:
        ax.plot([p["cpp"] for p in filt], [p["acc"] for p in filt],
                "o-", color="blue", markersize=9, linewidth=2.5, alpha=0.8,
                label="Adaptive")
        for i, p in enumerate(filt):
            ax.annotate(f"{p[varying_key]:.3f}", (p["cpp"], p["acc"]),
                        textcoords="offset points",
                        xytext=(0, 20 if i % 2 == 0 else -26),
                        fontsize=10, ha="center", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8,
                                        shrinkA=0, shrinkB=3))

    ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Strong Verifier Calls / Puzzle", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")

    if fixed_param == 'alpha' and fixed_value is not None:
        ax.set_title(f"Sudoku: α={fixed_value} (fixed), β swept",
                     fontsize=16, fontweight="bold")
    elif fixed_param == 'beta' and fixed_value is not None:
        ax.set_title(f"Sudoku: β={fixed_value} (fixed), α swept",
                     fontsize=16, fontweight="bold")
    else:
        ax.set_title("Sudoku: Accuracy vs Latency", fontsize=16, fontweight="bold")

    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(max(0, min(all_accs) - y_padding), min(100, max(all_accs) + y_padding))
    ax.set_xlim(-0.1, strong_cpp * 1.1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    plt.show()
    return fig


def plot_accuracy_vs_latency_multi_beta_sudoku(
    results_dict, save_path=None, weak_n_values_to_plot=None,
    y_padding=3, annotate_labels=True,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    first_results = list(results_dict.values())[0]
    strong_acc, strong_cpp, all_accs = _setup_baselines(ax, first_results, weak_n_values_to_plot)
    all_annotations = []

    for idx, (beta_val, results) in enumerate(sorted(results_dict.items())):
        color = _COLORS_FIXED.get(beta_val, _FALLBACK[idx % len(_FALLBACK)])
        pts = sorted(
            [{"cpp": d["calls_per_puzzle_mean"], "acc": d["accuracy_mean"]*100,
              "alpha": a, "beta": b}
             for (a, b), d in results["adaptive"].items()],
            key=lambda x: x["alpha"],
        )
        filt = _filter_curve(pts)
        all_accs.extend(p["acc"] for p in filt)
        if filt:
            ax.plot([p["cpp"] for p in filt], [p["acc"] for p in filt],
                    "o-", color=color, markersize=8, linewidth=2.5, alpha=0.85,
                    label=f"Adaptive (β={beta_val})")
            for p in filt:
                all_annotations.append({"x": p["cpp"], "y": p["acc"],
                                        "text": f"{p['alpha']:.2f}", "color": color})

    if annotate_labels and all_annotations:
        _place_annotations_smart(ax, all_annotations, strong_cpp)
    ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Strong Verifier Calls / Puzzle", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Sudoku: Multiple β values (varying α)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3); ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(max(0, min(all_accs) - y_padding - 3), min(100, max(all_accs) + y_padding + 3))
    ax.set_xlim(-0.15, strong_cpp * 1.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

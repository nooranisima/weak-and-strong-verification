"""
Sudoku step-wise simulation — adaptive runner, baselines, and error-rate plots.

Operates on pre-generated tree data (no LLM calls at experiment time).
"""

import json
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Helpers
# ============================================================================

def get_candidates_at_node(puzzle_data: dict, parent_node_id: str, depth: int) -> list:
    """Get candidate nodes at a given depth with a specific parent."""
    nodes = puzzle_data["nodes"]
    candidates = []
    for node_id, node in nodes.items():
        if node["parent_id"] == parent_node_id and node["depth"] == depth:
            candidates.append({
                "node_id": node_id,
                "weak_score": node["weak_score"] if node["weak_score"] is not None else 0.0,
                "is_correct": node["is_correct"] if node["is_correct"] is not None else False,
                "sample_index": node.get("sample_index", 0),
            })
    candidates.sort(key=lambda x: x["sample_index"])
    return candidates


def add_noise_to_score(score: float, noise_scale: float, rng=None) -> float:
    """Add uniform noise U(-noise_scale, +noise_scale), clamped to [0, 1]."""
    if noise_scale <= 0:
        return score
    if rng is None:
        noisy = score + np.random.uniform(-noise_scale, noise_scale)
    else:
        noisy = score + rng.uniform(-noise_scale, noise_scale)
    return float(np.clip(noisy, 0.0, 1.0))


# ============================================================================
# Adaptive Runner
# ============================================================================

class SudokuAdaptiveRunner:
    """Adaptive Best-of-N for step-wise Sudoku with noise support."""

    def __init__(self, config: dict, noise_scale: float = 0.0):
        self.config = config
        self.noise_scale = noise_scale
        self.max_attempts_per_step = config.get("max_attempts_per_step", None)

        self.alpha = config.get("alpha", 0.10)
        self.beta = config.get("beta", 0.10)

        self.tau_A = config.get("tau_A_init", 1 - self.alpha)
        self.tau_R = config.get("tau_R_init", self.beta)

        self.eta = config.get("eta", 0.1)
        self.eta_R = config.get("eta_R", 0.1)

        self.P_a_init = config.get("P_a_init", 0.3)
        self.P_r_init = config.get("P_r_init", 0.3)
        self.P_a_min = config.get("P_a_min", 0.05)
        self.P_r_min = config.get("P_r_min", 0.05)

        self.global_step = 0
        self.tau_A_history: List[float] = []
        self.tau_R_history: List[float] = []
        self.all_samples: List[dict] = []

    def get_P_a(self, t: int) -> float:
        return max(self.P_a_min, self.P_a_init / np.sqrt(t + 1))

    def get_P_r(self, t: int) -> float:
        return max(self.P_r_min, self.P_r_init / np.sqrt(t + 1))

    def update_thresholds(self, w_t: float, H_t: bool, region: str):
        t = self.global_step

        # τ_A update
        g_t = 0.0
        if region == "accept":
            g_t = ((1 if not H_t else 0) * (1 - self.alpha)) / self.get_P_a(t)
        elif region == "uncertainty":
            g_t = (1 if not H_t else 0) * (-self.alpha)
        else:
            g_t = ((1 if not H_t else 0) * (-self.alpha)) / self.get_P_r(t)
        self.tau_A = max(self.tau_R + 0.05, min(1.0, self.tau_A + self.eta * g_t))

        # τ_R update
        g_beta = 0.0
        if region == "accept":
            g_beta = ((1 if H_t else 0) * self.beta) / self.get_P_a(t)
        elif region == "uncertainty":
            g_beta = (1 if H_t else 0) * self.beta
        else:
            g_beta = ((1 if H_t else 0) * (-(1 - self.beta))) / self.get_P_r(t)
        self.tau_R = max(0.0, min(self.tau_A - 0.05, self.tau_R + self.eta_R * g_beta))

        self.tau_A_history.append(self.tau_A)
        self.tau_R_history.append(self.tau_R)

    def solve_puzzle(self, puzzle_data: dict) -> dict:
        depth_n = puzzle_data["depth_n"]
        max_per_step = self.max_attempts_per_step or puzzle_data["branching_m"]

        result = {"num_strong_verifier_calls": 0, "num_steps_correct": 0,
                  "num_weak_verifier_calls": 0}
        current_parent_id = puzzle_data["root_id"]

        for step_idx in range(depth_n):
            candidates = get_candidates_at_node(puzzle_data, current_parent_id, step_idx + 1)
            if not candidates:
                break
            candidates = candidates[:max_per_step]
            chosen = None

            for cand in candidates:
                self.global_step += 1
                result["num_weak_verifier_calls"] += 1

                w_t = add_noise_to_score(cand["weak_score"], self.noise_scale)
                H_t = cand["is_correct"]

                queried = False
                region = None
                decision = None

                if w_t > self.tau_A:
                    region = "accept"
                    if np.random.random() < self.get_P_a(self.global_step):
                        queried = True
                        result["num_strong_verifier_calls"] += 1
                        decision = "accept" if H_t else "reject"
                    else:
                        decision = "accept"
                elif w_t > self.tau_R:
                    region = "uncertainty"
                    queried = True
                    result["num_strong_verifier_calls"] += 1
                    decision = "accept" if H_t else "reject"
                else:
                    region = "reject"
                    if np.random.random() < self.get_P_r(self.global_step):
                        queried = True
                        result["num_strong_verifier_calls"] += 1
                        decision = "accept" if H_t else "reject"
                    else:
                        decision = "reject"

                if queried:
                    self.update_thresholds(w_t, H_t, region)

                self.all_samples.append({"is_correct": H_t, "accepted": decision == "accept"})

                if decision == "accept":
                    chosen = cand
                    break

            if chosen is None:
                chosen = candidates[-1]
            if chosen["is_correct"]:
                result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]

        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result

    def get_cumulative_errors(self) -> Tuple[List[float], List[float]]:
        accept_errors, reject_errors = [], []
        false_accepts = false_rejects = 0
        total_incorrect = total_correct = 0

        for s in self.all_samples:
            if s["is_correct"]:
                total_correct += 1
                if not s["accepted"]:
                    false_rejects += 1
            else:
                total_incorrect += 1
                if s["accepted"]:
                    false_accepts += 1
            accept_errors.append(false_accepts / total_incorrect if total_incorrect > 0 else 0)
            reject_errors.append(false_rejects / total_correct if total_correct > 0 else 0)
        return accept_errors, reject_errors

    def get_thresholds(self) -> dict:
        return {"tau_A": self.tau_A, "tau_R": self.tau_R}

    def reset(self):
        self.tau_A = self.config.get("tau_A_init", 1 - self.alpha)
        self.tau_R = self.config.get("tau_R_init", self.beta)
        self.global_step = 0
        self.tau_A_history = []
        self.tau_R_history = []
        self.all_samples = []


# ============================================================================
# Baselines
# ============================================================================

class SudokuStrongBaseline:
    """Strong baseline: query strong verifier until finding correct."""

    def __init__(self, max_attempts_per_step=None):
        self.max_attempts_per_step = max_attempts_per_step

    def solve_puzzle(self, puzzle_data):
        depth_n = puzzle_data["depth_n"]
        max_per_step = self.max_attempts_per_step or puzzle_data["branching_m"]
        result = {"num_strong_verifier_calls": 0, "num_steps_correct": 0}
        current_parent_id = puzzle_data["root_id"]

        for step_idx in range(depth_n):
            candidates = get_candidates_at_node(puzzle_data, current_parent_id, step_idx + 1)
            if not candidates:
                break
            candidates = candidates[:max_per_step]
            chosen = None
            for cand in candidates:
                result["num_strong_verifier_calls"] += 1
                if cand["is_correct"]:
                    chosen = cand
                    break
            if chosen is None:
                chosen = candidates[-1]
            if chosen["is_correct"]:
                result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]

        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result


class SudokuWeakBaseline:
    """Weak baseline: pick highest weak_score candidate (no strong verifier)."""

    def __init__(self, max_attempts_per_step=None, noise_scale=0.0):
        self.max_attempts_per_step = max_attempts_per_step
        self.noise_scale = noise_scale

    def solve_puzzle(self, puzzle_data):
        depth_n = puzzle_data["depth_n"]
        max_per_step = self.max_attempts_per_step or puzzle_data["branching_m"]
        result = {"num_strong_verifier_calls": 0, "num_steps_correct": 0}
        current_parent_id = puzzle_data["root_id"]

        for step_idx in range(depth_n):
            candidates = get_candidates_at_node(puzzle_data, current_parent_id, step_idx + 1)
            if not candidates:
                break
            candidates = candidates[:max_per_step]
            scored = [
                (add_noise_to_score(c["weak_score"], self.noise_scale), c)
                for c in candidates
            ]
            chosen = max(scored, key=lambda x: x[0])[1]
            if chosen["is_correct"]:
                result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]

        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(
    tree_data: dict,
    alpha_beta_configs: list,
    config: dict,
    noise_scale: float = 0.0,
    n_passes: int = 1,
    seed: int = 42,
) -> dict:
    """
    Run full experiment with baselines and adaptive configurations.

    Args:
        tree_data: Output from load_sudoku_tree_data().
        alpha_beta_configs: List of (alpha, beta) tuples.
        config: Algorithm config dict.
        noise_scale: Noise to add to weak scores.
        n_passes: Number of passes over data.
        seed: Random seed.
    """
    np.random.seed(seed)
    puzzles = tree_data["trees"]

    results = {
        "metadata": {"noise_scale": noise_scale, "n_passes": n_passes},
        "strong": {},
        "weak": {},
        "adaptive": {},
    }

    print(f"Running experiment: noise_scale={noise_scale}, n_passes={n_passes}")
    print(f"  Puzzles: {len(puzzles)}, Effective: {len(puzzles) * n_passes}")

    # Strong baseline
    print("\nRunning Strong Baseline...")
    strong = SudokuStrongBaseline()
    strong_res = [strong.solve_puzzle(p) for p in puzzles]
    results["strong"] = {
        "accuracy": sum(r["all_steps_correct"] for r in strong_res) / len(puzzles),
        "calls": sum(r["num_strong_verifier_calls"] for r in strong_res),
    }
    print(f"  Accuracy: {100 * results['strong']['accuracy']:.1f}%, Calls: {results['strong']['calls']}")

    # Weak baseline
    print("Running Weak Baseline...")
    weak = SudokuWeakBaseline(max_attempts_per_step=2, noise_scale=noise_scale)
    weak_res = [weak.solve_puzzle(p) for p in puzzles]
    results["weak"] = {
        "accuracy": sum(r["all_steps_correct"] for r in weak_res) / len(puzzles),
        "calls": 0,
    }
    print(f"  Accuracy: {100 * results['weak']['accuracy']:.1f}%, Calls: 0")

    # Adaptive with multi-pass
    print(f"\nRunning Adaptive ({n_passes} passes)...")
    for alpha, beta in alpha_beta_configs:
        cfg = config.copy()
        cfg["alpha"], cfg["beta"] = alpha, beta

        runner = SudokuAdaptiveRunner(cfg, noise_scale=noise_scale)
        for pass_idx in range(n_passes):
            shuffled = puzzles.copy()
            if pass_idx > 0:
                np.random.shuffle(shuffled)
            for p in shuffled:
                runner.solve_puzzle(p)

        acc_errs, rej_errs = runner.get_cumulative_errors()
        results["adaptive"][(alpha, beta)] = {
            "accept_errors": acc_errs,
            "reject_errors": rej_errs,
            "thresholds": runner.get_thresholds(),
            "tau_A_history": runner.tau_A_history,
            "tau_R_history": runner.tau_R_history,
            "total_samples": len(runner.all_samples),
        }
        print(
            f"  α={alpha}, β={beta}: "
            f"Final α̂={acc_errs[-1]:.3f}, β̂={rej_errs[-1]:.3f}, "
            f"τ_A={runner.tau_A:.3f}, τ_R={runner.tau_R:.3f}"
        )

    return results


# ============================================================================
# Data loading convenience
# ============================================================================

def load_sudoku_tree_data(path: str) -> dict:
    """Load Sudoku tree data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    print(f"✓ Loaded from {path}")
    print(f"  Puzzles: {len(data['trees'])}")
    print(f"  Depth: {data['metadata'].get('depth_n', 'N/A')}")
    print(f"  Branching: {data['metadata'].get('branching_m', 'N/A')}")

    total_nodes = sum(len(t["nodes"]) - 1 for t in data["trees"])
    total_correct = sum(
        sum(1 for n in t["nodes"].values() if n.get("is_correct", False) and n["depth"] > 0)
        for t in data["trees"]
    )
    print(f"  Total nodes: {total_nodes}")
    if total_nodes > 0:
        print(f"  Accuracy: {100 * total_correct / total_nodes:.1f}%")
    return data


# ============================================================================
# Plotting
# ============================================================================

def plot_error_rates(results: dict, alpha: float, beta: float, save_path: str = None):
    """Plot cumulative error rates over time for one (α, β)."""
    data = results["adaptive"].get((alpha, beta))
    if not data:
        print(f"No results for α={alpha}, β={beta}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(data["accept_errors"], color="blue", linewidth=1.5)
    axes[0].axhline(y=alpha, color="blue", linestyle="--", linewidth=2, label=f"Target α = {alpha}")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative Accept Error")
    axes[0].set_title("Accept Error Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["reject_errors"], color="red", linewidth=1.5)
    axes[1].axhline(y=beta, color="red", linestyle="--", linewidth=2, label=f"Target β = {beta}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative Reject Error")
    axes[1].set_title("Reject Error Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    noise = results.get("metadata", {}).get("noise_scale", 0)
    n_passes = results.get("metadata", {}).get("n_passes", 1)
    plt.suptitle(f"Error Rates (α={alpha}, β={beta}, noise={noise}, passes={n_passes})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_error_rates(results: dict, save_path: str = None):
    """Plot error rates for all (α, β) configs (normalized x-axis)."""
    n_configs = len(results["adaptive"])
    if n_configs == 0:
        print("No adaptive results")
        return

    fig, axes = plt.subplots(n_configs, 1, figsize=(8, 4 * n_configs))
    if n_configs == 1:
        axes = [axes]

    for idx, ((alpha, beta), data) in enumerate(results["adaptive"].items()):
        ax = axes[idx]
        n_steps = len(data["accept_errors"])
        x_norm = np.linspace(0, 1, n_steps)

        ax.plot(x_norm, data["accept_errors"], color="#377eb8", linewidth=1.5, label="Accept Error")
        ax.plot(x_norm, data["reject_errors"], color="red", linewidth=1.5, label="Reject Error")
        ax.axhline(y=alpha, color="#377eb8", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Target α={alpha}")
        ax.axhline(y=beta, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Target β={beta}")
        ax.set_xlabel("Fraction of Total Steps", fontsize=11)
        ax.set_ylabel("Cumulative Error Rate", fontsize=11)
        ax.set_title("Accept/Reject Error Convergence", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(alpha * 3, beta * 2, 0.4))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_threshold_evolution(results: dict, alpha: float, beta: float, save_path: str = None):
    """Plot threshold evolution over time."""
    data = results["adaptive"].get((alpha, beta))
    if not data:
        print(f"No results for α={alpha}, β={beta}")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["tau_A_history"], label="τ_A (accept)", color="blue", linewidth=1.5)
    ax.plot(data["tau_R_history"], label="τ_R (reject)", color="orange", linewidth=1.5)
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Threshold Value")
    ax.set_title(f"Threshold Evolution (α={alpha}, β={beta})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

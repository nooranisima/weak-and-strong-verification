"""
Sudoku: accuracy-vs-latency pipeline with grid search.

Full pipeline:
1. Compute baselines (strong, multiple weak Best-of-N).
2. For each (α, β), grid-search best hyperparams.
3. Run adaptive algorithm and measure accuracy / strong-verifier calls.
4. Plot and save results.
"""

import itertools
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .simulation import (
    add_noise_to_score,
    get_candidates_at_node,
)


# ============================================================================
# Adaptive runner with explicit RNG (for reproducible parallel runs)
# ============================================================================

class SudokuAdaptiveRunnerWithRNG:
    """Adaptive Best-of-N with explicit RNG for parallelisation."""

    def __init__(self, config: dict, rng: np.random.RandomState, noise_scale: float = 0.0):
        self.config = config
        self.rng = rng
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
        self.all_samples: List[dict] = []

    def get_P_a(self, t): return max(self.P_a_min, self.P_a_init / np.sqrt(t + 1))
    def get_P_r(self, t): return max(self.P_r_min, self.P_r_init / np.sqrt(t + 1))

    def update_thresholds(self, w_t, H_t, region):
        t = self.global_step
        g_t = 0.0
        if region == "accept":
            g_t = ((1 if not H_t else 0) * (1 - self.alpha)) / self.get_P_a(t)
        elif region == "uncertainty":
            g_t = (1 if not H_t else 0) * (-self.alpha)
        else:
            g_t = ((1 if not H_t else 0) * (-self.alpha)) / self.get_P_r(t)
        self.tau_A = max(self.tau_R + 0.05, min(1.0, self.tau_A + self.eta * g_t))

        g_beta = 0.0
        if region == "accept":
            g_beta = ((1 if H_t else 0) * self.beta) / self.get_P_a(t)
        elif region == "uncertainty":
            g_beta = (1 if H_t else 0) * self.beta
        else:
            g_beta = ((1 if H_t else 0) * (-(1 - self.beta))) / self.get_P_r(t)
        self.tau_R = max(0.0, min(self.tau_A - 0.05, self.tau_R + self.eta_R * g_beta))

    def solve_puzzle(self, puzzle_data):
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
                w_t = add_noise_to_score(cand["weak_score"], self.noise_scale, self.rng)
                H_t = cand["is_correct"]
                queried = False; region = None; decision = None

                if w_t > self.tau_A:
                    region = "accept"
                    if self.rng.random() < self.get_P_a(self.global_step):
                        queried = True; result["num_strong_verifier_calls"] += 1
                        decision = "accept" if H_t else "reject"
                    else:
                        decision = "accept"
                elif w_t > self.tau_R:
                    region = "uncertainty"; queried = True
                    result["num_strong_verifier_calls"] += 1
                    decision = "accept" if H_t else "reject"
                else:
                    region = "reject"
                    if self.rng.random() < self.get_P_r(self.global_step):
                        queried = True; result["num_strong_verifier_calls"] += 1
                        decision = "accept" if H_t else "reject"
                    else:
                        decision = "reject"

                if queried:
                    self.update_thresholds(w_t, H_t, region)
                self.all_samples.append({"is_correct": H_t, "accepted": decision == "accept"})
                if decision == "accept":
                    chosen = cand; break

            if chosen is None:
                chosen = candidates[-1]
            if chosen["is_correct"]:
                result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]

        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result

    def get_empirical_errors(self):
        fa = fr = ti = tc = 0
        for s in self.all_samples:
            if s["is_correct"]:
                tc += 1
                if not s["accepted"]: fr += 1
            else:
                ti += 1
                if s["accepted"]: fa += 1
        return {
            "accept_error": fa / ti if ti > 0 else 0,
            "reject_error": fr / tc if tc > 0 else 0,
            "total_correct": tc, "total_incorrect": ti,
        }


# ============================================================================
# Baselines (with weak-call tracking)
# ============================================================================

class SudokuStrongBaseline:
    def __init__(self, max_attempts_per_step=None):
        self.max_attempts_per_step = max_attempts_per_step

    def solve_puzzle(self, puzzle_data):
        depth_n = puzzle_data["depth_n"]
        max_per_step = self.max_attempts_per_step or puzzle_data["branching_m"]
        result = {"num_strong_verifier_calls": 0, "num_steps_correct": 0, "num_weak_verifier_calls": 0}
        current_parent_id = puzzle_data["root_id"]
        for step_idx in range(depth_n):
            candidates = get_candidates_at_node(puzzle_data, current_parent_id, step_idx + 1)
            if not candidates: break
            candidates = candidates[:max_per_step]
            chosen = None
            for cand in candidates:
                result["num_strong_verifier_calls"] += 1
                result["num_weak_verifier_calls"] += 1
                if cand["is_correct"]: chosen = cand; break
            if chosen is None: chosen = candidates[-1]
            if chosen["is_correct"]: result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]
        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result


class SudokuWeakBaseline:
    def __init__(self, max_attempts_per_step=None, noise_scale=0.0, rng=None):
        self.max_attempts_per_step = max_attempts_per_step
        self.noise_scale = noise_scale
        self.rng = rng or np.random.RandomState()

    def solve_puzzle(self, puzzle_data):
        depth_n = puzzle_data["depth_n"]
        max_per_step = self.max_attempts_per_step or puzzle_data["branching_m"]
        result = {"num_strong_verifier_calls": 0, "num_steps_correct": 0, "num_weak_verifier_calls": 0}
        current_parent_id = puzzle_data["root_id"]
        for step_idx in range(depth_n):
            candidates = get_candidates_at_node(puzzle_data, current_parent_id, step_idx + 1)
            if not candidates: break
            candidates = candidates[:max_per_step]
            result["num_weak_verifier_calls"] += len(candidates)
            scored = [(add_noise_to_score(c["weak_score"], self.noise_scale, self.rng), c) for c in candidates]
            chosen = max(scored, key=lambda x: x[0])[1]
            if chosen["is_correct"]: result["num_steps_correct"] += 1
            current_parent_id = chosen["node_id"]
        result["all_steps_correct"] = result["num_steps_correct"] == depth_n
        return result


# ============================================================================
# Grid search
# ============================================================================

def _grid_search_worker(params, puzzles, alpha, beta, noise_scale, seed, max_attempts_per_step):
    eta, eta_R, tau_A_init, tau_R_init = params
    if tau_R_init >= tau_A_init - 0.05:
        return None
    rng = np.random.RandomState(seed)
    config = {
        "max_attempts_per_step": max_attempts_per_step,
        "eta": eta, "eta_R": eta_R,
        "tau_A_init": tau_A_init, "tau_R_init": tau_R_init,
        "alpha": alpha, "beta": beta,
        "P_a_init": 0.3, "P_r_init": 0.3, "P_a_min": 0.05, "P_r_min": 0.05,
    }
    runner = SudokuAdaptiveRunnerWithRNG(config, rng, noise_scale)
    for puzzle in puzzles:
        runner.solve_puzzle(puzzle)
    errors = runner.get_empirical_errors()
    ae, re = errors["accept_error"], errors["reject_error"]
    return {
        "eta": eta, "eta_R": eta_R,
        "tau_A_init": tau_A_init, "tau_R_init": tau_R_init,
        "accept_error": ae, "reject_error": re,
        "score": abs(ae - alpha) + abs(re - beta),
        "params": params,
    }


def grid_search_for_config_sudoku(
    puzzles, alpha, beta, noise_scale=0.0,
    eta_values=None, eta_R_values=None, tau_A_offsets=None, tau_R_offsets=None,
    seed=42, n_workers=None, verbose=False, max_attempts_per_step=5,
):
    if eta_values is None:
        eta_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    if eta_R_values is None:
        eta_R_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    if tau_A_offsets is None:
        tau_A_offsets = [-0.1, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.1]
    if tau_R_offsets is None:
        tau_R_offsets = [-0.1, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, 0.1]

    tau_A_values = [np.clip(1 - alpha + off, 0.1, 0.99) for off in tau_A_offsets]
    tau_R_values = [np.clip(beta + off, 0.01, 0.9) for off in tau_R_offsets]
    all_params = list(itertools.product(eta_values, eta_R_values, tau_A_values, tau_R_values))

    if verbose:
        print(f"    Grid search: {len(all_params)} combinations...")

    worker_fn = partial(
        _grid_search_worker, puzzles=puzzles, alpha=alpha, beta=beta,
        noise_scale=noise_scale, seed=seed, max_attempts_per_step=max_attempts_per_step,
    )

    best_score, best_config = float("inf"), None
    if n_workers == 1:
        for params in all_params:
            result = worker_fn(params)
            if result and result["score"] < best_score:
                best_score, best_config = result["score"], result
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for future in as_completed(
                {executor.submit(worker_fn, p): p for p in all_params}
            ):
                result = future.result()
                if result and result["score"] < best_score:
                    best_score, best_config = result["score"], result

    if verbose and best_config:
        print(f"    Best: η={best_config['eta']:.2f}, η_R={best_config['eta_R']:.2f}, "
              f"ae={best_config['accept_error']:.3f}, re={best_config['reject_error']:.3f}")
    return best_config


def grid_search_fast(puzzles, alpha, beta, noise_scale=0.0, seed=42,
                     n_workers=None, verbose=False, max_attempts_per_step=5):
    return grid_search_for_config_sudoku(
        puzzles, alpha, beta, noise_scale,
        eta_values=[0.02, 0.05, 0.1, 0.15],
        eta_R_values=[0.02, 0.05, 0.1, 0.15],
        tau_A_offsets=[-0.05, 0, 0.05],
        tau_R_offsets=[-0.05, 0, 0.05],
        seed=seed, n_workers=n_workers, verbose=verbose,
        max_attempts_per_step=max_attempts_per_step,
    )


def grid_search_coarse_to_fine(puzzles, alpha, beta, noise_scale=0.0, seed=42,
                               n_workers=None, verbose=False, max_attempts_per_step=5):
    best_coarse = grid_search_for_config_sudoku(
        puzzles, alpha, beta, noise_scale,
        eta_values=[0.02, 0.05, 0.1, 0.2],
        eta_R_values=[0.02, 0.05, 0.1, 0.2],
        tau_A_offsets=[-0.1, 0, 0.1],
        tau_R_offsets=[-0.1, 0, 0.1],
        seed=seed, n_workers=n_workers, verbose=False,
        max_attempts_per_step=max_attempts_per_step,
    )
    if best_coarse is None:
        return None

    be, ber = best_coarse["eta"], best_coarse["eta_R"]
    fine_eta = list(set([max(0.01, be - 0.02), be, min(0.3, be + 0.02)]))
    fine_eta_R = list(set([max(0.01, ber - 0.02), ber, min(0.3, ber + 0.02)]))
    best_a_off = best_coarse["tau_A_init"] - (1 - alpha)
    best_r_off = best_coarse["tau_R_init"] - beta

    best_fine = grid_search_for_config_sudoku(
        puzzles, alpha, beta, noise_scale,
        fine_eta, fine_eta_R,
        [best_a_off - 0.03, best_a_off, best_a_off + 0.03],
        [best_r_off - 0.03, best_r_off, best_r_off + 0.03],
        seed=seed, n_workers=n_workers, verbose=False,
        max_attempts_per_step=max_attempts_per_step,
    )
    if best_fine is None or best_coarse["score"] <= best_fine["score"]:
        return best_coarse
    return best_fine


# ============================================================================
# Full pipeline
# ============================================================================

def run_full_pipeline_sudoku(
    tree_data, alpha_beta_values, noise_scale=0.0, seed=42, n_runs=3,
    n_workers=None, max_attempts_per_step=5, verbose=True,
    grid_search_mode="fast", weak_baseline_n_values=None,
):
    puzzles = tree_data["trees"]
    n_puzzles = len(puzzles)
    depth_n = tree_data["metadata"].get("depth_n", 3)
    branching_m = tree_data["metadata"].get("branching_m", 3)
    if weak_baseline_n_values is None:
        weak_baseline_n_values = list(range(1, branching_m + 1))

    results = {
        "metadata": {
            "noise_scale": noise_scale, "n_puzzles": n_puzzles, "n_runs": n_runs,
            "max_attempts_per_step": max_attempts_per_step,
            "grid_search_mode": grid_search_mode,
            "depth_n": depth_n, "branching_m": branching_m,
        },
        "strong_baseline": {}, "weak_baseline": {}, "adaptive": {},
        "configs": {}, "weak_baseline_n_values": weak_baseline_n_values,
    }

    print(f"Running Sudoku pipeline: {n_puzzles} puzzles, noise={noise_scale}")

    # Strong baseline
    strong = SudokuStrongBaseline(max_attempts_per_step=max_attempts_per_step)
    sr = [strong.solve_puzzle(p) for p in puzzles]
    results["strong_baseline"] = {
        "accuracy": sum(r["all_steps_correct"] for r in sr) / n_puzzles,
        "strong_calls": sum(r["num_strong_verifier_calls"] for r in sr),
        "weak_calls": sum(r["num_weak_verifier_calls"] for r in sr),
        "calls_per_puzzle": sum(r["num_strong_verifier_calls"] for r in sr) / n_puzzles,
        "weak_calls_per_puzzle": sum(r["num_weak_verifier_calls"] for r in sr) / n_puzzles,
        "n_puzzles": n_puzzles,
    }
    print(f"  Strong: {results['strong_baseline']['accuracy']*100:.1f}% @ "
          f"{results['strong_baseline']['calls_per_puzzle']:.2f} strong calls/puzzle")

    # Multiple weak baselines
    for n_val in weak_baseline_n_values:
        rng = np.random.RandomState(seed + n_val)
        weak = SudokuWeakBaseline(max_attempts_per_step=n_val, noise_scale=noise_scale, rng=rng)
        wr = [weak.solve_puzzle(p) for p in puzzles]
        results["weak_baseline"][n_val] = {
            "accuracy": sum(r["all_steps_correct"] for r in wr) / n_puzzles,
            "strong_calls": 0,
            "weak_calls": sum(r["num_weak_verifier_calls"] for r in wr),
            "calls_per_puzzle": 0,
            "weak_calls_per_puzzle": sum(r["num_weak_verifier_calls"] for r in wr) / n_puzzles,
            "n_puzzles": n_puzzles,
        }
        print(f"  Weak (BoN-{n_val}): {results['weak_baseline'][n_val]['accuracy']*100:.1f}%")

    # Grid search + adaptive
    gs_fn = {"fast": grid_search_fast, "coarse_to_fine": grid_search_coarse_to_fine}.get(
        grid_search_mode, grid_search_for_config_sudoku
    )

    for alpha, beta in alpha_beta_values:
        print(f"\n  (α={alpha:.3f}, β={beta:.3f}):")
        best_cfg = gs_fn(puzzles, alpha, beta, noise_scale, seed, n_workers, verbose, max_attempts_per_step)
        results["configs"][(alpha, beta)] = best_cfg

        accs, sc_list, wc_list = [], [], []
        for run_idx in range(n_runs):
            rng = np.random.RandomState(seed * 1000 + run_idx)
            cfg = {
                "max_attempts_per_step": max_attempts_per_step,
                "eta": best_cfg["eta"], "eta_R": best_cfg["eta_R"],
                "tau_A_init": best_cfg["tau_A_init"], "tau_R_init": best_cfg["tau_R_init"],
                "alpha": alpha, "beta": beta,
                "P_a_init": 0.1, "P_r_init": 0.1, "P_a_min": 0.05, "P_r_min": 0.05,
            }
            runner = SudokuAdaptiveRunnerWithRNG(cfg, rng, noise_scale)
            correct = sc = wc = 0
            for p in puzzles:
                r = runner.solve_puzzle(p)
                if r["all_steps_correct"]: correct += 1
                sc += r["num_strong_verifier_calls"]
                wc += r["num_weak_verifier_calls"]
            accs.append(correct / n_puzzles)
            sc_list.append(sc); wc_list.append(wc)

        results["adaptive"][(alpha, beta)] = {
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "strong_calls_mean": np.mean(sc_list), "strong_calls_std": np.std(sc_list),
            "calls_per_puzzle_mean": np.mean(sc_list) / n_puzzles,
            "calls_per_puzzle_std": np.std(sc_list) / n_puzzles,
            "weak_calls_mean": np.mean(wc_list), "weak_calls_std": np.std(wc_list),
            "weak_calls_per_puzzle_mean": np.mean(wc_list) / n_puzzles,
            "weak_calls_per_puzzle_std": np.std(wc_list) / n_puzzles,
        }
        print(f"    acc={np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%, "
              f"strong={np.mean(sc_list)/n_puzzles:.2f} calls/puzzle")

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_accuracy_vs_latency_sudoku(results, save_path=None, weak_n_values_to_plot=None, y_padding=3):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    strong = results["strong_baseline"]
    weak_baselines = results["weak_baseline"]
    adaptive = results["adaptive"]

    avail = sorted(results.get("weak_baseline_n_values", list(weak_baselines.keys())))
    if weak_n_values_to_plot is None:
        weak_n_values_to_plot = avail

    weak_colors = {1: "#FF6B6B", 2: "#4ECDC4", 3: "#9B59B6", 4: "#F39C12", 5: "#1ABC9C"}
    strong_acc = strong["accuracy"] * 100
    strong_cpp = strong["calls_per_puzzle"]
    all_accs = [strong_acc]

    # Adaptive points (symmetric only)
    pts = sorted(
        [{"cpp": d["calls_per_puzzle_mean"], "acc": d["accuracy_mean"]*100,
          "alpha": a} for (a, b), d in adaptive.items() if a == b],
        key=lambda x: x["alpha"],
    )
    # Filter monotonic
    if pts:
        best_i = max(range(len(pts)), key=lambda i: pts[i]["acc"])
        filt = [pts[best_i]]
        for i in range(best_i + 1, len(pts)):
            if pts[i]["acc"] <= filt[-1]["acc"] and pts[i]["cpp"] <= filt[-1]["cpp"]:
                filt.append(pts[i])
        filt.sort(key=lambda x: x["cpp"])
        all_accs.extend(p["acc"] for p in filt)
    else:
        filt = []

    for n_val in sorted(weak_n_values_to_plot):
        if n_val in weak_baselines:
            wa = weak_baselines[n_val]["accuracy"] * 100
            all_accs.append(wa)
            ax.scatter(0, wa, color=weak_colors.get(n_val, "gray"), marker="o", s=120,
                       zorder=9, edgecolors="black", linewidths=0.5, label=f"BoN-{n_val} ({wa:.1f}%)")

    ax.scatter(strong_cpp, strong_acc, color="black", marker="*", s=400, zorder=10,
               label=f"Strong ({strong_acc:.1f}%)")

    if filt:
        ax.plot([p["cpp"] for p in filt], [p["acc"] for p in filt], "o-", color="blue",
                markersize=9, linewidth=2.5, alpha=0.8, label="Adaptive")
        for i, p in enumerate(filt):
            ax.annotate(f"{p['alpha']:.2f}", (p["cpp"], p["acc"]),
                        textcoords="offset points", xytext=(0, 20 if i % 2 == 0 else -26),
                        fontsize=12, ha="center", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, shrinkA=0, shrinkB=3))

    ax.axhline(y=strong_acc, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Strong Verifier Calls / Puzzle", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Sudoku: Accuracy vs Latency Tradeoff", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(max(0, min(all_accs) - y_padding), min(100, max(all_accs) + y_padding))
    ax.set_xlim(-0.1, strong_cpp * 1.1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ============================================================================
# Save / load results
# ============================================================================

def save_results(results: dict, path: str, use_pickle: bool = True):
    """Save pipeline results. Pickle preserves tuple keys; JSON converts them."""
    if use_pickle or path.endswith('.pkl'):
        with open(path, "wb") as f:
            pickle.dump(results, f)
    else:
        results_json = _convert_tuple_keys_to_str(results)
        with open(path, "w") as f:
            json.dump(results_json, f, indent=2)
    print(f"✓ Results saved to: {path}")


def load_results(path: str) -> dict:
    """Load pipeline results from .pkl or .json."""
    if path.endswith('.pkl'):
        with open(path, "rb") as f:
            results = pickle.load(f)
    else:
        with open(path, "r") as f:
            results = json.load(f)
        results = _convert_str_keys_to_tuple(results)
    print(f"✓ Results loaded from: {path}")
    return results


def _convert_tuple_keys_to_str(obj):
    if isinstance(obj, dict):
        return {(f"({k[0]},{k[1]})" if isinstance(k, tuple) else k):
                _convert_tuple_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_tuple_keys_to_str(i) for i in obj]
    return obj


def _convert_str_keys_to_tuple(obj):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                parts = k[1:-1].split(',')
                k = (float(parts[0]), float(parts[1]))
            new[k] = _convert_str_keys_to_tuple(v)
        return new
    elif isinstance(obj, list):
        return [_convert_str_keys_to_tuple(i) for i in obj]
    return obj

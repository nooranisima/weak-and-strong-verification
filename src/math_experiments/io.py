"""Save and load pipeline results with proper key conversion."""

import json
import os


def save_pipeline_results(
    results: dict, save_dir: str, filename: str = "pipeline_results.json"
):
    """
    Save pipeline results to JSON file.
    Handles tuple keys by converting to strings.
    """
    os.makedirs(save_dir, exist_ok=True)

    serializable = {
        "strong_baseline": {},
        "weak_baseline": {},
        "adaptive": {},
        "configs": {},
        "weak_baseline_n_values": results.get("weak_baseline_n_values", [5]),
    }

    for diff, data in results["strong_baseline"].items():
        serializable["strong_baseline"][str(diff)] = data

    for n_val, diff_data in results["weak_baseline"].items():
        serializable["weak_baseline"][str(n_val)] = {}
        for diff, data in diff_data.items():
            serializable["weak_baseline"][str(n_val)][str(diff)] = data

    for diff, ab_data in results["adaptive"].items():
        serializable["adaptive"][str(diff)] = {}
        for (alpha, beta), data in ab_data.items():
            key = f"{alpha},{beta}"
            serializable["adaptive"][str(diff)][key] = data

    if "configs" in results:
        for diff, ab_data in results["configs"].items():
            serializable["configs"][str(diff)] = {}
            for (alpha, beta), data in ab_data.items():
                key = f"{alpha},{beta}"
                serializable["configs"][str(diff)][key] = data

    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"✓ Saved pipeline results to: {filepath}")
    return filepath


def load_pipeline_results(
    save_dir: str, filename: str = "pipeline_results.json"
) -> dict:
    """
    Load pipeline results from JSON file.
    Converts string keys back to proper types.
    """
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "r") as f:
        serializable = json.load(f)

    results = {
        "strong_baseline": {},
        "weak_baseline": {},
        "adaptive": {},
        "configs": {},
        "weak_baseline_n_values": serializable.get("weak_baseline_n_values", [5]),
    }

    for diff_str, data in serializable["strong_baseline"].items():
        results["strong_baseline"][int(diff_str)] = data

    for n_val_str, diff_data in serializable["weak_baseline"].items():
        n_val = int(n_val_str)
        results["weak_baseline"][n_val] = {}
        for diff_str, data in diff_data.items():
            results["weak_baseline"][n_val][int(diff_str)] = data

    for diff_str, ab_data in serializable["adaptive"].items():
        diff = int(diff_str)
        results["adaptive"][diff] = {}
        for key, data in ab_data.items():
            alpha, beta = map(float, key.split(","))
            results["adaptive"][diff][(alpha, beta)] = data

    if "configs" in serializable:
        for diff_str, ab_data in serializable["configs"].items():
            diff = int(diff_str)
            results["configs"][diff] = {}
            for key, data in ab_data.items():
                alpha, beta = map(float, key.split(","))
                results["configs"][diff][(alpha, beta)] = data

    print(f"✓ Loaded pipeline results from: {filepath}")
    print(f"  Difficulties: {sorted(results['strong_baseline'].keys())}")
    print(f"  Weak baseline N values: {results['weak_baseline_n_values']}")
    print(
        f"  Number of (α,β) configs: "
        f"{len(list(results['adaptive'].values())[0])}"
    )

    return results

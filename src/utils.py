"""Utility functions for saving, loading, and converting experiment data."""

import json
import numpy as np
from typing import Any


def convert_results_for_json(results: dict) -> dict:
    """
    Convert experiment results to JSON-serializable format.
    Handles numpy types, tuple keys, etc.
    """
    return _make_serializable(results)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to be JSON-serializable."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Convert tuple keys to strings
            if isinstance(k, tuple):
                k = str(k)
            new_dict[str(k)] = _make_serializable(v)
        return new_dict
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def load_pregenerated_data(path: str) -> dict:
    """Load pre-generated data from JSON, converting keys to proper types."""
    with open(path, "r") as f:
        data = json.load(f)

    # Convert string keys back to integers
    new_data = {}
    for diff_key, diff_val in data["data"].items():
        diff_int = int(diff_key)
        new_data[diff_int] = {}
        for prob_key, prob_val in diff_val.items():
            prob_int = int(prob_key)
            new_data[diff_int][prob_int] = prob_val
    data["data"] = new_data

    return data


def save_results_json(results: dict, path: str):
    """Save experiment results to JSON."""
    serializable = convert_results_for_json(results)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"✓ Saved results to {path}")

"""Experiment configuration for adaptive verification."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ExperimentConfig:
    """Configuration for the adaptive threshold experiment (Best-of-N)."""

    # API Keys (set via environment variables or directly)
    openai_api_key: str = ""
    deepseek_api_key: str = ""

    # Model Configuration
    generator_model: str = "gpt-4o-mini"
    strong_verifier_model: str = "gpt-4o-mini"

    # Weak verifier selection
    use_deepseek_weak_verifier: bool = True
    use_math_shepherd: bool = False

    # Threshold Configuration
    tau_A_init: float = 0.7
    tau_R_init: float = 0.3

    # Learning rates
    eta: float = 0.1
    eta_R: float = 0.1

    # Target error rates
    alpha: float = 0.10
    beta: float = 0.10

    # Query probabilities
    P_a: float = 0.3
    P_r: float = 0.3

    # Best-of-N settings
    max_attempts: int = 5

    # Legacy step-level settings (kept for compatibility)
    max_steps_per_problem: int = 15
    max_retries_per_step: int = 3
    collect_all_ground_truth: bool = True

    @property
    def verifier_model(self):
        """Legacy compatibility alias."""
        return self.strong_verifier_model

    def get_P_a(self, t: int) -> float:
        """Get P_a for step t (can be made dynamic)."""
        return self.P_a

    def get_P_r(self, t: int) -> float:
        """Get P_r for step t (can be made dynamic)."""
        return self.P_r


def load_config(yaml_path: str) -> dict:
    """Load experiment config from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def make_experiment_config(
    openai_api_key: str = None,
    deepseek_api_key: str = None,
    **overrides,
) -> ExperimentConfig:
    """Create ExperimentConfig with API keys from env or arguments."""
    cfg = ExperimentConfig()
    cfg.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    cfg.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg

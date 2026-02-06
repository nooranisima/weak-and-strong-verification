"""Adaptive threshold algorithm for Best-of-N verification."""

import numpy as np
from typing import Dict, List, Tuple


class SimulatedAdaptiveRunner:
    """
    Adaptive Best-of-N with threshold updates.

    Maintains accept threshold (tau_A) and reject threshold (tau_R),
    updating them based on strong verifier feedback to control error rates.
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_attempts = config.get("max_attempts", 5)

        # Target error rates
        self.alpha = config.get("alpha", 0.10)
        self.beta = config.get("beta", 0.10)

        # Initial thresholds
        self.tau_A = config.get("initial_tau_A", 1 - self.alpha)
        self.tau_R = config.get("initial_tau_R", self.beta)

        # Learning rates
        self.eta = config.get("eta", 0.1)
        self.eta_R = config.get("eta_R", 0.1)

        # Query probabilities
        self.P_a_init = config.get("P_a_init", 0.3)
        self.P_r_init = config.get("P_r_init", 0.3)
        self.P_a_min = config.get("P_a_min", 0.05)
        self.P_r_min = config.get("P_r_min", 0.05)

        # Tracking
        self.global_step = 0
        self.tau_A_history: List[float] = []
        self.tau_R_history: List[float] = []
        self.all_step_ground_truth: List[Tuple[float, bool, bool]] = []

    def get_P_a(self, t: int) -> float:
        """Query probability in accept region (decreases over time)."""
        return max(self.P_a_min, self.P_a_init / np.sqrt(t + 1))

    def get_P_r(self, t: int) -> float:
        """Query probability in reject region (decreases over time)."""
        return max(self.P_r_min, self.P_r_init / np.sqrt(t + 1))

    def update_thresholds(self, w_t: float, H_t: bool, queried: bool):
        """
        Update both thresholds based on verification result.

        Args:
            w_t: Weak verifier score
            H_t: Strong verifier result (True if correct)
            queried: Whether strong verifier was actually queried
        """
        t = self.global_step

        # ================================================================
        # UPDATE τ_A
        # ================================================================
        g_t = 0.0

        if w_t > self.tau_A:
            # Accept region
            if queried:
                P_a = self.get_P_a(t)
                indicator_H_false = 1 if not H_t else 0
                g_t = (indicator_H_false * (1 - self.alpha)) / P_a
        elif w_t > self.tau_R:
            # Uncertainty region (always queried)
            indicator_H_false = 1 if not H_t else 0
            g_t = indicator_H_false * (-self.alpha)
        else:
            # Reject region
            if queried:
                P_r = self.get_P_r(t)
                indicator_H_false = 1 if not H_t else 0
                g_t = (indicator_H_false * (-self.alpha)) / P_r

        self.tau_A = self.tau_A + self.eta * g_t
        self.tau_A = max(self.tau_R + 0.05, min(1.0, self.tau_A))

        # ================================================================
        # UPDATE τ_R
        # ================================================================
        g_beta_t = 0.0

        if w_t > self.tau_A:
            # Accept region
            if queried:
                P_a = self.get_P_a(t)
                indicator_H_true = 1 if H_t else 0
                g_beta_t = (indicator_H_true * self.beta) / P_a
        elif w_t > self.tau_R:
            # Uncertainty region
            indicator_H_true = 1 if H_t else 0
            g_beta_t = indicator_H_true * self.beta
        else:
            # Reject region
            if queried:
                P_r = self.get_P_r(t)
                indicator_H_true = 1 if H_t else 0
                g_beta_t = (indicator_H_true * (-(1 - self.beta))) / P_r

        self.tau_R = self.tau_R + self.eta_R * g_beta_t
        self.tau_R = max(0.0, min(self.tau_A - 0.05, self.tau_R))

        # Record history
        self.tau_A_history.append(self.tau_A)
        self.tau_R_history.append(self.tau_R)

    def solve_problem(self, problem_data: dict) -> dict:
        """
        Run adaptive algorithm on a single problem's pre-generated attempts.

        Args:
            problem_data: Dict with "attempts" list

        Returns:
            Result dict with decision details
        """
        attempts = problem_data["attempts"]
        max_attempts = min(self.max_attempts, len(attempts))

        result = {
            "completed": False,
            "final_answer_correct": None,
            "attempt_details": [],
            "num_attempts": 0,
            "num_strong_verifier_calls": 0,
        }

        for attempt_idx in range(max_attempts):
            attempt = attempts[attempt_idx]
            result["num_attempts"] = attempt_idx + 1
            self.global_step += 1

            w_t = attempt["weak_score"]
            H_t = attempt["is_correct"]

            queried = False
            decision = None

            if w_t > self.tau_A:
                # ACCEPT REGION
                region = "accept"
                decision = "accept"
                P_a = self.get_P_a(self.global_step)
                if np.random.random() < P_a:
                    queried = True
                    result["num_strong_verifier_calls"] += 1

            elif w_t > self.tau_R:
                # UNCERTAINTY REGION - always query
                region = "uncertainty"
                queried = True
                result["num_strong_verifier_calls"] += 1
                decision = "accept" if H_t else "reject"

            else:
                # REJECT REGION
                region = "reject"
                decision = "reject"
                P_r = self.get_P_r(self.global_step)
                if np.random.random() < P_r:
                    queried = True
                    result["num_strong_verifier_calls"] += 1

            # Update thresholds if we have ground truth
            if queried:
                self.update_thresholds(w_t, H_t, queried=True)

            # Record for error computation (ALL steps)
            self.all_step_ground_truth.append((w_t, H_t, decision == "accept"))

            result["attempt_details"].append({
                "attempt_idx": attempt_idx,
                "weak_score": w_t,
                "is_correct": H_t,
                "region": region,
                "decision": decision,
                "queried": queried,
                "tau_A": self.tau_A,
                "tau_R": self.tau_R,
            })

            if decision == "accept":
                result["completed"] = True
                result["final_answer_correct"] = H_t
                break

        if not result["completed"]:
            result["final_answer_correct"] = attempts[max_attempts - 1]["is_correct"]

        return result

    def get_empirical_errors(self) -> dict:
        """Compute empirical error rates from all steps."""
        if not self.all_step_ground_truth:
            return {"accept_error": 0, "reject_error": 0}

        # Accept error: P(accepted | H_t = False)
        incorrect_steps = [
            (w, acc) for w, H, acc in self.all_step_ground_truth if not H
        ]
        accept_error = (
            sum(acc for _, acc in incorrect_steps) / len(incorrect_steps)
            if incorrect_steps
            else 0
        )

        # Reject error: P(rejected | H_t = True)
        correct_steps = [
            (w, acc) for w, H, acc in self.all_step_ground_truth if H
        ]
        reject_error = (
            sum(1 - acc for _, acc in correct_steps) / len(correct_steps)
            if correct_steps
            else 0
        )

        return {"accept_error": accept_error, "reject_error": reject_error}

    def get_thresholds(self) -> dict:
        return {"tau_A": self.tau_A, "tau_R": self.tau_R}

    def set_thresholds(self, tau_A: float, tau_R: float):
        """Set thresholds (e.g., from warmup phase)."""
        self.tau_A = tau_A
        self.tau_R = tau_R

    def reset(self):
        """Full reset."""
        self.tau_A = self.config.get("initial_tau_A", 1 - self.alpha)
        self.tau_R = self.config.get("initial_tau_R", self.beta)
        self.global_step = 0
        self.tau_A_history = []
        self.tau_R_history = []
        self.all_step_ground_truth = []

    def reset_tracking_only(self):
        """Reset tracking but keep learned thresholds."""
        self.global_step = 0
        self.tau_A_history = []
        self.tau_R_history = []
        self.all_step_ground_truth = []


class SimulatedAdaptiveRunnerWithRNG:
    """
    Adaptive Best-of-N with CORRECT threshold update logic.

    Key differences from SimulatedAdaptiveRunner:
    - Uses explicit numpy RandomState for reproducible parallel runs
    - NO clamping on threshold updates (matches ThresholdUpdater theory exactly)
    - Uses tau_A_init / tau_R_init config keys
    """

    def __init__(self, config: dict, rng: np.random.RandomState):
        self.config = config
        self.rng = rng
        self.max_attempts = config.get("max_attempts", 5)

        self.alpha = config.get("alpha", 0.10)
        self.beta = config.get("beta", 0.10)

        # Initial thresholds
        self.tau_A = (
            config["tau_A_init"]
            if config.get("tau_A_init") is not None
            else (1 - self.alpha)
        )
        self.tau_R = (
            config["tau_R_init"]
            if config.get("tau_R_init") is not None
            else self.beta
        )

        # Learning rates
        self.eta = config.get("eta", 0.05)
        self.eta_R = config.get("eta_R", 0.05)

        # Query probabilities
        self.P_a_init = config.get("P_a_init", 0.3)
        self.P_r_init = config.get("P_r_init", 0.3)
        self.P_a_min = config.get("P_a_min", 0.05)
        self.P_r_min = config.get("P_r_min", 0.05)

        self.global_step = 0
        self.tau_A_history: List[float] = []
        self.tau_R_history: List[float] = []
        self.all_step_ground_truth: List[Tuple[float, bool, bool]] = []

    def get_P_a(self, t: int) -> float:
        return max(self.P_a_min, self.P_a_init / np.sqrt(t + 1))

    def get_P_r(self, t: int) -> float:
        return max(self.P_r_min, self.P_r_init / np.sqrt(t + 1))

    def update_accept_threshold(
        self, w_t: float, H_t: bool, t: int, queried: bool
    ) -> float:
        """Update τ_A: τ_A^{t+1} = τ_A^t + η * g_t (no clamping)."""
        tau_A = self.tau_A
        tau_R = self.tau_R
        alpha = self.alpha

        g_t = 0.0

        if w_t > tau_A:
            if queried:
                P_a = self.get_P_a(t)
                indicator_H_false = 1 if not H_t else 0
                g_t = (indicator_H_false * (1 - alpha)) / P_a
        elif tau_R < w_t <= tau_A:
            indicator_H_false = 1 if not H_t else 0
            g_t = indicator_H_false * (-alpha)
        else:
            if queried:
                P_r = self.get_P_r(t)
                indicator_H_false = 1 if not H_t else 0
                g_t = (indicator_H_false * (-alpha)) / P_r

        self.tau_A = tau_A + self.eta * g_t
        return g_t

    def update_reject_threshold(
        self, w_t: float, H_t: bool, t: int, queried: bool
    ) -> float:
        """Update τ_R: τ_R^{t+1} = τ_R^t + η_R * g_β^t (no clamping)."""
        tau_A = self.tau_A
        tau_R = self.tau_R
        beta = self.beta

        g_beta_t = 0.0

        if w_t > tau_A:
            if queried:
                P_a = self.get_P_a(t)
                indicator_H_true = 1 if H_t else 0
                g_beta_t = (indicator_H_true * beta) / P_a
        elif tau_R < w_t <= tau_A:
            indicator_H_true = 1 if H_t else 0
            g_beta_t = indicator_H_true * beta
        else:
            if queried:
                P_r = self.get_P_r(t)
                indicator_H_true = 1 if H_t else 0
                g_beta_t = (indicator_H_true * (-(1 - beta))) / P_r

        self.tau_R = tau_R + self.eta_R * g_beta_t
        return g_beta_t

    def solve_problem(self, problem_data: dict) -> dict:
        """Run adaptive algorithm on a single problem's pre-generated attempts."""
        attempts = problem_data["attempts"]
        max_attempts = min(self.max_attempts, len(attempts))

        result = {
            "completed": False,
            "final_answer_correct": None,
            "num_attempts": 0,
            "num_strong_verifier_calls": 0,
        }

        for attempt_idx in range(max_attempts):
            attempt = attempts[attempt_idx]
            result["num_attempts"] = attempt_idx + 1
            self.global_step += 1
            t = self.global_step

            w_t = attempt["weak_score"]
            H_t = attempt["is_correct"]

            queried = False
            decision = None

            if w_t > self.tau_A:
                decision = "accept"
                P_a = self.get_P_a(t)
                if self.rng.random() < P_a:
                    queried = True
                    result["num_strong_verifier_calls"] += 1
            elif self.tau_R < w_t <= self.tau_A:
                queried = True
                result["num_strong_verifier_calls"] += 1
                decision = "accept" if H_t else "reject"
            else:
                decision = "reject"
                P_r = self.get_P_r(t)
                if self.rng.random() < P_r:
                    queried = True
                    result["num_strong_verifier_calls"] += 1

            if queried:
                self.update_accept_threshold(w_t, H_t, t, queried=True)
                self.update_reject_threshold(w_t, H_t, t, queried=True)

            self.tau_A_history.append(self.tau_A)
            self.tau_R_history.append(self.tau_R)
            self.all_step_ground_truth.append((w_t, H_t, decision == "accept"))

            if decision == "accept":
                result["completed"] = True
                result["final_answer_correct"] = H_t
                break

        if not result["completed"]:
            result["final_answer_correct"] = attempts[max_attempts - 1]["is_correct"]

        return result

    def get_empirical_errors(self) -> dict:
        """Compute empirical error rates from ground truth."""
        if not self.all_step_ground_truth:
            return {"accept_error": 0, "reject_error": 0}

        incorrect_steps = [
            (w, acc) for w, H, acc in self.all_step_ground_truth if not H
        ]
        accept_error = (
            sum(acc for _, acc in incorrect_steps) / len(incorrect_steps)
            if incorrect_steps
            else 0
        )

        correct_steps = [
            (w, acc) for w, H, acc in self.all_step_ground_truth if H
        ]
        reject_error = (
            sum(1 - acc for _, acc in correct_steps) / len(correct_steps)
            if correct_steps
            else 0
        )

        return {"accept_error": accept_error, "reject_error": reject_error}

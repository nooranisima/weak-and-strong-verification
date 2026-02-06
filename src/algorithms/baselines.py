"""Baseline algorithms for Best-of-N verification."""


class SimulatedStrongBaseline:
    """Strong baseline: Always query strong verifier, accept first correct."""

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts

    def solve_problem(self, problem_data: dict) -> dict:
        """Use strong verifier on every attempt."""
        attempts = problem_data["attempts"]
        max_attempts = min(self.max_attempts, len(attempts))

        result = {
            "completed": False,
            "final_answer_correct": None,
            "num_attempts": 0,
            "num_strong_verifier_calls": 0,
            "attempt_details": [],
        }

        for attempt_idx in range(max_attempts):
            attempt = attempts[attempt_idx]
            result["num_attempts"] = attempt_idx + 1
            result["num_strong_verifier_calls"] += 1  # Always queries

            decision = "accept" if attempt["is_correct"] else "reject"

            result["attempt_details"].append({
                "attempt_idx": attempt_idx,
                "weak_score": attempt["weak_score"],
                "is_correct": attempt["is_correct"],
                "decision": decision,
            })

            if decision == "accept":
                result["completed"] = True
                result["final_answer_correct"] = True
                break

        if not result["completed"]:
            # Take last attempt
            result["final_answer_correct"] = attempts[max_attempts - 1]["is_correct"]

        return result


class SimulatedWeakBaseline:
    """Weak baseline: Accept if weak_score > threshold (no strong verifier)."""

    def __init__(self, threshold: float = 0.5, max_attempts: int = 5):
        self.threshold = threshold
        self.max_attempts = max_attempts

    def solve_problem(self, problem_data: dict) -> dict:
        """Use only weak verifier score."""
        attempts = problem_data["attempts"]
        max_attempts = min(self.max_attempts, len(attempts))

        result = {
            "completed": False,
            "final_answer_correct": None,
            "num_attempts": 0,
            "num_strong_verifier_calls": 0,  # Never queries strong
            "attempt_details": [],
        }

        for attempt_idx in range(max_attempts):
            attempt = attempts[attempt_idx]
            result["num_attempts"] = attempt_idx + 1

            decision = (
                "accept" if attempt["weak_score"] > self.threshold else "reject"
            )

            result["attempt_details"].append({
                "attempt_idx": attempt_idx,
                "weak_score": attempt["weak_score"],
                "is_correct": attempt["is_correct"],
                "decision": decision,
            })

            if decision == "accept":
                result["completed"] = True
                result["final_answer_correct"] = attempt["is_correct"]
                break

        if not result["completed"]:
            result["final_answer_correct"] = attempts[max_attempts - 1]["is_correct"]

        return result


class SimulatedWeakBaselineBestOfN:
    """Best-of-N using weak scores only (pick highest weak score)."""

    def __init__(self, max_attempts: int = 5, rng=None):
        self.max_attempts = max_attempts
        self.total_weak_calls = 0
        self.rng = rng

    def solve_problem(self, problem_data: dict) -> dict:
        attempts = problem_data["attempts"]
        n_attempts = min(self.max_attempts, len(attempts))

        self.total_weak_calls += n_attempts

        best_score = -1
        best_indices = []
        for i in range(n_attempts):
            score = attempts[i]["weak_score"]
            if score > best_score:
                best_score = score
                best_indices = [i]
            elif score == best_score:
                best_indices.append(i)

        if self.rng is not None:
            chosen_idx = self.rng.choice(best_indices)
        else:
            chosen_idx = best_indices[0]

        return {
            "final_answer_correct": attempts[chosen_idx]["is_correct"],
            "num_attempts": n_attempts,
            "num_weak_calls": n_attempts,
            "num_strong_calls": 0,
        }

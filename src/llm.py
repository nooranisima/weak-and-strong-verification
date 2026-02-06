"""LLM wrapper for solution generation and verification (Best-of-N)."""

import json
import re
import numpy as np
from typing import Tuple
from openai import OpenAI

from .config import ExperimentConfig
from .verifiers import DeepSeekWeakVerifier


class LLMWrapper:
    """Wrapper for OpenAI API calls - Best-of-N version."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_api_key)

        # Initialize weak verifier
        if config.use_math_shepherd:
            from .verifiers import MathShepherdScorer
            self.weak_verifier = MathShepherdScorer()
        elif config.use_deepseek_weak_verifier:
            self.weak_verifier = DeepSeekWeakVerifier(config.deepseek_api_key)
        else:
            self.weak_verifier = None

        self.call_counts = {
            "generator": 0,
            "verifier": 0,
            "ground_truth": 0,
            "weak_verifier": 0,
        }

    def _is_new_model(self, model_name: str) -> bool:
        """Check if model requires new API parameters (no temperature, etc.)."""
        new_models = ["o1", "o3", "gpt-4.5", "gpt-5", "gpt5"]
        model_lower = model_name.lower().replace("-", "")
        return any(m in model_lower for m in new_models)

    # ========================================================================
    # Generate full solution (Best-of-N)
    # ========================================================================

    def generate_solution(
        self, problem: str, temperature: float = 0.7
    ) -> Tuple[str, str, float]:
        """
        Generate a complete solution with reasoning and final answer.

        Returns:
            (reasoning, final_answer, weak_score)
        """
        self.call_counts["generator"] += 1

        prompt = f"""Solve this math problem step by step. Show your reasoning, then give the final answer.

PROBLEM: {problem}

Respond with JSON:
{{
  "reasoning": "Your step-by-step reasoning here...",
  "final_answer": "Your final answer (just the answer, e.g., '42' or '3/4' or 'x=5')"
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.generator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"Error in API call: {e}")
            return "[Error generating solution]", "[Error]", 0.5

        result = self._parse_solution_json(content)
        reasoning = result.get("reasoning", "")
        final_answer = result.get("final_answer", "")

        # Get confidence from weak verifier
        if self.weak_verifier:
            self.call_counts["weak_verifier"] += 1
            weak_score = self.weak_verifier.score_solution(
                problem, reasoning, final_answer
            )
        else:
            weak_score = 0.5

        weak_score = float(np.clip(weak_score, 0.0, 1.0))
        noise = np.random.uniform(-0.02, 0.02)
        weak_score = float(np.clip(weak_score + noise, 0.0, 1.0))

        return reasoning, final_answer, weak_score

    def _parse_solution_json(self, content: str) -> dict:
        """Robust JSON parsing for solution response."""
        # Try 1: Direct parse
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try 2: Regex for JSON object
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

        # Try 3: Extract fields manually
        try:
            reasoning_match = re.search(
                r'"reasoning"\s*:\s*"(.*?)"(?=\s*,|\s*\})', content, re.DOTALL
            )
            answer_match = re.search(
                r'"final_answer"\s*:\s*"(.*?)"(?=\s*,|\s*\})', content, re.DOTALL
            )
            reasoning = reasoning_match.group(1) if reasoning_match else ""
            final_answer = answer_match.group(1) if answer_match else ""
            reasoning = reasoning.replace('\\"', '"').replace("\\n", "\n")
            final_answer = final_answer.replace('\\"', '"').replace("\\n", "\n")
            return {"reasoning": reasoning, "final_answer": final_answer}
        except Exception:
            pass

        # Try 4: Use raw content
        return {"reasoning": content[:500], "final_answer": "[Could not parse]"}

    # ========================================================================
    # Verify final answer against ground truth
    # ========================================================================

    def verify_answer(
        self,
        problem: str,
        reasoning: str,
        generated_answer: str,
        ground_truth_answer: str,
        is_ground_truth_call: bool = False,
    ) -> Tuple[bool, str]:
        """
        Verify if the generated answer matches the ground truth.

        Returns:
            (is_correct, explanation)
        """
        if is_ground_truth_call:
            self.call_counts["ground_truth"] += 1
        else:
            self.call_counts["verifier"] += 1

        model_name = self.config.strong_verifier_model

        prompt = f"""Determine if two math answers are equivalent.

PROBLEM: {problem}

STUDENT'S REASONING:
{reasoning}

STUDENT'S ANSWER: {generated_answer}

CORRECT ANSWER: {ground_truth_answer}

Are these answers equivalent? Consider:
- Different formats: "5", "5.0", "5/1", "$5$", "\\boxed{{5}}" are all equivalent
- Simplified vs unsimplified: "2/4" = "1/2"
- Different notation: "x=3" vs "3" (if asking for x)

Respond with JSON:
{{"is_correct": true/false, "explanation": "brief reason"}}"""

        try:
            if self._is_new_model(model_name):
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=200,
                )
                content = response.choices[0].message.content
                return ("true" in content.lower()), "Answers match"
            else:
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content

                try:
                    result = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    match = re.search(r"\{[^}]+\}", content, re.DOTALL)
                    result = (
                        json.loads(match.group())
                        if match
                        else {"is_correct": False}
                    )

                return bool(result.get("is_correct", False)), result.get(
                    "explanation", ""
                )

        except Exception as e:
            print(f"Error in verify_answer: {e}")
            return False, f"Error: {e}"

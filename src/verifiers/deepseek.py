# ============================================================================
# FIXED: DeepSeek Weak Verifier with Robust JSON Parsing
# ============================================================================

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
from tqdm.auto import tqdm
from openai import OpenAI
from datasets import load_dataset
from huggingface_hub import InferenceClient
from IPython.display import clear_output
from datetime import datetime
import pickle
from scipy import stats
class DeepSeekWeakVerifier:
    """DeepSeek-based weak verifier for scoring complete solutions."""

    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

    def score_solution(self, problem: str, reasoning: str, final_answer: str) -> float:
        """
        Score a complete solution (reasoning + final answer).

        Returns:
            float: Score between 0 and 1
        """
        # Truncate reasoning if too long to avoid issues
        if len(reasoning) > 1500:
            reasoning = reasoning[:1500] + "..."

        prompt = f"""Rate the correctness of this math solution on a scale from 0.0 to 1.0.

PROBLEM: {problem}

REASONING:
{reasoning}

FINAL ANSWER: {final_answer}

Consider:
1. Is the reasoning mathematically sound?
2. Are all calculations correct?
3. Does the final answer follow from the reasoning?

Respond with ONLY a JSON object, nothing else:
{{"score": 0.XX}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50  # Keep it short to avoid truncation issues
            )

            content = response.choices[0].message.content.strip()

            # Try multiple parsing strategies
            score = self._parse_score(content)
            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"DeepSeek error: {e}")
            return 0.5

    def _parse_score(self, content: str) -> float:
        """Robust score parsing with multiple fallbacks."""

        # Strategy 1: Direct JSON parse
        try:
            result = json.loads(content)
            if "score" in result:
                return float(result["score"])
        except:
            pass

        # Strategy 2: Find JSON object with regex
        try:
            match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', content)
            if match:
                return float(match.group(1))
        except:
            pass

        # Strategy 3: Find any decimal number after "score"
        try:
            match = re.search(r'"?score"?\s*[:\s]\s*([\d.]+)', content, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except:
            pass

        # Strategy 4: Find any decimal number in the response
        try:
            match = re.search(r'(0\.\d+|1\.0|1|0)', content)
            if match:
                return float(match.group(1))
        except:
            pass

        # Strategy 5: Look for keywords
        content_lower = content.lower()
        if "incorrect" in content_lower or "wrong" in content_lower:
            return 0.3
        if "correct" in content_lower or "right" in content_lower:
            return 0.8

        # Default fallback
        return 0.5

    # Backward compatibility
    def score_step(self, problem: str, previous_steps: List[str], current_step: str) -> float:
        """Backward compatibility - redirect to score_solution."""
        reasoning = "\n".join(previous_steps + [current_step])
        return self.score_solution(problem, reasoning, current_step)
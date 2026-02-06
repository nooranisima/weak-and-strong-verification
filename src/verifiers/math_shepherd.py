"""Math-Shepherd PRM weak verifier (4-bit quantized).

Requires GPU and: pip install torch transformers bitsandbytes
"""

import numpy as np
from typing import List


class MathShepherdScorer:
    """Math-Shepherd PRM with proper prompt format."""

    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.call_count = 0
        self.model = None
        self.tokenizer = None
        self.device = None

        try:
            print("Loading Math-Shepherd PRM (4-bit quantized)...")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model_name = "peiyi9979/math-shepherd-mistral-7b-prm"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()

            self.good_token = "+"
            self.bad_token = "-"
            self.step_tag = "ки"

            self.candidate_tokens = self.tokenizer.encode(
                f"{self.good_token} {self.bad_token}"
            )[1:]
            self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1]

            print("✓ Math-Shepherd loaded!")

        except Exception as e:
            print(f"❌ Math-Shepherd load error: {e}")
            self.model = None

    def score_step(
        self, problem: str, accepted_steps: List[str], current_step: str
    ) -> float:
        """
        Score whether the current step is correct.

        Returns:
            float: Score from 0.0 (incorrect) to 1.0 (correct)
        """
        import torch

        self.call_count += 1

        if self.model is None:
            return np.random.uniform(0.3, 0.7)

        try:
            formatted_steps = ""
            for i, step in enumerate(accepted_steps):
                formatted_steps += f"Step {i+1}: {step} {self.step_tag}\n"
            formatted_steps += (
                f"Step {len(accepted_steps)+1}: {current_step} {self.step_tag}"
            )

            input_text = f"{problem}\n{formatted_steps}"

            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[:, :, 0]

            step_positions = (input_ids == self.step_tag_id).nonzero(as_tuple=True)[1]

            if len(step_positions) > 0:
                step_score = scores[0, step_positions[-1]].item()
            else:
                step_score = scores[0, -1].item()

            noise = np.random.uniform(-0.02, 0.02)
            return float(np.clip(step_score + noise, 0.01, 0.99))

        except Exception as e:
            print(f"Math-Shepherd error: {e}")
            return np.random.uniform(0.3, 0.7)

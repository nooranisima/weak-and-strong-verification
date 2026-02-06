"""
Sudoku experiment configuration.

Supports OpenAI and DeepSeek providers for generator and weak verifier models.
API keys are read from environment variables by default.
"""

from dataclasses import dataclass
import os


@dataclass
class SudokuConfig:
    """Configuration for Sudoku experiments."""

    # API provider: "openai" or "deepseek"
    provider: str = "openai"
    openai_api_key: str = ""
    deepseek_api_key: str = ""

    # Model selection
    generator_model: str = "gpt-4o-mini"
    weak_verifier_model: str = "gpt-4o-mini"

    # Generation settings
    num_problems: int = 50
    temperature: float = 0.7
    max_tokens: int = 2000

    # Dataset
    dataset_url: str = "https://huggingface.co/datasets/asadshahab/mini-sudoku"

    def __post_init__(self):
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.deepseek_api_key:
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

    def get_generator_client(self):
        """Get the appropriate API client for the generator."""
        from openai import OpenAI

        if self.provider == "deepseek":
            return OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return OpenAI(api_key=self.openai_api_key)

    def get_verifier_client(self):
        """Get the appropriate API client for the weak verifier."""
        from openai import OpenAI

        if self.provider == "deepseek":
            return OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
        return OpenAI(api_key=self.openai_api_key)

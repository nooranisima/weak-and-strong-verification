# Adaptive Verification: Best-of-N with Weak and Strong Verifiers

This repository implements the SSV algorithm.

## Repository Structure

```
adaptive-verification/
├── README.md
├── requirements.txt
├── configs/
│   └── math_bon.yaml              # Config for MATH Best-of-N experiments
├── src/
│   ├── __init__.py
│   ├── config.py                   # ExperimentConfig dataclass
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── deepseek.py             # DeepSeek weak verifier
│   │   └── math_shepherd.py        # Math-Shepherd PRM (optional)
│   ├── llm.py                      # LLMWrapper for generation & verification
│   ├── data/
│   │   ├── __init__.py
│   │   ├── math_dataset.py         # MATH dataset loading
│   │   └── pregeneration.py        # Pre-generate solutions pipeline
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── adaptive.py             # Adaptive threshold runner
│   │   └── baselines.py            # Strong & weak baselines
│   ├── experiment.py               # Run full experiment on pregenerated data
│   └── utils.py                    # Save/load, JSON conversion helpers
├── notebooks/
│   ├── math_pregenerate.ipynb      # Step 1: Generate data for MATH
│   ├── math_experiment.ipynb       # Step 2: Run algorithms & plot
│   └── (future dataset notebooks)
└── results/                        # Pre-generated data & experiment outputs (gitignored)
```

## Workflow

### Step 1: Pre-generate Data
```bash
# In notebooks/math_pregenerate.ipynb or via script:
python -m src.data.pregeneration --config configs/math_bon.yaml
```
This queries the LLMs and saves all solutions, weak scores, and ground truth labels to `results/`.

### Step 2: Run Experiments
```bash
# In notebooks/math_experiment.ipynb:
# Load pregenerated data, run adaptive + baselines, generate plots
```

## Key Components

| Module | Purpose |
|--------|---------|
| `src/config.py` | All experiment hyperparameters |
| `src/llm.py` | OpenAI API wrapper (generate solutions, verify answers) |
| `src/verifiers/deepseek.py` | DeepSeek as weak verifier |
| `src/algorithms/adaptive.py` | Adaptive threshold algorithm (SimulatedAdaptiveRunner) |
| `src/algorithms/baselines.py` | Strong baseline (always query) and weak baseline (threshold only) |
| `src/data/pregeneration.py` | Pre-generate all solutions + scores |
| `src/experiment.py` | Run all algorithm configs on pregenerated data |

## Adding a New Dataset

1. Create a new loader in `src/data/` (e.g., `gsm8k_dataset.py`)
2. Create a new config in `configs/`
3. Create notebooks following the same 2-step pattern

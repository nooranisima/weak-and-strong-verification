# Adaptive Verification: Best-of-N with Weak and Strong Verifiers

This repository implements the **SSV (Selective Strong Verification)** algorithm, which adaptively decides when to invoke an expensive strong verifier versus relying on a cheap weak verifier. The approach learns accept/reject thresholds online to guarantee user-specified error rates (α for false accepts, β for false rejects) while minimizing strong verifier calls.

Experiments are run on two domains: **MATH** (mathematical reasoning) and **Sudoku** (constraint satisfaction).

## Repository Structure

```
weak-and-strong-verification/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                        # Template for API keys
├── configs/
│   ├── math_bon.yaml                   # Config for MATH Best-of-N experiments
│   └── sudoku.yaml                     # Config for Sudoku experiments
├── src/
│   ├── __init__.py
│   ├── config.py                       # MATH ExperimentConfig dataclass
│   ├── llm.py                          # LLMWrapper for generation & verification
│   ├── experiment.py                   # Legacy experiment runner
│   ├── utils.py                        # Save/load, JSON conversion helpers
│   ├── verifiers/
│   │   ├── deepseek.py                 # DeepSeek weak verifier
│   │   └── math_shepherd.py            # Math-Shepherd PRM (optional, requires GPU)
│   ├── data/
│   │   ├── math_dataset.py             # MATH dataset loading & sampling
│   │   └── pregeneration.py            # Pre-generate solutions pipeline
│   ├── algorithms/
│   │   ├── adaptive.py                 # SimulatedAdaptiveRunner + WithRNG variant
│   │   └── baselines.py               # Strong, Weak, and WeakBestOfN baselines
│   ├── math_experiments/
│   │   ├── simulation.py               # Error convergence tracking & plotting
│   │   ├── pipeline.py                 # Grid search, full pipeline, accuracy-vs-latency
│   │   ├── sweep_plots.py              # Multi-α/β overlay plots
│   │   ├── analysis.py                 # Weak verifier quality (distributions, calibration)
│   │   └── io.py                       # Save/load pipeline results (JSON with tuple keys)
│   ├── sudoku/
│   │   ├── config.py                   # SudokuConfig dataclass
│   │   ├── constants.py                # Sudoku rules and formatting
│   │   ├── dataset.py                  # Load puzzles from HuggingFace
│   │   ├── generator.py                # LLM-based solution generator
│   │   ├── strong_verifier.py          # Exact grid comparison
│   │   ├── weak_verifier.py            # LLM-based weak verifier
│   │   └── pregeneration.py            # Tree-structured candidate generation
│   └── sudoku_experiments/
│       ├── simulation.py               # Sudoku adaptive runner & error tracking
│       ├── pipeline.py                 # Grid search, full pipeline, plotting
│       └── sweep_plots.py              # Multi-α/β overlays + individual sweep plots
├── notebooks/
│   ├── math_pregenerate.ipynb          # Step 1: Generate data for MATH
│   ├── math_experiment.ipynb           # Step 2: Run algorithms & plot (MATH)
│   ├── sudoku_pregenerate.ipynb        # Step 1: Generate tree data for Sudoku
│   └── sudoku_experiment.ipynb         # Step 2: Run algorithms & plot (Sudoku)
└── results/                            # Experiment outputs (gitignored)
    └── .gitkeep
```

## Setup

```bash
# Clone
git clone https://github.com/nooranisima/weak-and-strong-verification.git
cd weak-and-strong-verification

# Install dependencies
pip install -r requirements.txt

# Set API keys (needed only for pre-generation, not for running experiments)
cp .env.example .env
# Edit .env with your keys, or export them directly:
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

## Workflow

Each domain follows the same two-step pattern:

### Step 1: Pre-generate Data

Pre-generation queries LLMs to produce candidate solutions with weak scores and ground truth labels. This is the expensive step (API calls). Results are saved to `results/` so experiments can be re-run without re-querying.

**MATH:**
```bash
# Run notebooks/math_pregenerate.ipynb
# Outputs: results/pregenerated_difficulty{2,3,5}.json
```

**Sudoku:**
```bash
# Run notebooks/sudoku_pregenerate.ipynb
# Outputs: results/sudoku_tree_data.json
```

### Step 2: Run Experiments

Experiments run entirely on pre-generated data (no API calls). They perform grid search over hyperparameters, run the adaptive algorithm and baselines, and produce all paper figures.

**MATH** (`notebooks/math_experiment.ipynb`):
1. Weak verifier quality analysis (score distributions, calibration)
2. Error convergence plots (running accept/reject error rates)
3. Main pipeline: symmetric α=β sweep → accuracy vs latency
4. Appendix: fixed-α sweeps (varying β) and fixed-β sweeps (varying α)
5. Multi-sweep overlay plots

**Sudoku** (`notebooks/sudoku_experiment.ipynb`):
1. Error convergence experiment
2. Main pipeline: symmetric α=β sweep → accuracy vs latency
3. Appendix: fixed-α and fixed-β sweeps
4. Multi-sweep overlay plots
5. Reload & re-plot section (for rebuttals)

## Key Components

### Algorithms (`src/algorithms/`)

| Class | Description |
|-------|-------------|
| `SimulatedAdaptiveRunner` | Adaptive threshold algorithm with clamping |
| `SimulatedAdaptiveRunnerWithRNG` | Same but with explicit RNG, no clamping (matches theory) |
| `SimulatedStrongBaseline` | Always queries the strong verifier |
| `SimulatedWeakBaseline` | Uses only weak scores with a fixed threshold |
| `SimulatedWeakBaselineBestOfN` | Picks the attempt with the highest weak score |

### MATH Experiments (`src/math_experiments/`)

| Module | Key Functions |
|--------|---------------|
| `simulation.py` | `run_with_error_tracking()`, `plot_error_convergence()` |
| `pipeline.py` | `grid_search_for_config()`, `run_full_pipeline()`, `plot_accuracy_vs_latency()` |
| `sweep_plots.py` | `plot_accuracy_vs_latency_multi_alpha()`, `plot_accuracy_vs_latency_multi_beta()` |
| `analysis.py` | `plot_score_distribution()`, `plot_calibration()`, `print_weak_verifier_summary()` |
| `io.py` | `save_pipeline_results()`, `load_pipeline_results()` |

### Sudoku Experiments (`src/sudoku_experiments/`)

| Module | Key Functions |
|--------|---------------|
| `simulation.py` | `SudokuAdaptiveRunner`, `run_experiment()`, `plot_all_error_rates()` |
| `pipeline.py` | `run_full_pipeline_sudoku()`, `plot_accuracy_vs_latency_sudoku()`, `save_results()` |
| `sweep_plots.py` | `plot_accuracy_vs_latency_multi_alpha_sudoku()`, `plot_sweep_results()` |

## Reproducibility

All experiments use explicit RNG seeding for reproducibility:
- MATH: `seed * 100 + difficulty` per difficulty, `seed * 1000 + difficulty * 100 + run_idx` per run
- Sudoku: Configurable seed passed through grid search and pipeline

Pre-generated data files contain all information needed to reproduce results without API calls. The reload-and-replot sections in each notebook allow regenerating all figures from saved `.pkl` files.

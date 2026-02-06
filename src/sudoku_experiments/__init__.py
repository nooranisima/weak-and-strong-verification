"""Sudoku experiment runners: simulation, pipeline, and sweep plotting."""

from .simulation import (
    SudokuAdaptiveRunner,
    SudokuStrongBaseline,
    SudokuWeakBaseline,
    run_experiment,
    load_sudoku_tree_data,
    plot_error_rates,
    plot_all_error_rates,
    plot_threshold_evolution,
)
from .pipeline import (
    run_full_pipeline_sudoku,
    plot_accuracy_vs_latency_sudoku,
    save_results,
    load_results,
)
from .sweep_plots import (
    plot_accuracy_vs_latency_multi_alpha_sudoku,
    plot_accuracy_vs_latency_multi_beta_sudoku,
    plot_sweep_results,
)
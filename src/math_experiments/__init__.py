"""MATH dataset experiment modules: simulation, pipeline, analysis, and plotting."""

from .simulation import (
    run_with_error_tracking,
    run_with_error_tracking_per_diff_config,
    plot_error_convergence,
    plot_error_summary_with_convergence,
    plot_accept_error_convergence,
    plot_reject_error_convergence,
    plot_both_error_convergence,
)

from .pipeline import (
    grid_search_for_config,
    run_full_pipeline,
    plot_accuracy_vs_latency,
)

from .sweep_plots import (
    plot_accuracy_vs_latency_multi_alpha,
    plot_accuracy_vs_latency_multi_beta,
)

from .analysis import (
    collect_scores,
    plot_score_distribution,
    plot_calibration,
    print_weak_verifier_summary,
)

from .io import (
    save_pipeline_results,
    load_pipeline_results,
)

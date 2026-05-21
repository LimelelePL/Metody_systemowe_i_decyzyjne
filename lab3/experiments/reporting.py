from IPython.display import display

from lab3.plots.plots import (
    plot_linear_regularization_curves,
    plot_regression_baseline_vs_regularized,
    plot_tree_baseline_vs_regularized,
    plot_tree_regularization_by_parameter,
    plot_weight_shrinkage,
    plot_zero_weights_by_alpha,
)


def show_linear_regularization_report(experiment):
    results_df = experiment["results"]
    summary_df = experiment["summary"]
    comparison_df = experiment["comparison"]
    weights_df = experiment["weights"]

    display(results_df.round(4))
    display(summary_df.round(4))
    display(comparison_df[["model", "train_mse", "test_mse", "zero_weights"]].round(2))
    display(
        weights_df[
            [
                "feature",
                "baseline_abs_weight",
                "ridge_abs_weight",
                "lasso_abs_weight",
                "lasso_zero",
            ]
        ].round(4)
    )

    plot_linear_regularization_curves(results_df)
    plot_zero_weights_by_alpha(results_df)
    plot_weight_shrinkage(weights_df)
    plot_regression_baseline_vs_regularized(comparison_df)


def show_tree_regularization_report(experiment):
    results_df = experiment["results"]
    summary_df = experiment["summary"]
    comparison_df = experiment["comparison"]

    display(results_df.round(4))
    display(summary_df.round(4))
    display(comparison_df[["model", "train_error", "test_error"]].round(4))

    plot_tree_regularization_by_parameter(results_df)
    plot_tree_baseline_vs_regularized(comparison_df)

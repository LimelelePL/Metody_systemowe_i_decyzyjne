from IPython.display import display

try:
    from lab3.algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from lab3.algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from lab3.algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from lab3.plots.plots import plot_linear_regularization_all, plot_tree_regularization_all
except ModuleNotFoundError:
    from algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from plots.plots import plot_linear_regularization_all, plot_tree_regularization_all


def show_linear_regularization_report(experiment):
    comparison = LinearRegularizationComparison()
    results_df = comparison.build_results_df(experiment)
    summary_df = comparison.build_summary_df(experiment)
    comparison_df = comparison.build_comparison_df(experiment)
    weights_df = comparison.build_weights_df(experiment)

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

    plot_linear_regularization_all(experiment)


def show_tree_regularization_report(experiment):
    comparison = TreeRegularizationComparison()
    results_df = comparison.build_results_df(experiment)
    summary_df = comparison.build_summary_df(experiment)
    comparison_df = comparison.build_comparison_df(experiment)

    display(results_df.round(4))
    display(summary_df.round(4))
    display(comparison_df[["model", "train_error", "test_error"]].round(4))

    plot_tree_regularization_all(experiment)


def show_sweet_spot_report(linear_experiment, tree_experiment):
    summary_df = RegularizationSweetSpotComparison.build_comparison_df(linear_experiment, tree_experiment)
    display(summary_df.round(4))

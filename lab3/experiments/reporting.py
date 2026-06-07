from IPython.display import display
import pandas as pd

try:
    from lab3.algorithms.boosting.boosting_comparisions import BoostingComparison
    from lab3.algorithms.experts.mixture_of_experts_comparisions import MixtureOfExpertsComparison
    from lab3.algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from lab3.algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from lab3.algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from lab3.algorithms.stacking.stacking_comparisions import StackingComparison
    from lab3.plots.plots import (
        plot_boosting_all,
        plot_final_classification_summary,
        plot_final_regression_ensemble_summary,
        plot_linear_regularization_all,
        plot_mixture_of_experts_all,
        plot_stacking_level_0_diagnostics,
        plot_stacking_model_comparison,
        plot_stacking_vs_boosting,
        plot_tree_regularization_all,
    )
except ModuleNotFoundError:
    from algorithms.boosting.boosting_comparisions import BoostingComparison
    from algorithms.experts.mixture_of_experts_comparisions import MixtureOfExpertsComparison
    from algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from algorithms.stacking.stacking_comparisions import StackingComparison
    from plots.plots import (
        plot_boosting_all,
        plot_final_classification_summary,
        plot_final_regression_ensemble_summary,
        plot_linear_regularization_all,
        plot_mixture_of_experts_all,
        plot_stacking_level_0_diagnostics,
        plot_stacking_model_comparison,
        plot_stacking_vs_boosting,
        plot_tree_regularization_all,
    )


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


def show_stacking_report(experiment):
    comparison = StackingComparison()
    metrics_df = comparison.build_metrics_df(experiment)
    diagnostics_df = comparison.build_level_0_diagnostics_df(experiment)
    stacking_vs_boosting_df = comparison.build_stacking_vs_boosting_df(experiment)

    display(metrics_df.round(4))
    display(stacking_vs_boosting_df.round(4))

    plot_stacking_model_comparison(metrics_df)
    plot_stacking_vs_boosting(stacking_vs_boosting_df)

    display(diagnostics_df.round(4))
    plot_stacking_level_0_diagnostics(diagnostics_df)


def show_boosting_report(experiment):
    comparison = BoostingComparison()
    results_df = comparison.build_results_df(experiment)
    summary_df = comparison.build_summary_df(experiment)
    comparison_df = comparison.build_comparison_df(experiment)

    display(results_df.round(4))
    display(summary_df.round(4))
    display(comparison_df.round(4))

    plot_boosting_all(experiment)


def show_mixture_of_experts_report(experiment):
    comparison = MixtureOfExpertsComparison()
    cluster_summary_df = comparison.build_cluster_summary_df(experiment)
    global_comparison_df = comparison.build_global_comparison_df(experiment)
    comparison_df = comparison.build_comparison_df(experiment)

    display(cluster_summary_df.round(4))
    display(global_comparison_df.round(4))
    display(comparison_df.round(4))

    plot_mixture_of_experts_all(experiment)


def show_final_summary_report(classification_df, stacking_experiment, mixture_of_experts_experiment):
    stacking_comparison = StackingComparison()
    mixture_comparison = MixtureOfExpertsComparison()

    stacking_vs_boosting_df = stacking_comparison.build_stacking_vs_boosting_df(stacking_experiment).copy()
    stacking_vs_boosting_df = stacking_vs_boosting_df.rename(
        columns={
            "Test MAE": "Test MAE",
            "Test RMSE": "Test RMSE",
            "Test R2": "Test R2",
        }
    )

    mixture_df = mixture_comparison.build_comparison_df(mixture_of_experts_experiment).copy()
    mixture_df = mixture_df[mixture_df["model"] == "Mixture of Experts"].rename(
        columns={
            "model": "Model",
            "test_mae": "Test MAE",
            "test_rmse": "Test RMSE",
            "test_r2": "Test R2",
        }
    )

    regression_summary_df = pd.concat(
        [
            stacking_vs_boosting_df[["Model", "Test MAE", "Test RMSE", "Test R2"]],
            mixture_df[["Model", "Test MAE", "Test RMSE", "Test R2"]],
        ],
        ignore_index=True,
    )

    display(classification_df.round(4))
    plot_final_classification_summary(classification_df)

    display(regression_summary_df.round(4))
    plot_final_regression_ensemble_summary(regression_summary_df)

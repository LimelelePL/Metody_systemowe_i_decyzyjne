import pandas as pd
from IPython.display import display

try:
    from lab3.algorithms.boosting.boosting_comparisions import BoostingComparison
    from lab3.algorithms.experts.mixture_of_experts_comparisions import MixtureOfExpertsComparison
    from lab3.algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from lab3.algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from lab3.algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from lab3.algorithms.stacking.stacking_comparisions import StackingComparison
    from lab3.plots.plots import (
        plot_boosting_all,
        plot_boosting_model_comparison,
        plot_boosting_n_estimators_curves,
        plot_classification_model_comparison,
        plot_final_classification_summary,
        plot_final_regression_ensemble_summary,
        plot_linear_regularization_all,
        plot_linear_regularization_curves,
        plot_mixture_of_experts_all,
        plot_mixture_of_experts_clusters,
        plot_mixture_of_experts_model_comparison,
        plot_regression_baseline_vs_regularized,
        plot_stacking_level_0_diagnostics,
        plot_stacking_model_comparison,
        plot_stacking_vs_boosting,
        plot_tree_baseline_vs_regularized,
        plot_tree_regularization_all,
        plot_tree_regularization_by_parameter,
        plot_weight_shrinkage,
        plot_zero_weights_by_alpha,
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
        plot_boosting_model_comparison,
        plot_boosting_n_estimators_curves,
        plot_classification_model_comparison,
        plot_final_classification_summary,
        plot_final_regression_ensemble_summary,
        plot_linear_regularization_all,
        plot_linear_regularization_curves,
        plot_mixture_of_experts_all,
        plot_mixture_of_experts_clusters,
        plot_mixture_of_experts_model_comparison,
        plot_regression_baseline_vs_regularized,
        plot_stacking_level_0_diagnostics,
        plot_stacking_model_comparison,
        plot_stacking_vs_boosting,
        plot_tree_baseline_vs_regularized,
        plot_tree_regularization_all,
        plot_tree_regularization_by_parameter,
        plot_weight_shrinkage,
        plot_zero_weights_by_alpha,
    )


def _display_rounded(df, digits=4):
    display(df.round(digits))


def _build_final_regression_summary(stacking_experiment, mixture_of_experts_experiment):
    stacking_comparison = StackingComparison()
    mixture_comparison = MixtureOfExpertsComparison()

    stacking_vs_boosting_df = stacking_comparison.build_stacking_vs_boosting_df(stacking_experiment).copy()
    mixture_df = mixture_comparison.build_comparison_df(mixture_of_experts_experiment).copy()
    mixture_df = mixture_df[mixture_df["model"] == "Mixture of Experts"].rename(
        columns={
            "model": "Model",
            "test_mae": "Test MAE",
            "test_rmse": "Test RMSE",
            "test_r2": "Test R2",
        }
    )

    return pd.concat(
        [
            stacking_vs_boosting_df[["Model", "Test MAE", "Test RMSE", "Test R2"]],
            mixture_df[["Model", "Test MAE", "Test RMSE", "Test R2"]],
        ],
        ignore_index=True,
    )


def display_linear_regularization_results(experiment):
    comparison = LinearRegularizationComparison()
    _display_rounded(comparison.build_results_df(experiment))


def display_linear_regularization_summary(experiment):
    comparison = LinearRegularizationComparison()
    _display_rounded(comparison.build_summary_df(experiment))


def display_linear_regularization_comparison(experiment):
    comparison = LinearRegularizationComparison()
    comparison_df = comparison.build_comparison_df(experiment)
    _display_rounded(comparison_df[["model", "train_mse", "test_mse", "zero_weights"]], digits=2)


def display_linear_regularization_weights(experiment):
    comparison = LinearRegularizationComparison()
    weights_df = comparison.build_weights_df(experiment)
    _display_rounded(
        weights_df[
            [
                "feature",
                "baseline_abs_weight",
                "ridge_abs_weight",
                "lasso_abs_weight",
                "lasso_zero",
            ]
        ]
    )


def plot_linear_regularization_curves_report(experiment):
    plot_linear_regularization_curves(experiment["results"])


def plot_linear_zero_weights_report(experiment):
    plot_zero_weights_by_alpha(experiment["results"])


def plot_linear_weight_shrinkage_report(experiment):
    plot_weight_shrinkage(experiment["weights"])


def plot_linear_regularization_comparison_report(experiment):
    plot_regression_baseline_vs_regularized(experiment["comparison"])


def display_tree_regularization_results(experiment):
    comparison = TreeRegularizationComparison()
    _display_rounded(comparison.build_results_df(experiment))


def display_tree_regularization_summary(experiment):
    comparison = TreeRegularizationComparison()
    _display_rounded(comparison.build_summary_df(experiment))


def display_tree_regularization_comparison(experiment):
    comparison = TreeRegularizationComparison()
    comparison_df = comparison.build_comparison_df(experiment)
    _display_rounded(comparison_df[["model", "train_error", "test_error"]])


def plot_tree_regularization_parameters_report(experiment):
    plot_tree_regularization_by_parameter(experiment["results"])


def plot_tree_regularization_comparison_report(experiment):
    plot_tree_baseline_vs_regularized(experiment["comparison"])


def display_sweet_spot_summary(linear_experiment, tree_experiment):
    summary_df = RegularizationSweetSpotComparison.build_comparison_df(linear_experiment, tree_experiment)
    _display_rounded(summary_df)


def display_stacking_metrics(experiment):
    comparison = StackingComparison()
    _display_rounded(comparison.build_level_0_and_stacking_df(experiment))


def display_stacking_vs_boosting(experiment):
    comparison = StackingComparison()
    _display_rounded(comparison.build_stacking_vs_boosting_df(experiment))


def display_stacking_diagnostics(experiment):
    comparison = StackingComparison()
    _display_rounded(comparison.build_level_0_diagnostics_df(experiment))


def plot_stacking_models_report(experiment):
    comparison = StackingComparison()
    plot_stacking_model_comparison(comparison.build_level_0_and_stacking_df(experiment))


def plot_stacking_vs_boosting_report(experiment):
    comparison = StackingComparison()
    plot_stacking_vs_boosting(comparison.build_stacking_vs_boosting_df(experiment))


def plot_stacking_diagnostics_report(experiment):
    comparison = StackingComparison()
    plot_stacking_level_0_diagnostics(comparison.build_level_0_diagnostics_df(experiment))


def display_boosting_results(experiment):
    comparison = BoostingComparison()
    _display_rounded(comparison.build_results_df(experiment))


def display_boosting_summary(experiment):
    comparison = BoostingComparison()
    _display_rounded(comparison.build_summary_df(experiment))


def display_boosting_comparison(experiment):
    comparison = BoostingComparison()
    _display_rounded(comparison.build_comparison_df(experiment))


def plot_boosting_n_estimators_report(experiment):
    plot_boosting_n_estimators_curves(experiment["results"])


def plot_boosting_comparison_report(experiment):
    plot_boosting_model_comparison(experiment["comparison"])


def display_mixture_cluster_summary(experiment):
    comparison = MixtureOfExpertsComparison()
    _display_rounded(comparison.build_cluster_summary_df(experiment))


def display_mixture_global_comparison(experiment):
    comparison = MixtureOfExpertsComparison()
    _display_rounded(comparison.build_global_comparison_df(experiment))


def display_mixture_model_comparison(experiment):
    comparison = MixtureOfExpertsComparison()
    _display_rounded(comparison.build_comparison_df(experiment))


def plot_mixture_clusters_report(experiment):
    plot_mixture_of_experts_clusters(experiment["cluster_summary"])


def plot_mixture_model_comparison_report(experiment):
    plot_mixture_of_experts_model_comparison(experiment["comparison"])


def display_final_classification_summary(classification_df):
    _display_rounded(classification_df)


def display_classification_comparison(comparison_df):
    _display_rounded(comparison_df)


def plot_classification_comparison_report(comparison_df):
    plot_classification_model_comparison(comparison_df)


def plot_final_classification_summary_report(classification_df):
    plot_final_classification_summary(classification_df)


def display_final_regression_summary(stacking_experiment, mixture_of_experts_experiment):
    _display_rounded(_build_final_regression_summary(stacking_experiment, mixture_of_experts_experiment))


def plot_final_regression_summary_report(stacking_experiment, mixture_of_experts_experiment):
    regression_summary_df = _build_final_regression_summary(stacking_experiment, mixture_of_experts_experiment)
    plot_final_regression_ensemble_summary(regression_summary_df)


def show_linear_regularization_report(experiment):
    display_linear_regularization_results(experiment)
    plot_linear_regularization_curves_report(experiment)
    display_linear_regularization_summary(experiment)
    display_linear_regularization_comparison(experiment)
    plot_linear_regularization_comparison_report(experiment)
    display_linear_regularization_weights(experiment)
    plot_linear_zero_weights_report(experiment)
    plot_linear_weight_shrinkage_report(experiment)


def show_tree_regularization_report(experiment):
    display_tree_regularization_results(experiment)
    plot_tree_regularization_parameters_report(experiment)
    display_tree_regularization_summary(experiment)
    display_tree_regularization_comparison(experiment)
    plot_tree_regularization_comparison_report(experiment)


def show_sweet_spot_report(linear_experiment, tree_experiment):
    display_sweet_spot_summary(linear_experiment, tree_experiment)


def show_stacking_report(experiment):
    display_stacking_metrics(experiment)
    plot_stacking_models_report(experiment)
    display_stacking_vs_boosting(experiment)
    plot_stacking_vs_boosting_report(experiment)
    display_stacking_diagnostics(experiment)
    plot_stacking_diagnostics_report(experiment)


def show_boosting_report(experiment):
    display_boosting_results(experiment)
    plot_boosting_n_estimators_report(experiment)
    display_boosting_summary(experiment)
    display_boosting_comparison(experiment)
    plot_boosting_comparison_report(experiment)


def show_mixture_of_experts_report(experiment):
    display_mixture_cluster_summary(experiment)
    plot_mixture_clusters_report(experiment)
    display_mixture_global_comparison(experiment)
    display_mixture_model_comparison(experiment)
    plot_mixture_model_comparison_report(experiment)


def show_final_summary_report(classification_df, stacking_experiment, mixture_of_experts_experiment):
    display_final_classification_summary(classification_df)
    plot_final_classification_summary_report(classification_df)
    display_final_regression_summary(stacking_experiment, mixture_of_experts_experiment)
    plot_final_regression_summary_report(stacking_experiment, mixture_of_experts_experiment)

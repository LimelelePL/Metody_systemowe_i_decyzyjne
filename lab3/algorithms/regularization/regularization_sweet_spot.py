import pandas as pd

try:
    from lab3.algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from lab3.algorithms.regularization.tree_regularization import TreeRegularizationComparison
except ModuleNotFoundError:
    from algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from algorithms.regularization.tree_regularization import TreeRegularizationComparison


class RegularizationSweetSpotComparison:
    def __init__(self, linear_comparison=None, tree_comparison=None):
        self.linear_comparison = linear_comparison or LinearRegularizationComparison()
        self.tree_comparison = tree_comparison or TreeRegularizationComparison()

    def compare(
        self,
        X_train_regression,
        y_train_regression,
        X_test_regression,
        y_test_regression,
        X_train_classification,
        y_train_classification,
        X_test_classification,
        y_test_classification,
    ):
        linear_results = self.linear_comparison.compare(
            X_train_regression,
            y_train_regression,
            X_test_regression,
            y_test_regression,
        )
        tree_results = self.tree_comparison.compare(
            X_train_classification,
            y_train_classification,
            X_test_classification,
            y_test_classification,
        )

        return {
            "linear": linear_results,
            "tree": tree_results,
            "summary": self.build_summary(linear_results, tree_results),
        }

    @staticmethod
    def build_comparison_df(linear_results, tree_results):
        return RegularizationSweetSpotComparison.build_summary(linear_results, tree_results)

    @staticmethod
    def build_summary(linear_results, tree_results):
        linear_comparison = linear_results["comparison"].copy()
        linear_baseline = linear_comparison[linear_comparison["model"] == "baseline GD"].iloc[0]
        linear_best = linear_comparison[linear_comparison["model"] != "baseline GD"].sort_values("test_mse").iloc[0]

        tree_comparison = tree_results["comparison"].copy()
        tree_baseline = tree_comparison[tree_comparison["model"] == "baseline tree"].iloc[0]
        tree_best = tree_comparison[tree_comparison["model"] != "baseline tree"].sort_values("test_error").iloc[0]

        return pd.DataFrame([
            {
                "task": "regression",
                "baseline_model": linear_baseline["model"],
                "regularized_model": linear_best["model"],
                "baseline_test_metric": linear_baseline["test_mse"],
                "regularized_test_metric": linear_best["test_mse"],
                "improvement": linear_baseline["test_mse"] - linear_best["test_mse"],
            },
            {
                "task": "classification",
                "baseline_model": tree_baseline["model"],
                "regularized_model": tree_best["model"],
                "baseline_test_metric": tree_baseline["test_error"],
                "regularized_test_metric": tree_best["test_error"],
                "improvement": tree_baseline["test_error"] - tree_best["test_error"],
            },
        ])

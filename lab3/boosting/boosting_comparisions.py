import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from lab3.boosting.boosting import MyGrandientBoosting
except ModuleNotFoundError:
    from boosting.boosting import MyGrandientBoosting


class BoostingComparison:
    def __init__(self, n_estimators_values=None, learning_rate=0.1, max_depth=3):
        self.n_estimators_values = n_estimators_values or [1, 5, 10, 20, 30, 50]
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    @staticmethod
    def _build_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_true, y_pred),
        }

    def compare(self, X_train, X_test, y_train, y_test):
        baseline_model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        baseline_model.fit(X_train, y_train)

        baseline_train_predictions = baseline_model.predict(X_train)
        baseline_test_predictions = baseline_model.predict(X_test)
        baseline_train_metrics = self._build_metrics(y_train, baseline_train_predictions)
        baseline_test_metrics = self._build_metrics(y_test, baseline_test_predictions)

        boosting_rows = []
        for n_estimators in self.n_estimators_values:
            model = MyGrandientBoosting(
                n_estimators=n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
            )
            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            train_metrics = self._build_metrics(y_train, train_predictions)
            test_metrics = self._build_metrics(y_test, test_predictions)

            boosting_rows.append({
                "n_estimators": n_estimators,
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
            })

        results_df = pd.DataFrame(boosting_rows)
        best_row = results_df.loc[results_df["test_rmse"].idxmin()].copy()
        best_row["n_estimators"] = int(best_row["n_estimators"])

        comparison_df = pd.DataFrame([
            {
                "model": f"DecisionTreeRegressor depth={self.max_depth}",
                "train_mae": baseline_train_metrics["mae"],
                "test_mae": baseline_test_metrics["mae"],
                "train_rmse": baseline_train_metrics["rmse"],
                "test_rmse": baseline_test_metrics["rmse"],
                "train_r2": baseline_train_metrics["r2"],
                "test_r2": baseline_test_metrics["r2"],
            },
            {
                "model": f"MyGrandientBoosting n={int(best_row['n_estimators'])}",
                "train_mae": best_row["train_mae"],
                "test_mae": best_row["test_mae"],
                "train_rmse": best_row["train_rmse"],
                "test_rmse": best_row["test_rmse"],
                "train_r2": best_row["train_r2"],
                "test_r2": best_row["test_r2"],
            },
        ])

        return {
            "results": results_df,
            "summary": pd.DataFrame([best_row]),
            "comparison": comparison_df,
        }

    @staticmethod
    def build_results_df(experiment):
        return experiment["results"].copy()

    @staticmethod
    def build_summary_df(experiment):
        return experiment["summary"].copy()

    @staticmethod
    def build_comparison_df(experiment):
        return experiment["comparison"].copy()

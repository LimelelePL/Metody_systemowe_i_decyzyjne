import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    from lab3.algorithms.regularization.regularization_alg import Lasso, LinearRegressionGD, Ridge
except ModuleNotFoundError:
    from algorithms.regularization.regularization_alg import Lasso, LinearRegressionGD, Ridge


class LinearRegularizationComparison:
    def __init__(
        self,
        ridge_alphas=None,
        lasso_alphas=None,
        learning_rate=0.03,
        epochs=2000,
        zero_threshold=1e-6,
    ):
        self.ridge_alphas = ridge_alphas if ridge_alphas is not None else np.logspace(-8, 1, 10)
        self.lasso_alphas = lasso_alphas if lasso_alphas is not None else np.logspace(-5, -1, 9)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.zero_threshold = zero_threshold
        self.fitted_models = {}

    def compare(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_original=None,
        y_test_original=None,
        y_mean=0.0,
        y_std=1.0,
        **_,
    ):
        baseline_model = LinearRegressionGD(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )
        baseline_score = self._score_model(
            baseline_model,
            X_train,
            y_train,
            X_test,
            y_test,
            y_train_original,
            y_test_original,
            y_mean,
            y_std,
        )
        baseline_row = {
            "model_type": "baseline",
            "alpha": 0.0,
            "train_mse": baseline_score["train_mse"],
            "test_mse": baseline_score["test_mse"],
            **self._weights_summary(baseline_model),
        }
        self.fitted_models = {"baseline": baseline_model}

        result_rows = []
        for alpha in self.ridge_alphas:
            ridge_model = Ridge(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                alpha=float(alpha),
            )
            result_rows.append(
                self._fit_regularized_model(
                    "Ridge",
                    float(alpha),
                    ridge_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    y_train_original,
                    y_test_original,
                    y_mean,
                    y_std,
                )
            )

        for alpha in self.lasso_alphas:
            lasso_model = Lasso(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                alpha=float(alpha),
            )
            result_rows.append(
                self._fit_regularized_model(
                    "Lasso",
                    float(alpha),
                    lasso_model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    y_train_original,
                    y_test_original,
                    y_mean,
                    y_std,
                )
            )

        results_df = pd.DataFrame(result_rows)
        summary_df = self._build_summary(results_df)
        comparison_df = self._build_comparison(baseline_row, summary_df)
        weights_df = self._build_weights_table(X_train, summary_df)

        return {
            "results": results_df,
            "summary": summary_df,
            "comparison": comparison_df,
            "weights": weights_df,
            "best_regularized": summary_df.iloc[0],
            "baseline": baseline_row,
        }

    @staticmethod
    def build_results_df(experiment):
        return experiment["results"]

    @staticmethod
    def build_summary_df(experiment):
        return experiment["summary"]

    @staticmethod
    def build_comparison_df(experiment):
        return experiment["comparison"]

    @staticmethod
    def build_weights_df(experiment):
        return experiment["weights"]

    def _fit_regularized_model(
        self,
        model_type,
        alpha,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_original,
        y_test_original,
        y_mean,
        y_std,
    ):
        score = self._score_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            y_train_original,
            y_test_original,
            y_mean,
            y_std,
        )
        self.fitted_models[model_type, alpha] = model  # type: ignore

        return {
            "model_type": model_type,
            "alpha": alpha,
            "train_mse": score["train_mse"],
            "test_mse": score["test_mse"],
            **self._weights_summary(model),
        }

    def _score_model(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_original,
        y_test_original,
        y_mean,
        y_std,
    ):
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train) * y_std + y_mean
        test_predictions = model.predict(X_test) * y_std + y_mean
        train_true = (
            y_train_original if y_train_original is not None else np.asarray(y_train, dtype=float) * y_std + y_mean
        )
        test_true = y_test_original if y_test_original is not None else np.asarray(y_test, dtype=float) * y_std + y_mean

        return {
            "train_mse": mean_squared_error(train_true, train_predictions),
            "test_mse": mean_squared_error(test_true, test_predictions),
        }

    def _weights_summary(self, model):
        weights = np.asarray(model.weights, dtype=float)
        return {
            "nonzero_weights": int(np.sum(np.abs(weights) > self.zero_threshold)),
            "zero_weights": int(np.sum(np.abs(weights) <= self.zero_threshold)),
            "mean_abs_weight": float(np.mean(np.abs(weights))),
            "max_abs_weight": float(np.max(np.abs(weights))),
        }

    def _build_summary(self, results_df):
        return (
            results_df
            .loc[results_df.groupby("model_type")["test_mse"].idxmin()]
            .sort_values("test_mse")
            .reset_index(drop=True)
        )

    def _build_comparison(self, baseline_row, summary_df):
        best_ridge = summary_df[summary_df["model_type"] == "Ridge"].iloc[0]
        best_lasso = summary_df[summary_df["model_type"] == "Lasso"].iloc[0]

        return pd.DataFrame([
            {"model": "baseline GD", **baseline_row},
            {"model": f"Ridge alpha={best_ridge['alpha']:.0e}", **best_ridge.to_dict()},
            {"model": f"Lasso alpha={best_lasso['alpha']:.0e}", **best_lasso.to_dict()},
        ])

    def _build_weights_table(self, X_train, summary_df):
        best_ridge = summary_df[summary_df["model_type"] == "Ridge"].iloc[0]
        best_lasso = summary_df[summary_df["model_type"] == "Lasso"].iloc[0]
        baseline_model = self.fitted_models["baseline"]
        ridge_model = self.fitted_models["Ridge", float(best_ridge["alpha"])]  # type: ignore
        lasso_model = self.fitted_models["Lasso", float(best_lasso["alpha"])]  # type: ignore
        feature_names = (
            list(X_train.columns)
            if hasattr(X_train, "columns")
            else [f"x{i}" for i in range(len(baseline_model.weights))]  # type: ignore
        )

        weights_df = pd.DataFrame({
            "feature": feature_names,
            "degree": np.arange(1, len(feature_names) + 1),
            "baseline_weight": baseline_model.weights,
            "ridge_weight": ridge_model.weights,
            "lasso_weight": lasso_model.weights,
        })
        weights_df["baseline_abs_weight"] = weights_df["baseline_weight"].abs()
        weights_df["ridge_abs_weight"] = weights_df["ridge_weight"].abs()
        weights_df["lasso_abs_weight"] = weights_df["lasso_weight"].abs()
        weights_df["lasso_zero"] = weights_df["lasso_abs_weight"] <= self.zero_threshold
        weights_df["ridge_zero"] = weights_df["ridge_abs_weight"] <= self.zero_threshold

        return weights_df

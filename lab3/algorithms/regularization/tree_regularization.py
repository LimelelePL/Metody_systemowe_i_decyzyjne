from typing import Literal

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None

    def fit(self, X, y):
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,  # type: ignore
            random_state=42,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model nie zostal wytrenowany.")

        return self.model.predict(X)


class TreeRegularizationComparison:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.parameter_values = {
            "min_samples_split": [2, 5, 10, 20, 40, 80, 120, 200, 400],
            "min_samples_leaf": [1, 2, 5, 10, 20, 40, 80, 120, 200],
            "max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75],
        }
        self.defaults = {
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
        }

    def compare(self, X_train, y_train, X_test, y_test):
        baseline = {
            "model": "baseline tree",
            "parameter": "baseline",
            "value": None,
            "label": "default",
            **self._score_tree(X_train, y_train, X_test, y_test),
        }

        rows = []
        for parameter, values in self.parameter_values.items():
            for value in values:
                rows.append({
                    "parameter": parameter,
                    "value": value,
                    "label": self._value_label(value),
                    "is_default": value == self.defaults[parameter],
                    **self._score_tree(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        **{parameter: value},
                    ),
                })

        results_df = pd.DataFrame(rows)
        summary_df = self._build_summary(results_df)
        best_regularized = self._find_best_regularized(results_df)
        comparison_df = self._build_comparison(baseline, best_regularized)

        return {
            "results": results_df,
            "summary": summary_df,
            "comparison": comparison_df,
            "best_regularized": best_regularized,
            "baseline": baseline,
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

    def _score_tree(self, X_train, y_train, X_test, y_test, **tree_params):
        tree = DecisionTreeClassifier(random_state=self.random_state, **tree_params)
        tree.fit(X_train, y_train)
        return {
            "train_error": 1 - accuracy_score(y_train, tree.predict(X_train)),
            "test_error": 1 - accuracy_score(y_test, tree.predict(X_test)),
        }

    @staticmethod
    def _value_label(value):
        if value is None:
            return "None"

        return str(value)

    @staticmethod
    def _build_summary(results_df):
        return (
            results_df
            .loc[results_df.groupby("parameter")["test_error"].idxmin()]
            .sort_values("test_error")
            .reset_index(drop=True)
        )

    @staticmethod
    def _find_best_regularized(results_df):
        regularized_results = results_df[~results_df["is_default"]].copy()
        return regularized_results.loc[regularized_results["test_error"].idxmin()]

    @staticmethod
    def _build_comparison(baseline, best_regularized):
        return pd.DataFrame([
            baseline,
            {
                "model": f"{best_regularized['parameter']}={best_regularized['label']}",
                **best_regularized.to_dict(),
            },
        ])

from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

try:
    from lab3.algorithms.bagging.bagging import MyBaggingClassifier
except ModuleNotFoundError:
    from algorithms.bagging.bagging import MyBaggingClassifier


class DecisionTreeTest:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


class RandomForrestTest:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


class ClassificationComparison:
    def __init__(self):
        self.models = {
            "Decision Tree": DecisionTreeTest(),
            "Random Forest": RandomForrestTest(),
            "Bagging": MyBaggingClassifier(),
        }

    def compare(self, X_train, X_test, y_train, y_test):
        rows = []

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def build_comparison_df(results: dict[str, Any] | DataFrame):
        if isinstance(results, pd.DataFrame):
            return results

        return (
            pd
            .DataFrame(results)
            .T.reset_index()
            .rename(
                columns={
                    "index": "Model",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1-score",
                }
            )
        )

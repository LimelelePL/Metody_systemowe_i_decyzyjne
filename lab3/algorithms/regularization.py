from typing import Literal

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.weights = None
        self.bias = None
        self.errors = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_values = X.to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)

        m, n = X_values.shape

        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X_values @ self.weights + self.bias

            self.errors = y_values - y_pred

            weights_gradient = self.calculate_weight_gradient(X_values, y_values, self.errors, m)
            bias_gradient = self.calculate_bias_gradient(self.errors, m)

            self.weights -= self.learning_rate * weights_gradient
            self.bias -= self.learning_rate * bias_gradient

        return self

    def predict(self, X_test: pd.DataFrame):
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet.")

        return X_test.to_numpy(dtype=float) @ self.weights + float(self.bias)

    def calculate_weight_gradient(self, X, y, errors, m):
        return (-2 / m) * X.T @ errors

    def calculate_bias_gradient(self, errors, m):
        return (-2 / m) * np.sum(errors)


class Lasso(LinearRegressionGD):
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        super().__init__(learning_rate, epochs)
        self.alpha = alpha

    def calculate_weight_gradient(self, X, y, errors, m):
        mse_gradient = (-2 / m) * X.T @ errors

        if self.weights is None:
            raise ValueError("wagi sa puste")

        lasso_gradient = self.alpha * np.sign(np.asarray(self.weights, dtype=float))

        return mse_gradient + lasso_gradient


class Ridge(LinearRegressionGD):
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        super().__init__(learning_rate, epochs)
        self.alpha = alpha

    def calculate_weight_gradient(self, X, y, errors, m):
        mse_gradient = (-2 / m) * X.T @ errors

        if self.weights is None:
            raise ValueError("wagi sa puste")

        ridge_gradient = -2 * self.alpha * np.asarray(self.weights, dtype=float)

        return mse_gradient + ridge_gradient


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 5,
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
            raise ValueError("model nie zostal wytrenowany")
        return self.model.predict(X)

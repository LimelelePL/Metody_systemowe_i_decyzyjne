import numpy as np
import pandas as pd


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
            X_values = X.to_numpy(dtype=float)
        else:
            X_values = np.asarray(X, dtype=float)
            self.feature_names = [f"x{i}" for i in range(X_values.shape[1])]

        y_values = np.asarray(y, dtype=float)
        m, n = X_values.shape

        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            predictions = X_values @ self.weights + self.bias
            errors = predictions - y_values

            weights_gradient = self.calculate_weight_gradient(X_values, errors, m)
            bias_gradient = self.calculate_bias_gradient(errors, m)

            self.weights -= self.learning_rate * weights_gradient
            self.weights = self.apply_regularization_step(self.weights)
            self.bias -= self.learning_rate * bias_gradient
            self.loss_history.append(float(np.mean(errors**2)))

        return self

    def predict(self, X_test: pd.DataFrame):
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet.")

        if hasattr(X_test, "to_numpy"):
            X_values = X_test.to_numpy(dtype=float)
        else:
            X_values = np.asarray(X_test, dtype=float)

        return X_values @ self.weights + float(self.bias)

    def calculate_weight_gradient(self, X, errors, m):
        return (2 / m) * X.T @ errors

    def calculate_bias_gradient(self, errors, m):
        return (2 / m) * np.sum(errors)

    def apply_regularization_step(self, weights):
        return weights

    def weights_frame(self, zero_threshold=1e-8):
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")

        return pd.DataFrame({
            "feature": self.feature_names,
            "weight": self.weights,
            "abs_weight": np.abs(self.weights),
            "is_zero": np.abs(self.weights) <= zero_threshold,
        }).sort_values("abs_weight", ascending=False)


class Lasso(LinearRegressionGD):
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        super().__init__(learning_rate, epochs)
        self.alpha = alpha

    def apply_regularization_step(self, weights):
        threshold = self.learning_rate * self.alpha
        return np.sign(weights) * np.maximum(np.abs(weights) - threshold, 0.0)


class Ridge(LinearRegressionGD):
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        super().__init__(learning_rate, epochs)
        self.alpha = alpha

    def calculate_weight_gradient(self, X, errors, m):
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")

        mse_gradient = super().calculate_weight_gradient(X, errors, m)
        ridge_gradient = 2 * self.alpha * np.asarray(self.weights, dtype=float)
        return mse_gradient + ridge_gradient

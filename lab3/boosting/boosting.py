import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class MyGrandientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=None):
        self.initial_prediction = None
        self.current_prediciton = None
        self.trees = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_columns = None
        self.numeric_fill_values = None
        self.categorical_fill_values = None

    def _prepare_features(self, X, fit=False):
        X_processed = X.copy()

        numeric_columns = X_processed.select_dtypes(include=["number"]).columns
        categorical_columns = X_processed.select_dtypes(exclude=["number"]).columns

        if fit:
            self.numeric_fill_values = X_processed[numeric_columns].median()
            self.categorical_fill_values = {
                column: X_processed[column].mode().iloc[0] for column in categorical_columns
            }

        if len(numeric_columns) > 0 and self.numeric_fill_values is not None:
            X_processed.loc[:, numeric_columns] = X_processed[numeric_columns].fillna(self.numeric_fill_values)

        for column in categorical_columns:
            if self.categorical_fill_values is not None:
                X_processed[column] = X_processed[column].fillna(self.categorical_fill_values[column])

        X_processed = pd.get_dummies(X_processed)

        if fit:
            self.feature_columns = X_processed.columns
        else:
            X_processed = X_processed.reindex(columns=self.feature_columns, fill_value=0)

        return X_processed

    def fit(self, X_train, y_train):
        X_train_processed = self._prepare_features(X_train, fit=True)
        y_train_processed = np.asarray(y_train, dtype=float)

        self.initial_prediction = float(np.mean(y_train_processed))
        self.current_prediciton = np.full(y_train_processed.shape, self.initial_prediction)
        self.trees = []

        for _ in range(self.n_estimators):
            rests = y_train_processed - self.current_prediciton
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X_train_processed, rests)

            correction = tree.predict(X_train_processed)
            self.current_prediciton += self.learning_rate * correction
            self.trees.append(tree)

        return self

    def predict(self, X_test):
        if self.initial_prediction is None:
            raise ValueError("Model nie zostal wytrenowany. Najpierw uzyj fit().")

        X_test_processed = self._prepare_features(X_test, fit=False)
        prediction = np.full(X_test_processed.shape[0], self.initial_prediction)

        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X_test_processed)

        return prediction

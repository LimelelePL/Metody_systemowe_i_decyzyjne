import numpy as np
from sklearn.tree import DecisionTreeRegressor

try:
    from lab3.utils.preprocessing import fit_feature_preprocessor, transform_features
except ModuleNotFoundError:
    from utils.preprocessing import fit_feature_preprocessor, transform_features


class MyGrandientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=None):
        self.initial_prediction = None
        self.current_prediciton = None
        self.trees = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.preprocessing_artifacts = None

    def fit_prepared(self, X_train_processed, y_train):
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

    def fit(self, X_train, y_train):
        X_train_processed, self.preprocessing_artifacts = fit_feature_preprocessor(X_train)
        return self.fit_prepared(X_train_processed, y_train)

    def predict_prepared(self, X_test_processed, n_estimators=None):
        if self.initial_prediction is None:
            raise ValueError("Model nie zostal wytrenowany. Najpierw uzyj fit().")

        prediction = np.full(X_test_processed.shape[0], self.initial_prediction)

        estimators_to_use = self.trees if n_estimators is None else self.trees[:n_estimators]
        for tree in estimators_to_use:
            prediction += self.learning_rate * tree.predict(X_test_processed)

        return prediction

    def predict(self, X_test, n_estimators=None):
        if self.preprocessing_artifacts is None:
            raise ValueError("Brak zapisanego preprocessingu. Najpierw uzyj fit().")

        X_test_processed = transform_features(X_test, self.preprocessing_artifacts)
        return self.predict_prepared(X_test_processed, n_estimators=n_estimators)

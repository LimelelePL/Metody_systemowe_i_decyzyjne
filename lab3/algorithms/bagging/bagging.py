from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MyBaggingClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models: list[DecisionTreeClassifier] = []

    def fit(self, X_train, y_train):
        n_samples = X_train.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.models = []

        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, size=n_samples, replace=True)

            if hasattr(X_train, "iloc"):
                X_sample = X_train.iloc[indices]
            else:
                X_sample = X_train[indices]

            if hasattr(y_train, "iloc"):
                y_sample = y_train.iloc[indices]
            else:
                y_sample = y_train[indices]

            model = DecisionTreeClassifier(random_state=42, max_depth=self.max_depth)
            model.fit(X_sample, y_sample)

            self.models.append(model)

        return self

    def predict(self, X_test):
        all_predictions = []

        for model in self.models:
            prediction = model.predict(X_test)
            all_predictions.append(prediction)

        all_predictions = np.array(all_predictions)

        return self.vote(all_predictions, X_test.shape[0])

    def vote(self, all_predictions: np.ndarray, shape: int):
        final_predictions = []
        for i in range(shape):
            votes = all_predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)

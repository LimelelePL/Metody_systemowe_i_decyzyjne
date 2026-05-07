import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class TelcoDecisionTreeModel:
    def __init__(self, random_state: int = 42, max_depth: int | None = None):
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
        )
        self.columns_: list[str] | None = None

    def fit(self, X, y):
        x_encoded = pd.get_dummies(X, drop_first=False)
        self.columns_ = x_encoded.columns.tolist()
        self.model.fit(x_encoded, y)
        return self

    def predict(self, X):
        x_encoded = pd.get_dummies(X, drop_first=False)
        x_encoded = x_encoded.reindex(columns=self.columns_, fill_value=0)
        return self.model.predict(x_encoded)

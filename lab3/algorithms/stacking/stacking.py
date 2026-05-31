import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


class MyStacking:
    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.level_0_names = ["Decision Tree", "KNN", "Random Forest"]
        self.tree = DecisionTreeClassifier(random_state=random_state)
        self.kn = KNeighborsClassifier()
        self.random_forest = RandomForestClassifier(random_state=random_state)
        self.meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.label_encoder = LabelEncoder()

    def _make_cross_val_predicts(self, X_train, y_train):
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return np.column_stack([
            self.label_encoder.transform(cross_val_predict(self.tree, X_train, y_train, cv=cv)),
            self.label_encoder.transform(cross_val_predict(self.kn, X_train, y_train, cv=cv)),
            self.label_encoder.transform(cross_val_predict(self.random_forest, X_train, y_train, cv=cv)),
        ])

    def _make_test_predicts(self, X_test):
        return np.column_stack([
            self.label_encoder.transform(self.tree.predict(X_test)),
            self.label_encoder.transform(self.kn.predict(X_test)),
            self.label_encoder.transform(self.random_forest.predict(X_test)),
        ])

    def fit(self, X_train, y_train):
        self.label_encoder.fit(y_train)
        cross_predicts = self._make_cross_val_predicts(X_train, y_train)
        self.meta_model.fit(cross_predicts, y_train)

        self.tree.fit(X_train, y_train)
        self.kn.fit(X_train, y_train)
        self.random_forest.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        low_model_predicts = self._make_test_predicts(X_test)
        return self.meta_model.predict(low_model_predicts)

    def predict_level_0(self, X_test):
        return self._make_test_predicts(X_test)

    def get_meta_model_coefficients(self):
        return self.meta_model.coef_.ravel()

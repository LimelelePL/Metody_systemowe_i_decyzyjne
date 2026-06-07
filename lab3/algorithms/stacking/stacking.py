import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

try:
    from lab3.utils.preprocessing import fit_feature_preprocessor, transform_features
except ModuleNotFoundError:
    from utils.preprocessing import fit_feature_preprocessor, transform_features


class MyStacking:
    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.level_0_names = ["Decision Tree", "Random Forest", "Extra Trees"]
        self.tree = DecisionTreeRegressor(max_depth=3, random_state=random_state)
        self.random_forest = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.meta_model = Ridge(alpha=1.0)
        self.preprocessing_artifacts = None

    def _prepare_features(self, X, fit=False):
        if fit:
            X_processed, self.preprocessing_artifacts = fit_feature_preprocessor(X)
            return X_processed

        if self.preprocessing_artifacts is None:
            raise ValueError("Model nie zostal wytrenowany. Najpierw uzyj fit().")

        return transform_features(X, self.preprocessing_artifacts)

    def _make_cross_val_predicts(self, X_train, y_train):
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return np.column_stack([
            cross_val_predict(self.tree, X_train, y_train, cv=cv),
            cross_val_predict(self.random_forest, X_train, y_train, cv=cv, n_jobs=-1),
            cross_val_predict(self.extra_trees, X_train, y_train, cv=cv, n_jobs=-1),
        ])

    def _make_test_predicts(self, X_test):
        return np.column_stack([
            self.tree.predict(X_test),
            self.random_forest.predict(X_test),
            self.extra_trees.predict(X_test),
        ])

    def fit(self, X_train, y_train):
        X_train_processed = self._prepare_features(X_train, fit=True)
        y_train_values = np.asarray(y_train, dtype=float)

        cross_predicts = self._make_cross_val_predicts(X_train_processed, y_train_values)
        self.meta_model.fit(cross_predicts, y_train_values)

        self.tree.fit(X_train_processed, y_train_values)
        self.random_forest.fit(X_train_processed, y_train_values)
        self.extra_trees.fit(X_train_processed, y_train_values)
        return self

    def predict(self, X_test):
        X_test_processed = self._prepare_features(X_test, fit=False)
        low_model_predicts = self._make_test_predicts(X_test_processed)
        return self.meta_model.predict(low_model_predicts)

    def predict_level_0(self, X_test):
        X_test_processed = self._prepare_features(X_test, fit=False)
        return self._make_test_predicts(X_test_processed)

    def get_meta_model_coefficients(self):
        return self.meta_model.coef_.ravel()

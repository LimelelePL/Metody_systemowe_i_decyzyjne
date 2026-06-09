import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from lab3.constants import CARAT, DEPTH, TABLE, X, Y, Z
    from lab3.utils.preprocessing import fit_feature_preprocessor, transform_features
except ModuleNotFoundError:
    from constants import CARAT, DEPTH, TABLE, X, Y, Z
    from utils.preprocessing import fit_feature_preprocessor, transform_features


class MixtureOfExperts:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.gate = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        self.global_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        self.scaler = StandardScaler()
        self.experts = {}
        self.expert_names = {}
        self.preprocessing_artifacts = None
        self.cluster_summary = None

    def _prepare_features(self, X, fit=False):
        if fit:
            X_prepared, self.preprocessing_artifacts = fit_feature_preprocessor(X)
            return X_prepared

        if self.preprocessing_artifacts is None:
            raise ValueError("Model nie zostal wytrenowany. Najpierw uzyj fit().")

        return transform_features(X, self.preprocessing_artifacts)

    def _scale_for_clustering(self, X, fit=False):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _build_expert(self, cluster):
        expert_builders = [
            ("Ridge", lambda: Ridge(alpha=1.0)),
            (
                "RandomForestRegressor",
                lambda: RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            ),
            (
                "GradientBoostingRegressor",
                lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=self.random_state),
            ),
        ]
        expert_name, builder = expert_builders[cluster % len(expert_builders)]
        return expert_name, builder()

    @staticmethod
    def _metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred),
        }

    def _build_cluster_summary(self, X_train, y_train, cluster_labels):
        summary_df = X_train.copy()
        summary_df["price"] = np.asarray(y_train, dtype=float)
        summary_df["cluster"] = cluster_labels

        summary = (
            summary_df
            .groupby("cluster")
            .agg(
                count=("price", "size"),
                mean_price=("price", "mean"),
                mean_carat=(CARAT, "mean"),
                mean_depth=(DEPTH, "mean"),
                mean_table=(TABLE, "mean"),
                mean_x=(X, "mean"),
                mean_y=(Y, "mean"),
                mean_z=(Z, "mean"),
            )
            .reset_index()
        )
        summary["expert_model"] = summary["cluster"].map(self.expert_names)

        return summary

    def fit(self, X_train, y_train):
        X_train_prepared = self._prepare_features(X_train, fit=True)
        X_train_scaled = self._scale_for_clustering(X_train_prepared, fit=True)
        y_train_values = np.asarray(y_train, dtype=float)

        cluster_labels = self.kmeans.fit_predict(X_train_scaled)
        self.gate.fit(X_train_scaled, cluster_labels)

        self.experts = {}
        self.expert_names = {}
        for cluster in range(self.n_clusters):
            mask = cluster_labels == cluster
            X_cluster = X_train_prepared.loc[mask]
            y_cluster = y_train_values[mask]

            expert_name, expert = self._build_expert(cluster)
            expert.fit(X_cluster, y_cluster)
            self.experts[cluster] = expert
            self.expert_names[cluster] = expert_name

        self.global_model.fit(X_train_prepared, y_train_values)
        self.cluster_summary = self._build_cluster_summary(X_train, y_train, cluster_labels)
        return self

    def predict(self, X_test):
        if not self.experts:
            raise ValueError("Model nie zostal wytrenowany. Najpierw uzyj fit().")

        X_test_prepared = self._prepare_features(X_test, fit=False)
        X_test_scaled = self._scale_for_clustering(X_test_prepared, fit=False)
        gate_clusters = self.gate.predict(X_test_scaled)

        predictions = np.empty(X_test_prepared.shape[0], dtype=float)
        for cluster in range(self.n_clusters):
            mask = gate_clusters == cluster
            if not np.any(mask):
                continue

            predictions[mask] = self.experts[cluster].predict(X_test_prepared.loc[mask])

        return predictions

    def predict_global_model(self, X_test):
        X_test_prepared = self._prepare_features(X_test, fit=False)
        return self.global_model.predict(X_test_prepared)

    def compare_with_global_model(self, X_test, y_test):
        y_true = np.asarray(y_test, dtype=float)
        moe_predictions = self.predict(X_test)
        global_predictions = self.predict_global_model(X_test)

        return pd.DataFrame([
            {"Model": "Mixture of Experts", **self._metrics(y_true, moe_predictions)},
            {"Model": "Global Random Forest", **self._metrics(y_true, global_predictions)},
        ])

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

try:
    from lab3.algorithms.boosting.boosting import MyGrandientBoosting
    from lab3.algorithms.stacking.stacking import MyStacking
    from lab3.utils.preprocessing import fit_feature_preprocessor, transform_features
except ModuleNotFoundError:
    from algorithms.boosting.boosting import MyGrandientBoosting
    from algorithms.stacking.stacking import MyStacking
    from utils.preprocessing import fit_feature_preprocessor, transform_features


class StackingComparison:
    def __init__(self, boosting_n_estimators=50, boosting_learning_rate=0.1, boosting_max_depth=3):
        self.level_0_models = {
            "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }
        self.stacking = MyStacking(random_state=42)
        self.boosting = MyGrandientBoosting(
            n_estimators=boosting_n_estimators,
            learning_rate=boosting_learning_rate,
            max_depth=boosting_max_depth,
        )
        self.boosting_n_estimators = boosting_n_estimators

    @staticmethod
    def _build_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred),
        }

    def compare(self, X_train, X_test, y_train, y_test):
        X_train_prepared, preprocessing_artifacts = fit_feature_preprocessor(X_train)
        X_test_prepared = transform_features(X_test, preprocessing_artifacts)

        y_train_values = np.asarray(y_train, dtype=float)
        y_test_values = np.asarray(y_test, dtype=float)

        metrics_rows = []
        level_0_predictions = {}

        for model_name, model in self.level_0_models.items():
            model.fit(X_train_prepared, y_train_values)
            train_predictions = model.predict(X_train_prepared)
            test_predictions = model.predict(X_test_prepared)
            level_0_predictions[model_name] = test_predictions

            train_metrics = self._build_metrics(y_train_values, train_predictions)
            test_metrics = self._build_metrics(y_test_values, test_predictions)
            metrics_rows.append({
                "Model": model_name,
                "Train MAE": train_metrics["MAE"],
                "Test MAE": test_metrics["MAE"],
                "Train RMSE": train_metrics["RMSE"],
                "Test RMSE": test_metrics["RMSE"],
                "Train R2": train_metrics["R2"],
                "Test R2": test_metrics["R2"],
            })

        self.stacking.fit(X_train, y_train_values)
        stacking_train_predictions = self.stacking.predict(X_train)
        stacking_test_predictions = self.stacking.predict(X_test)
        stacking_train_metrics = self._build_metrics(y_train_values, stacking_train_predictions)
        stacking_test_metrics = self._build_metrics(y_test_values, stacking_test_predictions)
        metrics_rows.append({
            "Model": "Stacking",
            "Train MAE": stacking_train_metrics["MAE"],
            "Test MAE": stacking_test_metrics["MAE"],
            "Train RMSE": stacking_train_metrics["RMSE"],
            "Test RMSE": stacking_test_metrics["RMSE"],
            "Train R2": stacking_train_metrics["R2"],
            "Test R2": stacking_test_metrics["R2"],
        })

        self.boosting.fit(X_train, y_train_values)
        boosting_train_predictions = self.boosting.predict(X_train)
        boosting_test_predictions = self.boosting.predict(X_test)
        boosting_train_metrics = self._build_metrics(y_train_values, boosting_train_predictions)
        boosting_test_metrics = self._build_metrics(y_test_values, boosting_test_predictions)
        metrics_rows.append({
            "Model": f"MyGrandientBoosting n={self.boosting_n_estimators}",
            "Train MAE": boosting_train_metrics["MAE"],
            "Test MAE": boosting_test_metrics["MAE"],
            "Train RMSE": boosting_train_metrics["RMSE"],
            "Test RMSE": boosting_test_metrics["RMSE"],
            "Train R2": boosting_train_metrics["R2"],
            "Test R2": boosting_test_metrics["R2"],
        })

        metrics_df = pd.DataFrame(metrics_rows)
        diagnostics_df = self._build_level_0_diagnostics(level_0_predictions, y_test_values)

        return {
            "metrics": metrics_df,
            "level_0_diagnostics": diagnostics_df,
            "stacking_vs_boosting": metrics_df[
                metrics_df["Model"].isin(["Stacking", f"MyGrandientBoosting n={self.boosting_n_estimators}"])
            ].reset_index(drop=True),
        }

    def _build_level_0_diagnostics(self, level_0_predictions, y_test):
        diagnostics_rows = []
        meta_coefficients = self.stacking.get_meta_model_coefficients()

        for index, model_name in enumerate(self.stacking.level_0_names):
            base_predictions = np.asarray(level_0_predictions[model_name], dtype=float)
            base_metrics = self._build_metrics(y_test, base_predictions)

            diagnostics_rows.append({
                "Model": model_name,
                "Meta coefficient": float(meta_coefficients[index]),
                "Test MAE": base_metrics["MAE"],
                "Test RMSE": base_metrics["RMSE"],
                "Test R2": base_metrics["R2"],
            })

        return pd.DataFrame(diagnostics_rows)

    @staticmethod
    def build_metrics_df(experiment):
        return experiment["metrics"].copy()

    @staticmethod
    def build_level_0_diagnostics_df(experiment):
        return experiment["level_0_diagnostics"].copy()

    @staticmethod
    def build_stacking_vs_boosting_df(experiment):
        return experiment["stacking_vs_boosting"].copy()

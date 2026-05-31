import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from lab3.algorithms.bagging.bagging import MyBaggingClassifier
    from lab3.algorithms.stacking.stacking import MyStacking
except ModuleNotFoundError:
    from algorithms.bagging.bagging import MyBaggingClassifier
    from algorithms.stacking.stacking import MyStacking


class StackingComparison:
    def __init__(self):
        self.level_0_models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
        }
        self.bagging = MyBaggingClassifier(random_state=42)
        self.stacking = MyStacking(random_state=42)

    @staticmethod
    def _build_metrics_row(model_name, y_true, y_pred):
        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def compare(self, X_train, X_test, y_train, y_test):
        metrics_rows = []
        level_0_predictions = {}

        for model_name, model in self.level_0_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            level_0_predictions[model_name] = y_pred
            metrics_rows.append(self._build_metrics_row(model_name, y_test, y_pred))

        self.bagging.fit(X_train, y_train)
        bagging_predictions = self.bagging.predict(X_test)
        metrics_rows.append(self._build_metrics_row("Bagging", y_test, bagging_predictions))

        self.stacking.fit(X_train, y_train)
        stacking_predictions = self.stacking.predict(X_test)
        metrics_rows.append(self._build_metrics_row("Stacking", y_test, stacking_predictions))

        metrics_df = pd.DataFrame(metrics_rows)
        diagnostics_df = self._build_level_0_diagnostics(level_0_predictions, stacking_predictions, y_test)

        return {
            "metrics": metrics_df,
            "level_0_diagnostics": diagnostics_df,
            "stacking_vs_bagging": metrics_df[metrics_df["Model"].isin(["Bagging", "Stacking"])].reset_index(drop=True),
        }

    def _build_level_0_diagnostics(self, level_0_predictions, stacking_predictions, y_test):
        y_true = pd.Series(y_test).reset_index(drop=True)
        diagnostics_rows = []
        meta_coefficients = self.stacking.get_meta_model_coefficients()

        for index, model_name in enumerate(self.stacking.level_0_names):
            base_predictions = pd.Series(level_0_predictions[model_name]).reset_index(drop=True)
            wrong_mask = base_predictions != y_true
            wrong_count = int(wrong_mask.sum())
            corrected_by_stacking = int(((stacking_predictions == y_true) & wrong_mask).sum())
            corrected_rate = corrected_by_stacking / wrong_count if wrong_count else 0.0
            agreement_with_stacking = float((base_predictions == stacking_predictions).mean())

            diagnostics_rows.append({
                "Model": model_name,
                "Meta coefficient": float(meta_coefficients[index]),
                "Base accuracy": accuracy_score(y_true, base_predictions),
                "Wrong predictions": wrong_count,
                "Corrected by stacking": corrected_by_stacking,
                "Corrected rate": corrected_rate,
                "Agreement with stacking": agreement_with_stacking,
            })

        return pd.DataFrame(diagnostics_rows)

    @staticmethod
    def build_metrics_df(experiment):
        return experiment["metrics"].copy()

    @staticmethod
    def build_level_0_diagnostics_df(experiment):
        return experiment["level_0_diagnostics"].copy()

    @staticmethod
    def build_stacking_vs_bagging_df(experiment):
        return experiment["stacking_vs_bagging"].copy()

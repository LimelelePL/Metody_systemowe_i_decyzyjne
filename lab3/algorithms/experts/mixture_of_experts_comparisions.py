import pandas as pd

try:
    from lab3.algorithms.boosting.boosting import MyGrandientBoosting
    from lab3.algorithms.experts.mixture_of_experts import MixtureOfExperts
except ModuleNotFoundError:
    from algorithms.boosting.boosting import MyGrandientBoosting
    from algorithms.experts.mixture_of_experts import MixtureOfExperts


class MixtureOfExpertsComparison:
    def __init__(self, n_clusters=3, boosting_n_estimators=50, boosting_learning_rate=0.1, boosting_max_depth=3):
        self.n_clusters = n_clusters
        self.boosting_n_estimators = boosting_n_estimators
        self.boosting_learning_rate = boosting_learning_rate
        self.boosting_max_depth = boosting_max_depth

    @staticmethod
    def _build_metrics_row(model_name, y_true, y_pred):
        return {
            "Model": model_name,
            **MixtureOfExperts._metrics(y_true, y_pred),
        }

    def compare(self, X_train, X_test, y_train, y_test):
        moe = MixtureOfExperts(n_clusters=self.n_clusters)
        moe.fit(X_train, y_train)

        moe_train_predictions = moe.predict(X_train)
        moe_test_predictions = moe.predict(X_test)

        global_train_predictions = moe.predict_global_model(X_train)
        global_test_predictions = moe.predict_global_model(X_test)

        boosting = MyGrandientBoosting(
            n_estimators=self.boosting_n_estimators,
            learning_rate=self.boosting_learning_rate,
            max_depth=self.boosting_max_depth,
        )
        boosting.fit(X_train, y_train)
        boosting_train_predictions = boosting.predict(X_train)
        boosting_test_predictions = boosting.predict(X_test)

        comparison_df = pd.DataFrame([
            {
                "model": "Mixture of Experts",
                "train_mae": MixtureOfExperts._metrics(y_train, moe_train_predictions)["MAE"],
                "test_mae": MixtureOfExperts._metrics(y_test, moe_test_predictions)["MAE"],
                "train_rmse": MixtureOfExperts._metrics(y_train, moe_train_predictions)["RMSE"],
                "test_rmse": MixtureOfExperts._metrics(y_test, moe_test_predictions)["RMSE"],
                "train_r2": MixtureOfExperts._metrics(y_train, moe_train_predictions)["R2"],
                "test_r2": MixtureOfExperts._metrics(y_test, moe_test_predictions)["R2"],
            },
            {
                "model": "Global Random Forest",
                "train_mae": MixtureOfExperts._metrics(y_train, global_train_predictions)["MAE"],
                "test_mae": MixtureOfExperts._metrics(y_test, global_test_predictions)["MAE"],
                "train_rmse": MixtureOfExperts._metrics(y_train, global_train_predictions)["RMSE"],
                "test_rmse": MixtureOfExperts._metrics(y_test, global_test_predictions)["RMSE"],
                "train_r2": MixtureOfExperts._metrics(y_train, global_train_predictions)["R2"],
                "test_r2": MixtureOfExperts._metrics(y_test, global_test_predictions)["R2"],
            },
            {
                "model": f"MyGrandientBoosting n={self.boosting_n_estimators}",
                "train_mae": MixtureOfExperts._metrics(y_train, boosting_train_predictions)["MAE"],
                "test_mae": MixtureOfExperts._metrics(y_test, boosting_test_predictions)["MAE"],
                "train_rmse": MixtureOfExperts._metrics(y_train, boosting_train_predictions)["RMSE"],
                "test_rmse": MixtureOfExperts._metrics(y_test, boosting_test_predictions)["RMSE"],
                "train_r2": MixtureOfExperts._metrics(y_train, boosting_train_predictions)["R2"],
                "test_r2": MixtureOfExperts._metrics(y_test, boosting_test_predictions)["R2"],
            },
        ])

        return {
            "cluster_summary": moe.cluster_summary.copy(),
            "global_comparison": moe.compare_with_global_model(X_test, y_test),
            "comparison": comparison_df,
        }

    @staticmethod
    def build_cluster_summary_df(experiment):
        return experiment["cluster_summary"].copy()

    @staticmethod
    def build_global_comparison_df(experiment):
        return experiment["global_comparison"].copy()

    @staticmethod
    def build_comparison_df(experiment):
        return experiment["comparison"].copy()

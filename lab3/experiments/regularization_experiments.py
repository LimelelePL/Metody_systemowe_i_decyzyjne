import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lab3.algorithms.regularization import Lasso, LinearRegressionGD, Ridge

RANDOM_STATE = 42


def _sample_data(X, y, sample_size):
    if sample_size is None or len(X) <= sample_size:
        return X.copy(), y.copy()

    sampled_index = X.sample(n=sample_size, random_state=RANDOM_STATE).index
    return X.loc[sampled_index].copy(), y.loc[sampled_index].copy()


def _prepare_polynomial_regression_data(
    X_train,
    y_train,
    X_test,
    y_test,
    feature="carat",
    degree=20,
    train_sample_size=6000,
    test_sample_size=6000,
):
    X_train_sample, y_train_sample = _sample_data(X_train[[feature]], y_train, train_sample_size)
    X_test_sample, y_test_sample = _sample_data(X_test[[feature]], y_test, test_sample_size)

    fill_value = X_train_sample[feature].median()
    X_train_sample[feature] = X_train_sample[feature].fillna(fill_value)
    X_test_sample[feature] = X_test_sample[feature].fillna(fill_value)

    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = polynomial.fit_transform(X_train_sample)
    X_test_poly = polynomial.transform(X_test_sample)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    feature_names = polynomial.get_feature_names_out([feature])
    y_train_values = y_train_sample.to_numpy(dtype=float)
    y_test_values = y_test_sample.to_numpy(dtype=float)
    y_mean = y_train_values.mean()
    y_std = y_train_values.std()

    return {
        "X_train": pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_sample.index),
        "X_test": pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test_sample.index),
        "y_train": y_train_values,
        "y_test": y_test_values,
        "y_train_scaled": (y_train_values - y_mean) / y_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_names": feature_names,
    }


def _score_regression_model(model, data):
    model.fit(data["X_train"], data["y_train_scaled"])

    train_predictions = model.predict(data["X_train"]) * data["y_std"] + data["y_mean"]
    test_predictions = model.predict(data["X_test"]) * data["y_std"] + data["y_mean"]

    return {
        "train_mse": mean_squared_error(data["y_train"], train_predictions),
        "test_mse": mean_squared_error(data["y_test"], test_predictions),
        "model": model,
    }


def _weights_summary(model, zero_threshold):
    weights = np.asarray(model.weights, dtype=float)
    return {
        "nonzero_weights": int(np.sum(np.abs(weights) > zero_threshold)),
        "zero_weights": int(np.sum(np.abs(weights) <= zero_threshold)),
        "mean_abs_weight": float(np.mean(np.abs(weights))),
        "max_abs_weight": float(np.max(np.abs(weights))),
    }


def run_linear_regularization_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    feature="carat",
    degree=20,
    ridge_alphas=None,
    lasso_alphas=None,
    learning_rate=0.03,
    epochs=2000,
    zero_threshold=1e-6,
    train_sample_size=6000,
    test_sample_size=6000,
):
    if ridge_alphas is None:
        ridge_alphas = np.logspace(-8, 1, 10)
    if lasso_alphas is None:
        lasso_alphas = np.logspace(-5, -1, 9)

    data = _prepare_polynomial_regression_data(
        X_train,
        y_train,
        X_test,
        y_test,
        feature=feature,
        degree=degree,
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
    )

    baseline_model = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs)
    baseline_score = _score_regression_model(baseline_model, data)
    baseline_row = {
        "model_type": "baseline",
        "alpha": 0.0,
        **{key: baseline_score[key] for key in ["train_mse", "test_mse"]},
        **_weights_summary(baseline_model, zero_threshold),
    }

    result_rows = []
    fitted_models = {"baseline": baseline_model}

    for alpha in ridge_alphas:
        ridge_model = Ridge(learning_rate=learning_rate, epochs=epochs, alpha=float(alpha))
        score = _score_regression_model(ridge_model, data)
        result_rows.append({
            "model_type": "Ridge",
            "alpha": float(alpha),
            **{key: score[key] for key in ["train_mse", "test_mse"]},
            **_weights_summary(ridge_model, zero_threshold),
        })
        fitted_models["Ridge", float(alpha)] = ridge_model

    for alpha in lasso_alphas:
        lasso_model = Lasso(learning_rate=learning_rate, epochs=epochs, alpha=float(alpha))
        score = _score_regression_model(lasso_model, data)
        result_rows.append({
            "model_type": "Lasso",
            "alpha": float(alpha),
            **{key: score[key] for key in ["train_mse", "test_mse"]},
            **_weights_summary(lasso_model, zero_threshold),
        })
        fitted_models["Lasso", float(alpha)] = lasso_model

    results_df = pd.DataFrame(result_rows)
    summary_df = (
        results_df
        .loc[results_df.groupby("model_type")["test_mse"].idxmin()]
        .sort_values("test_mse")
        .reset_index(drop=True)
    )

    best_ridge = summary_df[summary_df["model_type"] == "Ridge"].iloc[0]
    best_lasso = summary_df[summary_df["model_type"] == "Lasso"].iloc[0]
    best_regularized = summary_df.iloc[0]

    comparison_df = pd.DataFrame([
        {"model": "baseline GD", **baseline_row},
        {
            "model": f"Ridge alpha={best_ridge['alpha']:.0e}",
            **best_ridge.to_dict(),
        },
        {
            "model": f"Lasso alpha={best_lasso['alpha']:.0e}",
            **best_lasso.to_dict(),
        },
    ])

    ridge_model = fitted_models["Ridge", float(best_ridge["alpha"])]
    lasso_model = fitted_models["Lasso", float(best_lasso["alpha"])]
    weights_df = pd.DataFrame({
        "feature": data["feature_names"],
        "degree": np.arange(1, len(data["feature_names"]) + 1),
        "baseline_weight": baseline_model.weights,
        "ridge_weight": ridge_model.weights,
        "lasso_weight": lasso_model.weights,
    })
    weights_df["baseline_abs_weight"] = weights_df["baseline_weight"].abs()
    weights_df["ridge_abs_weight"] = weights_df["ridge_weight"].abs()
    weights_df["lasso_abs_weight"] = weights_df["lasso_weight"].abs()
    weights_df["lasso_zero"] = weights_df["lasso_abs_weight"] <= zero_threshold
    weights_df["ridge_zero"] = weights_df["ridge_abs_weight"] <= zero_threshold

    return {
        "results": results_df,
        "summary": summary_df,
        "comparison": comparison_df,
        "weights": weights_df,
        "best_regularized": best_regularized,
        "baseline": baseline_row,
    }


def _prepare_tree_data(X_train, X_test):
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test).reindex(columns=X_train_encoded.columns, fill_value=0)
    return X_train_encoded, X_test_encoded


def _score_tree(X_train, y_train, X_test, y_test, **tree_params):
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, **tree_params)
    tree.fit(X_train, y_train)
    return {
        "train_error": 1 - accuracy_score(y_train, tree.predict(X_train)),
        "test_error": 1 - accuracy_score(y_test, tree.predict(X_test)),
    }


def _value_label(value):
    if value is None:
        return "None"
    return str(value)


def run_tree_regularization_experiment(X_train, y_train, X_test, y_test):
    X_train_encoded, X_test_encoded = _prepare_tree_data(X_train, X_test)

    baseline = {
        "model": "baseline tree",
        "parameter": "baseline",
        "value": None,
        "label": "default",
        **_score_tree(X_train_encoded, y_train, X_test_encoded, y_test),
    }

    parameter_values = {
        "min_samples_split": [2, 5, 10, 20, 40, 80, 120, 200, 400],
        "min_samples_leaf": [1, 2, 5, 10, 20, 40, 80, 120, 200],
        "max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75],
    }
    defaults = {
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
    }

    rows = []
    for parameter, values in parameter_values.items():
        for value in values:
            scores = _score_tree(
                X_train_encoded,
                y_train,
                X_test_encoded,
                y_test,
                **{parameter: value},
            )
            rows.append({
                "parameter": parameter,
                "value": value,
                "label": _value_label(value),
                "is_default": value == defaults[parameter],
                **scores,
            })

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df
        .loc[results_df.groupby("parameter")["test_error"].idxmin()]
        .sort_values("test_error")
        .reset_index(drop=True)
    )

    regularized_results = results_df[~results_df["is_default"]].copy()
    best_regularized = regularized_results.loc[regularized_results["test_error"].idxmin()]

    comparison_df = pd.DataFrame([
        baseline,
        {
            "model": f"{best_regularized['parameter']}={best_regularized['label']}",
            **best_regularized.to_dict(),
        },
    ])

    return {
        "results": results_df,
        "summary": summary_df,
        "comparison": comparison_df,
        "best_regularized": best_regularized,
        "baseline": baseline,
    }

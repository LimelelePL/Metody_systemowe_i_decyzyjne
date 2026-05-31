import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def _format_number(value, _):
    return f"{value:,.0f}".replace(",", " ")


def _get_sweet_spot(results_df, parameter_column, test_column, lower_is_better=True):
    if lower_is_better:
        sweet_spot_index = results_df[test_column].idxmin()
    else:
        sweet_spot_index = results_df[test_column].idxmax()

    return (
        results_df.loc[sweet_spot_index, parameter_column],
        results_df.loc[sweet_spot_index, test_column],
    )


# Zadanie 3.0 - wspolne wykresy: model bazowy vs model zregularyzowany


def plot_tree_regularization_curve(
    results_df,
    parameter_column="max_depth",
    train_column="train_error",
    test_column="test_error",
    metric_label="blad klasyfikacji",
    lower_is_better=True,
):
    sweet_spot_parameter, sweet_spot_value = _get_sweet_spot(
        results_df,
        parameter_column,
        test_column,
        lower_is_better,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        results_df[parameter_column],
        results_df[train_column],
        marker="o",
        linewidth=2,
        label="train",
    )
    plt.plot(
        results_df[parameter_column],
        results_df[test_column],
        marker="o",
        linewidth=2,
        label="test",
    )
    plt.scatter(
        sweet_spot_parameter,
        sweet_spot_value,
        color="red",
        s=70,
        zorder=5,
        label=f"sweet spot: {parameter_column}={sweet_spot_parameter}",
    )

    plt.title("Regularyzacja drzewa decyzyjnego")
    plt.xlabel(parameter_column)
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regression_regularization_curve(
    results_df,
    parameter_column="alpha",
    train_column="train_mse",
    test_column="test_mse",
    metric_label="MSE",
    lower_is_better=True,
    log_x=True,
):
    sweet_spot_parameter, sweet_spot_value = _get_sweet_spot(
        results_df,
        parameter_column,
        test_column,
        lower_is_better,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        results_df[parameter_column],
        results_df[train_column],
        marker="o",
        linewidth=2,
        label="train",
    )
    plt.plot(
        results_df[parameter_column],
        results_df[test_column],
        marker="o",
        linewidth=2,
        label="test",
    )
    plt.scatter(
        sweet_spot_parameter,
        sweet_spot_value,
        color="red",
        s=70,
        zorder=5,
        label=f"sweet spot: {parameter_column}={sweet_spot_parameter:g}",
    )

    if log_x:
        plt.xscale("log")

    plt.title("Regularyzacja regresji")
    plt.xlabel(parameter_column)
    plt.ylabel(metric_label)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_format_number))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_baseline_vs_regularized(
    comparison_df,
    model_column="model",
    train_column="train_metric",
    test_column="test_metric",
    metric_label="metryka",
    title="Model bazowy vs model zregularyzowany",
):
    positions = range(len(comparison_df))
    bar_width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(
        [position - bar_width / 2 for position in positions],
        comparison_df[train_column],
        width=bar_width,
        color="steelblue",
        label="train",
    )
    plt.bar(
        [position + bar_width / 2 for position in positions],
        comparison_df[test_column],
        width=bar_width,
        color="coral",
        label="test",
    )

    plt.xticks(list(positions), comparison_df[model_column])
    plt.title(title)
    plt.ylabel(metric_label)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tree_baseline_vs_regularized(comparison_df):
    plot_baseline_vs_regularized(
        comparison_df,
        train_column="train_error",
        test_column="test_error",
        metric_label="blad klasyfikacji",
        title="Drzewo: model bazowy vs zregularyzowany",
    )


def plot_regression_baseline_vs_regularized(comparison_df):
    plot_baseline_vs_regularized(
        comparison_df,
        train_column="train_mse",
        test_column="test_mse",
        metric_label="MSE",
        title="Regresja: model bazowy vs zregularyzowany",
    )


# Zadanie 3.0 - podpunkt: Regularyzacja L1/L2 dla regresji


def plot_linear_regularization_curves(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, model_type in zip(axes, ["Ridge", "Lasso"]):
        model_df = results_df[results_df["model_type"] == model_type].copy()
        sweet_spot_alpha, sweet_spot_value = _get_sweet_spot(model_df, "alpha", "test_mse")

        ax.plot(model_df["alpha"], model_df["train_mse"], marker="o", label="train")
        ax.plot(model_df["alpha"], model_df["test_mse"], marker="o", label="test")
        ax.scatter(
            sweet_spot_alpha,
            sweet_spot_value,
            color="red",
            s=70,
            zorder=5,
            label=f"sweet spot: alpha={sweet_spot_alpha:.0e}",
        )
        ax.set_xscale("log")
        ax.set_title(f"{model_type} - wplyw alpha")
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("MSE")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_format_number))
    plt.tight_layout()
    plt.show()


def plot_zero_weights_by_alpha(results_df):
    plt.figure(figsize=(9, 5))

    for model_type in ["Ridge", "Lasso"]:
        model_df = results_df[results_df["model_type"] == model_type]
        plt.plot(
            model_df["alpha"],
            model_df["zero_weights"],
            marker="o",
            linewidth=2,
            label=model_type,
        )

    plt.xscale("log")
    plt.title("Liczba wag wyzerowanych dla roznych alpha")
    plt.xlabel("alpha")
    plt.ylabel("liczba wag rownych zero")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weight_shrinkage(weights_df):
    plt.figure(figsize=(11, 5))
    plt.plot(
        weights_df["degree"],
        weights_df["baseline_abs_weight"],
        marker="o",
        linewidth=2,
        label="baseline",
    )
    plt.plot(
        weights_df["degree"],
        weights_df["ridge_abs_weight"],
        marker="o",
        linewidth=2,
        label="Ridge",
    )
    plt.plot(
        weights_df["degree"],
        weights_df["lasso_abs_weight"],
        marker="o",
        linewidth=2,
        label="Lasso",
    )

    plt.title("Wplyw regularyzacji na wartosci bezwzgledne wag")
    plt.xlabel("stopien cechy wielomianowej")
    plt.ylabel("|waga|")
    plt.xticks(weights_df["degree"])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_linear_regularization_all(experiment):
    plot_linear_regularization_curves(experiment["results"])
    plot_zero_weights_by_alpha(experiment["results"])
    plot_weight_shrinkage(experiment["weights"])
    plot_regression_baseline_vs_regularized(experiment["comparison"])


# Zadanie 3.0 - podpunkt: Regularyzacja drzew decyzyjnych


def plot_tree_regularization_by_parameter(results_df):
    parameters = ["min_samples_split", "min_samples_leaf", "max_features"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, parameter in zip(axes, parameters):
        parameter_df = results_df[results_df["parameter"] == parameter].reset_index(drop=True)
        positions = list(range(len(parameter_df)))
        sweet_spot_index = parameter_df["test_error"].idxmin()

        ax.plot(positions, parameter_df["train_error"], marker="o", linewidth=2, label="train")
        ax.plot(positions, parameter_df["test_error"], marker="o", linewidth=2, label="test")
        ax.scatter(
            sweet_spot_index,
            parameter_df.loc[sweet_spot_index, "test_error"],
            color="red",
            s=70,
            zorder=5,
            label="sweet spot",
        )

        ax.set_title(parameter)
        ax.set_xlabel(parameter)
        ax.set_xticks(positions)
        ax.set_xticklabels(parameter_df["label"], rotation=35)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("blad klasyfikacji")
    plt.tight_layout()
    plt.show()


def plot_tree_regularization_all(experiment):
    plot_tree_regularization_by_parameter(experiment["results"])
    plot_tree_baseline_vs_regularized(experiment["comparison"])


# Zadanie 3.5 - podpunkt: Decision Tree vs Random Forest, vs bagging


def plot_classification_model_comparison(results):
    if isinstance(results, pd.DataFrame):
        comparison_df = results
    else:
        comparison_df = (
            pd
            .DataFrame(results)
            .T.reset_index()
            .rename(
                columns={
                    "index": "Model",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1-score",
                }
            )
        )

    ax = comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar",
        figsize=(10, 5),
    )
    plt.title("Porownanie modeli")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(title="Metryka", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


# Zadanie 4.0 - podpunkt: Stacking


def plot_stacking_model_comparison(metrics_df):
    ax = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar",
        figsize=(11, 5),
    )
    plt.title("Porownanie modeli poziomu 0, baggingu i stackingu")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(title="Metryka", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def plot_stacking_level_0_diagnostics(diagnostics_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(diagnostics_df["Model"], diagnostics_df["Meta coefficient"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Wspolczynniki meta-modelu")
    axes[0].set_ylabel("wartosc wspolczynnika")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        diagnostics_df["Model"],
        diagnostics_df["Corrected rate"],
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    axes[1].set_title("Jak czesto stacking naprawia bledy modelu poziomu 0")
    axes[1].set_ylabel("odsetek poprawionych bledow")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_stacking_vs_bagging(metrics_df):
    ax = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar",
        figsize=(9, 5),
        color=["#4c78a8", "#f58518", "#54a24b", "#e45756"],
    )
    plt.title("Stacking vs bagging")
    plt.ylabel("Wartosc metryki")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(title="Metryka", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()

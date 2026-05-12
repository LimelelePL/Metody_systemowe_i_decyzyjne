import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree


def plot_classification_metrics(metrics_df):
    plt.figure(figsize=(7, 4))
    plt.bar(
        metrics_df["metric"],
        metrics_df["value"],
        color=["steelblue", "coral", "seagreen", "mediumpurple"],
    )
    plt.ylim(0, 1)
    plt.title("Metryki modelu na zbiorze testowym")
    plt.ylabel("wartosc")
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=["No", "Yes"],
        cmap="Blues",
    )
    plt.title("Macierz pomylek dla drzewa decyzyjnego")
    plt.show()


def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Wizualizacja drzewa decyzyjnego")
    plt.show()


def plot_feature_importance(top_features_df):
    plt.figure(figsize=(14, 5))
    plt.barh(top_features_df["feature"], top_features_df["importance"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title("Najwazniejsze cechy w drzewie")
    plt.xlabel("feature importance")
    plt.show()


def plot_information_gain_comparison(comparison_plot_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        comparison_plot_df["feature"],
        comparison_plot_df["feature_importance"],
        color="steelblue",
    )
    axes[0].set_title("Feature importance")
    axes[0].set_ylabel("wartosc")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(
        comparison_plot_df["feature"],
        comparison_plot_df["manual_information_gain"],
        color="coral",
    )
    axes[1].set_title("Reczny Information Gain")
    axes[1].set_ylabel("wartosc")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def plot_regression_predictions(y_true, predictions_list):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, predictions) in zip(axes, predictions_list):
        ax.scatter(y_true, predictions, alpha=0.25)
        min_value = min(y_true.min(), predictions.min())
        max_value = max(y_true.max(), predictions.max())
        ax.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("wartosc rzeczywista")
        ax.set_ylabel("predykcja")

    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history_by_learning_rate):
    plt.figure(figsize=(10, 5))

    for learning_rate, history in loss_history_by_learning_rate.items():
        plt.plot(history, linewidth=2, label=f"learning rate = {learning_rate}")

    plt.title("Porownanie spadku MSE dla roznych learning rate")
    plt.xlabel("iteracja")
    plt.ylabel("MSE na treningu")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_top_weights(weight_df, abs_column_name, title, xlabel, top_n=10):
    plot_df = weight_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df[abs_column_name], color="teal")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("cecha")
    plt.show()


def plot_weight_comparison(plot_compare_df):
    plot_compare_df = plot_compare_df.head(8).copy().iloc[::-1]

    plt.figure(figsize=(11, 6))
    bar_width = 0.4
    positions = range(len(plot_compare_df))

    plt.barh(
        [position - bar_width / 2 for position in positions],
        plot_compare_df["abs_weight_after_scaling"],
        height=bar_width,
        color="darkorange",
        label="po skalowaniu",
    )
    plt.barh(
        [position + bar_width / 2 for position in positions],
        plot_compare_df["abs_weight_without_scaling"],
        height=bar_width,
        color="steelblue",
        label="bez skalowania",
    )
    plt.yticks(list(positions), plot_compare_df["feature"])
    plt.title("Porownanie wag dla tych samych cech")
    plt.xlabel("wartosc bezwzgledna wagi")
    plt.ylabel("cecha")
    plt.legend()
    plt.show()


def plot_numeric_weight_comparison(numeric_plot_df):
    numeric_plot_df = numeric_plot_df.sort_values("abs_weight_after_scaling")

    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    positions = range(len(numeric_plot_df))

    plt.bar(
        [position - bar_width / 2 for position in positions],
        numeric_plot_df["abs_weight_without_scaling"],
        width=bar_width,
        color="slateblue",
        label="bez skalowania",
    )
    plt.bar(
        [position + bar_width / 2 for position in positions],
        numeric_plot_df["abs_weight_after_scaling"],
        width=bar_width,
        color="seagreen",
        label="po skalowaniu",
    )
    plt.xticks(list(positions), numeric_plot_df["feature"])
    plt.title("Wplyw skalowania na interpretacje cech liczbowych")
    plt.xlabel("cecha liczbowa")
    plt.ylabel("wartosc bezwzgledna wagi")
    plt.legend()
    plt.show()


def plot_complexity_curves(complexity_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sweet_spot_index = complexity_df["test_mse"].idxmin()
    sweet_spot_degree = complexity_df.loc[sweet_spot_index, "degree"]
    sweet_spot_value = complexity_df.loc[sweet_spot_index, "test_mse"]

    axes[0].plot(complexity_df["degree"], complexity_df["train_mse"], marker="o", label="blad treningowy")
    axes[0].plot(complexity_df["degree"], complexity_df["test_mse"], marker="o", label="blad testowy")
    axes[0].scatter(sweet_spot_degree, sweet_spot_value, color="red", s=60, zorder=5, label="sweet spot")
    axes[0].set_title("Krzywe zlozonosci - pelna skala")
    axes[0].set_xlabel("stopien wielomianu")
    axes[0].set_ylabel("MSE")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    zoom_df = complexity_df[complexity_df["degree"] <= 12].copy()
    zoom_y_max = max(zoom_df["train_mse"].max(), zoom_df["test_mse"].max()) * 1.03

    axes[1].plot(zoom_df["degree"], zoom_df["train_mse"], marker="o", label="blad treningowy")
    axes[1].plot(zoom_df["degree"], zoom_df["test_mse"], marker="o", label="blad testowy")
    if sweet_spot_degree <= 12:
        axes[1].scatter(sweet_spot_degree, sweet_spot_value, color="red", s=60, zorder=5, label="sweet spot")
    axes[1].set_title("Krzywe zlozonosci - zoom dla malych stopni")
    axes[1].set_xlabel("stopien wielomianu")
    axes[1].set_ylabel("MSE")
    axes[1].set_ylim(2000000, zoom_y_max)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_tree_complexity_curves(complexity_df):
    sweet_spot_index = complexity_df["test_error"].idxmin()
    sweet_spot_depth = complexity_df.loc[sweet_spot_index, "max_depth"]
    sweet_spot_value = complexity_df.loc[sweet_spot_index, "test_error"]

    plt.figure(figsize=(10, 5))
    plt.plot(complexity_df["max_depth"], complexity_df["train_error"], marker="o", label="blad treningowy")
    plt.plot(complexity_df["max_depth"], complexity_df["test_error"], marker="o", label="blad testowy")
    plt.scatter(sweet_spot_depth, sweet_spot_value, color="red", s=60, zorder=5, label="sweet spot")
    plt.title("Krzywe zlozonosci dla drzewa decyzyjnego")
    plt.xlabel("max_depth")
    plt.ylabel("blad klasyfikacji")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_polynomial_fit(train_x, train_y, grid_x, prediction_dict, title, training_limit=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(train_x, train_y, s=12, alpha=0.25, color="black", label="dane treningowe")

    for label, values in prediction_dict.items():
        plt.plot(grid_x, values, linewidth=2, label=label)

    if training_limit is not None:
        plt.axvline(training_limit, color="gray", linestyle="--", label="koniec zakresu treningowego")

    plt.title(title)
    plt.xlabel("carat")
    plt.ylabel("price")
    plt.legend()
    plt.show()


def plot_polynomial_fit_comparison(train_x, train_y, grid_x, left_prediction_dict, right_prediction_dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_specs = [
        (axes[0], left_prediction_dict, "Porownanie stopnia 1 i 8"),
        (axes[1], right_prediction_dict, "Porownanie stopnia 1, 8 i 20"),
    ]

    for ax, prediction_dict, title in plot_specs:
        ax.scatter(train_x, train_y, s=12, alpha=0.25, color="black", label="dane treningowe")
        for label, values in prediction_dict.items():
            ax.plot(grid_x, values, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("carat")
        ax.set_ylabel("price")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_black_swan_comparison(train_x, train_y, grid_x, prediction_dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax in axes:
        ax.scatter(train_x, train_y, s=12, alpha=0.25, color="black", label="dane treningowe")
        for label, values in prediction_dict.items():
            ax.plot(grid_x, values, linewidth=2, label=label)
        ax.set_xlabel("carat")
        ax.set_ylabel("price")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " ")))

    axes[0].set_title("Czarny Labedz - pelna skala")

    non_extreme_prediction_max = min(values.max() for values in prediction_dict.values())
    zoom_y_max = max(train_y.max(), non_extreme_prediction_max) * 1.15
    axes[1].set_title("Czarny Labedz - zoom na sensowny zakres")
    axes[1].set_ylim(0, zoom_y_max)

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

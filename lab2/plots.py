import matplotlib.pyplot as plt
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


def plot_loss_history(loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="seagreen")
    plt.title("Spadek MSE w czasie dla gradient descent")
    plt.xlabel("epoka")
    plt.ylabel("MSE na treningu")
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
    plt.figure(figsize=(10, 5))
    plt.plot(complexity_df["degree"], complexity_df["train_mse"], marker="o", label="blad treningowy")
    plt.plot(complexity_df["degree"], complexity_df["test_mse"], marker="o", label="blad testowy")
    plt.title("Krzywe zlozonosci dla regresji wielomianowej")
    plt.xlabel("stopien wielomianu")
    plt.ylabel("MSE")
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

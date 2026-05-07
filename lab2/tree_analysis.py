import math

import numpy as np
import pandas as pd
from sklearn.tree import _tree, export_text


def extract_tree_rules(model, feature_names: list[str]) -> str:
    return export_text(model, feature_names=feature_names)


def entropy(y: pd.Series) -> float:
    probabilities = y.value_counts(normalize=True)
    return float(-sum(p * math.log2(p) for p in probabilities if p > 0))


def information_gain_for_split(
    X: pd.DataFrame,
    y: pd.Series,
    feature_name: str,
    threshold: float,
) -> float:
    parent_entropy = entropy(y)

    left_mask = X[feature_name] <= threshold
    right_mask = X[feature_name] > threshold

    y_left = y[left_mask]
    y_right = y[right_mask]

    if y_left.empty or y_right.empty:
        return 0.0

    left_weight = len(y_left) / len(y)
    right_weight = len(y_right) / len(y)

    child_entropy = (left_weight * entropy(y_left)) + (right_weight * entropy(y_right))
    return float(parent_entropy - child_entropy)


def get_tree_feature_thresholds(model, feature_names: list[str]) -> dict[str, list[float]]:
    feature_thresholds: dict[str, list[float]] = {}
    tree = model.tree_

    for node_index in range(tree.node_count):
        feature_index = tree.feature[node_index]
        if feature_index == _tree.TREE_UNDEFINED:
            continue

        feature_name = feature_names[feature_index]
        threshold = float(tree.threshold[node_index])
        feature_thresholds.setdefault(feature_name, []).append(threshold)

    return feature_thresholds


def get_positive_importance_features(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.asarray(model.feature_importances_, dtype=float),
    })
    importance_df = importance_df[importance_df["importance"] > 0].copy()
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_all_feature_importances(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": np.asarray(model.feature_importances_, dtype=float),
    })
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

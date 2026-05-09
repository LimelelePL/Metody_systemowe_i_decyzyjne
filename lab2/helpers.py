import numpy as np
import pandas as pd


def add_intercept(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def get_thresholds_from_tree(tree_model, feature_names):
    feature_thresholds = {}

    for feature_index, threshold in zip(tree_model.tree_.feature, tree_model.tree_.threshold):
        if feature_index != -2:
            feature_name = feature_names[feature_index]
            if feature_name not in feature_thresholds:
                feature_thresholds[feature_name] = []
            feature_thresholds[feature_name].append(float(threshold))

    return feature_thresholds


def build_weight_table(feature_names, weights, weight_column_name):
    weight_table = pd.DataFrame({
        "feature": feature_names,
        weight_column_name: weights,
    })
    weight_table[f"abs_{weight_column_name}"] = weight_table[weight_column_name].abs()
    weight_table = weight_table.sort_values(f"abs_{weight_column_name}", ascending=False)
    weight_table = weight_table.reset_index(drop=True)
    return weight_table

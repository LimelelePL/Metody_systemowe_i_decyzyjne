try:
    from lab3.constants import CATEGORICAL_COLUMNS, CHURN, CUSTOMER_ID, D_NUMERIC, NUMERIC_COLUMNS, PRICE
except ModuleNotFoundError:
    from constants import CATEGORICAL_COLUMNS, CHURN, CUSTOMER_ID, D_NUMERIC, NUMERIC_COLUMNS, PRICE

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _scale_selected_columns(X_train, X_test, columns):
    available_columns = [column for column in columns if column in X_train.columns]
    if not available_columns:
        return X_train, X_test

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[available_columns] = X_train[available_columns].astype(float)
    X_test[available_columns] = X_test[available_columns].astype(float)

    scaler = StandardScaler()
    X_train.loc[:, available_columns] = scaler.fit_transform(X_train[available_columns])
    X_test.loc[:, available_columns] = scaler.transform(X_test[available_columns])
    return X_train, X_test


def prepare_telco_data(train_df, test_df, scale_numeric=True):
    X_train = train_df.drop(columns=[CHURN, CUSTOMER_ID]).copy()
    y_train = train_df[CHURN].copy()
    X_test = test_df.drop(columns=[CHURN, CUSTOMER_ID]).copy()
    y_test = test_df[CHURN].copy()

    numeric_fill_values = X_train[NUMERIC_COLUMNS].median()
    for column in NUMERIC_COLUMNS:
        X_train[column] = X_train[column].fillna(numeric_fill_values[column])
        X_test[column] = X_test[column].fillna(numeric_fill_values[column])

    for column in CATEGORICAL_COLUMNS:
        train_mode = X_train[column].mode().iloc[0]
        X_train[column] = X_train[column].fillna(train_mode)
        X_test[column] = X_test[column].fillna(train_mode)

    if scale_numeric:
        X_train, X_test = _scale_selected_columns(X_train, X_test, NUMERIC_COLUMNS)

    return X_train, y_train, X_test, y_test


def prepare_diamonds_data(train_df, test_df, scale_numeric=True):
    X_train = train_df.drop(columns=[PRICE]).copy()
    y_train = train_df[PRICE].copy()
    X_test = test_df.drop(columns=[PRICE]).copy()
    y_test = test_df[PRICE].copy()

    if scale_numeric:
        diamond_numeric_columns = [column for column in D_NUMERIC if column != PRICE]
        X_train, X_test = _scale_selected_columns(X_train, X_test, diamond_numeric_columns)

    return X_train, y_train, X_test, y_test


def encode_train_test_features(X_train, X_test):
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test).reindex(columns=X_train_encoded.columns, fill_value=0)

    return X_train_encoded, X_test_encoded


def prepare_linear_regularization_data(
    X_train,
    y_train,
    X_test,
    y_test,
    feature="carat",
    degree=20,
    train_sample_size=6000,
    test_sample_size=6000,
    random_state=42,
):
    X_train_feature = X_train[[feature]].copy()
    X_test_feature = X_test[[feature]].copy()

    if train_sample_size is not None and len(X_train_feature) > train_sample_size:
        train_index = X_train_feature.sample(n=train_sample_size, random_state=random_state).index
        X_train_feature = X_train_feature.loc[train_index]
        y_train = y_train.loc[train_index]

    if test_sample_size is not None and len(X_test_feature) > test_sample_size:
        test_index = X_test_feature.sample(n=test_sample_size, random_state=random_state).index
        X_test_feature = X_test_feature.loc[test_index]
        y_test = y_test.loc[test_index]

    fill_value = X_train_feature[feature].median()
    X_train_feature[feature] = X_train_feature[feature].fillna(fill_value)
    X_test_feature[feature] = X_test_feature[feature].fillna(fill_value)

    polynomial = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = polynomial.fit_transform(X_train_feature)
    X_test_poly = polynomial.transform(X_test_feature)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    feature_names = polynomial.get_feature_names_out([feature])
    y_train_values = y_train.to_numpy(dtype=float)
    y_test_values = y_test.to_numpy(dtype=float)
    y_mean = y_train_values.mean()
    y_std = y_train_values.std()

    return {
        "X_train": pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train_feature.index),
        "y_train": pd.Series((y_train_values - y_mean) / y_std, index=X_train_feature.index),
        "X_test": pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test_feature.index),  # type: ignore
        "y_test": pd.Series((y_test_values - y_mean) / y_std, index=X_test_feature.index),
        "y_train_original": pd.Series(y_train_values, index=X_train_feature.index),
        "y_test_original": pd.Series(y_test_values, index=X_test_feature.index),
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_names": feature_names,
    }


def prepare_data(X_train: DataFrame, X_test: DataFrame, scale_numeric=True):
    X_train_prepared = X_train.copy()
    X_test_prepared = X_test.copy()

    if scale_numeric:
        numeric_columns = [column for column in NUMERIC_COLUMNS if column in X_train_prepared.columns]
        X_train_prepared, X_test_prepared = _scale_selected_columns(
            X_train_prepared,
            X_test_prepared,
            numeric_columns,
        )

    return encode_train_test_features(X_train_prepared, X_test_prepared)


def fit_feature_preprocessor(X: DataFrame):
    X_processed = X.copy()

    numeric_columns = X_processed.select_dtypes(include=["number"]).columns
    categorical_columns = X_processed.select_dtypes(exclude=["number"]).columns

    numeric_fill_values = X_processed[numeric_columns].median()
    categorical_fill_values = {column: X_processed[column].mode().iloc[0] for column in categorical_columns}

    if len(numeric_columns) > 0:
        X_processed.loc[:, numeric_columns] = X_processed[numeric_columns].fillna(numeric_fill_values)

    for column in categorical_columns:
        X_processed[column] = X_processed[column].fillna(categorical_fill_values[column])

    X_processed = pd.get_dummies(X_processed)

    artifacts = {
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "feature_columns": X_processed.columns,
    }

    return X_processed, artifacts


def transform_features(X: DataFrame, artifacts: dict):
    X_processed = X.copy()

    numeric_columns = X_processed.select_dtypes(include=["number"]).columns
    categorical_columns = X_processed.select_dtypes(exclude=["number"]).columns

    numeric_fill_values = artifacts["numeric_fill_values"]
    categorical_fill_values = artifacts["categorical_fill_values"]
    feature_columns = artifacts["feature_columns"]

    if len(numeric_columns) > 0:
        X_processed.loc[:, numeric_columns] = X_processed[numeric_columns].fillna(numeric_fill_values)

    for column in categorical_columns:
        if column in categorical_fill_values:
            X_processed[column] = X_processed[column].fillna(categorical_fill_values[column])

    X_processed = pd.get_dummies(X_processed)
    X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)

    return X_processed

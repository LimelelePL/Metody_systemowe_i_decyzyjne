import pandas as pd

try:
    from constants import CATEGORICAL_COLUMNS, CHURN, CUSTOMER_ID, NUMERIC_COLUMNS
except ModuleNotFoundError:
    from lab2.constants import CATEGORICAL_COLUMNS, CHURN, CUSTOMER_ID, NUMERIC_COLUMNS


def prepare_telco_classification_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

    return X_train, y_train, X_test, y_test

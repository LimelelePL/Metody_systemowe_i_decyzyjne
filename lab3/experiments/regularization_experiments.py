try:
    from lab3.algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from lab3.algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from lab3.algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from lab3.utils.preprocessing import encode_train_test_features, prepare_linear_regularization_data
except ModuleNotFoundError:
    from algorithms.regularization.linear_regularization_comparison import LinearRegularizationComparison
    from algorithms.regularization.regularization_sweet_spot import RegularizationSweetSpotComparison
    from algorithms.regularization.tree_regularization import TreeRegularizationComparison
    from utils.preprocessing import encode_train_test_features, prepare_linear_regularization_data


def run_linear_regularization_experiment(X_train, y_train, X_test, y_test, **kwargs):
    preprocessing_keys = {"feature", "degree", "train_sample_size", "test_sample_size", "random_state"}
    preprocessing_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in preprocessing_keys}
    linear_data = prepare_linear_regularization_data(X_train, y_train, X_test, y_test, **preprocessing_kwargs)
    return LinearRegularizationComparison(**kwargs).compare(**linear_data)


def run_tree_regularization_experiment(X_train, y_train, X_test, y_test, **kwargs):
    X_train_encoded, X_test_encoded = encode_train_test_features(X_train, X_test)
    return TreeRegularizationComparison(**kwargs).compare(X_train_encoded, y_train, X_test_encoded, y_test)


def run_regularization_sweet_spot_experiment(
    X_train_regression,
    y_train_regression,
    X_test_regression,
    y_test_regression,
    X_train_classification,
    y_train_classification,
    X_test_classification,
    y_test_classification,
):
    return RegularizationSweetSpotComparison().compare(
        X_train_regression,
        y_train_regression,
        X_test_regression,
        y_test_regression,
        X_train_classification,
        y_train_classification,
        X_test_classification,
        y_test_classification,
    )

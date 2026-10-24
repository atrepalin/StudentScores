import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator


def test_models(
    model_specs: dict[str, tuple[object, pd.DataFrame, pd.DataFrame]],
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    cv: BaseCrossValidator,
):
    """
    Test sklearn model predictions

    Args:
        model_specs (dict[str, tuple[object, pd.DataFrame, pd.DataFrame]]): model specs
        y_train (pd.DataFrame): training targets
        y_test (pd.DataFrame): test targets
        cv (BaseCrossValidator): cross validation strategy

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: test predictions, test errors, test results
    """
    test_predictions = pd.DataFrame(index=y_test.index)
    test_predictions["y_true"] = y_test
    error_rows = []
    test_rows = []

    for model_name, (model, X_fit, X_holdout) in model_specs.items():
        fold_test_predictions = []

        for fold, (fit_idx, _) in enumerate(cv.split(X_fit), start=1):
            fold_model = clone(model)
            fold_model.fit(X_fit.iloc[fit_idx], y_train.iloc[fit_idx])
            fold_test_predictions.append(fold_model.predict(X_holdout))

        test_pred = np.mean(np.vstack(fold_test_predictions), axis=0)
        test_predictions[model_name] = test_pred

        # Calculate MAE and RMSE
        residual = y_test.to_numpy() - test_pred
        abs_error = np.abs(residual)
        test_rows.append(
            {
                "Model": model_name,
                "Test RMSE": np.sqrt(np.mean(residual**2)),
                "Test MAE": np.mean(abs_error),
            }
        )

        error_rows.extend(
            {
                "Model": model_name,
                "Test row": row_idx,
                "Absolute error": err,
                "Residual (true - pred)": res,
            }
            for row_idx, err, res in zip(y_test.index, abs_error, residual)
        )

    test_errors = pd.DataFrame(error_rows)
    test_results = (
        pd.DataFrame(test_rows).sort_values("Test RMSE").reset_index(drop=True)
    )

    return test_predictions, test_errors, test_results


def test_models_manual(
    model_specs: dict,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    cv: BaseCrossValidator,
    should_clone: bool = True,
):
    """
    Test any model predictions
    Uses manual fit and predict functions

    Args:
        model_specs (dict[model_name: (model, X_fit, X_holdout, fit_fn, predict_fn)]): model specs
        y_train (pd.DataFrame): training targets
        y_test (pd.DataFrame): test targets
        cv (BaseCrossValidator): cross validation strategy
        should_clone (bool): whether to clone models

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: test predictions, test errors, test results
    """
    test_predictions = pd.DataFrame(index=y_test.index)
    test_predictions["y_true"] = y_test.squeeze()

    error_rows = []
    test_rows = []

    for model_name, spec in model_specs.items():
        model, X_fit, X_holdout, fit_fn, predict_fn = spec

        fold_test_predictions = []

        for fold, (fit_idx, _) in enumerate(cv.split(X_fit), start=1):
            fold_model = clone(model) if should_clone else model

            X_fold = X_fit.iloc[fit_idx]
            y_fold = y_train.iloc[fit_idx]

            fitted_model = fit_fn(
                fold_model,
                X_fold,
                y_fold,
                X_holdout,
                y_test,
            )

            if fitted_model is not None:
                fold_model = fitted_model

            predictions = predict_fn(
                fold_model,
                X_holdout,
            )

            fold_test_predictions.append(np.asarray(predictions).reshape(-1))

        test_pred = np.mean(
            np.vstack(fold_test_predictions),
            axis=0,
        )

        test_predictions[model_name] = test_pred

        residual = y_test.to_numpy().reshape(-1) - test_pred
        abs_error = np.abs(residual)

        test_rows.append(
            {
                "Model": model_name,
                "Test RMSE": np.sqrt(np.mean(residual**2)),
                "Test MAE": np.mean(abs_error),
            }
        )

        error_rows.extend(
            {
                "Model": model_name,
                "Test row": row_idx,
                "Absolute error": err,
                "Residual (true - pred)": res,
            }
            for row_idx, err, res in zip(y_test.index, abs_error, residual)
        )

    test_errors = pd.DataFrame(error_rows)

    test_results = (
        pd.DataFrame(test_rows).sort_values("Test RMSE").reset_index(drop=True)
    )

    return test_predictions, test_errors, test_results

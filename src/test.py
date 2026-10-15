import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def test_models(
    model_specs: dict[str, tuple[object, pd.DataFrame, pd.DataFrame]],
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    kf: KFold,
):
    test_predictions = pd.DataFrame(index=y_test.index)
    test_predictions["y_true"] = y_test
    error_rows = []
    test_rows = []

    for model_name, (model, X_fit, X_holdout) in model_specs.items():
        fold_test_predictions = []

        for fold, (fit_idx, _) in enumerate(kf.split(X_fit), start=1):
            fold_model = clone(model)
            fold_model.fit(X_fit.iloc[fit_idx], y_train.iloc[fit_idx])
            fold_test_predictions.append(fold_model.predict(X_holdout))

        test_pred = np.mean(np.vstack(fold_test_predictions), axis=0)
        test_predictions[model_name] = test_pred

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

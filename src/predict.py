from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def ensemble_predict(
    model_specs: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    kf: BaseCrossValidator,
    numerical_features: list = [],
    checkpoint_dir: str | Path | None = None,
    recompute: bool | str | Iterable[str] = False,
    ignore: str | Iterable[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_size = len(X_train)
    test_size = len(X_test)

    if isinstance(recompute, bool):
        recompute_models = set(model_specs) if recompute else set()
    elif isinstance(recompute, str):
        recompute_models = set([recompute])
    else:
        recompute_models = set(recompute)

    if isinstance(ignore, str):
        ignored_models = set([ignore])
    else:
        ignored_models = set(ignore)

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    oof_preds = []
    test_preds = []

    def checkpoint_paths(model_name: str):
        safe_name = quote(str(model_name), safe="")

        return (
            checkpoint_dir / f"{safe_name}__oof.parquet",
            checkpoint_dir / f"{safe_name}__test.parquet",
        )

    for model_name, spec in model_specs.items():
        if model_name in ignored_models:
            continue

        oof_path = test_path = None

        if checkpoint_dir is not None:
            oof_path, test_path = checkpoint_paths(model_name)

            checkpoint_exists = oof_path.exists() and test_path.exists()
            should_recompute = model_name in recompute_models

            if checkpoint_exists and not should_recompute:
                oof_pred = pd.read_parquet(oof_path)
                test_pred = pd.read_parquet(test_path)

                if (
                    len(oof_pred) == train_size
                    and len(test_pred) == test_size
                    and model_name in oof_pred.columns
                    and model_name in test_pred.columns
                ):
                    print(f"[checkpoint] {model_name}: loaded")

                    oof_preds.append(oof_pred[[model_name]])
                    test_preds.append(test_pred[[model_name]])
                    continue

                print(
                    f"[checkpoint] {model_name}: "
                    "incompatible checkpoint, recomputing"
                )

        model, fit_fn, predict_fn, scale, should_clone = spec

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy().drop("id", axis=1, errors="ignore")

        if scale:
            scaler: StandardScaler = StandardScaler().set_output(transform="pandas")

            X_train_processed[numerical_features] = scaler.fit_transform(
                X_train[numerical_features]
            )

            X_test_processed[numerical_features] = scaler.transform(
                X_test[numerical_features]
            )

        oof_pred = pd.DataFrame(
            index=range(train_size),
            columns=[model_name],
        )

        fold_test_predictions = []

        for fit_idx, valid_idx in tqdm(
            kf.split(X_train),
            desc=model_name,
            leave=False,
            total=kf.n_splits,
        ):
            fold_model = clone(model) if should_clone else model

            fitted_model = fit_fn(
                fold_model,
                X_train_processed.iloc[fit_idx],
                y_train.iloc[fit_idx],
                X_train_processed.iloc[valid_idx],
                y_train.iloc[valid_idx],
            )

            validation_predictions = predict_fn(
                fitted_model,
                X_train_processed.iloc[valid_idx],
            )

            oof_pred.loc[valid_idx, model_name] = np.asarray(
                validation_predictions
            ).reshape(-1)

            test_predictions = predict_fn(
                fitted_model,
                X_test_processed,
            )

            fold_test_predictions.append(np.asarray(test_predictions).reshape(-1))

        test_pred = pd.DataFrame(
            {
                model_name: np.mean(
                    np.vstack(fold_test_predictions),
                    axis=0,
                )
            }
        )

        oof_pred = oof_pred.astype(float)

        if checkpoint_dir is not None:
            oof_pred.to_parquet(oof_path, index=False)
            test_pred.to_parquet(test_path, index=False)

            print(f"[checkpoint] {model_name}: saved")

        oof_preds.append(oof_pred)
        test_preds.append(test_pred)

    train = X_train.copy()
    train["target"] = y_train

    return (train, pd.concat(oof_preds, axis=1), pd.concat(test_preds, axis=1))

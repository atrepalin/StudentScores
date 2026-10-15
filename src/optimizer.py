import gc
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.utils import _safe_indexing


def optimize_model(
    model_class,
    param_space: Callable[[optuna.Trial], dict],
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: BaseCrossValidator,
    n_trials=50,
):
    def objective(trial: optuna.Trial):
        params = param_space(trial)

        model = model_class(**params)

        scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        gc.collect()

        trial.set_user_attr("scores", scores.tolist())

        return np.mean(scores)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study


def optimize_model_manual(
    model_class,
    param_space: Callable[[optuna.Trial], dict],
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: BaseCrossValidator,
    fit: Callable,
    predict: Callable,
    n_trials=50,
):
    def objective(trial: optuna.Trial):
        params = param_space(trial)
        scores = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            X_train = _safe_indexing(X, train_idx)
            y_train = _safe_indexing(y, train_idx)

            X_valid = _safe_indexing(X, valid_idx)
            y_valid = _safe_indexing(y, valid_idx)

            model = model_class(**params)

            fitted_model = fit(
                model,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )

            if fitted_model is not None:
                model = fitted_model

            predictions = predict(model, X_valid)

            rmse = np.sqrt(mean_squared_error(y_valid, predictions))

            scores.append(float(rmse))

            trial.report(float(np.mean(scores)), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

            del model, X_train, y_train, X_valid, y_valid
            gc.collect()

        trial.set_user_attr("scores", scores)

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study

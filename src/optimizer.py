import gc
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_model(
    model_class,
    param_space: Callable[[optuna.Trial], dict],
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: BaseCrossValidator,
    n_trials=50,
):
    """
    Optimize sklearn model hyperparameters using optuna

    Args:
        model_class: sklearn model class
        param_space (Callable): hyperparameter space factory
        X (pd.DataFrame): input features
        y (pd.DataFrame): target
        cv (BaseCrossValidator): cross validation strategy
        n_trials (int): number of trials

    Returns:
        optuna study
    """

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

        # Force garbage collection to avoid memory leaks
        gc.collect()

        trial.set_user_attr("scores", scores.tolist())

        return np.mean(scores)

    # Use sampler for reproducibility
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
    fit_fn: Callable,
    predict_fn: Callable,
    n_trials=50,
):
    """
    Optimize any model hyperparameters using optuna
    Uses manual fit and predict functions

    Args:
        model_class: sklearn model class
        param_space (Callable): hyperparameter space factory
        X (pd.DataFrame): input features
        y (pd.DataFrame): target
        cv (BaseCrossValidator): cross validation strategy
        fit_fn (Callable): fit function
        predict_fn (Callable): predict function
        n_trials (int): number of trials

    Returns:
        optuna study
    """

    def objective(trial: optuna.Trial):
        params = param_space(trial)
        scores = []

        # Manually split data using cross validation strategy
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            model = model_class(**params)

            fitted_model = fit_fn(
                model,
                X.iloc[train_idx],
                y.iloc[train_idx],
                X.iloc[valid_idx],
                y.iloc[valid_idx],
            )

            if fitted_model is not None:
                model = fitted_model

            predictions = predict_fn(model, X.iloc[valid_idx])

            rmse = np.sqrt(mean_squared_error(y.iloc[valid_idx], predictions))

            scores.append(float(rmse))

            trial.report(float(np.mean(scores)), step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Force garbage collection to avoid memory leaks
            del model
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

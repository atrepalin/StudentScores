import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
import gc

def optimize_model(model_class, param_space, X, y, cv, n_trials=50):
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

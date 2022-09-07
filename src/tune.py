"""
Tune hyperparameters using optuna
"""
import optuna


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x * x + x - 1


# TODO: pruning? Maybe if gradients explode or collapse?
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_params)

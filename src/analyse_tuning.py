import pickle
import sys
from pprint import pprint

import optuna


def main(study: optuna.Study) -> None:
    param_cardinality = (
        study.trials_dataframe()
        .filter(regex="^params_")
        .rename(lambda name: name.split("params_", 1)[1], axis=1)
        .nunique()
    )
    non_constant_params = param_cardinality.loc[
        param_cardinality > 1
    ].index.to_list()

    optuna.visualization.plot_contour(study, non_constant_params).show()
    optuna.visualization.plot_edf(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()

    print("Best params:")
    pprint(study.best_params)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please provide a path to a tuning results file"

    with open(sys.argv[1], "rb") as f:
        study = pickle.load(f)

    main(study)

import json
from pathlib import Path

import warnings
from optuna.exceptions import ExperimentalWarning

from Pipelines.KNN import build_KNN_optuna

from src.utils.data import read_train_data


warnings.filterwarnings("ignore", category=ExperimentalWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTEFACTS_DIR = BASE_DIR / 'artefacts'
PARAMS_PATH = ARTEFACTS_DIR / 'best_params.json'


def tuner(grid, X_tr, y_tr):
    grid = grid.fit(X_tr, y_tr)
    return grid.best_params_

def save_params(model_name: str, params: dict) -> None:
    if PARAMS_PATH.exists():
        with PARAMS_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
    else:
        data = {}
    data[model_name] = params
    with PARAMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main(X, y) -> None:
    grid_tuple = (build_KNN_optuna(),)
    grid_names = ('KNN',)
    for grid, name in zip(grid_tuple, grid_names):
        print(f"\n{name}: {type(grid)}")
        if hasattr(grid, "estimator"):
            print("  estimator obj:", grid.estimator)
            print("  estimator type:", type(grid.estimator))
        params = tuner(grid, X, y)
        save_params(model_name=name, params=params)

if __name__ == '__main__':
    X_train, y_train = read_train_data()
    main(X_train, y_train)
from src.utils.data import read_train_data
from src.utils.finalists_cv import compute_cv
from src.Pipelines.KNN import build_KNN_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTS_DIR = BASE_DIR / "artefacts"

def load_best_params(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing params file: {path}")
    with path.open('r', encoding='utf-8') as file:
        best_params = json.load(file)
    return best_params

def prepare_model(model, params: dict):
    model.set_params(**params)
    return model

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def save_model(model, model_name) -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f'{model_name}.joblib'
    joblib.dump(model, filename=path)
    print('Saved: ', path)

def main() -> None:
    X_train, y_train = read_train_data()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    params = load_best_params(ARTEFACTS_DIR / 'best_params.json')
    models = {'KNN': build_KNN_pipeline(),
              'RF': RandomForestClassifier(random_state=333093),
              'ET': ExtraTreesClassifier(random_state=333093),
              'GB': HistGradientBoostingClassifier(random_state=333093)}

    CV_results = {}
    for name, model in models.items():
        if name not in params:
            raise KeyError(f"Missing hyperparameters for model '{name}' in best_params.json")
        model_prepared = prepare_model(model, params=params[name])
        mean, std = compute_cv(model_prepared, X_train=X_train, y_train=y_train)

        CV_results[name] = {'cv_mean': mean, 'cv_std': std}

        model_trained = train_model(X_train=X_train,
                                    y_train=y_train,
                                    model=model_prepared)
        save_model(model=model_trained,
                   model_name=name)
    with open(ARTEFACTS_DIR / 'CV_results.json', 'w', encoding='utf-8') as file:
        json.dump(CV_results, file)
    print('Done')

if __name__ == '__main__':
    main()
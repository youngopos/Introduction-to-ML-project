from src.utils.data import read_train_data
from src.utils.finalists_cv import compute_cv
from Pipelines.KNN import build_KNN_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARTEFACTS_DIR = BASE_DIR / "artefacts"

def load_best_params():
    with open('../artefacts/best_params.json', 'r', encoding='utf-8') as file:
        best_params = json.load(file)
    return best_params

def prepare_model(model_function, params: dict):
    model_function.set_params(**params)
    return model_function

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

def save_model(model, model_name) -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTEFACTS_DIR / f'{model_name}.joblib'
    joblib.dump(model, filename=path)
    print('Saved: ', path)

def main():
    X_train, y_train = read_train_data()
    params = load_best_params()
    models = (build_KNN_pipeline(),
              RandomForestClassifier(random_state=333093),
              ExtraTreesClassifier(random_state=333093),
              HistGradientBoostingClassifier(random_state=333093))
    models_names = tuple(
        params.keys()
    )
    CV_results = []
    for name, model in zip(models_names, models):

        model_prepared = prepare_model(model, params=params[name])
        model_cv_mean, model_cv_std = compute_cv(model_prepared, X_train=X_train, y_train=y_train)

        CV_results.append({
            name: [model_cv_mean, model_cv_std]
        })

        model_trained = train_model(X_train=X_train,
                                    y_train=y_train,
                                    model=model_prepared)
        save_model(model=model_trained,
                   model_name=name)

    with open(ARTEFACTS_DIR / 'CV_results.json', 'w', encoding='utf-8') as file:
        json.dump(CV_results, file)

if __name__ == '__main__':
    main()
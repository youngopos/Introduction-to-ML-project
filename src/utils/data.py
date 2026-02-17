import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

p = Path(__file__).resolve()
BASE_DIR = next(parent for parent in p.parents if (parent / "src").exists())
DATA_DIR = BASE_DIR / "data"
ARTEFACTS_DIR = BASE_DIR / 'artefacts'
PARAMS_PATH = ARTEFACTS_DIR / 'best_params.json'

def load_top_features():
    with open(ARTEFACTS_DIR / 'top_features.json', 'r', encoding='utf-8') as file:
        top_features = json.load(file)
    return top_features

def load_data():
    X = pd.read_csv(DATA_DIR / "artifical_train_data.csv")
    y = pd.read_csv(DATA_DIR / "artifical_train_labels.csv").squeeze('columns')
    X = X[load_top_features()]
    return X,y

def split_data(X, y, random_state = 333093, test_size = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=random_state,
                                                        test_size=test_size,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def save_test_data(X_test, y_test) -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(ARTEFACTS_DIR / 'test_set_X.csv', index=True)
    y_test.to_csv(ARTEFACTS_DIR / 'test_set_y.csv', index=True)

def save_train_data(X_train, y_train) -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(ARTEFACTS_DIR / 'train_set_X.csv', index=True)
    y_train.to_csv(ARTEFACTS_DIR / 'train_set_y.csv', index=True)

def read_test_data() -> tuple:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_test = pd.read_csv(ARTEFACTS_DIR / 'test_set_X.csv', index_col=0)
    y_test = pd.read_csv(ARTEFACTS_DIR / 'test_set_y.csv',
                         index_col=0).squeeze("columns")
    return X_test[load_top_features()], y_test

def read_train_data() -> tuple:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train = pd.read_csv(ARTEFACTS_DIR / 'train_set_X.csv', index_col=0)
    y_train = pd.read_csv(ARTEFACTS_DIR / 'train_set_y.csv',
                         index_col=0).squeeze("columns")
    return X_train[load_top_features()], y_train

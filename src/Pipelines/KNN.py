from sklearn.neighbors import  KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.integration import OptunaSearchCV

param_distributions = {'KNN__n_neighbors': optuna.distributions.CategoricalDistribution([i for i in range(1,31)])}

def build_KNN_pipeline() -> Pipeline:
    return Pipeline([
        ('sca', StandardScaler()),
        ('KNN', KNeighborsClassifier(n_jobs=1))
    ])

def build_KNN_optuna():
    return OptunaSearchCV(
        estimator=build_KNN_pipeline(),
        param_distributions=param_distributions,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc',
        n_trials=60,
        verbose=1,
        random_state=333093
    )
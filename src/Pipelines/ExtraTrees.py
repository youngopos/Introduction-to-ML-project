from sklearn.ensemble import ExtraTreesClassifier
import optuna
from optuna.integration import OptunaSearchCV

param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution(200,2000,step=200),
    'max_depth': optuna.distributions.CategoricalDistribution([None, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25]),
    'min_samples_leaf': optuna.distributions.CategoricalDistribution([1,2,5,10, 20, 50]),
    'criterion': optuna.distributions.CategoricalDistribution(['gini', 'entropy', 'log_loss']),
    'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2', 0.2, 0.3 ,0.5, 0.7]),
    'min_samples_split': optuna.distributions.CategoricalDistribution([2, 5, 10, 20])
}

def build_ET_optuna():
    return OptunaSearchCV(
        estimator = ExtraTreesClassifier(n_jobs=1, random_state=333093),
        param_distributions = param_distributions,
        cv = 5,
        n_jobs = -1,
        scoring = 'roc_auc',
        n_trials = 400,
        verbose = 1,
        random_state = 333093
    )
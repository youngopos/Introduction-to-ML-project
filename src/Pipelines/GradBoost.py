from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from optuna.integration import OptunaSearchCV

param_distributions = {
    'learning_rate': optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
    'max_iter': optuna.distributions.IntDistribution(200,2_000, step=200),
    'max_depth': optuna.distributions.CategoricalDistribution([None, 2, 3, 4, 5, 6, 7, 8, 12, 16, 25]),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 200, log=True),
    'max_bins': optuna.distributions.CategoricalDistribution([64, 128, 255]),
    'l2_regularization': optuna.distributions.FloatDistribution(1e-8, 1.0, log=True)
}

def build_GB_optuna():
    return OptunaSearchCV(
        estimator = HistGradientBoostingClassifier(random_state=333093),
        n_jobs = -1,
        param_distributions = param_distributions,
        cv = 5,
        n_trials=400,
        verbose=1,
        scoring = 'roc_auc',
        random_state = 333093
    )
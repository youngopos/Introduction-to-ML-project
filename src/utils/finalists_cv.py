from sklearn.model_selection import cross_val_score

def compute_cv(model, X_train, y_train):
    cv_score = cross_val_score(estimator=model, X=X_train, y=y_train,
                               cv=10,
                               scoring='roc_auc')
    return [cv_score.mean(), cv_score.std()]
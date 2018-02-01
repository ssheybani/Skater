from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics


def validation_curve(estimator, n_folds, x=None, y=None, param_name=None, param_range=None):
    c_v = StratifiedKFold(n_splits=n_folds)
    for v in param_range:
        for train_idx, test_idx in c_v.split(x, y):
            param_dict = {param_name: v}
            estimator = estimator.set_params(param_dict)
            model_training = estimator.fit(x[train_idx], y[train_idx])
            train_scores = model_training.predict_prob(x[train_idx])
            train_metric = metrics.roc_auc_score(y[train_idx] ,train_scores[1], pos_label=1)

            test_scores = model_training.predict_prob(x[test_idx])
            test_metric = metrics.roc_auc_score(y[test_idx] ,test_scores[1], pos_label=1)
    return   train_metric, test_metric

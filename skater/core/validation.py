from sklearn.cross_validation import StratifiedKFold
from skater.core.global_interpretation.rule_list import SBRL

def validation_curve(estimator, n_folds, X=None, y=None, param_name=None, param_range=None):
    c_v = StratifiedKFold(n_splits=n_folds)
    for v in param_range:
        for train, test in c_v.split(X, y):
            param_dict = {param_name: v}
            estimator = estimator.set_params(param_dict)
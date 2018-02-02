# coding=utf-8
# Reference:
# Referencing the work done by Professor Cynthia/Hongyuy/Others with modifications

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import numpy as np
import pandas as pd
import rpy2.robjects as ro
pandas2ri.activate()


class SBRL(object):
    _estimator_type = "classifier"

    def __init__(self, iterations=30000, pos_sign=1, neg_sign=0, min_rule_len=1,
                 max_rule_len=8, min_support_pos=0.10, min_support_neg=0.10,
                 eta=1.0, n_chains=50, alpha=1, lambda_=8):
        """
        SBRL is a scalable generative estimator to build interpretable decision lists

        Parameters
        ----------
        iterations: the number of iterations for each MCMC chain (default 30000)
        pos_sign: sign for the positive labels in the "label" column.(default "1")
        neg_sign: sign for the negative labels in the "label" column.(default "0")
        min_rule_len: the minimum number of cardinality for rules to be mined from the data-frame(default 1)
        max_rule_len: the maximum number of cardinality for rules to be mined from the data-frame(default 1)
        min_support_pos: a number between 0 and 1, for the minimum percentage support for the positive
        observations.(default 0.1)
        min_support_neg: a number between 0 and 1, for the minimum percentage support for the negative
        observations.(default 0.1)
        eta:  default 1
        n_chains: default 10
        alpha: a prior pseudo-count for the positive and negative classes. fixed at 1’s
        lambda_: a hyper-parameter for the expected length of the rule list(default 10)

        References
        ----------
        .. [1] Letham et.al(2015) Interpretable classifiers using rules and Bayesian analysis:
               Building a better stroke prediction model (https://arxiv.org/abs/1511.01644)
        .. [2] Yang et.al(2016) Scalable Bayesian Rule Lists (https://arxiv.org/abs/1602.08610)
        .. [3] https://github.com/Hongyuy/sbrl-python-wrapper/blob/master/sbrl/C_sbrl.py

        """
        self.r_sbrl = importr('sbrl')
        self.model = None
        self.as_factor = ro.r['as.factor']
        self.s_apply = ro.r['lapply']
        self.r_frame = ro.r['data.frame']
        self.model_params = {
            "iters":iterations, "pos_sign":pos_sign, "neg_sign":neg_sign, "rule_minlen":min_rule_len,
            "rule_maxlen":max_rule_len, "minsupport_pos":min_support_pos, "minsupport_neg":min_support_neg,
            "eta":eta, "nchain":n_chains, "lambda":lambda_, "alpha":alpha
        }


    def set_params(self, params):
        self.model_params[list(params.keys())[0]] = list(params.values())[0]


    def fit(self, X, y_true):
        """
        Parameters:
            X: pandas.DataFrame object that could be used by the model for training.
                 It must not have a column named 'label'
            y_true: pandas.Series, 1-D array to store ground truth labels
        """
        data = X.assign(label=y_true)
        data_as_r_frame = self.r_frame(self.s_apply(data, self.as_factor))
        self.model = self.r_sbrl.sbrl(data_as_r_frame, **self.model_params)
        return self.model


    def predict_prob(self, X):
        """
        Parameters:
            X:  pandas.DataFrame object, representing the data to be making predictions on.
            `type`  whether the prediction is discrete or probabilistic.
            return a numpy.ndarray of shape (#datapoints, 2), the probability for each observations
        """
        data_as_r_frame = self.r_frame(self.s_apply(X, self.as_factor))
        results = self.r_sbrl.predict_sbrl(self.model, data_as_r_frame)
        return pandas2ri.ri2py_dataframe(results).T


    def predict(self, X=None, prob_score=None, threshold=0.5, pos_label=1):
        """
        Binary Classification
        Adjust threshold to balance between sensitivity and specificity
        """
        probability_df = self.predict_prob(X) if X is not None and prob_score is None else prob_score
        y_prob = probability_df.loc[:, pos_label]
        y_prob.loc[np.where(y_prob.values > threshold)] = 1
        y_prob.loc[np.where(y_prob.values < threshold)] = 0
        return y_prob


    def print_model(self):
        self.r_sbrl.print_sbrl(self.model)


    def access_learned_rules(self, rule_indexes):
        if not isinstance(rule_indexes, str):
            raise TypeError('Expected type string {} provided'.format(type(rule_indexes)))

        # Convert model properties into a readable python dict
        result_dict = dict(zip(self.model.names, map(list,list(self.model))))
        # Enable the ability to access single or multiple sequential model learned decisions
        indexes = [int(v) for v in rule_indexes.split(':')]

        rules_result = lambda rules: result_dict['rulenames'][indexes[0]:indexes[1]] if rule_indexes.find(':') > -1\
            else result_dict['rulenames'][indexes[0]]
        return rules_result(self.model)





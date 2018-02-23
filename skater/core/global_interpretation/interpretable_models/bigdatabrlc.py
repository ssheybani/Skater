from skater.core.global_interpretation.interpretable_models.brlc \
    import BRLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from ..util import exceptions

import pandas as pd
import numpy as np


class BigDataBRLC(BRLC):

    def __init__(self, sub_sample_percentage=0.1, iterations=30000, pos_sign=1, neg_sign=0, min_rule_len=1,
                 max_rule_len=8, min_support_pos=0.10, min_support_neg=0.10, eta=1.0, n_chains=50, alpha=1,
                 lambda_=8, discretize=True, threshold=0.5, surrogate_estimator=None):
        self.sample_percentage = sub_sample_percentage
        self.threshold = threshold
        self.surrogate_estimator = RandomForestClassifier(warm_start=True, oob_score=True,
                                                          max_features="sqrt",
                                                          random_state=0) \
            if surrogate_estimator is None else surrogate_estimator

        super(BRLC, self).__init__(iterations, pos_sign, neg_sign, min_rule_len,
                                   max_rule_len, min_support_pos, min_support_neg, eta, n_chains, alpha,
                                   lambda_, discretize)


    def subsample(self, X, y, pos_label=1, neg_label=0):
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise exceptions.DataSetError("Only pandas.DataFrame as input type is currently supported")

        # validate the consistency of the input data
        if not X.shape[0] == y.shape[0]:
            raise exceptions.DataSetError("mismatch in the shape of X and y")

        try:
            self.surrogate_estimator.predict_proba(X[0:1])
        except NotFittedError:
            self.surrogate_estimator.fit(X, y)

        est_prob_scores = pd.DataFrame(self.surrogate_estimator.predict_proba(X))

        # compute the distance from the decision boundary
        distance_from_threshold = est_prob_scores[pos_label].apply(lambda x: np.abs(self.threshold - x))
        pos_label_index = np.where(y == pos_label)[0]
        neg_label_index = np.where(y == neg_label)[0]

        pos_label_dist = distance_from_threshold[pos_label_index]
        neg_label_dist = distance_from_threshold[neg_label_index]

        # Sort the neighboring distances from the threshold in the ascending order to select points which
        # are closer to the decision boundary
        sorted_dist_pos_label = pos_label_dist.sort_values()
        sorted_dist_neg_label = neg_label_dist.sort_values()

        # sub-sample the data
        number_of_rows = len(y) * self.sample_percentage
        pos_fraction = len(pos_label_index) / float(len(y))
        neg_fraction = 1 - pos_fraction

        pos_df = pd.DataFrame(X.iloc[sorted_dist_pos_label[:int(number_of_rows * pos_fraction) + 1].index])
        neg_df = pd.DataFrame(X.iloc[sorted_dist_neg_label[:int(number_of_rows * neg_fraction) + 1].index])

        new_X = pd.concat([pos_df, neg_df], axis=0)
        return new_X, y[new_X.index]

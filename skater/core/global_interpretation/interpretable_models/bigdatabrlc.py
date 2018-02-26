from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

from skater.core.global_interpretation.interpretable_models.brlc import BRLC
from skater.util import exceptions


class BigDataBRLC(BRLC):

    def __init__(self, sub_sample_percentage=0.1, iterations=30000, pos_sign=1, neg_sign=0, min_rule_len=1,
                 max_rule_len=8, min_support_pos=0.10, min_support_neg=0.10, eta=1.0, n_chains=50, alpha=1,
                 lambda_=8, discretize=True, drop_features=False, tolerance=0.5, penalty_param_svm=0.01,
                 calibration_type='sigmoid', cv_calibration=3, random_state=0, surrogate_estimator='SVM'):

        """ BigDataBRLC is a BRLC to handle large data-sets. Advisable to be used when the number of
        input examples>1k. It approximates large datasets with the help of surrogate(metamodel) estimators. For example, it uses
         surrogate estimator such as SVC(Support Vector Classifier) or RandomForest by default to filter the data
         points which are closest to the decision boundary. The idea is to identify the minimum training set size
         (controlled by the parameter sub_sample_percentage) with the goal to maximize accuracy.
         This helps in reducing the computation time while building the final SBRL.

        :param sub_sample_percentage:
        :param iterations:
        :param pos_sign:
        :param neg_sign:
        :param min_rule_len:
        :param max_rule_len:
        :param min_support_pos:
        :param min_support_neg:
        :param eta:
        :param n_chains:
        :param alpha:
        :param lambda_:
        :param discretize:
        :param drop_features:
        :param tolerance:
        :param penalty_param_svm:
        :param calibration_type:
            Reference: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
        :param cv_calibration:
        :param random_state:
        :param surrogate_estimator:
        :return:

        References:
        ------------
        .. [1] Tamas Madl, https://github.com/tmadl/sklearn-expertsys/blob/master/BigDataRuleListClassifier.py
        .. [2] https://pdfs.semanticscholar.org/e44c/9dcf90d5a9a7e74a1d74c9900ff69142c67f.pdf
        .. [3] Surrogate model: https://en.wikipedia.org/wiki/Surrogate_model
        .. [4] W. Andrew Pruett , Robert L. Hester(2016),
        The Creation of Surrogate Models for Fast Estimation of Complex Model Outcomes
        (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0156574)
        """

        self.sample_percentage = sub_sample_percentage
        self.tol = tolerance

        self.surrogate_estimator = RandomForestClassifier(warm_start=True, oob_score=True,
                                                          max_features="sqrt",
                                                          random_state=random_state) \
            if surrogate_estimator is 'RF' else surrogate_estimator

        self.surrogate_estimator = CalibratedClassifierCV(LinearSVC(C=penalty_param_svm, random_state=random_state),
                                                          method=calibration_type, cv=cv_calibration) \
            if surrogate_estimator is 'SVM' else surrogate_estimator

        super(BigDataBRLC, self).__init__(iterations, pos_sign, neg_sign, min_rule_len,
                                          max_rule_len, min_support_pos, min_support_neg, eta, n_chains, alpha,
                                          lambda_, discretize, drop_features)


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
        distance_from_threshold = est_prob_scores[pos_label].apply(lambda x: np.abs(self.tol - x))
        pos_label_index = np.where(y == pos_label)[0]
        neg_label_index = np.where(y == neg_label)[0]

        pos_label_dist = distance_from_threshold[pos_label_index]
        neg_label_dist = distance_from_threshold[neg_label_index]

        # sort the neighboring distances from the threshold in the ascending order to select points which
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

        # Randomly shuffle the newly formed data-set
        X_, y_ = shuffle(new_X, y[new_X.index])
        return X_, y_

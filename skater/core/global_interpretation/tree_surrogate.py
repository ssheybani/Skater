from .base import BaseGlobalInterpretation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


class TreeSurrogate(BaseGlobalInterpretation):
    # Reference: http://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf
    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False,
                 feature_name=None, impurity_threshold=0.01):
        self.model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                            random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                                            class_weight=class_weight, presort=presort)
        self.feature_names = feature_name
        self.impurity_threshold = impurity_threshold


    def learn(self, X, Y, original_y, cv=True, n_iter_search=10, param_grid=None):
        if cv is False:
            self.model.fit(X, Y)
        else:
            # apply randomized cross validation
            default_grid = {
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
            search_space = param_grid if param_grid is not None else default_grid
            random_search_estimator = RandomizedSearchCV(estimator=self.model, param_distributions=search_space,
                                                         n_iter_search=n_iter_search)
            self.model = random_search_estimator.best_estimator_
        y_hat_surrogate = self.predict(X)
        # TODO This should be abstracted by the model scorer factory
        base_precision, base_recall, base_fscore, base_support = score(original_y, Y)
        surrogate_precision, surrogate_recall, surrogate_fscore, surrogate_support = score(Y, y_hat_surrogate)
        # Check on the length of any of the metric to determine the number of classes
        # if all is selected then compare against all metrics
        avg_score = np.sqrt(np.sum((base_fscore - surrogate_fscore)**2))
        return avg_score


    def predict(self, X, prob_score=False):
        predict_values = self.model.predict(X)
        predict_prob_values = self.predict_proba(X) if prob_score is True else None
        return predict_values if predict_prob_values is None else predict_prob_values


    def get_params(self):
        pass


    def plot_global_decisions(self):
        pass


    def plot_local_decisions(self):
        pass

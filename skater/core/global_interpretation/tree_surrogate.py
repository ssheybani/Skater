from .base import BaseGlobalInterpretation
from sklearn.tree import DecisionTreeClassifier


class TreeSurrogate(BaseGlobalInterpretation):
    # Reference: http://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf
    def __init__(self, criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False,
                 feature_name=None):
        self.model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                       min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                                       class_weight=class_weight, presort=presort)
        self.feature_names = feature_name


    def fit(self, X, Y):
        self.model.fit(X, Y)


    def predict(self, prob_score=False):
        pass


    def get_params(self):
        pass


    def plot_global_decisions(self):
        pass


    def plot_local_decisions(self):
        pass
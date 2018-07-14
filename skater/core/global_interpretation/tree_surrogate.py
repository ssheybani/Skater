from .base import BaseGlobalInterpretation
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from skater.model.base import ModelType
from skater.core.visualizer.tree_visualizer import plot_tree



class TreeSurrogate(BaseGlobalInterpretation):
    # Reference: http://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf
    def __init__(self):
        super(TreeSurrogate, self).__init__()
        self.__model = None
        self.feature_names = None
        self.class_names = None
        self.impurity_threshold = 0.01
        self.criterion_types = {'classification': {'criterion': ['gini', 'entropy']},
                                'regression': {'criterion': ['mse', 'friedman_mse', 'mae']}
                                }
        self.splitter_types = ['best', 'random']


    def apply(self, estimator_type='classification', criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, seed=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, class_names=None,
              presort=False, feature_name=None, impurity_threshold=0.01):

        # TODO validate the parameters based on estimator type
        if estimator_type == 'classification':
            self.__model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                  max_features=max_features, random_state=seed,
                                                  max_leaf_nodes=max_leaf_nodes,
                                                  min_impurity_decrease=min_impurity_decrease,
                                                  min_impurity_split=min_impurity_split,
                                                  class_weight=class_weight, presort=presort)
        else:
            self.__model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=None,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf,
                                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                 max_features=max_features,
                                                 random_state=seed, max_leaf_nodes=max_leaf_nodes,
                                                 min_impurity_decrease=min_impurity_decrease,
                                                 min_impurity_split=min_impurity_split, presort=presort)

            self.feature_names = feature_name
            self.impurity_threshold = impurity_threshold
            self.class_names = class_names


    def learn(self, X, Y, original_y, cv=True, n_iter_search=10, param_grid=None):
        if cv is False:
            self.__model.fit(X, Y)
        else:
            # apply randomized cross validation
            default_grid = {
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 5, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_leaf_nodes": [2, 4, 6]
            }
            search_space = param_grid if param_grid is not None else default_grid
            # Default scoring Function used by RandomizedSearchCV
            # Classification: accuracy
            # Regression: r2_score
            # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
            # References for pruning:
            # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
            # Using Randomize Search here to prune the trees for readability
            random_search_estimator = RandomizedSearchCV(estimator=self.__model, param_distributions=search_space,
                                                         n_iter=n_iter_search)
            random_search_estimator.fit(X, Y)
            self.__model = random_search_estimator.best_estimator_
        y_hat_surrogate = self.predict(X)

        model_inst = ModelType(model_type='classifier')
        scorer = model_inst.scorers.get_scorer_function(scorer_type='f1')
        # TODO This should be abstracted by the model scorer factory
        metric_score = scorer(original_y, Y)
        surrogate_metric_score = scorer(Y, y_hat_surrogate)
        # Check on the length of any of the metric to determine the number of classes
        # if all is selected then compare against all metrics
        fidelity_score = np.sqrt(np.sum((metric_score - surrogate_metric_score)**2))
        return fidelity_score


    @property
    def estimator(self):
        return self.__model


    def predict(self, X, prob_score=False):
        predict_values = self.__model.predict(X)
        predict_prob_values = self.predict_proba(X) if prob_score is True else None
        return predict_values if predict_prob_values is None else predict_prob_values


    def get_params(self):
        pass


    def plot_global_decisions(self, model, colors=None, enable_node_id=True, random_state=5,
                              persist=True, file_name="interpretable_tree.png"):
        g = plot_tree(model, feature_names=self.feature_names, color_list=colors, class_names=self.class_name,
                      enable_node_id=enable_node_id, seed=random_state)
        f_name = "interpretable_tree.png" if file_name is None else file_name

        if persist is True:
            g.write_png(f_name)
            return 0
        else:
            return g


    def plot_local_decisions(self):
        pass

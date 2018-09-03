from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from skater.model.base import ModelType
from skater.core.visualizer.tree_visualizer import plot_tree, tree_to_text

from skater.util.logger import build_logger
from skater.util.logger import _INFO
from skater.util import exceptions

logger = build_logger(_INFO, __name__)


class TreeSurrogate(object):
    """ :: Experimental :: The implementation is currently experimental and might change in future.
    The idea of using TreeSurrogates as means for explaining a model's(Oracle or the base model)
    learned decision policies(for inductive learning tasks) is inspired by the work of Mark W. Craven
    described as the TREPAN algorithm. In this explanation learning hypothesis, the base estimator(Oracle)
    could be any form of supervised learning predictive models. The explanations are approximated using
    DecisionTrees(both for Classification/Regression) by learning decision boundaries similar to that learned by
    the Oracle(predictions from the base model are used for learning the DecisionTree representation).
    The implementation also generates a fidelity score to quantify tree based surrogate model's
    approximation to the Oracle. Ideally, the score should be 0 for truthful explanation both globally and locally.

    Parameters
    ----------
    estimator_type='classifier'
    splitter='best'
    max_depth=None
    min_samples_split=2
    min_samples_leaf=1
    min_weight_fraction_leaf=0.0
    max_features=None, seed=None
    max_leaf_nodes=None
    min_impurity_decrease=0.0
    min_impurity_split=None
    class_weight=None
    class_names=None
    presort=False
    feature_names=None
    impurity_threshold=0.01
    log_level=_INFO


    References
    ----------
    .. [1] Mark W. Craven(1996) EXTRACTING COMPREHENSIBLE MODELS FROM TRAINED NEURAL NETWORKS
           (http://ftp.cs.wisc.edu/machine-learning/shavlik-group/craven.thesis.pdf)
    .. [2] Mark W. Craven and Jude W. Shavlik(NIPS, 96). Extracting Thee-Structured Representations of Thained Networks
           (https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
    """
    __name__ = "TreeSurrogate"
    
    def __init__(self, estimator_type='classifier', splitter='best', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, seed=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced", class_names=None,
                 presort=False, feature_names=None, impurity_threshold=0.01, log_level=_INFO):
        self.logger = build_logger(log_level, __name__)
        self.__model = None
        self.__model_type = None

        self.feature_names = feature_names
        self.class_names = class_names
        self.impurity_threshold = impurity_threshold
        self.criterion_types = {'classifier': {'criterion': ['gini', 'entropy']},
                                'regressor': {'criterion': ['mse', 'friedman_mse', 'mae']}}
        self.splitter_types = ['best', 'random']
        self.splitter = splitter if any(splitter in item for item in self.splitter_types) else 'best'
        self.seed = seed
        self.__model_type = estimator_type
        # TODO validate the parameters based on estimator type
        if estimator_type == 'classifier':
            self.__model = DecisionTreeClassifier(splitter=self.splitter, max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                  max_features=max_features, random_state=seed,
                                                  max_leaf_nodes=max_leaf_nodes,
                                                  min_impurity_decrease=min_impurity_decrease,
                                                  min_impurity_split=min_impurity_split,
                                                  class_weight=class_weight, presort=presort)
        elif estimator_type == 'regressor':
            self.__model = DecisionTreeRegressor(splitter=self.splitter, max_depth=None,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf,
                                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                 max_features=max_features,
                                                 random_state=seed, max_leaf_nodes=max_leaf_nodes,
                                                 min_impurity_decrease=min_impurity_decrease,
                                                 min_impurity_split=min_impurity_split, presort=presort)
        else:
            raise exceptions.ModelError("Model type not supported. Supported options types{'classifier', 'regressor'}")


    def _post_pruning(self, X, Y, scorer_type, impurity_threshold, needs_prob=False):
        pred_func = lambda prob: self.__model.predict(X) if prob is False else self.__model.predict_proba(X)
        y_pred = pred_func(needs_prob)
        self.logger.info("Unique Labels in ground truth provided {}".format(np.unique(Y)))
        self.logger.info("Unique Labels in predictions generated {}".format(np.unique(y_pred)))

        model_inst = ModelType(model_type=self.__model_type, probability=False)
        scorer = model_inst.scorers.get_scorer_function(scorer_type=scorer_type)
        current_score = scorer(Y, y_pred, average='weighted')
        self.logger.info("current score {}".format(current_score))

        tree = self.__model.tree_
        no_of_nodes = tree.node_count
        tree_leaf = -1  # value to identify a leaf node in a tree

        removed_node_index = []
        for index in range(no_of_nodes):
            current_left, current_right = tree.children_left[index], tree.children_right[index]
            if tree.children_left[index] != tree_leaf or tree.children_right[index] != tree_leaf:
                tree.children_left[index], tree.children_right[index] = -1, -1
                new_score = scorer(Y, pred_func(needs_prob), average='weighted')

                if current_score - new_score <= impurity_threshold:
                    current_score = new_score
                    removed_node_index.append(index)
                    self.logger.info("Removed index {}".format(removed_node_index))
            else:
                tree.children_left[index], tree.children_right[index] = current_left, current_right
        self.logger.info("Node indexes removed {}".format(removed_node_index))


    def _pre_pruning(self, X, Y, cv=5, n_iter_search=10, n_jobs=1, param_grid=None):
        default_grid = {
            "criterion": self.criterion_types[self.__model_type]['criterion'],
            "max_depth": [2, 4, 6, 8],  # helps in reducing the depth of the tree
            "min_samples_leaf": [2, 4],  # restrict the number of samples in a leaf
            "max_leaf_nodes": [2, 4, 6, 8, 10]  # reduce the number of leaf nodes
        }
        search_space = param_grid if param_grid is not None else default_grid
        # Cost function aiming to optimize(Total Cost) = measure of fit + measure of complexity
        # References for pruning:
        # 1. http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # 2. https://www.coursera.org/lecture/ml-classification/optional-pruning-decision-trees-to-avoid-overfitting-qvf6v
        # Using Randomize Search here to prune the trees to improve readability without
        # comprising on model's performance
        random_search_estimator = RandomizedSearchCV(estimator=self.__model, cv=cv, param_distributions=search_space,
                                                     n_iter=n_iter_search, n_jobs=n_jobs, random_state=self.seed)
        # train a surrogate DT
        random_search_estimator.fit(X, Y)
        # access the best estimator
        self.__model = random_search_estimator.best_estimator_


    def learn(self, X, Y, oracle_y, prune='post', cv=5, n_iter_search=10,
              scorer_type='default', n_jobs=1, param_grid=None, impurity_threshold=0.01):
        """ Learn an approximate representation by constructing a Decision Tree based on the results retrieved by
        querying the Oracle(base model). Instances used for training should belong to the base learners instance space.

        Parameters
        ----------
        X:
        Y:
        oracle_y:
        prune: None, 'pre', 'post'
        cv: used only for 'pre-pruning' right now
        n_iter_search:
        param_grid:
        scorer_type:
        n_jobs:

        """
        if prune is None:
            self.__model.fit(X, Y)
        elif prune == 'pre':
            # apply randomized cross validation for pruning
            self._pre_pruning(X, Y, cv, n_iter_search, n_jobs, param_grid)
        else:
            # Since, this is post pruning, we first learn a model
            # and then try to prune the tree controling the model's score using the impurity_threshold
            self.__model.fit(X, Y)
            self._post_pruning(X, Y, scorer_type, impurity_threshold, needs_prob=False)
        y_hat_surrogate = self.__model.predict(X)
        self.logger.info('Done generating prediction using the surrogate, shape {}'.format(y_hat_surrogate.shape))

        model_inst = ModelType(model_type=self.__model_type)
        # Default metrics:
        # {Classification: if probability score used --> cross entropy(log-loss) else --> F1 score}
        # {Regression: Mean Absolute Error (MAE)}
        scorer = model_inst.scorers.get_scorer_function(scorer_type=scorer_type)
        oracle_score = scorer(oracle_y, Y)
        surrogate_metric_score = scorer(Y, y_hat_surrogate)
        self.logger.info('Done scoring ...')

        impurity_score = np.abs(surrogate_metric_score - oracle_score)
        if impurity_score > self.impurity_threshold:
            self.logger.warning('impurity score:{} of the surrogate model is higher than the impurity threshold: {}. '
                                'The higher the impurity score, lower is the fidelity/faithfulness '
                                'of the surrogate model')
        return impurity_score


    @property
    def estimator(self):
        """ Learned approximate surrogate estimator
        """
        return self.__model


    @property
    def estimator_type(self):
        """ Estimator type
        """
        return self.__model_type


    def predict(self, X, prob_score=False):
        """ Predict for input X
        """
        predict_values = self.__model.predict(X)
        predict_prob_values = self.__model.predict_proba(X) if prob_score is True else None
        return predict_values if predict_prob_values is None else predict_prob_values


    def plot_global_decisions(self, colors=None, enable_node_id=True, random_state=0, file_name="interpretable_tree.png",
                              show_img=False, fig_size=(20, 8)):
        """ Visualizes the decision nodes of the surrogate tree.
        """
        graph_inst = plot_tree(self.__model, self.__model_type, feature_names=self.feature_names, color_list=colors,
                               class_names=self.class_names, enable_node_id=enable_node_id, seed=random_state)
        f_name = "interpretable_tree.png" if file_name is None else file_name
        graph_inst.write_png(f_name)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise exceptions.MatplotlibUnavailableError("Matplotlib is required but unavailable on your system.")
        except RuntimeError:
            raise exceptions.MatplotlibDisplayError("Matplotlib unable to open display")

        if show_img:
            plt.rcParams["figure.figsize"] = fig_size
            img = plt.imread(f_name)
            if self.__model_type == 'regressor':
                cax = plt.imshow(img, cmap=plt.cm.get_cmap(graph_inst.get_colorscheme()))
                plt.colorbar(cax)
            else:
                plt.imshow(img)
        return graph_inst


    def decisions_as_txt(self, scope='global', X=None):
        tree_to_text(self.__model, self.feature_names, self.__model_type, scope, X)

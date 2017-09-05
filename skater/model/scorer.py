from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.multiclass import type_of_target
from abc import ABCMeta, abstractmethod

from .base import ModelType
from ..util.static_types import StaticTypes


class Scorer(object):
    """
    Base Class for all skater scoring functions.

    Any Scoring function must consume a model.

    Any scorer must determine the types of models that are compatible.

    """

    __metaclass__ = ABCMeta


    model_types = None
    prediction_types = None
    label_types = None

    @classmethod
    def check_params(cls):
        assert all([i in StaticTypes.model_types._valid_ for i in cls.model_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.prediction_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.label_types])

    @classmethod
    def check_model(cls, model):
        assert issubclass(model, ModelType), "Expected object of type " \
                                             "skater.model.ModelType, got {}".format(type(model))
        assert model.model_type in cls.model_types, "Scorer {0} not valid for models of type {1}, " \
                                                     "only {2}".format(cls,
                                                                       model.model_type,
                                                                       cls.model_types)
    @classmethod
    def __call__(cls, model, inputs, y_true):
        cls.check_model(model)
        y_predicted = model(inputs)
        cls.check_data(y_predicted, inputs)
        return cls._score(y_predicted, y_true)

    @staticmethod
    @abstractmethod
    def _score(model, inputs, y_true):
        """
        Private method for getting scores
        :param model:
        :param inputs:
        :param y_true:
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def check_data(y_predicted, y_true):
        pass



class RegressionScorer(Scorer):
    model_types = [StaticTypes.model_types.regressor]
    prediction_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]
    label_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]

    @staticmethod
    def check_data(y_predicted, y_true):
        assert hasattr(y_predicted, 'shape'), \
            'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), \
            'y_true must have a shape attribute'
        assert (len(y_predicted.shape)==1) or (y_predicted.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_predicted.shape)
        assert (len(y_true.shape)==1) or (y_true.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_true.shape)

class ClassifierProbaScorer(Scorer):

    """
    * predictions must be N x K matrix with N rows and K classes.
    * labels must be be N x K matrix with N rows and K classes.
    """

    model_types = [StaticTypes.model_types.classifier]
    prediction_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]
    label_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]

    @staticmethod
    def check_data(y_predicted, y_true):
        assert hasattr(y_predicted, 'shape'), 'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), 'y_true must have a shape attribute'

        assert y_true.shape == y_predicted.shape, "Labels and Predictions must have same shape for" \
                                                  "classification models. " \
                                                  "Labels Shape: {0} and " \
                                                  "Predictions Shape: {1}".format(y_true.shape, y_predicted.shape)

### Regression Scorers
class MeanSquaredError(RegressionScorer):
    @staticmethod
    def _score(model, inputs, y_true, sample_weights=None):
        return mean_squared_error(y_true, model(inputs), sample_weights=sample_weights)

class MeanAbsoluteError(RegressionScorer):
    @staticmethod
    def _score(model, inputs, y_true, sample_weights=None):
        return mean_absolute_error(y_true, model(inputs), sample_weights=sample_weights)

class RSquared(RegressionScorer):
    @staticmethod
    def _score(model, inputs, y_true, sample_weights=None):
        return r2_score(y_true, model(inputs), sample_weights=sample_weights)

class CrossEntropy(ClassifierProbaScorer):

    @staticmethod
    def _score(y_true, y_pred, sample_weights=None):
        """

        :param X: Dense X of probabilities, or binary indicator
        :param y:
        :param sample_weights:
        :return:
        """



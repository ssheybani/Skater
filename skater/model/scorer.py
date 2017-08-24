from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.utils.multiclass import type_of_target

from ..util.static_types import StaticTypes


class Scorer(object):
    """Skater's model scorer"""

    def __init__(self, predictions, labels, model_type):
        """
        Parameters
        ------------
        predictions: array type
        labels: array type
        model_instance: subtype of skater.util.static_types.ModelTypes
        """
        assert predictions.shape[0] == labels.shape[0], "labels and prediction shapes dont match" \
                                                        "predictions: {0}, labels: {1}".format(predictions.shape,
                                                                                               labels.shape)
        # continuous, binary, continuous multioutput, multiclass, multilabel-indicator
        # see https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/multiclass.py#L175
        # for details
        self.prediction_type = type_of_target(predictions)
        self.label_type = type_of_target(labels)
        self.labels = labels
        self.predictions = predictions
        self.model_type = model_type


    def score(self, sample_weights=None):
        if self.model_type == StaticTypes.model_types.classifier:
            return log_loss(self.labels, self.predictions, sample_weight=sample_weights)
        elif self.model_type == StaticTypes.model_types.regressor:
            return mean_absolute_error(self.labels, self.predictions, sample_weight=sample_weights)
        else:
            raise ValueError("Model type unknown")

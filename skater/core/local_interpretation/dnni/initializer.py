import sys, os
import numpy as np
import warnings
from scipy.misc import imread

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad
from collections import OrderedDict


ACTIVATIONS_OPS = [
    'Relu', 'Elu',  'Softplus', 'Tanh', 'Sigmoid']

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0


class Initializer(object):
    """
    Attribution method base class
    """
    def __init__(self, feature_coefficients, X_placeholder, xs, session, keras_learning_phase=None):
        self.feature_coefficients = feature_coefficients
        self.X_placeholder = X_placeholder
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase


    def session_run(self, feature_coefficients, xs):
        feed_dict = {}
        feed_dict[self.X_placeholder] = xs

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(feature_coefficients, feed_dict)


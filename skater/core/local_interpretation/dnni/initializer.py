# -*- coding: UTF-8 -*-

from tensorflow.python.ops import nn_grad, math_grad
import warnings
import numpy as np



class Initializer(object):
    """
    """
    activation_ops = ['Relu', 'Elu', 'Softplus', 'Tanh', 'Sigmoid']
    enabled_method_class = None
    grad_override_checkflag = 0

    def __init__(self, feature_wts, X, xs, session):
        self.feature_wts = feature_wts
        self.X = X
        self.xs = xs
        self.session = session


    def session_run(self, feature_wts, xs):
        feed_dict = {}
        feed_dict[self.X] = xs
        return self.session.run(feature_wts, feed_dict)


    @classmethod
    def original_grad(cls, op, grad):
        if op.type not in cls.activation_ops:
            warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
        op_name = '_{}Grad'.format(op.type)
        ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
        return ops_func(op, grad)

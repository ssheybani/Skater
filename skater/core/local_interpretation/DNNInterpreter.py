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


def original_grad(op, grad):

    if op.type not in ACTIVATIONS_OPS:
        warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
    op_name = '_{}Grad'.format(op.type)

    ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
    return ops_func(op, grad)




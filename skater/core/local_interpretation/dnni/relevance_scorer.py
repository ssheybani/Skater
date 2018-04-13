# -*- coding: UTF-8 -*-
from skater.core.local_interpretation.dnni.initializer import Initializer
import tensorflow as tf


class GradientBased(Initializer):
    """
    Base class for gradient-based relevance computation

    Reference
    - https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    """
    def default_relevance_score(self):
        return tf.gradients(self.feature_wts, self.X)


    def run(self):
        relevance_scores = self.default_relevance_score()
        results = self.session_run(relevance_scores, self.xs)
        return results[0]


    @classmethod
    def non_linear_grad(cls, op, grad):
        return cls.original_grad(op, grad)


class LRP(GradientBased):
    """ LRP Implementation computed using backpropagation by applying change rule on a modified gradient function.
    LRP could be implemented in different ways. This version implements the epsilon-LRP(Eq (58) as stated in [1]
    or Eq (2) in [2]. Epsilon acts as a numerical stabilizer. 

    Parameters
    __________


    Reference
    _________
    .. [1] Bach S, Binder A, Montavon G, Klauschen F, Müller K-R, Samek W (2015)
       On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.
       PLoS ONE 10(7): e0130140. https://doi.org/10.1371/journal.pone.0130140
    .. [2] Ancona M, Ceolini E, Öztireli C, Gross M:
           Towards better understanding of gradient-based attribution methods for Deep Neural Networks. ICLR, 2018
    """
    eps = None

    def __init__(self, feature_wts, X, xs, session, epsilon=1e-4):
        super(LRP, self).__init__(feature_wts, X, xs, session)
        assert epsilon > 0.0, 'LRP epsilon must be > 0'
        LRP.eps = epsilon


    def default_relevance_score(self):
        return [g * x for g, x in zip(
                tf.gradients(self.feature_wts, self.X), [self.X])]


    @classmethod
    def non_linear_grad(cls, op, grad):
        op_out = op.outputs[0]
        op_in = op.inputs[0]
        stabilizer_epsilon = cls.eps * tf.where(op_in >= 0, tf.ones_like(op_in, dtype=tf.float32),
                                                -1 * tf.ones_like(op_in, dtype=tf.float32))
        op_in += stabilizer_epsilon
        return grad * op_out / op_in


class IntegratedGradients(GradientBased):

    def __init__(self, T, X, xs, session, steps=100, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, xs, session)
        self.steps = steps
        self.baseline = baseline


    def run(self):
        # Apply the baseline specified or use default
        self._set_check_baseline()

        relevance_scores = self.default_relevance_score()
        pass

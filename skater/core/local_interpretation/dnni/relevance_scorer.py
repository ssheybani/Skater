from skater.core.local_interpretation.dnni.initializer import Initializer
import tensorflow as tf


class GradientBased(Initializer):
    """
    Base class for gradient-based relevance computation
    """
    def compute_gradients(self):
        return tf.gradients(self.feature_coefficients, self.X)


    def run(self):
        relevance_scores = self.compute_gradients()
        results =  self.session_run(relevance_scores, self.xs)
        return results[0]


    @classmethod
    def non_linear_grad(cls, op, grad):
        return original_grad(op, grad)


class LRP(GradientBased):
    eps = None
    def __init__(self, feature_coefficients, X, xs, session, keras_learning_phase, epsilon=1e-4):
        super(LRP, self).__init__(feature_coefficients, X, xs, session, keras_learning_phase)
        assert epsilon > 0.0, 'LRP epsilon must be > 0'
        LRP.eps = epsilon


    def compute_gradients(self):
        return [g * x for g, x in zip(
                tf.gradients(self.feature_coefficients, self.X), [self.X])]


    @classmethod
    def non_linear_grad(cls, op, grad):
        op_out = op.outputs[0]
        op_in = op.inputs[0]
        stabilizer_epsilon = cls.eps * tf.where(op_in >= 0, tf.ones_like(op_in, dtype=tf.float32),
                                            -1* tf.ones_like(op_in, dtype=tf.float32))
        op_in += stabilizer_epsilon
        return grad * op_out / op_in
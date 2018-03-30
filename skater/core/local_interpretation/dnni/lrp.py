from skater.core.local_interpretation.dnni.initializer import Initializer
import tensorflow as tf

def original_grad(op, grad):

    if op.type not in ACTIVATIONS_OPS:
        warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
    op_name = '_{}Grad'.format(op.type)

    ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
    return ops_func(op, grad)


class GradientBased(Initializer):
    """
    Base class for gradient-based relevance computation
    """
    def compute_gradients(self):
        return tf.gradients(self.T, self.X)

    def run(self):
        relevance_scores = self.compute_gradients()
        results =  self.session_run(relevance_scores, self.xs)
        return results[0]

    @classmethod
    def non_linearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class LRP(GradientBased):

    def __init__(self, feature_coefficients, X, xs, session, keras_learning_phase, epsilon=1e-4):
        super(LRP, self).__init__(feature_coefficients, X, xs, session, keras_learning_phase)
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        self.eps = epsilon

    def compute_gradients(self):
        return [g * x for g, x in zip(
                tf.gradients(self.feature_coefficients, self.X), [self.X])]

    @classmethod
    def non_linear_grad_override(cls, op, grad):
        output = op.outputs[0]
        zs = op.inputs[0]
        stablizer_epsilon = cls.eps * (tf.where(zs >= 0, tf.ones_like(zs, dtype=tf.float32), -1
                                               * tf.ones_like(zs, dtype=tf.float32)))
        zs += stablizer_epsilon

        return grad * output / zs
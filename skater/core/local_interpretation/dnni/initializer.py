from tensorflow.python.ops import nn_grad, math_grad
import warnings



class Initializer(object):
    """
    """
    activation_ops = ['Relu', 'Elu',  'Softplus', 'Tanh', 'Sigmoid']
    enabled_method_class = None
    grad_override_checkflag = 0

    def __init__(self, feature_wts, X, xs, session, keras_learning_phase=None):
        self.feature_wts = feature_wts
        self.X = X
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase


    def session_run(self, feature_wts, xs):
        feed_dict = {}
        feed_dict[self.X] = xs

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(feature_wts, feed_dict)


    def original_grad(self, op, grad):
        if op.type not in self.activation_ops:
            warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
        op_name = '_{}Grad'.format(op.type)
        ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
        return ops_func(op, grad)





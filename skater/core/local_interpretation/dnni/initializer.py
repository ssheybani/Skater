from skater.core.local_interpretation.dnni.lrp import LRP
from collections import OrderedDict

ACTIVATIONS_OPS = [
    'Relu', 'Elu',  'Softplus', 'Tanh', 'Sigmoid']

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0

relevance_scorer_type = OrderedDict({
    'elrp': (LRP, 0)
})


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


    def original_grad(self, op, grad):
        if op.type not in ACTIVATIONS_OPS:
            warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
        op_name = '_{}Grad'.format(op.type)

        ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
        return ops_func(op, grad)


    @staticmethod
    def get_override_map():
        return dict((ops_item, 'DeepInterpretGrad') for ops_item in ACTIVATIONS_OPS)





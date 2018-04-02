from skater.core.local_interpretation.dnni.relevance_scorer import GradientBased
from skater.core.local_interpretation.dnni.relevance_scorer import LRP
from skater.core.local_interpretation.dnni.initializer import Initializer
from skater.core.local_interpretation.dnni.initializer import ACTIVATIONS_OPS

from tensorflow.python.framework import ops
import tensorflow as tf

from collections import OrderedDict
import warnings
from skater.util.logger import build_logger
from skater.util.logger import _INFO

@ops.RegisterGradient("DeepInterpretGrad")
def deep_interpreter_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBased):
        return _ENABLED_METHOD_CLASS.non_linear_grad(op, grad)
    else:
        return Initializer.original_grad(op, grad)


class DeepInterpreter(object):

    def __init__(self, graph=None, session=tf.get_default_session(), log_level=_INFO):
        self.logger = build_logger(log_level, __name__)
        self.relevance_type = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self._get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        self.relevance_scorer_type = OrderedDict({
            'elrp': LRP })
        if self.session is None:
            raise RuntimeError('Relevant session not retrieved')


    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self


    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False


    @staticmethod
    def _get_gradient_override_map():
        return dict((ops_item, 'DeepInterpretGrad') for ops_item in ACTIVATIONS_OPS)


    def explain(self, relevance_type, T, X, xs, **kwargs):
        if not self.context_on:
            raise RuntimeError('explain can be invoked only within a DeepInterpreter context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        self.relevance_type = relevance_type
        self.logger.info("all supported relevancy scorers {}".format(relevance_scorer_type))

        relevance_type_class = self.relevance_scorer_type[self.relevance_type] \
                                        if self.relevance_type in self.relevance_scorer_type.keys() \
                                        else None
        if relevance_type_class is None:
            raise RuntimeError('Method type not found in {}'.formatlist(self.relevance_scorer_type.keys()))
        self.logger.info('DeepInterpreter: executing relevance type class {}'.format(self.relevance_type_class))

        _GRAD_OVERRIDE_CHECKFLAG = 0
        _ENABLED_METHOD_CLASS = relevance_type_class

        method = _ENABLED_METHOD_CLASS(T, X, xs, self.session, self.keras_phase_placeholder, **kwargs)
        result = method.run()
        if issubclass(_ENABLED_METHOD_CLASS, GradientBased) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('Results may not reliable, as default gradient seems to have been used. Be careful...')

        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return result
from skater.core.local_interpretation.dnni.lrp import GradientBased
from skater.core.local_interpretation.dnni.initializer import relevance_scorer_type
from skater.core.local_interpretation.dnni.initializer import Initializer

from tensorflow.python.framework import ops
import tensorflow as tf
import warnings

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

    def __init__(self, graph=None, session=tf.get_default_session()):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
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


    def explain(self, method, T, X, xs, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        self.method = method

        method_class, method_flag = relevance_scorer_type[self.method] if self.method in relevance_scorer_type \
                                        else None, None
        if method_class and method_flag is None:
            raise RuntimeError('Method must be in %s' % list(relevance_scorer_type.keys()))

        print('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        _GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X, xs, self.session, self.keras_phase_placeholder, **kwargs)
        result = method.run()
        if issubclass(_ENABLED_METHOD_CLASS, GradientBased) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepInterpreter detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepInterpreter context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return result
# -*- coding: UTF-8 -*-
from skater.core.local_interpretation.dnni.relevance_scorer import GradientBased
from skater.core.local_interpretation.dnni.relevance_scorer import LRP
from skater.core.local_interpretation.dnni.initializer import Initializer


from tensorflow.python.framework import ops
import tensorflow as tf

from collections import OrderedDict
import warnings
from skater.util.logger import build_logger
from skater.util.logger import _INFO


@ops.RegisterGradient("DeepInterpretGrad")
def deep_interpreter_grad(op, grad):
    Initializer.grad_override_checkflag = 1
    if Initializer.enabled_method_class is not None \
            and issubclass(Initializer.enabled_method_class, GradientBased):
        return Initializer.enabled_method_class.non_linear_grad(op, grad)
    else:
        return Initializer.original_grad(op, grad)


class DeepInterpreter(object):
    """
    :: Experimental :: The implementation is currently experimental and might change in future
    Interpreter for inferring Deep Learning Models. Given a trained NN model and an input vector X, DeepInterpreter
    is responsible for providing relevance scores w.r.t a target class to analyze most contributing features driving
    an estimator's decision for or against the respective class

    Parameters
    __________
    graph
    session
    log_level

    Reference
    ---------
    .. [1] Marco Ancona, Enea Ceolini, Cengiz Öztireli, Markus Gross:
           Towards better understanding of gradient-based attribution methods for Deep Neural Networks. ICLR, 2018
    .. [2] https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py

    """

    def __init__(self, graph=None, session=tf.get_default_session(), log_level=_INFO):
        self.logger = build_logger(log_level, __name__)
        self.relevance_type = None
        self.batch_size = None
        self.session = session
        if self.session is None:
            raise RuntimeError('Relevant session not retrieved')
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self._get_gradient_override_map())
        self.context_on = False
        self.__supported_relevance_type_dict = OrderedDict({
            'elrp': LRP})


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
        return dict((ops_item, 'DeepInterpretGrad') for ops_item in Initializer.activation_ops)


    def explain(self, relevance_type, T, X, xs, **kwargs):
        if not self.context_on:
            raise RuntimeError('explain can be invoked only within a DeepInterpreter context.')
        self.relevance_type = relevance_type
        self.logger.info("all supported relevancy scorers {}".format(self.__supported_relevance_type_dict))

        relevance_type_class = self.__supported_relevance_type_dict[self.relevance_type] \
            if self.relevance_type in self.__supported_relevance_type_dict.keys() else None
        if relevance_type_class is None:
            raise RuntimeError('Method type not found in {}'.formatlist(self.__supported_relevance_type_dict.keys()))
        self.logger.info('DeepInterpreter: executing relevance type class {}'.format(relevance_type_class))

        Initializer.grad_override_checkflag = 0
        Initializer.enabled_method_class = relevance_type_class

        method = Initializer.enabled_method_class(T, X, xs, self.session, **kwargs)
        self.logger.info('DeepInterpreter: executing method {}'.format(method))

        result = method.run()
        if issubclass(Initializer.enabled_method_class, GradientBased) and Initializer.grad_override_checkflag == 0:
            warnings.warn('Results may not reliable: As default gradient seems to have been used. '
                          'or you might have forgotten to create the graph within the DeepInterpreter context. '
                          'Be careful...')

        Initializer.enabled_method_class = None
        Initializer.grad_override_checkflag = 0
        return result

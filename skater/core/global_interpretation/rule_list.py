# Reference: https://github.com/Hongyuy/sbrl-python-wrapper/blob/master/sbrl/C_sbrl.py
# Referencing the work done by Professor Cynthia/Hongyuy with modification

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


class SBRL():
    def __init__(self):
        self.r_sbrl = importr('sbrl')
        self.model = None


    def fit(self, X, y, **kwargs):
        """
        Parameters:
            X: pandas.DataFrame object, used to fit the model.
                 It must not have a column named 'label'
            y: pandas.Series, 1-D array to store ground truth labels
            **kwarg: key word arguments including the following keys:
            'iters':    default 30000
            'pos_sign': default "1"
            'neg_sign': default "0"
            'rule_minlen': default 1
            'rule_maxlen': default 1
            'minsupport_pos': 0.1
            'minsupport_neg': 0.1
            'lambda':   default 10
            'eta':  default 1
            'nchain': default 10

        """
        data = X.assign(label=y)
        self.model = self.r_sbrl.sbrl(pandas2ri.py2ri(data), **kwargs)


    def predict(self, X, type='prob'):
        '''
            `X`  an instance of pandas.DataFrame object, representing the data to be making predictions on.
            `type`  whether the prediction is discrete or probabilistic.
            return a numpy.ndarray of shape (#datapoints, 2), the probability for each observations
        '''
        results = self.r_sbrl.predict_sbrl(self.model, pandas2ri.py2ri(X))
        return np.asarray(map(pandas2ri.ri2py, results)).transpose()


    def print_model(self):
        self.r_sbrl.print_sbrl(self.model)
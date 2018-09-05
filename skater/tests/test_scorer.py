from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection._split import train_test_split
import unittest

from skater.model import InMemoryModel


class TestScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_moons(1000, noise=0.5)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y)
        cls.classifier_est = DecisionTreeClassifier(max_depth=5)
        cls.classifier_est.fit(cls.X_train, cls.y_train)


    def test_compute_default_scores(self):
        # For classification default scorer is weighted F1-score
        model_inst = InMemoryModel(self.classifier_est.predict_proba, examples=self.X_train, model_type='classifier')
        scorer = model_inst.scorers.get_scorer_function(scorer_type='default')
        y_hat = self.classifier_est.predict(self.X_test)
        value = scorer(self.y_test, y_hat, average='weighted')
        self.assertEqual(scorer.__name__ == 'f1-score', True)
        self.assertEquals(value > 0, True)

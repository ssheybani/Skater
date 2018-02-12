import pandas as pd
import unittest
import sys

if sys.version_info >= (3, 5):
    from skater.core.global_interpretation.interpretable_models.rule_lists import BayesianRuleLists
    from skater.core.validation import compute_validation_curve


@unittest.skipIf(sys.version_info < (3, 5), "SBRL supported only for python 3.5/3.6")
class TestRuleList(unittest.TestCase):

    def setUp(self):
        self.sbrl_inst = BayesianRuleLists()
        self.input_data = pd.read_csv('skater/tests/data/sample_data.csv')
        # data transformation and cleaning ...
        self.input_data["Sex"] = self.input_data["Sex"].astype('category')
        self.input_data["Sex_Encoded"] = self.input_data["Sex"].cat.codes
        self.input_data["Embarked"] = self.input_data["Embarked"].astype('category')
        self.input_data["Embarked_Encoded"] = self.input_data["Embarked"].cat.codes
        self.input_data = self.input_data.drop(['Ticket', 'Cabin', 'Name', 'Sex', 'Embarked'], axis=1)
        # Remove NaN values
        self.input_data = self.input_data.dropna()
        self.y = self.input_data['Survived']
        self.input_data = self.input_data.drop(['Survived'], axis=1)


    def test_model_build(self):
        self.sbrl_inst.fit(self.input_data, self.y)
        result_score = self.sbrl_inst.predict_prob(self.input_data)
        self.assertEquals(result_score.shape, (77, 2))


    def test_model_output(self):
        self.sbrl_inst.fit(self.input_data, self.y)
        result = self.sbrl_inst.access_learned_rules('23:25')
        self.assertEquals(len(result), 2)


    def test_validation(self):
        param_range = [1, 3]
        train_scores, test_scores = compute_validation_curve(self.sbrl_inst, n_folds=2, x=self.input_data, y=self.y,
                                                             param_name="rule_minlen", param_range=param_range)
        self.assertEquals(train_scores.shape[0], 2)
        self.assertEquals(test_scores.shape[0], 2)


if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
        unittest.TextTestRunner(verbosity=2).run(suite)

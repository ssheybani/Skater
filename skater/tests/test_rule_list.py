import unittest
import pandas as pd
from skater.core.global_interpretation.rule_list import SBRL


class TestRuleList(unittest.TestCase):

    def setUp(self):
        self.sbrl_inst = SBRL()
        self.kwargs = {"iters":50000, "pos_sign":1, "neg_sign":0, "rule_minlen":3,
                  "rule_maxlen":6, "minsupport_pos":0.10, "minsupport_neg":0.10, "eta":1.0, "nchain":40, "lambda":12}
        self.input_data = pd.read_csv('skater/tests/data/sample_data.csv')
        # data transformation and cleaning ...
        self.input_data["Sex"] = self.input_data["Sex"].astype('category')
        self.input_data["Sex_Encoded"] = self.input_data["Sex"].cat.codes
        self.input_data["Embarked"] = self.input_data["Embarked"].astype('category')
        self.input_data["Embarked_Encoded"] = self.input_data["Embarked"].cat.codes
        self.input_data = self.input_data.drop(['Ticket','Cabin', 'Name', 'Sex', 'Embarked'], axis=1)
        # Remove NaN values
        self.input_data = self.input_data.dropna()
        self.y = self.input_data['Survived']
        self.input_data = self.input_data.drop(['Survived'], axis=1)


    def test_model_build(self):
        self.sbrl_inst.fit(self.input_data, self.y, **self.kwargs)
        result_score = self.sbrl_inst.predict_prob(self.input_data)
        self.assertEquals(result_score.shape, (77, 2))


    def test_model_output(self):
        self.sbrl_inst.fit(self.input_data, self.y, **self.kwargs)
        result = self.sbrl_inst.access_learned_rules('23:25')
        self.assertEquals(len(result), 2)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
    unittest.TextTestRunner(verbosity=2).run(suite)
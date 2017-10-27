import unittest
import pandas as pd
from skater.core.global_interpretation.rule_list import SBRL


class TestRuleList(unittest.TestCase):

    def setup(self):
        self.sbrl_inst = SBRL()
        self.input_data = pd.read_csv('data/sample.csv')
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
        self.sbrl_inst.fit(self.input_data, self.y)
        result_score = self.sbrl_inst.predict_prob(self.input_data)
        print(result_score.shape)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
    unittest.TextTestRunner(verbosity=2).run(suite)
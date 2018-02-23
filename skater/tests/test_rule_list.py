import pandas as pd
import unittest
import sys


if sys.version_info >= (3, 5):
    from skater.core.global_interpretation.interpretable_models.brlc \
        import BRLC
    from skater.core.validation import compute_validation_curve


# When in Dev mode, a consistent mode to validate test, that would keep track of weird segmentation fault is using gdb
# (GNU Debugger). This is a temporary workaround. Follow the below mentioned steps
# 1. sudo apt install gdb
# 2. gdb python
# 3. r skater/tests/test_rule_list.py
# 4. Result: All the tests should succeed, and if it fails it will point out the trace for failure
@unittest.skipIf(sys.version_info < (3, 5), "only Bayesian Rule List supported only for python 3.5/3.6")
class TestRuleList(unittest.TestCase):

    def setUp(self):
        self.sbrl_inst = BRLC(min_rule_len=1, max_rule_len=2, iterations=10000, n_chains=3, lambda_=12)
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


    def test_discretizer(self):
        new_df = self.sbrl_inst.discretizer(self.input_data, column_list=["Age"])
        self.assertEquals(new_df["Age_q_label"].shape[0] > 0, True)


    def test_model_build(self):
        self.sbrl_inst.fit(self.input_data, self.y, undiscretize_feature_list=["PassengerId", "Pclass",
                                                                               "SibSp", "Parch", "Sex_Encoded",
                                                                               "Embarked_Encoded"])

        new_data = self.sbrl_inst.discretizer(self.input_data, column_list=["Age", "Fare"])
        result_score = self.sbrl_inst.predict_prob(new_data)
        self.assertEquals(result_score.shape, (77, 2))


    @unittest.skip("Disabling these tests as running them could be computationally expensive. But, recommended to run "
                   "these tests during development")
    def test_model_predict(self):
        self.sbrl_inst.predict(self.input_data)


    @unittest.skip("Disabling these tests as running them could be computationally expensive. But, recommended to run "
                   "these tests during development")
    def test_model_output(self):
        self.sbrl_inst.fit(self.input_data, self.y)
        result = self.sbrl_inst.access_learned_rules('23:25')
        self.assertEquals(len(result), 2)


    @unittest.skip("Disabling these tests as running them could be computationally expensive. But, recommended to run "
                   "these tests during development")
    def test_validation(self):
        param_range = [3, 4]
        train_scores, test_scores = compute_validation_curve(self.sbrl_inst, n_folds=2, x=self.input_data, y=self.y,
                                                             param_name="rule_minlen", param_range=param_range)
        self.assertEquals(train_scores.shape[0], 2)
        self.assertEquals(test_scores.shape[0], 2)


if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
        unittest.TextTestRunner(verbosity=2).run(suite)
